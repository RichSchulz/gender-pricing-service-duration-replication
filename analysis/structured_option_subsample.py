#!/usr/bin/env python3
"""Diagnose structured option menus using raw Treatwell menu metadata.

This script:
1. rebuilds the current adult baseline sample used in the paper;
2. matches each listing to its raw ``menuHighlight`` payload;
3. expands matched highlights to the option-group / option level;
4. classifies option structures as service structure, staff tier, or generic;
5. flags title-based bundle/conflict issues as a secondary audit layer;
6. prints a summary to stdout; and
7. writes a single audit CSV to ``data/audits/structured_option_audit.csv``.
"""

from __future__ import annotations

import json
import re
from collections import Counter

import pandas as pd

from generate_paper_tables import DATA_DIR, normalize, preprocess_adults, restrict_unisex_salons


RAW_ADULT_DATA = DATA_DIR / "snapshots" / "treatwell-all-2025-06-02.csv"
OUTPUT_FILE = DATA_DIR / "audits" / "structured_option_audit.csv"

KEY_COLS = [
    "id",
    "simpleCutName",
    "simpleCutSalePrice",
    "simpleCutDurationMin",
    "simpleCutDurationMax",
    "is_male",
    "is_female",
]

NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
MULTISPACE_PATTERN = re.compile(r"\s+")

TIME_PATTERNS = [
    ("time_15", re.compile(r"\b15\s*(?:min|mins|minutes|minuten|minuti)\b|\bquarter hour\b")),
    ("time_20", re.compile(r"\b20\s*(?:min|mins|minutes|minuten|minuti)\b")),
    ("time_25", re.compile(r"\b25\s*(?:min|mins|minutes|minuten|minuti)\b")),
    ("time_30", re.compile(r"\b30\s*(?:min|mins|minutes|minuten|minuti)\b|\bhalf hour\b")),
    ("time_40", re.compile(r"\b40\s*(?:min|mins|minutes|minuten|minuti)\b")),
    ("time_45", re.compile(r"\b45\s*(?:min|mins|minutes|minuten|minuti)\b|\bthree quarter hour\b")),
    (
        "time_60",
        re.compile(
            r"\b60\s*(?:min|mins|minutes|minuten|minuti)\b|\b1\s*(?:hour|hr|stunde|ora)\b"
        ),
    ),
]

LENGTH_PATTERNS = [
    ("length_short", re.compile(r"\b(?:short hair|short|kurz|kurze haare|kurzes haar|cheveux courts|capelli corti|kort haar|pelo corto)\b")),
    ("length_medium", re.compile(r"\b(?:medium hair|medium|mittel|mittellang|medio|moyen)\b")),
    ("length_long", re.compile(r"\b(?:long hair|long|lang|lange haare|cheveux longs|capelli lunghi|lang haar|pelo largo)\b")),
]

STAFF_TIER_TERMS = [
    "stylist",
    "master stylist",
    "senior stylist",
    "junior stylist",
    "creative director",
    "salon director",
    "art director",
    "director",
    "manager",
    "store manager",
    "technician",
    "top stylist",
    "top-stylist",
    "collaboratori",
    "altri collaboratori",
]

BUNDLE_MARKERS = {
    "bundle_beard": [
        "beard",
        "bart",
        "barba",
        "rasur",
        "shave",
        "shaving",
        "moustache",
        "mustache",
        "grooming",
    ],
    "bundle_color": [
        "color",
        "colour",
        "tint",
        "bleach",
        "highlight",
        "balayage",
        "meches",
        "strahnen",
        "blond",
        "ombr",
        "patine",
        "gloss",
    ],
    "bundle_treatment": [
        "keratin",
        "mask",
        "repair",
        "treatment",
        "pflege",
        "soin",
        "tratamiento",
        "botox",
        "olaplex",
        "plex",
    ],
    "bundle_styling": [
        "styling",
        "coiffage",
        "stylen",
        "style",
        "finish",
    ],
    "bundle_blowdry": [
        "blow dry",
        "blowdry",
        "fohnen",
        "fohnen",
        "brushing",
        "sechage",
        "piega",
    ],
    "bundle_premium": [
        "premium",
        "deluxe",
        "luxury",
        "signature",
        "makeover",
        "forfait",
        "package",
        "paket",
    ],
    "bundle_student": [
        "student",
        "junior",
        "senior",
        "schuler",
        "etudiant",
        "idoso",
    ],
    "bundle_restyle": [
        "restyle",
        "re style",
        "reshape",
        "new style",
        "style change",
        "cambio de look",
        "trasformazione",
    ],
    "bundle_consultation": [
        "consultation",
        "beratung",
        "diagnostic",
        "diagnosis",
        "consulenza",
    ],
}

CORE_BUNDLE_COLUMNS = [
    "bundle_beard",
    "bundle_color",
    "bundle_treatment",
    "bundle_premium",
    "bundle_student",
    "bundle_restyle",
    "bundle_consultation",
]

LENGTH_MARKER_TERMS = {
    "short": ["short", "kurz", "court", "corto", "curto", "kort"],
    "medium": ["medium", "mittel", "medio", "moyen", "middellang"],
    "long": ["long", "lang", "largo", "longue"],
}

WASH_MARKER_TERMS = {
    "wash": ["wash", "waschen", "shampoo", "shampoing", "lavado", "lavage", "lavaggio", "wassen"],
    "dry": ["dry", "trocken", "a sec", "a secco", "sin lavado", "zonder wassen"],
    "wet": ["wet", "nass", "humido", "umido"],
}

MACHINE_TERMS = ["machine", "maschinen", "maschinenschnitt", "maquina", "macchina", "clipper", "tondeuse", "macchinetta"]


def normalize_text(text: object) -> str:
    value = normalize(text)
    value = NON_ALNUM_PATTERN.sub(" ", value)
    return MULTISPACE_PATTERN.sub(" ", value).strip()


def contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def add_match_sequence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["match_seq"] = out.groupby(KEY_COLS).cumcount()
    return out


def title_length_marker(text: str) -> str:
    for label, terms in LENGTH_MARKER_TERMS.items():
        if contains_any(text, [normalize_text(term) for term in terms]):
            return label
    return ""


def title_wash_marker(text: str) -> str:
    for label, terms in WASH_MARKER_TERMS.items():
        if contains_any(text, [normalize_text(term) for term in terms]):
            return label
    return ""


def title_machine_marker(text: str) -> bool:
    return contains_any(text, [normalize_text(term) for term in MACHINE_TERMS])


def detect_family(text: str) -> str | None:
    if not text:
        return None
    for family, pattern in TIME_PATTERNS:
        if pattern.search(text):
            return family
    for family, pattern in LENGTH_PATTERNS:
        if pattern.search(text):
            return family
    return None


def is_staff_tier(text: str) -> bool:
    if not text:
        return False
    return contains_any(text, [normalize_text(term) for term in STAFF_TIER_TERMS])


def classify_structure(group_norm: str, option_norm: str, title_norm: str) -> tuple[str, str | None, str, str]:
    group_family = detect_family(group_norm)
    option_family = detect_family(option_norm)
    group_staff = is_staff_tier(group_norm)
    option_staff = is_staff_tier(option_norm)

    if group_staff or option_staff:
        if group_family and option_staff:
            return "staff_tier", group_family, "group_and_option", "high"
        if group_family:
            return "staff_tier", group_family, "group", "medium"
        if option_family:
            return "staff_tier", option_family, "option", "medium"
        source = "option" if option_staff else "group"
        return "staff_tier", None, source, "medium"

    if group_family and option_family and group_family == option_family:
        return "service_structure", group_family, "group_and_option", "high"
    if group_family:
        return "service_structure", group_family, "group", "high"
    if option_family:
        return "service_structure", option_family, "option", "medium"

    group_is_generic = not group_norm or group_norm == title_norm
    option_is_generic = not option_norm or option_norm == title_norm or option_norm == group_norm
    if group_is_generic and option_is_generic:
        return "generic", None, "group_and_option", "low"
    if group_is_generic:
        return "generic", None, "option", "low"
    if option_is_generic:
        return "generic", None, "group", "low"
    return "generic", None, "group_and_option", "low"


def parse_raw_option_rows() -> tuple[pd.DataFrame, dict[str, int]]:
    records: list[dict[str, object]] = []
    stats: Counter[str] = Counter()
    seen_keys: Counter[tuple[object, ...]] = Counter()

    for chunk in pd.read_csv(RAW_ADULT_DATA, low_memory=False, chunksize=1000):
        for row in chunk.itertuples(index=False):
            key_tuple = tuple(getattr(row, column) for column in KEY_COLS)
            match_seq = seen_keys[key_tuple]
            seen_keys[key_tuple] += 1
            base = {
                "id": row.id,
                "simpleCutName": row.simpleCutName,
                "simpleCutSalePrice": row.simpleCutSalePrice,
                "simpleCutDurationMin": row.simpleCutDurationMin,
                "simpleCutDurationMax": row.simpleCutDurationMax,
                "is_male": row.is_male,
                "is_female": row.is_female,
                "match_seq": match_seq,
                "menuHighlightName": None,
                "primaryTreatmentCategoryId": None,
                "treatmentCategoryIds": None,
                "optionGroupName": None,
                "optionGroupIndex": None,
                "optionName": None,
                "optionIndex": None,
                "optionDurationMinutes": None,
                "raw_match_found": False,
            }
            try:
                raw = json.loads(row.raw)
            except Exception:
                stats["json_parse_failed"] += 1
                records.append(base)
                continue

            menu = raw.get("data", {}).get("menuHighlights", []) or []
            if not menu:
                stats["missing_menu_highlights"] += 1

            matched_item = None
            for item in menu:
                data = item.get("data", {}) or {}
                if data.get("name") == row.simpleCutName:
                    matched_item = data
                    break

            if matched_item is None:
                stats["raw_name_match_missing"] += 1
                records.append(base)
                continue

            stats["raw_name_match_found"] += 1
            highlight_base = dict(base)
            highlight_base["raw_match_found"] = True
            highlight_base["menuHighlightName"] = matched_item.get("name")
            highlight_base["primaryTreatmentCategoryId"] = matched_item.get("primaryTreatmentCategoryId")
            category_ids = matched_item.get("treatmentCategoryIds")
            highlight_base["treatmentCategoryIds"] = (
                json.dumps(category_ids) if category_ids is not None else None
            )

            option_groups = matched_item.get("optionGroups") or []
            if not option_groups:
                stats["matched_without_option_groups"] += 1
                records.append(highlight_base)
                continue

            stats["matched_with_option_groups"] += 1
            for group_index, group in enumerate(option_groups):
                group_name = group.get("name")
                options = group.get("options") or []
                if not options:
                    stats["option_groups_without_options"] += 1
                    record = dict(highlight_base)
                    record["optionGroupName"] = group_name
                    record["optionGroupIndex"] = group_index
                    records.append(record)
                    continue

                stats["option_groups_with_options"] += 1
                for option_index, option in enumerate(options):
                    stats["option_rows"] += 1
                    record = dict(highlight_base)
                    record["optionGroupName"] = group_name
                    record["optionGroupIndex"] = group_index
                    record["optionName"] = option.get("name")
                    record["optionIndex"] = option_index
                    record["optionDurationMinutes"] = option.get("durationMinutes")
                    records.append(record)

    return pd.DataFrame.from_records(records), dict(stats)


def add_title_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["title_norm"] = out["simpleCutName"].map(normalize_text)
    out["title_stem"] = out["title_norm"]

    for column, terms in BUNDLE_MARKERS.items():
        normalized_terms = [normalize_text(term) for term in terms]
        out[column] = out["title_norm"].apply(lambda value: contains_any(value, normalized_terms))
    out["noncomparable_bundle_flag"] = out[CORE_BUNDLE_COLUMNS].any(axis=1)

    out["title_length_marker"] = out["title_norm"].map(title_length_marker)
    out["title_wash_marker"] = out["title_norm"].map(title_wash_marker)
    out["title_machine_marker"] = out["title_norm"].map(title_machine_marker)
    return out


def summarize_marker_conflicts(service_rows: pd.DataFrame) -> pd.DataFrame:
    def conflict_for_group(group: pd.DataFrame) -> pd.Series:
        male = group[group["is_male"] == True]
        female = group[group["is_female"] == True]

        male_lengths = {value for value in male["title_length_marker"] if value}
        female_lengths = {value for value in female["title_length_marker"] if value}
        male_wash = {value for value in male["title_wash_marker"] if value}
        female_wash = {value for value in female["title_wash_marker"] if value}
        male_machine = bool(male["title_machine_marker"].any())
        female_machine = bool(female["title_machine_marker"].any())

        return pd.Series(
            {
                "length_conflict": bool(male_lengths and female_lengths and male_lengths.isdisjoint(female_lengths)),
                "wash_dry_conflict": bool(male_wash and female_wash and male_wash.isdisjoint(female_wash)),
                "machine_conflict": bool((male_machine or female_machine) and male_machine != female_machine),
                "cell_bundle_conflict": bool(group["noncomparable_bundle_flag"].any()),
            }
        )

    conflicts = (
        service_rows.groupby(
            ["id", "structure_family", "structure_value_raw"],
            dropna=False,
        )
        .apply(conflict_for_group, include_groups=False)
        .reset_index()
    )
    return conflicts


def family_bucket(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .map(
            lambda value: "time"
            if value.startswith("time_")
            else ("length" if value.startswith("length_") else "")
        )
    )


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    baseline = preprocess_adults()
    unisex = restrict_unisex_salons(baseline).reset_index(drop=True)
    unisex["row_id"] = range(len(unisex))
    unisex_keyed = add_match_sequence(unisex)

    raw_options, raw_stats = parse_raw_option_rows()
    audit = unisex_keyed.merge(
        raw_options,
        on=KEY_COLS + ["match_seq"],
        how="left",
        validate="one_to_many",
    )
    audit = add_title_flags(audit)

    audit["group_norm"] = audit["optionGroupName"].map(normalize_text)
    audit["option_norm"] = audit["optionName"].map(normalize_text)
    audit["menuHighlightName"] = audit["menuHighlightName"].fillna(audit["simpleCutName"])

    classifications = audit.apply(
        lambda row: classify_structure(row["group_norm"], row["option_norm"], row["title_norm"]),
        axis=1,
        result_type="expand",
    )
    classifications.columns = [
        "structure_class",
        "structure_family",
        "structure_source",
        "structure_confidence",
    ]
    audit = pd.concat([audit, classifications], axis=1)

    audit["structure_value_raw"] = ""
    group_based = audit["structure_source"].isin(["group", "group_and_option"])
    option_based = audit["structure_source"].eq("option")
    audit.loc[group_based, "structure_value_raw"] = audit.loc[group_based, "group_norm"].fillna("")
    audit.loc[option_based, "structure_value_raw"] = audit.loc[option_based, "option_norm"].fillna("")
    audit.loc[
        (audit["structure_value_raw"] == "") & audit["group_norm"].notna(),
        "structure_value_raw",
    ] = audit.loc[(audit["structure_value_raw"] == "") & audit["group_norm"].notna(), "group_norm"]
    audit["structure_value_raw"] = audit["structure_value_raw"].fillna("")
    audit["family_bucket"] = family_bucket(audit["structure_family"])

    service = audit[audit["structure_class"] == "service_structure"].copy()
    raw_shared = (
        service.groupby(["id", "structure_family", "structure_value_raw"], dropna=False)
        .agg(has_male=("is_male", "max"), has_female=("is_female", "max"))
        .reset_index()
    )
    raw_shared["shared_cell_raw"] = raw_shared["has_male"] & raw_shared["has_female"]

    family_shared = (
        service.groupby(["id", "structure_family"], dropna=False)
        .agg(has_male=("is_male", "max"), has_female=("is_female", "max"))
        .reset_index()
    )
    family_shared["shared_cell_family"] = family_shared["has_male"] & family_shared["has_female"]

    conflicts = summarize_marker_conflicts(service) if not service.empty else pd.DataFrame()

    audit = audit.merge(
        raw_shared[["id", "structure_family", "structure_value_raw", "shared_cell_raw"]],
        on=["id", "structure_family", "structure_value_raw"],
        how="left",
    )
    audit = audit.merge(
        family_shared[["id", "structure_family", "shared_cell_family"]],
        on=["id", "structure_family"],
        how="left",
    )
    if not conflicts.empty:
        audit = audit.merge(
            conflicts,
            on=["id", "structure_family", "structure_value_raw"],
            how="left",
        )

    fill_false = [
        "shared_cell_raw",
        "shared_cell_family",
        "length_conflict",
        "machine_conflict",
        "wash_dry_conflict",
        "cell_bundle_conflict",
    ]
    for column in fill_false:
        if column not in audit:
            audit[column] = False
        audit[column] = audit[column].where(audit[column].notna(), False).astype(bool)

    audit["candidate_status"] = "candidate_service_structure"
    audit["drop_reason"] = ""

    no_option_mask = audit["optionGroupName"].isna() & audit["optionName"].isna()
    staff_mask = audit["structure_class"].eq("staff_tier")
    generic_mask = audit["structure_class"].eq("generic")
    bundle_mask = audit["structure_class"].eq("service_structure") & audit["cell_bundle_conflict"]
    marker_mask = (
        audit["structure_class"].eq("service_structure")
        & ~audit["cell_bundle_conflict"]
        & (audit["length_conflict"] | audit["machine_conflict"] | audit["wash_dry_conflict"])
    )
    clean_mask = (
        audit["structure_class"].eq("service_structure")
        & audit["shared_cell_family"]
        & ~audit["cell_bundle_conflict"]
        & ~audit["length_conflict"]
        & ~audit["machine_conflict"]
        & ~audit["wash_dry_conflict"]
    )

    audit.loc[no_option_mask, "candidate_status"] = "excluded_no_option_structure"
    audit.loc[no_option_mask, "drop_reason"] = "matched highlight has no option-group structure"
    audit.loc[staff_mask, "candidate_status"] = "excluded_staff_tier"
    audit.loc[staff_mask, "drop_reason"] = "option structure reflects staff tier"
    audit.loc[generic_mask, "candidate_status"] = "excluded_generic_option"
    audit.loc[generic_mask, "drop_reason"] = "option labels are generic or non-informative"
    audit.loc[bundle_mask, "candidate_status"] = "excluded_bundle_conflict"
    audit.loc[bundle_mask, "drop_reason"] = "title suggests bundled non-comparable services"
    audit.loc[marker_mask, "candidate_status"] = "excluded_marker_conflict"
    audit.loc[marker_mask, "drop_reason"] = "title markers conflict within shared option cell"
    audit.loc[clean_mask, "candidate_status"] = "candidate_clean_service_structure"
    audit.loc[clean_mask, "drop_reason"] = ""

    service_only_mask = audit["structure_class"].eq("service_structure") & ~clean_mask & audit["drop_reason"].eq("")
    audit.loc[
        service_only_mask & ~audit["shared_cell_family"],
        "drop_reason",
    ] = "service structure is not shared across genders within salon"

    output_columns = [
        "id",
        "country",
        "simpleCutName",
        "simpleCutSalePrice",
        "duration",
        "is_male",
        "is_female",
        "primaryTreatmentCategoryId",
        "menuHighlightName",
        "optionGroupName",
        "optionGroupIndex",
        "optionName",
        "optionIndex",
        "optionDurationMinutes",
        "structure_class",
        "structure_family",
        "structure_value_raw",
        "structure_source",
        "structure_confidence",
        "shared_cell_raw",
        "shared_cell_family",
        "noncomparable_bundle_flag",
        "length_conflict",
        "machine_conflict",
        "wash_dry_conflict",
        "candidate_status",
        "drop_reason",
        "title_norm",
        "title_stem",
    ]
    audit[output_columns].to_csv(OUTPUT_FILE, index=False)

    matched_listing_count = int(audit.loc[audit["raw_match_found"] == True, "row_id"].nunique())
    option_group_listing_count = int(
        audit.loc[audit["optionGroupName"].notna(), "row_id"].nunique()
    )
    option_level_rows = int(audit["optionGroupName"].notna().sum())

    structure_counts = audit["structure_class"].fillna("missing").value_counts()
    family_counts = (
        audit.loc[audit["structure_class"] == "service_structure", "structure_family"]
        .fillna("__none__")
        .value_counts()
    )

    shared_service_rows = audit[
        (audit["structure_class"] == "service_structure") & (audit["shared_cell_family"])
    ]
    shared_service_salons = int(shared_service_rows["id"].nunique())
    clean_candidates = audit[audit["candidate_status"] == "candidate_clean_service_structure"]
    clean_time = clean_candidates[clean_candidates["family_bucket"] == "time"]
    clean_length = clean_candidates[clean_candidates["family_bucket"] == "length"]

    multi_level_counts = (
        audit[
            (audit["structure_class"] == "service_structure") & (audit["shared_cell_raw"])
        ]
        .drop_duplicates(["id", "structure_family", "structure_value_raw"])
        .groupby(["id", "structure_family"])
        .size()
    )
    multi_level_salons = int(multi_level_counts[multi_level_counts > 1].reset_index()["id"].nunique())

    recommendation = "not promising"
    recommendation_reason = "clean structured-option cells are too sparse or too concentrated"
    recommended_rows = clean_candidates
    recommended_label = "pooled clean service-structure cells"

    def qualifies(frame: pd.DataFrame) -> bool:
        if frame.empty:
            return False
        if len(frame) < 300 or frame["id"].nunique() < 100:
            return False
        country_share = frame["country"].value_counts(normalize=True)
        return bool(country_share.empty or float(country_share.max()) <= 0.85)

    if qualifies(clean_time):
        recommendation = "promising only for time-based structured menus"
        recommendation_reason = "time-based clean cells clear the scale threshold"
        recommended_rows = clean_time
        recommended_label = "time-based clean service-structure cells"
    elif qualifies(clean_length):
        recommendation = "promising only for length-based structured menus"
        recommendation_reason = "length-based clean cells clear the scale threshold"
        recommended_rows = clean_length
        recommended_label = "length-based clean service-structure cells"
    elif qualifies(clean_candidates):
        recommendation = "promising for second-pass regression"
        recommendation_reason = "pooled clean service-structure cells clear the scale threshold"
    else:
        country_share = clean_candidates["country"].value_counts(normalize=True)
        if not clean_candidates.empty and not country_share.empty and float(country_share.max()) > 0.85:
            recommendation_reason = "clean cells are too concentrated in a single country"

    print("Structured-option diagnostic")
    print(f"- baseline adult analysis sample: {len(baseline)} listings")
    print(f"- baseline unisex-salon sample: {len(unisex)} listings across {unisex['id'].nunique()} salons")
    print(f"- matched raw menuHighlights: {matched_listing_count} listings")
    print(f"- matched listings with option groups: {option_group_listing_count}")
    print(f"- option-level rows in audit file: {option_level_rows}")
    print("- raw parsing summary:")
    for key in sorted(raw_stats):
        print(f"  - {key}: {raw_stats[key]}")

    print("- structure class distribution:")
    for key, value in structure_counts.items():
        print(f"  - {key}: {int(value)}")

    print("- service-structure family counts:")
    for key, value in family_counts.items():
        print(f"  - {key}: {int(value)}")

    print(
        f"- salons with shared service-structure cells across genders: {shared_service_salons}"
    )
    print(
        f"- salons with multiple shared levels within the same family: {multi_level_salons}"
    )
    print("- clean candidate counts after title-based screening:")
    print(
        f"  - pooled: {len(clean_candidates)} rows across {clean_candidates['id'].nunique()} salons"
    )
    print(f"  - time-based: {len(clean_time)} rows across {clean_time['id'].nunique()} salons")
    print(
        f"  - length-based: {len(clean_length)} rows across {clean_length['id'].nunique()} salons"
    )

    if not recommended_rows.empty:
        country_share = recommended_rows["country"].value_counts(normalize=True)
        top_country = country_share.index[0]
        top_share = float(country_share.iloc[0])
        print(
            f"- recommendation: {recommendation} ({recommended_label}; {len(recommended_rows)} rows, "
            f"{recommended_rows['id'].nunique()} salons, top country share {top_country}={top_share:.3f})"
        )
    else:
        print(f"- recommendation: {recommendation} ({recommendation_reason})")
    print(f"- wrote audit file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
