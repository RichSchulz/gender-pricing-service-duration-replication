#!/usr/bin/env python3
"""Generate the current paper tables from code.

This script rebuilds the regression tables used by the paper from the
current CSV extracts in ``data/`` and writes the LaTeX inputs used by the
paper into ``paper/tables/``. It also prints each table to stdout and
compares the generated numbers against the values currently reported in the
manuscript.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import reverse_geocoder as rg
from statsmodels.formula.api import ols

from utils import filter_words, ppp_per_usd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PAPER_DIR = ROOT / "paper"
TABLES_DIR = PAPER_DIR / "tables"
MANUSCRIPT_DIR = PAPER_DIR / "manuscript"

ADULT_DATA = DATA_DIR / "snapshots" / "treatwell_without_raw-all-2025-06-02.csv"
VENUE_INFO = DATA_DIR / "snapshots" / "venue_info-2025-11-23.csv"
MATCHED_MODEL_SUMMARY = DATA_DIR / "derived" / "matched_category_model_summary.json"

USD_TO_EUR = ppp_per_usd["DE"]

WET_CUT_KEYWORDS = [
    "waschen",
    "shampoo",
    "shampoing",
    "nass",
    "mit waschen",
    "wash",
    "wet",
    "with wash",
    "haircut & wash",
    "lavage",
    "avec shampoing",
    "lavado",
    "lavar",
    "con lavado",
    "corte y lavado",
    "lavaggio",
    "con shampoo",
    "taglio e shampoo",
    "wassen",
    "met wassen",
    "com lavado",
    "corte e lavado",
]

MACHINE_KEYWORDS = [
    "maschinen",
    "maschinenschnitt",
    "clipper",
    "rasur",
    "machine",
    "clipper cut",
    "tondeuse",
    "maquina",
    "máquina",
    "macchina",
    "macchinetta",
]


PAPER_TARGETS = {
    "main_1": {
        "coef": {"Duration (min)": 0.536, "Female": 4.534},
        "se": {"Duration (min)": 0.008, "Female": 0.156},
        "obs": 20758,
        "groups": 6459,
        "r2": 0.621,
    },
    "main_2": {
        "coef": {"Duration (min)": 0.462, "Female": 0.741, "Duration $\\times$ Female": 0.100},
        "se": {"Duration (min)": 0.011, "Female": 0.431, "Duration $\\times$ Female": 0.012},
        "obs": 20758,
        "groups": 6459,
        "r2": 0.625,
    },
    "robust_a_1": {
        "coef": {"Duration (min)": 0.556, "Female": 4.134},
        "se": {"Duration (min)": 0.016, "Female": 0.282},
        "obs": 6127,
        "groups": 1881,
        "r2": 0.605,
    },
    "robust_a_2": {
        "coef": {"Duration (min)": 0.481, "Female": 0.306, "Duration $\\times$ Female": 0.103},
        "se": {"Duration (min)": 0.020, "Female": 0.814, "Duration $\\times$ Female": 0.021},
        "obs": 6127,
        "groups": 1881,
        "r2": 0.608,
    },
    "robust_b_1": {
        "coef": {"Duration (min)": 0.437, "Female": 6.496},
        "se": {"Duration (min)": 0.018, "Female": 0.335},
        "obs": 7456,
        "groups": 3680,
        "r2": 0.594,
    },
    "robust_b_2": {
        "coef": {"Duration (min)": 0.343, "Female": 1.356, "Duration $\\times$ Female": 0.127},
        "se": {"Duration (min)": 0.027, "Female": 0.993, "Duration $\\times$ Female": 0.025},
        "obs": 7456,
        "groups": 3680,
        "r2": 0.599,
    },
    "robust_c_1": {
        "coef": {"Duration (min)": 0.304, "Female": 9.814},
        "se": {"Duration (min)": 0.031, "Female": 0.626},
        "obs": 1548,
        "groups": 725,
        "r2": 0.681,
    },
    "robust_c_2": {
        "coef": {"Duration (min)": 0.122, "Female": -0.632, "Duration $\\times$ Female": 0.252},
        "se": {"Duration (min)": 0.040, "Female": 1.452, "Duration $\\times$ Female": 0.035},
        "obs": 1548,
        "groups": 725,
        "r2": 0.701,
    },
}


@dataclass
class ModelResult:
    label: str
    model: object
    nobs: int
    n_groups: int
    coef_map: dict[str, str]

    def coef(self, row: str) -> float | None:
        key = self.coef_map.get(row)
        if key is None or key not in self.model.params:
            return None
        return float(self.model.params[key])

    def se(self, row: str) -> float | None:
        key = self.coef_map.get(row)
        if key is None or key not in self.model.bse:
            return None
        return float(self.model.bse[key])

    def pvalue(self, row: str) -> float | None:
        key = self.coef_map.get(row)
        if key is None or key not in self.model.pvalues:
            return None
        return float(self.model.pvalues[key])

    @property
    def r2(self) -> float:
        return float(self.model.rsquared)


@dataclass
class FrozenModelResult:
    label: str
    nobs: int
    n_groups: int
    r2_value: float
    coef_values: dict[str, float | None]
    se_values: dict[str, float | None]
    pvalue_values: dict[str, float | None]

    def coef(self, row: str) -> float | None:
        return self.coef_values.get(row)

    def se(self, row: str) -> float | None:
        return self.se_values.get(row)

    def pvalue(self, row: str) -> float | None:
        return self.pvalue_values.get(row)

    @property
    def r2(self) -> float:
        return self.r2_value


@dataclass
class SameDurationGap:
    label: str
    gap: float
    se: float
    pvalue: float
    listings: int
    salons: int
    cells: int


def normalize(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


FILTER_WORDS = list({normalize(word) for word in filter_words})
WET_KEYWORDS = [normalize(word) for word in WET_CUT_KEYWORDS]
MACHINE_KEYWORDS_N = [normalize(word) for word in MACHINE_KEYWORDS]


def contains_any_keyword(text: object, keywords: Iterable[str]) -> bool:
    value = normalize(text)
    return any(keyword in value for keyword in keywords)


def load_country_codes(df: pd.DataFrame) -> pd.Series:
    coords = df[["lat", "lon"]].dropna()
    results = rg.search(list(coords.itertuples(index=False, name=None)), mode=1, verbose=False)
    geo = pd.DataFrame(results, index=coords.index)[["cc"]]
    geo.columns = ["country"]
    return geo["country"]


def convert_price_to_eur(row: pd.Series) -> float:
    return row["simpleCutSalePrice"] / ppp_per_usd.get(row["country"], 1.0) * USD_TO_EUR


def preprocess_adults() -> pd.DataFrame:
    df = pd.read_csv(ADULT_DATA, low_memory=False)
    df["country"] = load_country_codes(df)
    df["duration"] = (df["simpleCutDurationMin"] + df["simpleCutDurationMax"]) / 2
    df = df[(df["duration"] > 10) & (df["duration"] < 120)].copy()
    df = df[~df["country"].isin(["LU", "SM"])].copy()
    df = df[df["simpleCutSalePrice"] < 200].copy()
    df = df[df["name"] != "Test Salon TEST PURPOSE'S ONLY"].copy()
    df["price"] = df.apply(convert_price_to_eur, axis=1)
    df = df[
        df["simpleCutName"].apply(
            lambda value: not any(word in normalize(value) for word in FILTER_WORDS)
        )
    ].copy()
    df = df[(df["is_male"] == True) | (df["is_female"] == True)].copy()
    df["female"] = df["is_female"].astype(int)
    return df.reset_index(drop=True)


def add_demeaned_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    means = df.groupby("id")[columns].transform("mean")
    out = df.copy()
    for column in columns:
        out[f"{column}_dm"] = out[column] - means[column]
    return out


def fit_clustered_fe(formula: str, df: pd.DataFrame, label: str, coef_map: dict[str, str]) -> ModelResult:
    model = ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["id"]})
    return ModelResult(
        label=label,
        model=model,
        nobs=len(df),
        n_groups=int(df["id"].nunique()),
        coef_map=coef_map,
    )


def fit_adult_models(df: pd.DataFrame) -> tuple[ModelResult, ModelResult]:
    work = df.copy()
    work["fem_dur"] = work["female"] * work["duration"]
    work = add_demeaned_columns(work, ["price", "female", "duration", "fem_dur"])
    m1 = fit_clustered_fe(
        "price_dm ~ female_dm + duration_dm - 1",
        work,
        label="FE (No Interaction)",
        coef_map={"Duration (min)": "duration_dm", "Female": "female_dm"},
    )
    m2 = fit_clustered_fe(
        "price_dm ~ female_dm + duration_dm + fem_dur_dm - 1",
        work,
        label="FE (w/ Interaction)",
        coef_map={
            "Duration (min)": "duration_dm",
            "Female": "female_dm",
            "Duration $\\times$ Female": "fem_dur_dm",
        },
    )
    return m1, m2


def restrict_unisex_salons(df: pd.DataFrame) -> pd.DataFrame:
    female_ids = set(df.loc[df["female"] == 1, "id"])
    male_ids = set(df.loc[df["female"] == 0, "id"])
    both_ids = female_ids & male_ids
    return df[df["id"].isin(both_ids)].copy()


def restrict_single_employee(df: pd.DataFrame) -> pd.DataFrame:
    venue = pd.read_csv(VENUE_INFO, usecols=["venue_id", "employee_count"])
    venue["employee_count"] = pd.to_numeric(venue["employee_count"], errors="coerce")
    merged = df.merge(venue.drop_duplicates("venue_id"), left_on="id", right_on="venue_id", how="left")
    return merged[merged["employee_count"].fillna(999) <= 1].copy()


def restrict_wet_cuts(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["is_wet_name"] = work["simpleCutName"].apply(lambda value: contains_any_keyword(value, WET_KEYWORDS))
    work["is_machine_name"] = work["simpleCutName"].apply(
        lambda value: contains_any_keyword(value, MACHINE_KEYWORDS_N)
    )
    return work[work["is_wet_name"] & ~work["is_machine_name"]].copy()


def fit_same_duration_gap(df: pd.DataFrame, label: str) -> SameDurationGap:
    grouped = (
        df.groupby(["id", "duration", "female"], observed=True)
        .agg(mean_price=("price", "mean"), listings=("price", "size"))
        .reset_index()
    )
    male = grouped[grouped["female"] == 0].drop(columns=["female"])
    female = grouped[grouped["female"] == 1].drop(columns=["female"])
    overlap_cells = male.merge(female, on=["id", "duration"], suffixes=("_male", "_female"))
    overlap_keys = overlap_cells[["id", "duration"]].drop_duplicates()
    work = df.merge(overlap_keys, on=["id", "duration"], how="inner", validate="many_to_one")
    cell_means = work.groupby(["id", "duration"], observed=True)[["price", "female"]].transform("mean")
    work["price_cell_dm"] = work["price"] - cell_means["price"]
    work["female_cell_dm"] = work["female"] - cell_means["female"]
    model = ols("price_cell_dm ~ female_cell_dm - 1", data=work).fit(
        cov_type="cluster",
        cov_kwds={"groups": work["id"]},
    )
    return SameDurationGap(
        label=label,
        gap=float(model.params["female_cell_dm"]),
        se=float(model.bse["female_cell_dm"]),
        pvalue=float(model.pvalues["female_cell_dm"]),
        listings=int(len(work)),
        salons=int(work["id"].nunique()),
        cells=int(len(overlap_cells)),
    )


def stars(pvalue: float | None) -> str:
    if pvalue is None or math.isnan(pvalue):
        return ""
    if pvalue < 0.01:
        return "$^{***}$"
    if pvalue < 0.05:
        return "$^{**}$"
    if pvalue < 0.1:
        return "$^{*}$"
    return "$^{}$"


def fmt_coef(value: float | None, pvalue: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}{stars(pvalue)}"


def fmt_se(value: float | None) -> str:
    if value is None:
        return ""
    return f"({value:.3f})"


def fmt_eur_coef(value: float | None, pvalue: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.1f}{stars(pvalue)}"


def fmt_eur_se(value: float | None) -> str:
    if value is None:
        return ""
    return f"({value:.1f})"


def fmt_int(value: int) -> str:
    return str(int(value))


def fmt_r2(value: float) -> str:
    return f"{value:.3f}"


def fmt_optional(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def render_main_table(models: list[ModelResult]) -> str:
    rows = ["Duration (min)", "Female", "Duration $\\times$ Female"]
    lines = [
        "",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Adult Haircuts: Posted Price--Duration Schedule Differences}",
        "  \\label{tab:main_results}",
        "  \\small",
        "  ",
        "\\begin{tabular}{@{\\extracolsep{5pt}}lcc}",
        "\\\\[-1.8ex]\\hline",
        "\\hline \\\\[-1.8ex]",
        "& \\multicolumn{2}{c}{\\textit{Dependent variable: price}} \\\\",
        "\\cr \\cline{2-3}",
        "\\\\[-1.8ex] & \\multicolumn{2}{c}{Unisex Salons Only}  \\\\",
        "\\\\[-1.8ex] & (1) & (2) \\\\",
        "\\hline \\\\[-1.8ex]",
    ]
    for row in rows:
        coef_cells = [fmt_coef(model.coef(row), model.pvalue(row)) for model in models]
        se_cells = [fmt_se(model.se(row)) for model in models]
        lines.append(f" {row} & " + " & ".join(coef_cells) + " \\\\")
        lines.append("& " + " & ".join(se_cells) + " \\\\")
    lines.extend(
        [
            "\\hline \\\\[-1.8ex]",
            " Salon Fixed Effects & Yes & Yes \\\\",
            " Observations & " + " & ".join(fmt_int(model.nobs) for model in models) + " \\\\",
            " Number of salons & " + " & ".join(fmt_int(model.n_groups) for model in models) + " \\\\",
            " $R^2$ & " + " & ".join(fmt_r2(model.r2) for model in models) + " \\\\",
            "\\hline",
            "\\hline \\\\[-1.8ex]",
            "",
            "\\end{tabular}",
            "",
            "  \\begin{minipage}{0.9\\linewidth}",
            "      \\footnotesize",
            "      \\textit{Notes:} The dependent variable is price. Standard errors, clustered at the salon level, are reported in parentheses.",
            "      The sample is restricted to unisex salons that offer services to both men and women.",
            "      Column (1) excludes the interaction; Column (2) includes it.",
            "      Statistical significance is denoted as: $^{*}$p$<$0.1; $^{**}$p$<$0.05; $^{***}$p$<$0.01.",
            "  \\end{minipage}\\end{table}",
        ]
    )
    return "\n".join(lines)


def render_robustness_table(
    robust_a: list[ModelResult],
    robust_b: list[ModelResult],
    robust_c: list[ModelResult | FrozenModelResult],
    same_duration_checks: list[SameDurationGap],
) -> str:
    interaction_row = "Duration $\\times$ Female"
    same_by_label = {item.label: item for item in same_duration_checks}
    rows = [
        ("Single employee", same_by_label["Single employee"], robust_a[1]),
        ("Wet cuts", same_by_label["Wet cuts"], robust_b[1]),
        ("Matched titles", same_by_label["Matched titles"], robust_c[1]),
    ]
    lines = [
        "",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Supplementary Same-Duration and Continuous Checks}",
        "  \\label{tab:combined_robustness_results}",
        "  \\footnotesize",
        "",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Sample restriction & Same-duration gap & \\makecell{Exact-overlap\\\\listings/salons} & Duration $\\times$ Female & \\makecell{Continuous\\\\listings/salons} \\\\",
        "\\hline",
    ]
    for label, same_duration, model in rows:
        exact_gap = f"{fmt_eur_coef(same_duration.gap, same_duration.pvalue)} {fmt_eur_se(same_duration.se)}"
        interaction = (
            f"{fmt_coef(model.coef(interaction_row), model.pvalue(interaction_row))} "
            f"{fmt_se(model.se(interaction_row))}"
        )
        lines.append(
            f"{label} & {exact_gap} & "
            f"{fmt_int(same_duration.listings)}/{fmt_int(same_duration.salons)} & "
            f"{interaction} & "
            f"{fmt_int(model.nobs)}/{fmt_int(model.n_groups)} \\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "",
            "  \\begin{minipage}{0.96\\linewidth}",
            "      \\footnotesize",
            "      \\textit{Notes:} Same-duration gaps use salon-by-exact-duration fixed effects and are identified by salon-duration cells containing both male- and female-coded listings.",
            "      The continuous-interaction column estimates the Table~\\ref{tab:main_results} interaction specification with salon fixed effects.",
            "      Standard errors are clustered at the salon level and reported in parentheses.",
            "      The single-employee and wet-cut rows use adult unisex salons; the matched-title row uses within-salon male/female listings with identical nongendered service titles and additional descriptive content.",
            "      Statistical significance is denoted as: $^{*}$p$<$0.1; $^{**}$p$<$0.05; $^{***}$p$<$0.01.",
            "  \\end{minipage}\\end{table}",
        ]
    )
    return "\n".join(lines)


def render_country_heterogeneity_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Country-Specific Continuous-Interaction Estimates}",
        "  \\label{tab:country_heterogeneity}",
        "  \\small",
        "  ",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Country & Duration $\\times$ Female & SE & Observations & Number of salons \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['name']} & {row['beta3']:.3f} & ({row['se']:.3f}) & {int(row['obs'])} & {int(row['salons'])} \\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "",
            "\\begin{minipage}{0.88\\linewidth}",
            "    \\footnotesize",
            "    \\textit{Notes:} Each row reports the interaction coefficient from the preferred unisex-salon specification estimated separately by country.",
            "    Standard errors are clustered at the salon level.",
            "\\end{minipage}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def compare_model(name: str, model: ModelResult | FrozenModelResult, target: dict[str, object]) -> list[str]:
    mismatches: list[str] = []
    for row_name, target_value in target["coef"].items():
        actual_value = model.coef(row_name)
        if actual_value is None or round(actual_value, 3) != round(target_value, 3):
            mismatches.append(
                f"{name} coef {row_name}: generated {fmt_optional(actual_value)} vs paper {target_value:.3f}"
            )
    for row_name, target_value in target["se"].items():
        actual_value = model.se(row_name)
        if actual_value is None or round(actual_value, 3) != round(target_value, 3):
            mismatches.append(
                f"{name} se {row_name}: generated {fmt_optional(actual_value)} vs paper {target_value:.3f}"
            )
    if model.nobs != target["obs"]:
        mismatches.append(f"{name} observations: generated {model.nobs} vs paper {target['obs']}")
    if model.n_groups != target["groups"]:
        mismatches.append(f"{name} number of salons: generated {model.n_groups} vs paper {target['groups']}")
    if round(model.r2, 3) != round(target["r2"], 3):
        mismatches.append(f"{name} R^2: generated {model.r2:.3f} vs paper {target['r2']:.3f}")
    return mismatches


def compare_all_tables(
    main_models: list[ModelResult],
    robust_a: list[ModelResult],
    robust_b: list[ModelResult],
    robust_c: list[ModelResult | FrozenModelResult],
) -> list[str]:
    checks = [
        ("main_1", main_models[0]),
        ("main_2", main_models[1]),
        ("robust_a_1", robust_a[0]),
        ("robust_a_2", robust_a[1]),
        ("robust_b_1", robust_b[0]),
        ("robust_b_2", robust_b[1]),
        ("robust_c_1", robust_c[0]),
        ("robust_c_2", robust_c[1]),
    ]
    mismatches: list[str] = []
    for key, model in checks:
        mismatches.extend(compare_model(key, model, PAPER_TARGETS[key]))
    return mismatches


def print_spec_summary() -> None:
    summary = {
        "adult_data": str(ADULT_DATA.relative_to(ROOT)),
        "single_employee_data": str(VENUE_INFO.relative_to(ROOT)),
        "adult_sample_restrictions": [
            "duration > 10 and < 120",
            "exclude LU and SM from reverse-geocoded country",
            "simpleCutSalePrice < 200",
            "drop Test Salon TEST PURPOSE'S ONLY",
            "drop names containing adult/non-cut keyword filters from analysis/utils.py",
            "keep rows flagged male or female",
        ],
        "robustness_panel_a": "adult unisex sample merged to data/snapshots/venue_info-2025-11-23.csv with employee_count <= 1",
        "robustness_panel_b": "adult unisex sample restricted to names containing wet-cut keywords and excluding machine-keyword names",
    }
    print("Specification summary:")
    print(json.dumps(summary, indent=2))


def write_file(path: Path, content: str) -> None:
    path.write_text(content + "\n", encoding="utf-8")


def refresh_matched_category_models() -> list[FrozenModelResult]:
    script = ROOT / "analysis" / "matched_category_robustness.py"
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    payload = json.loads(MATCHED_MODEL_SUMMARY.read_text(encoding="utf-8"))
    if not payload.get("available"):
        listings = payload.get("listings", "unknown")
        salons = payload.get("salons", "unknown")
        raise RuntimeError(
            f"Matched-category regression unavailable: listings={listings}, salons={salons}"
        )
    return [
        FrozenModelResult(
            label=str(model["label"]),
            nobs=int(model["nobs"]),
            n_groups=int(model["n_groups"]),
            r2_value=float(model["r2"]),
            coef_values=dict(model["coef"]),
            se_values=dict(model["se"]),
            pvalue_values=dict(model["pvalue"]),
        )
        for model in payload["models"]
    ]


def fit_country_rows(adults_unisex: pd.DataFrame) -> list[dict[str, object]]:
    country_names = {
        "AT": "Austria",
        "BE": "Belgium",
        "CH": "Switzerland",
        "DE": "Germany",
        "ES": "Spain",
        "FR": "France",
        "GB": "United Kingdom",
        "IT": "Italy",
        "NL": "Netherlands",
        "PT": "Portugal",
    }
    frame = adults_unisex.copy()
    frame["country"] = frame["country"].replace({"VA": "IT"})
    rows: list[dict[str, object]] = []
    for country, subset in sorted(frame.groupby("country")):
        _, model = fit_adult_models(subset)
        rows.append(
            {
                "country": country,
                "name": country_names.get(country, country),
                "obs": len(subset),
                "salons": int(subset["id"].nunique()),
                "beta3": float(model.model.params["fem_dur_dm"]),
                "se": float(model.model.bse["fem_dur_dm"]),
                "pvalue": float(model.model.pvalues["fem_dur_dm"]),
            }
        )
    return rows


def main() -> None:
    print_spec_summary()

    adults = preprocess_adults()
    adults_unisex = restrict_unisex_salons(adults)

    main_models = list(fit_adult_models(adults_unisex))

    single_employee = restrict_single_employee(adults_unisex)
    robust_a = list(fit_adult_models(single_employee))

    wet_cuts = restrict_wet_cuts(adults_unisex)
    robust_b = list(fit_adult_models(wet_cuts))
    robust_c = refresh_matched_category_models()
    matched_titles = pd.read_csv(DATA_DIR / "audits" / "matched_category_primary_sample.csv")
    same_duration_checks = [
        fit_same_duration_gap(single_employee, "Single employee"),
        fit_same_duration_gap(wet_cuts, "Wet cuts"),
        fit_same_duration_gap(matched_titles, "Matched titles"),
    ]

    country_rows = fit_country_rows(adults_unisex)

    main_tex = render_main_table(main_models)
    robustness_tex = render_robustness_table(robust_a, robust_b, robust_c, same_duration_checks)
    country_tex = render_country_heterogeneity_table(country_rows)

    write_file(TABLES_DIR / "table_main_results.tex", main_tex)
    write_file(TABLES_DIR / "table_robustness.tex", robustness_tex)
    write_file(TABLES_DIR / "table_country_heterogeneity.tex", country_tex)

    print("\n=== Generated paper/tables/table_main_results.tex ===")
    print(main_tex)
    print("\n=== Generated paper/tables/table_robustness.tex ===")
    print(robustness_tex)
    print("\n=== Generated paper/tables/table_country_heterogeneity.tex ===")
    print(country_tex)

    mismatches = compare_all_tables(main_models, robust_a, robust_b, robust_c)
    print("\n=== Comparison Against Current Paper Values ===")
    if not mismatches:
        print("Generated values match the current manuscript tables exactly.")
    else:
        for item in mismatches:
            print(f"- {item}")

    print("\nReplication-package outputs updated:")
    print("- paper/tables/table_main_results.tex")
    print("- paper/tables/table_robustness.tex")
    print("- paper/tables/table_country_heterogeneity.tex")


if __name__ == "__main__":
    main()
