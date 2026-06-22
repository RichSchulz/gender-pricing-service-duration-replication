#!/usr/bin/env python3
"""Duration-support and same-duration gap diagnostics for the R&R.

This script is additive to the existing table workflow. It reuses the
canonical adult preprocessing code, restricts to unisex salons, and writes
duration-overlap diagnostics requested by the second referee.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.formula.api import ols

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from generate_paper_tables import (  # noqa: E402
    DATA_DIR,
    PAPER_DIR,
    add_demeaned_columns,
    convert_price_to_eur,
    fmt_int,
    fmt_se,
    load_country_codes,
    normalize,
    preprocess_adults,
    restrict_unisex_salons,
    write_file,
)
from utils import filter_words_kids  # noqa: E402


DERIVED_DIR = DATA_DIR / "derived"
TABLES_DIR = PAPER_DIR / "tables"
FIGURES_DIR = PAPER_DIR / "figures"
KIDS_DATA = DATA_DIR / "snapshots" / "treatwell_without_raw_kids-2025-06-03.csv"
FILTER_WORDS_KIDS = list({normalize(word) for word in filter_words_kids})

COMMON_DURATIONS = [15, 20, 25, 30, 35, 40, 45, 52.5, 60, 75, 90]
REQUESTED_SLOT_DURATIONS = [30, 45, 60, 90]
DURATION_BIN_DEFS = [
    (0, 30, "<=30", "le_30"),
    (30, 45, "31-45", "31_45"),
    (45, 60, "46-60", "46_60"),
    (60, 75, "61-75", "61_75"),
    (75, 120, "76-119", "76_119"),
]
DURATION_BIN_LABELS = [item[2] for item in DURATION_BIN_DEFS]
DURATION_BIN_SAFE = {label: safe for _, _, label, safe in DURATION_BIN_DEFS}


def fmt_num(value: float | int | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_money(value: float | int | None) -> str:
    return fmt_num(value, digits=1)


def fmt_money_se(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"({float(value):.1f})"


def fmt_duration(value: float | int) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}\\%"


def latex_duration_label(label: str) -> str:
    if label == "<=30":
        return "$\\leq$30"
    return label.replace("-", "--")


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


def add_duration_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bins = [item[0] for item in DURATION_BIN_DEFS] + [DURATION_BIN_DEFS[-1][1]]
    out["duration_bin"] = pd.cut(
        out["duration"],
        bins=bins,
        labels=DURATION_BIN_LABELS,
        right=True,
        include_lowest=True,
    )
    out["duration_bin"] = out["duration_bin"].astype("category")
    out["duration_bin"] = out["duration_bin"].cat.set_categories(DURATION_BIN_LABELS, ordered=True)
    return out


def preprocess_children_boys_girls() -> pd.DataFrame:
    df = pd.read_csv(KIDS_DATA, low_memory=False)
    df["country"] = load_country_codes(df)
    df["duration"] = (df["simpleCutDurationMin"] + df["simpleCutDurationMax"]) / 2
    df = df[(df["duration"] > 10) & (df["duration"] < 120)].copy()
    df = df[~df["country"].isin(["LU", "SM"])].copy()
    df = df[df["simpleCutSalePrice"] < 200].copy()
    df = df[df["name"] != "Test Salon TEST PURPOSE'S ONLY"].copy()
    df["price"] = df.apply(convert_price_to_eur, axis=1)
    df = df[
        df["simpleCutName"].apply(
            lambda value: not any(word in normalize(value) for word in FILTER_WORDS_KIDS)
        )
    ].copy()
    df = df[(df["is_boys"] == True) | (df["is_girls"] == True)].copy()
    df["female"] = df["is_girls"].astype(int)
    return df.reset_index(drop=True)


def duration_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    duration_counts = df.groupby(["duration", "female"], observed=True).size().unstack(fill_value=0)
    common_exact = duration_counts[(duration_counts.get(0, 0) > 0) & (duration_counts.get(1, 0) > 0)].index

    rows: list[dict[str, object]] = []
    for female, label in [(0, "Male-coded"), (1, "Female-coded")]:
        subset = df[df["female"] == female]
        row: dict[str, object] = {
            "gender": label,
            "listings": int(len(subset)),
            "mean_duration": float(subset["duration"].mean()),
            "p10": float(subset["duration"].quantile(0.10)),
            "p25": float(subset["duration"].quantile(0.25)),
            "p50": float(subset["duration"].quantile(0.50)),
            "p75": float(subset["duration"].quantile(0.75)),
            "p90": float(subset["duration"].quantile(0.90)),
            "share_at_common_exact_durations": float(subset["duration"].isin(common_exact).mean()),
        }
        counts = subset["duration"].value_counts()
        for duration in COMMON_DURATIONS:
            row[f"count_{fmt_duration(duration).replace('.', '_')}"] = int(counts.get(float(duration), 0))
        rows.append(row)
    return pd.DataFrame(rows)


def build_overlap_cells(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["id", "duration", "female"], observed=True)
        .agg(mean_price=("price", "mean"), listings=("price", "size"))
        .reset_index()
    )
    male = grouped[grouped["female"] == 0].drop(columns=["female"])
    female = grouped[grouped["female"] == 1].drop(columns=["female"])
    cells = male.merge(female, on=["id", "duration"], suffixes=("_male", "_female"))
    cells = cells.rename(
        columns={
            "mean_price_male": "male_mean_price",
            "mean_price_female": "female_mean_price",
            "listings_male": "male_count",
            "listings_female": "female_count",
        }
    )
    cells["female_minus_male_gap"] = cells["female_mean_price"] - cells["male_mean_price"]
    return cells.sort_values(["duration", "id"]).reset_index(drop=True)


def paired_gap_rows(overlap_cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def summarize(label: str, subset: pd.DataFrame, duration: float | None = None) -> None:
        if subset.empty:
            return
        gaps = subset["female_minus_male_gap"]
        rows.append(
            {
                "estimate_type": "paired_exact_duration",
                "duration_or_bin": label,
                "duration": duration,
                "cells": int(len(subset)),
                "salons": int(subset["id"].nunique()),
                "listings": int(subset["male_count"].sum() + subset["female_count"].sum()),
                "male_listings": int(subset["male_count"].sum()),
                "female_listings": int(subset["female_count"].sum()),
                "gap": float(gaps.mean()),
                "se": float(gaps.std(ddof=1) / math.sqrt(len(gaps))) if len(gaps) > 1 else math.nan,
                "median_gap": float(gaps.median()),
                "share_positive": float((gaps > 0).mean()),
                "pvalue": math.nan,
                "r2": math.nan,
            }
        )

    summarize("All exact overlaps", overlap_cells)
    for duration in COMMON_DURATIONS:
        subset = overlap_cells[overlap_cells["duration"] == float(duration)]
        if len(subset) >= 20:
            summarize(fmt_duration(duration), subset, float(duration))
    return pd.DataFrame(rows)


def fit_duration_bin_salon_band_gaps(df: pd.DataFrame) -> pd.DataFrame:
    work = add_duration_bins(df)
    rows = []
    for label in DURATION_BIN_LABELS:
        subset = work[work["duration_bin"].astype(str) == label].copy()
        overlap_ids = subset.groupby("id", observed=True)["female"].nunique()
        overlap_ids = overlap_ids[overlap_ids == 2].index
        subset = subset[subset["id"].isin(overlap_ids)].copy()
        if subset.empty:
            continue

        subset = add_demeaned_columns(subset, ["price", "female", "duration"])
        model = ols("price_dm ~ female_dm - 1", data=subset).fit(
            cov_type="cluster",
            cov_kwds={"groups": subset["id"]},
        )
        exact_overlap = subset.groupby(["id", "duration"], observed=True)["female"].nunique()
        exact_overlap = exact_overlap[exact_overlap == 2].reset_index()[["id", "duration"]]
        exact_subset = subset.merge(
            exact_overlap,
            on=["id", "duration"],
            how="inner",
            validate="many_to_one",
        )
        if exact_subset.empty:
            model_exact_duration = None
            exact_listings = 0
            exact_salons = 0
            exact_cells = 0
        else:
            cell_means = exact_subset.groupby(["id", "duration"], observed=True)[["price", "female"]].transform("mean")
            exact_subset["price_cell_dm"] = exact_subset["price"] - cell_means["price"]
            exact_subset["female_cell_dm"] = exact_subset["female"] - cell_means["female"]
            model_exact_duration = ols("price_cell_dm ~ female_cell_dm - 1", data=exact_subset).fit(
                cov_type="cluster",
                cov_kwds={"groups": exact_subset["id"]},
            )
            exact_listings = int(len(exact_subset))
            exact_salons = int(exact_subset["id"].nunique())
            exact_cells = int(exact_subset[["id", "duration"]].drop_duplicates().shape[0])
        male_duration = subset.loc[subset["female"] == 0, "duration"].mean()
        female_duration = subset.loc[subset["female"] == 1, "duration"].mean()
        exact_gap = (
            float(model_exact_duration.params["female_cell_dm"])
            if model_exact_duration is not None
            else math.nan
        )
        exact_se = (
            float(model_exact_duration.bse["female_cell_dm"])
            if model_exact_duration is not None
            else math.nan
        )
        exact_pvalue = (
            float(model_exact_duration.pvalues["female_cell_dm"])
            if model_exact_duration is not None
            else math.nan
        )
        exact_r2 = float(model_exact_duration.rsquared) if model_exact_duration is not None else math.nan
        rows.append(
            {
                "estimate_type": "duration_bin_salon_band_fe",
                "duration_or_bin": label,
                "duration": math.nan,
                "cells": int(subset["id"].nunique()),
                "salons": int(subset["id"].nunique()),
                "listings": int(len(subset)),
                "male_listings": int((subset["female"] == 0).sum()),
                "female_listings": int((subset["female"] == 1).sum()),
                "gap": float(model.params["female_dm"]),
                "se": float(model.bse["female_dm"]),
                "gap_exact_duration_fe": exact_gap,
                "se_exact_duration_fe": exact_se,
                "pvalue_exact_duration_fe": exact_pvalue,
                "listings_exact_duration_fe": exact_listings,
                "salons_exact_duration_fe": exact_salons,
                "cells_exact_duration_fe": exact_cells,
                "male_mean_duration_within_bin": float(male_duration),
                "female_mean_duration_within_bin": float(female_duration),
                "female_minus_male_duration_within_bin": float(female_duration - male_duration),
                "median_gap": math.nan,
                "share_positive": math.nan,
                "pvalue": float(model.pvalues["female_dm"]),
                "r2": float(model.rsquared),
                "r2_exact_duration_fe": exact_r2,
            }
        )
    return pd.DataFrame(rows)


def fit_exact_duration_gaps(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("duration", observed=True)["female"].agg(["size", "sum"]).reset_index()
    counts["male"] = counts["size"] - counts["sum"]
    valid_durations = counts[
        (counts["size"] >= 300) & (counts["male"] >= 50) & (counts["sum"] >= 50)
    ]["duration"].tolist()
    work = df[df["duration"].isin(valid_durations)].copy()

    covariates: list[str] = []
    name_for_duration: dict[float, str] = {}
    for duration in valid_durations:
        safe = str(duration).replace(".", "_").replace("-", "m")
        if safe.endswith("_0"):
            safe = safe[:-2]
        name_for_duration[float(duration)] = safe
        dur_col = f"dur_{safe}"
        fem_col = f"fem_dur_{safe}"
        work[dur_col] = (work["duration"] == duration).astype(float)
        work[fem_col] = work["female"] * work[dur_col]
        if duration != valid_durations[0]:
            covariates.append(dur_col)
        covariates.append(fem_col)

    work = add_demeaned_columns(work, ["price", *covariates])
    formula = "price_dm ~ " + " + ".join(f"{column}_dm" for column in covariates) + " - 1"
    model = ols(formula, data=work).fit(cov_type="cluster", cov_kwds={"groups": work["id"]})

    rows = []
    for duration in valid_durations:
        safe = name_for_duration[float(duration)]
        key = f"fem_dur_{safe}_dm"
        subset = work[work["duration"] == duration]
        rows.append(
            {
                "estimate_type": "exact_duration_regression",
                "duration_or_bin": fmt_duration(duration),
                "duration": float(duration),
                "cells": math.nan,
                "salons": int(subset["id"].nunique()),
                "listings": int(len(subset)),
                "male_listings": int((subset["female"] == 0).sum()),
                "female_listings": int((subset["female"] == 1).sum()),
                "gap": float(model.params[key]),
                "se": float(model.bse[key]),
                "median_gap": math.nan,
                "share_positive": math.nan,
                "pvalue": float(model.pvalues[key]),
                "r2": float(model.rsquared),
            }
        )
    return pd.DataFrame(rows)


def fit_requested_slot_gaps(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df["duration"].isin([float(value) for value in REQUESTED_SLOT_DURATIONS])].copy()
    covariates: list[str] = []
    safe_names: dict[float, str] = {}
    valid_durations: list[float] = []

    for duration in REQUESTED_SLOT_DURATIONS:
        subset = work[work["duration"] == float(duration)]
        if subset["female"].nunique() != 2:
            continue
        valid_durations.append(float(duration))
        safe = str(duration).replace(".", "_")
        safe_names[float(duration)] = safe
        dur_col = f"slot_{safe}"
        fem_col = f"fem_slot_{safe}"
        work[dur_col] = (work["duration"] == float(duration)).astype(float)
        work[fem_col] = work["female"] * work[dur_col]
        if duration != valid_durations[0]:
            covariates.append(dur_col)
        covariates.append(fem_col)

    work = add_demeaned_columns(work, ["price", *covariates])
    formula = "price_dm ~ " + " + ".join(f"{column}_dm" for column in covariates) + " - 1"
    model = ols(formula, data=work).fit(cov_type="cluster", cov_kwds={"groups": work["id"]})

    rows = []
    for duration in valid_durations:
        safe = safe_names[duration]
        key = f"fem_slot_{safe}_dm"
        subset = work[work["duration"] == duration]
        rows.append(
            {
                "estimate_type": "requested_slot_regression",
                "duration_or_bin": fmt_duration(duration),
                "duration": duration,
                "cells": math.nan,
                "salons": int(subset["id"].nunique()),
                "listings": int(len(subset)),
                "male_listings": int((subset["female"] == 0).sum()),
                "female_listings": int((subset["female"] == 1).sum()),
                "gap": float(model.params[key]),
                "se": float(model.bse[key]),
                "median_gap": math.nan,
                "share_positive": math.nan,
                "pvalue": float(model.pvalues[key]),
                "r2": float(model.rsquared),
            }
        )
    return pd.DataFrame(rows)


def fit_same_duration_pair_regression(df: pd.DataFrame, overlap_cells: pd.DataFrame) -> pd.DataFrame:
    overlap_keys = overlap_cells[["id", "duration"]].drop_duplicates()
    work = df.merge(overlap_keys, on=["id", "duration"], how="inner", validate="many_to_one")
    cell_means = work.groupby(["id", "duration"], observed=True)[["price", "female"]].transform("mean")
    work["price_cell_dm"] = work["price"] - cell_means["price"]
    work["female_cell_dm"] = work["female"] - cell_means["female"]
    model = ols("price_cell_dm ~ female_cell_dm - 1", data=work).fit(
        cov_type="cluster",
        cov_kwds={"groups": work["id"]},
    )
    return pd.DataFrame(
        [
            {
                "estimate_type": "same_duration_pair_regression",
                "duration_or_bin": "All exact overlaps",
                "duration": math.nan,
                "cells": int(len(overlap_cells)),
                "salons": int(work["id"].nunique()),
                "listings": int(len(work)),
                "male_listings": int((work["female"] == 0).sum()),
                "female_listings": int((work["female"] == 1).sum()),
                "gap": float(model.params["female_cell_dm"]),
                "se": float(model.bse["female_cell_dm"]),
                "median_gap": math.nan,
                "share_positive": math.nan,
                "pvalue": float(model.pvalues["female_cell_dm"]),
                "r2": float(model.rsquared),
            }
        ]
    )


def render_overlap_table(
    df: pd.DataFrame,
    distribution: pd.DataFrame,
    overlap_cells: pd.DataFrame,
    paired_rows: pd.DataFrame,
) -> str:
    salons_with_overlap = overlap_cells["id"].nunique()
    listings_with_overlap = df[df["id"].isin(set(overlap_cells["id"]))].shape[0]
    male_common_share = distribution.loc[distribution["gender"] == "Male-coded", "share_at_common_exact_durations"].iloc[0]
    female_common_share = distribution.loc[
        distribution["gender"] == "Female-coded", "share_at_common_exact_durations"
    ].iloc[0]
    exact_rows = paired_rows[
        paired_rows["duration_or_bin"].isin(["All exact overlaps", "30", "45", "60"])
    ].copy()

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Duration Support and Exact Same-Duration Overlap in Unisex Salons}",
        "  \\label{tab:duration_overlap}",
        "  \\small",
        "\\begin{tabular}{lcc}",
        "\\hline",
        "Metric & Value & Share \\\\",
        "\\hline",
        f"Unisex-salon listings & {fmt_int(len(df))} &  \\\\",
        f"Unisex salons & {fmt_int(df['id'].nunique())} &  \\\\",
        f"Exact same-duration overlap cells & {fmt_int(len(overlap_cells))} &  \\\\",
        f"Salons with exact same-duration overlap & {fmt_int(salons_with_overlap)} & {fmt_pct(salons_with_overlap / df['id'].nunique())} \\\\",
        f"Listings in salons with exact overlap & {fmt_int(listings_with_overlap)} & {fmt_pct(listings_with_overlap / len(df))} \\\\",
        f"Male-coded listings at gender-overlap durations &  & {fmt_pct(male_common_share)} \\\\",
        f"Female-coded listings at gender-overlap durations &  & {fmt_pct(female_common_share)} \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\vspace{0.25cm}",
        "",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Duration & Cells & Salons & Mean gap & SE & Positive share \\\\",
        "\\hline",
    ]
    for row in exact_rows.itertuples(index=False):
        lines.append(
            f"{row.duration_or_bin} & {fmt_int(row.cells)} & {fmt_int(row.salons)} & "
            f"{fmt_money(row.gap)} & {fmt_money_se(row.se)} & {fmt_pct(row.share_positive)} \\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "",
            "\\begin{minipage}{0.94\\linewidth}",
            "\\footnotesize",
            "\\textit{Notes:} The sample is restricted to adult haircut listings from unisex salons.",
            "Exact same-duration overlap cells are salon-duration cells containing at least one male-coded and one female-coded listing.",
            "The mean gap is the cell-level female-coded mean price minus the male-coded mean price, in euros.",
            "\\end{minipage}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def render_category_gap_table(bin_rows: pd.DataFrame) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Female-Coded Price Gaps by Listed-Duration Category}",
        "  \\label{tab:duration_category_gaps}",
        "  \\small",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Duration category & Gap & SE & Listings & Salons & Female listings \\\\",
        "\\hline",
    ]
    for row in bin_rows.itertuples(index=False):
        coef = f"{fmt_money(row.gap)}{stars(row.pvalue)}"
        lines.append(
            f"{latex_duration_label(row.duration_or_bin)} & {coef} & {fmt_money_se(row.se)} & "
            f"{fmt_int(row.listings)} & {fmt_int(row.salons)} & {fmt_int(row.female_listings)} \\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "",
            "\\begin{minipage}{0.9\\linewidth}",
            "\\footnotesize",
            "\\textit{Notes:} Each row reports the coefficient on a female-coded listing indicator in the listed-duration category.",
            "Each row is restricted to salon-band cells containing both male- and female-coded listings and includes salon-by-duration-band fixed effects.",
            "Standard errors are clustered at the salon level.",
            "The estimates describe posted price differences within same-salon duration bands and do not estimate causal returns to time.",
            "\\end{minipage}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def render_within_duration_gap_table(bin_rows: pd.DataFrame, same_duration_row: pd.DataFrame) -> str:
    row = same_duration_row.iloc[0]
    same_duration_coef = f"{fmt_money(row['gap'])}{stars(row['pvalue'])}"
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Within-Duration Posted Price Differences in Unisex Salons}",
        "  \\label{tab:within_duration_gaps}",
        "  \\footnotesize",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Comparison & \\makecell{Salon $\\times$ band\\\\FE gap} & \\makecell{Salon $\\times$ exact-\\\\duration FE gap} & \\makecell{Band sample\\\\listings/salons} & \\makecell{Exact-FE sample\\\\listings/salons} \\\\",
        "\\hline",
        "\\multicolumn{5}{l}{\\textit{Panel A: Same salon and exact listed duration}} \\\\",
        f"All overlap cells & -- & {same_duration_coef} {fmt_money_se(row['se'])} & -- & "
        f"{fmt_int(row['listings'])}/{fmt_int(row['salons'])} \\\\",
        "\\hline",
        "\\multicolumn{5}{l}{\\textit{Panel B: Same salon and duration band}} \\\\",
    ]
    for row in bin_rows.itertuples(index=False):
        coef = f"{fmt_money(row.gap)}{stars(row.pvalue)}"
        coef_exact = f"{fmt_money(row.gap_exact_duration_fe)}{stars(row.pvalue_exact_duration_fe)}"
        lines.append(
            f"{latex_duration_label(row.duration_or_bin)} minutes & {coef} {fmt_money_se(row.se)} & "
            f"{coef_exact} {fmt_money_se(row.se_exact_duration_fe)} & "
            f"{fmt_int(row.listings)}/{fmt_int(row.salons)} & "
            f"{fmt_int(row.listings_exact_duration_fe)}/{fmt_int(row.salons_exact_duration_fe)} \\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "",
            "\\begin{minipage}{0.94\\linewidth}",
            "\\footnotesize",
            "\\textit{Notes:} Adult unisex listings.",
            "Panel A uses salon-duration cells with both genders and salon-by-exact-duration fixed effects.",
            "Panel B uses a separate, broader support rule: the band-FE column includes all salon-duration-band cells containing both genders and is not restricted to Panel A exact-overlap salons.",
            "The exact-duration FE column uses only same-salon exact-duration cells within each band, so those counts sum to Panel A.",
            "Standard errors are clustered by salon; estimates are euros and descriptive.",
            "\\end{minipage}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def render_children_within_duration_gap_table(
    bin_rows: pd.DataFrame,
    same_duration_row: pd.DataFrame,
) -> str:
    row = same_duration_row.iloc[0]
    same_duration_coef = f"{fmt_money(row['gap'])}{stars(row['pvalue'])}"
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Children's Haircuts: Same-Duration Price Differences}",
        "  \\label{tab:children_within_duration_gaps}",
        "  \\small",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Comparison & Girls--boys gap & SE & Listings/salons \\\\",
        "\\hline",
        f"All same-duration cells & {same_duration_coef} & {fmt_money_se(row['se'])} & "
        f"{fmt_int(row['listings'])}/{fmt_int(row['salons'])} \\\\",
        "\\hline",
    ]
    for row in bin_rows.itertuples(index=False):
        coef_exact = f"{fmt_money(row.gap_exact_duration_fe)}{stars(row.pvalue_exact_duration_fe)}"
        lines.append(
            f"{latex_duration_label(row.duration_or_bin)} minutes & {coef_exact} & "
            f"{fmt_money_se(row.se_exact_duration_fe)} & "
            f"{fmt_int(row.listings_exact_duration_fe)}/{fmt_int(row.salons_exact_duration_fe)} \\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "",
            "\\begin{minipage}{0.94\\linewidth}",
            "\\footnotesize",
            "\\textit{Notes:} Boys' and girls' haircut listings; unisex child listings are excluded.",
            "Each estimate uses salon-by-exact-duration fixed effects and is identified by salon-duration cells containing both boys' and girls' listings.",
            "Standard errors are clustered by salon; estimates are euros and descriptive.",
            "\\end{minipage}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def render_slot_gap_table(slot_rows: pd.DataFrame, same_duration_row: pd.DataFrame) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Exact Listed-Duration Slot Gaps in Unisex Salons}",
        "  \\label{tab:duration_slot_gaps}",
        "  \\small",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Exact duration & Gap & SE & Listings & Salons & Male/Female listings \\\\",
        "\\hline",
    ]
    for row in slot_rows.itertuples(index=False):
        coef = f"{fmt_money(row.gap)}{stars(row.pvalue)}"
        lines.append(
            f"{row.duration_or_bin} & {coef} & {fmt_money_se(row.se)} & "
            f"{fmt_int(row.listings)} & {fmt_int(row.salons)} & "
            f"{fmt_int(row.male_listings)}/{fmt_int(row.female_listings)} \\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\multicolumn{6}{l}{\\textit{Same-duration overlap sample}} \\\\",
        ]
    )
    row = same_duration_row.iloc[0]
    coef = f"{fmt_money(row['gap'])}{stars(row['pvalue'])}"
    lines.append(
        f"All overlap cells & {coef} & {fmt_money_se(row['se'])} & "
        f"{fmt_int(row['listings'])} & {fmt_int(row['salons'])} & "
        f"{fmt_int(row['male_listings'])}/{fmt_int(row['female_listings'])} \\\\"
    )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "",
            "\\begin{minipage}{0.94\\linewidth}",
            "\\footnotesize",
            "\\textit{Notes:} Exact-duration rows report coefficients on female-coded listing indicators interacted with exact listed-duration slots.",
            "The slot regression includes salon fixed effects and exact-duration fixed effects for 30, 45, 60, and 90 minutes.",
            "The overlap-sample row restricts to salon-duration cells containing both male- and female-coded listings and includes salon-duration fixed effects.",
            "Standard errors are clustered at the salon level.",
            "\\end{minipage}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def save_duration_figure(df: pd.DataFrame) -> None:
    counts = (
        df[df["duration"].isin([float(value) for value in COMMON_DURATIONS])]
        .groupby(["duration", "female"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex([float(value) for value in COMMON_DURATIONS], fill_value=0)
    )
    male_share = counts.get(0, pd.Series(0, index=counts.index)) / (df["female"] == 0).sum()
    female_share = counts.get(1, pd.Series(0, index=counts.index)) / (df["female"] == 1).sum()

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x_positions = range(len(COMMON_DURATIONS))
    width = 0.38
    ax.bar([x - width / 2 for x in x_positions], male_share * 100, width=width, label="Male-coded", color="#4C78A8")
    ax.bar(
        [x + width / 2 for x in x_positions],
        female_share * 100,
        width=width,
        label="Female-coded",
        color="#D65F5F",
    )
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([fmt_duration(value) for value in COMMON_DURATIONS])
    ax.set_xlabel("Listed duration (minutes)")
    ax.set_ylabel("Share of gender-coded listings (%)")
    ax.set_title("Duration distribution in unisex salons")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_duration_distribution_by_gender.png", dpi=300)
    plt.close(fig)


def main() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    adults = preprocess_adults()
    unisex = restrict_unisex_salons(adults)
    children = preprocess_children_boys_girls()
    children_mixed = restrict_unisex_salons(children)

    distribution = duration_distribution_summary(unisex)
    overlap_cells = build_overlap_cells(unisex)
    paired_rows = paired_gap_rows(overlap_cells)
    bin_rows = fit_duration_bin_salon_band_gaps(unisex)
    exact_rows = fit_exact_duration_gaps(unisex)
    slot_rows = fit_requested_slot_gaps(unisex)
    same_duration_row = fit_same_duration_pair_regression(unisex, overlap_cells)
    gap_rows = pd.concat([paired_rows, bin_rows, exact_rows, slot_rows, same_duration_row], ignore_index=True)

    children_overlap_cells = build_overlap_cells(children_mixed)
    children_bin_rows = fit_duration_bin_salon_band_gaps(children_mixed)
    children_same_duration_row = fit_same_duration_pair_regression(children_mixed, children_overlap_cells)

    distribution.to_csv(DERIVED_DIR / "duration_distribution_summary.csv", index=False)
    overlap_cells.to_csv(DERIVED_DIR / "duration_overlap_cells.csv", index=False)
    gap_rows.to_csv(DERIVED_DIR / "duration_category_gap_estimates.csv", index=False)
    slot_rows.to_csv(DERIVED_DIR / "duration_slot_gap_estimates.csv", index=False)
    children_overlap_cells.to_csv(DERIVED_DIR / "children_duration_overlap_cells.csv", index=False)
    children_gap_rows = pd.concat([children_bin_rows, children_same_duration_row], ignore_index=True)
    children_gap_rows.to_csv(DERIVED_DIR / "children_duration_gap_estimates.csv", index=False)

    write_file(TABLES_DIR / "table_duration_overlap.tex", render_overlap_table(unisex, distribution, overlap_cells, paired_rows))
    write_file(TABLES_DIR / "table_duration_category_gaps.tex", render_category_gap_table(bin_rows))
    write_file(TABLES_DIR / "table_within_duration_gaps.tex", render_within_duration_gap_table(bin_rows, same_duration_row))
    write_file(TABLES_DIR / "table_duration_slot_gaps.tex", render_slot_gap_table(slot_rows, same_duration_row))
    write_file(
        TABLES_DIR / "table_children_within_duration_gaps.tex",
        render_children_within_duration_gap_table(children_bin_rows, children_same_duration_row),
    )
    save_duration_figure(unisex)

    print("Duration-overlap analysis complete.")
    print(f"- Adult unisex sample: {len(unisex):,} listings, {unisex['id'].nunique():,} salons")
    print(f"- Exact same-duration overlap cells: {len(overlap_cells):,}")
    print(f"- Salons with exact overlap: {overlap_cells['id'].nunique():,}")
    print(f"- Children boys/girls mixed-salon sample: {len(children_mixed):,} listings, {children_mixed['id'].nunique():,} salons")
    print(f"- Children exact same-duration overlap cells: {len(children_overlap_cells):,}")
    print(f"- Children salons with exact overlap: {children_overlap_cells['id'].nunique():,}")
    print("\nDistribution summary:")
    print(distribution.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nPaired exact-duration gaps:")
    print(paired_rows.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nSame-salon duration-band regression gaps:")
    print(bin_rows.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nExact-duration regression gaps:")
    print(exact_rows.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nRequested 30/45/60/90 slot regression gaps:")
    print(slot_rows.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nSame-duration overlap-sample regression:")
    print(same_duration_row.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nChildren same-duration overlap-sample regression:")
    print(children_same_duration_row.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nChildren same-salon duration-band regression gaps:")
    print(children_bin_rows.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
