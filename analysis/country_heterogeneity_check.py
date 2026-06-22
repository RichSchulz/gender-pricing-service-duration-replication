"""Check country-specific slope heterogeneity in the preferred unisex sample."""

from __future__ import annotations

from generate_paper_tables import fit_adult_models, preprocess_adults, restrict_unisex_salons


COUNTRY_NAMES = {
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


def main() -> None:
    adults = preprocess_adults()
    # Reverse geocoding assigns a small set of Rome listings to Vatican City.
    adults["country"] = adults["country"].replace({"VA": "IT"})
    unisex = restrict_unisex_salons(adults)

    rows: list[tuple[str, str, int, int, float, float, float]] = []
    for country, frame in sorted(unisex.groupby("country")):
        _, model = fit_adult_models(frame)
        rows.append(
            (
                country,
                COUNTRY_NAMES.get(country, country),
                len(frame),
                int(frame["id"].nunique()),
                float(model.model.params["fem_dur_dm"]),
                float(model.model.bse["fem_dur_dm"]),
                float(model.model.pvalues["fem_dur_dm"]),
            )
        )

    print("country\tname\tobs\tsalons\tbeta3\tse\tp")
    for row in rows:
        print(
            f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t"
            f"{row[4]:.3f}\t{row[5]:.3f}\t{row[6]:.3g}"
        )

    positive = sum(row[4] > 0 for row in rows)
    print(f"\nPositive interactions: {positive} of {len(rows)} countries")


if __name__ == "__main__":
    main()
