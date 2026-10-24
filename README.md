# Gender Pricing and Service Duration: Evidence from European Salons — Replication Package

This repository contains the scraping and analysis code needed to reproduce the empirical workflow for the paper *Gender Pricing and Service Duration: Evidence from European Salons* without redistributing platform data.

## Repository Map

- `analysis/`: scripts required to regenerate the paper tables.
- `scraping/`: Treatwell collection scripts for adult, children, and venue metadata.
- `paper/`: final manuscript PDF and LaTeX source, bibliography, and generated table and figure inputs.
- `data/`: empty directory structure plus documentation for where locally generated files belong.

## Data Policy

Treatwell data are not included here. The platform's Terms of Service do not permit redistribution of the scraped datasets or raw API payloads.

The package therefore includes:

- code
- table-generation scripts
- the final manuscript PDF and its LaTeX source, bibliography, and JPEG figure
- generated LaTeX table files and the paper-facing duration figure
- directory-level documentation

The package does not include:

- raw platform extracts
- cleaned CSV snapshots
- venue-response dumps
- derived audit CSVs and JSON files

## Python Requirements

Install the main dependencies with:

```bash
pip install pandas numpy scipy statsmodels requests reverse_geocoder stargazer matplotlib jupyter
```

## Replication Workflow

### 1. Prepare the original historical inputs

Exact reproduction requires the original June 2025 adult and children's listings and November 2025 venue metadata. These restricted files are not distributed in this repository. If you already have access to the original inputs, place them at these paths:

```text
data/snapshots/treatwell_without_raw-all-2025-06-02.csv
data/snapshots/treatwell-all-2025-06-02.csv
data/snapshots/treatwell_without_raw_kids-2025-06-03.csv
data/snapshots/venue_info-2025-11-23.csv
```

The raw adult file is required for the matched-title robustness analysis. The analysis scripts use these historical filenames directly. Without the original inputs, the published tables cannot be regenerated exactly; the generated table files remain available in `paper/tables/`.

### 2. Rebuild paper tables

Run the duration-overlap step first because it writes the same-duration and children tables plus the duration-distribution figure:

```bash
python analysis/duration_overlap_analysis.py
```

Then rebuild the main and robustness tables:

```bash
python analysis/generate_paper_tables.py
```

Generated outputs are written to:

- `paper/tables/`
- `paper/figures/`
- `data/derived/`

Optional diagnostic scripts used during revision are also included:

- `analysis/analyze_treatment_ids.py`
- `analysis/country_heterogeneity_check.py`
- `analysis/linearity_check.py`
- `analysis/structured_option_subsample.py`

### Collecting a new sample

The collection scripts are included to document how the data were obtained:

```bash
python scraping/crawl-treatwell.py
python scraping/crawl-treatwell.py --kids
```

These commands collect current listings and write files dated on the collection day. A fresh scrape cannot recreate the historical sample. To analyze a new sample, update the input paths in the analysis scripts to its filenames; do not rename new extracts to the historical filenames. The venue metadata script, `scraping/scrape-venue-info.py`, also reads the historical adult filename by default, so its `csv_file` path must be updated before collecting venue metadata for a new sample.

## Manuscript

`paper/manuscript.pdf` is the final submission PDF. Its source is `paper/manuscript.tex`; all tables are embedded in that file. The bibliography and JPEG figure are included alongside it. To compile:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
```

## Notes On Scope

This package intentionally excludes internal review artifacts, scratch notebooks, local machine paths, and restricted data files.
