# Gender Pricing and Service Duration: Evidence from European Salons — Replication Package

This repository contains the scraping and analysis code needed to reproduce the empirical workflow for the paper *Gender Pricing and Service Duration: Evidence from European Salons* without redistributing platform data.

## Repository Map

- `analysis/`: scripts required to regenerate the paper tables.
- `scraping/`: Treatwell collection scripts for adult, children, and venue metadata.
- `paper/`: generated LaTeX table and figure inputs used by the paper.
- `data/`: empty directory structure plus documentation for where locally generated files belong.

## Data Policy

Treatwell data are not included here. The platform's Terms of Service do not permit redistribution of the scraped datasets or raw API payloads.

The package therefore includes:

- code
- table-generation scripts
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

### 1. Collect raw data locally

```bash
python scraping/crawl-treatwell.py
python scraping/crawl-treatwell.py --kids
python scraping/scrape-venue-info.py
```

These scripts write dated files into `data/snapshots/`.

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

## Notes On Scope

This package intentionally excludes internal review artifacts, scratch notebooks, local machine paths, and restricted data files.
