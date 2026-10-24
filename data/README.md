# Data Directory

This replication package does not ship Treatwell data. Exact reproduction requires access to the original historical inputs listed below. A fresh scrape collects current listings and cannot recreate the historical sample.

## Directory Layout

- `data/snapshots/`: dated scraper outputs and venue metadata files.
- `data/derived/`: regenerated intermediate analysis files required by the paper table workflow.
- `data/audits/`: regenerated audit samples and inspection outputs used by robustness diagnostics.
- `data/final/`: reserved for curated, reproducibility-critical local inputs if needed later.

## Required Historical Snapshot Files

- `snapshots/treatwell_without_raw-all-2025-06-02.csv`: adult listings.
- `snapshots/treatwell-all-2025-06-02.csv`: raw adult listings used for matched-title enrichment.
- `snapshots/treatwell_without_raw_kids-2025-06-03.csv`: children's listings.
- `snapshots/venue_info-2025-11-23.csv`: venue metadata used for the single-employee restriction.

The analysis scripts read these exact filenames. Do not substitute fresh extracts under the historical names. For a new sample, update the script input paths to point to its own dated files. See the root README for collection and replication instructions.

## Expected Derived Files

- `matched_category_model_summary.json`
- `duration_distribution_summary.csv`
- `duration_category_gap_estimates.csv`
- `duration_slot_gap_estimates.csv`
- `children_duration_gap_estimates.csv`

## Terms Of Service

Do not commit or redistribute generated platform data from these directories.
