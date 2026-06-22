# Data Directory

This replication package does not ship Treatwell data. Generate all files locally.

## Directory Layout

- `data/snapshots/`: dated scraper outputs and venue metadata files.
- `data/derived/`: regenerated intermediate analysis files required by the paper table workflow.
- `data/audits/`: regenerated audit samples and inspection outputs used by robustness diagnostics.
- `data/final/`: reserved for curated, reproducibility-critical local inputs if needed later.

## Expected Snapshot Files

- `treatwell-all-{date}.csv`
- `treatwell_without_raw-all-{date}.csv`
- `treatwell_kids-{date}.csv`
- `treatwell_without_raw_kids-{date}.csv`
- `venue_info-{date}.csv`

## Expected Derived Files

- `matched_category_model_summary.json`
- `duration_distribution_summary.csv`
- `duration_category_gap_estimates.csv`
- `duration_slot_gap_estimates.csv`
- `children_duration_gap_estimates.csv`

## Terms Of Service

Do not commit or redistribute generated platform data from these directories.
