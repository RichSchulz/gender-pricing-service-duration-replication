# Paper Outputs

This directory only keeps lightweight paper-facing artifacts that are safe to redistribute.

## Included

- `paper/tables/table_main_results.tex`
- `paper/tables/table_robustness.tex`
- `paper/tables/table_country_heterogeneity.tex`
- `paper/tables/table_within_duration_gaps.tex`
- `paper/tables/table_children_within_duration_gaps.tex`
- `paper/tables/table_duration_category_gaps.tex`
- `paper/tables/table_duration_overlap.tex`
- `paper/tables/table_duration_slot_gaps.tex`
- `paper/tables/table_matched_category_robustness.tex`
- `paper/figures/figure_duration_distribution_by_gender.png`
- this README

## Excluded

- manuscript drafts
- submission files
- internal review artifacts
- generated PDFs

Rebuild the main tables with:

```bash
python analysis/duration_overlap_analysis.py
python analysis/generate_paper_tables.py
```
