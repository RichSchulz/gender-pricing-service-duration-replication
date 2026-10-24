# Paper Outputs

This directory only keeps lightweight paper-facing artifacts that are safe to redistribute.

## Included

- `paper/manuscript.pdf`: final submission PDF.
- `paper/manuscript.tex`: final submission LaTeX source, with all manuscript tables embedded.
- `paper/references.bib`: manuscript bibliography.
- `paper/figure_duration_distribution_by_gender.jpeg`: figure used by the manuscript source.
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

- superseded manuscript drafts and submission bundles
- internal review artifacts
- intermediate LaTeX build files

## Build The Manuscript

From the repository root:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
```

The manuscript files were copied from the final submission. The PNG in `paper/figures/` is the analysis-generated figure; the manuscript uses the JPEG included alongside its source.

## Rebuild Analysis Outputs

From the repository root, with the historical inputs described in `data/README.md`:

```bash
python analysis/duration_overlap_analysis.py
python analysis/generate_paper_tables.py
```

These commands regenerate the separate table files and the PNG figure. They do not modify the tables embedded in `paper/manuscript.tex`.
