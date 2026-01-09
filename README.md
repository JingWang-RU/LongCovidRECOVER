# Long COVID Recovery Trajectories (RECOVER) — Analysis Notebooks (2026)

This repository contains two primary Jupyter notebooks supporting the manuscript:

**“Immunological Density Shapes Recoverty Trajectories in Long COVID”**

The notebooks implement (i) LLM-assisted trajectory review for hypothesis generation and data quality checks, and (ii) the statistical analyses and Nature-style figure generation used in the main text.

---

## Contents

### 1) `2026LLM_Recover.ipynb` — LLM-assisted trajectory review (hypothesis-generating)
This notebook demonstrates how large language models (LLMs) are used to read longitudinal patient timelines and produce concise, hypothesis-generating summaries.

**Scope**
- Constructs per-participant timelines from longitudinal observations (vaccination events + follow-up visits + PASC scores).
- Performs **data-quality–aware** preprocessing (e.g., sorting, de-duplication, missingness handling).
- Applies an LLM prompt template that explicitly:
  - treats outputs as **non-causal** and **non-diagnostic**
  - acknowledges measurement noise and incomplete follow-up
  - requests within-patient and pairwise pattern comparisons (when multiple patients are shown)

**Primary output**
- A structured summary block suitable for inclusion in the manuscript as “LLM insight (hypothesis-generating)”.

**Notes on interpretation**
- LLM outputs are used to **guide** statistical analysis decisions and narrative framing.
- LLM outputs **must not** be interpreted as evidence of causality, clinical recommendations, or individual-level diagnosis.

---

### 2) `2026plot.ipynb` — Statistical analyses + Nature-style figures
This notebook produces the quantitative results and figures used in the manuscript.

**Analyses implemented**
- **Phenotype stratification**: Protected / Responder / Refractory definitions based on PASC threshold and longitudinal pattern.
- **Longitudinal summary statistics**:
  - Initial vs. last PASC distributions
  - Outcome classification (Improved/Unchanged/Worsened) by phenotype
  - Follow-up duration distributions
- **Dose-associated effects**
  - Mean initial PASC by prior vaccine count
  - Dose-to-dose step changes (waterfall-style summary)
  - Dose-0–relative percent change by phenotype (with 95% CI)
- **Correlation analyses**
  - Pearson correlations between PASC severity and time since vaccination
  - Pearson correlations between PASC severity and cumulative prior dose count
  - Heatmap visualization with significance annotations
- **Survival analysis**
  - Kaplan–Meier curves for time-to-recovery (based on first recovered event)
  - Censoring at last follow-up for non-recovered participants
  - Optional annotations (median time-to-recovery, event counts, number-at-risk table)

**Outputs**
- Publication-quality, Nature-style figures exported as `.pdf` (and optionally `.png`) for manuscript integration.
- Tables or intermediate summaries required for figure annotation (e.g., subgroup sizes, dose strata counts).

---

## Data Requirements

Both notebooks assume access to a longitudinal dataset (e.g., `pasc_df`) containing, at minimum:

- `id`: participant identifier  
- `date`: record timestamp (visit date)  
- `name`: PASC score or record descriptor (depending on preprocessing)  
- `prior_vax_count` (or similarly named column): number of prior vaccine doses at enrollment or at a given observation  
- Additional columns as available (e.g., vaccine brand indicators, follow-up flags)

**Important**: All processing assumes that dates may be inconsistent in raw extracts; timelines are always constructed by sorting by `date` within participant.

---

## Reproducibility and Environment

# !pip install unsloth
# !pip install huggingface
# !pip install tf-keras

**Core packages**
- `numpy`, `pandas`
- `matplotlib` (Nature-style formatting settings)
- `scipy` (correlations and statistical tests)
- Optional: `seaborn` (used sparingly; most final figures use matplotlib for precise control)

---

## Figure Mapping (Manuscript Integration)

The following figure families are produced in `2026plot.ipynb`:

- **Dose–response by phenotype** (mean ± 95% CI vs cumulative doses)
- **Dose distribution at initial observation**
- **Raincloud distributions** (initial and peak PASC by phenotype)
- **Three-panel summary**: initial vs last, outcome composition, follow-up duration
- **Dose-0–relative percent change** (0–4 doses)
- **Waterfall / step-change plot** for prior-dose strata
- **Correlation heatmap** (Time–Severity and Dose–Severity across phenotypes)
- **Vaccine dose burden + brand composition** (bar + pies)
- **Kaplan–Meier time-to-recovery curves** (by phenotype)

All figures are exported in vector format (`.pdf`) for LaTeX submission.

---

## Recommended Workflow

1. Run **`2026LLM_Recover.ipynb`** to:
   - validate timeline construction logic
   - generate hypothesis-generating summaries
   - identify potential data quality issues that may affect modeling choices

2. Run **`2026plot.ipynb`** to:
   - compute final reported statistics
   - generate final figures for manuscript submission
   - export PDFs for LaTeX inclusion

---

## Scientific and Ethical Use Notes

- The LLM summaries are intended to be **hypothesis-generating only** and must not be interpreted as clinical decisions or causal inference.
- All statistical claims in the manuscript should be supported by the quantitative analyses in `2026plot.ipynb`.
- Avoid overclaiming causality from observational associations; correlations and survival curves reflect **association**, not randomized intervention effects.

---

## Contact

For questions about figure generation, phenotype definitions, or manuscript integration, refer to
jing.wang20@nih.gov
