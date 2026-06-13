# Aid Risk Modeling

This repository provides code-only replication materials for an analysis of post-aid violence risk in humanitarian settings.

The package is intentionally lightweight. It includes scripts for event-data preparation, survival-dataset construction, Cox proportional hazards model comparison, RSF/GBS survival-model evaluation, residual diagnostics, and robustness analyses. Raw event data, processed analytical datasets, trained model objects, generated result tables, figures, and plotting scripts are not included.

## Repository Contents

```text
core/
  config.py                         # Region definitions, time windows, feature labels
  data_processing.py                # Aid/attack filtering and survival-dataset construction
  modeling.py                       # Survival models, cross-validation, residual metrics
  run_stage1.py                     # Cox model comparison and scale selection
  run_stage2.py                     # RSF/GBS comparison with stratified 5-fold CV
  residual_diagnostics.py           # Cox-Snell residual diagnostics for RSF and GBS
  actor_proxy_robustness.py         # Actor-proxy sensitivity analysis
  dominant_country_robustness.py    # Dominant-country exclusion analysis
  reconstruct_polecat_enhanced.py   # Data-preparation audit script
docs/
  DATA_AVAILABILITY.md              # Data source links and local file layout
requirements.txt
```

## Data

Event data are not redistributed in this repository. See `docs/DATA_AVAILABILITY.md` for source links and expected local paths.

Expected local structure after downloading and preparing the data:

```text
data/
  POLECAT_merged_cleaned_enhanced.parquet
  dataverse_files/
    ngecEvents.DV.2018.txt
    ngecEvents.DV.2019.txt
    ngecEvents.DV.2020.txt
    ngecEvents.DV.2021.txt
    ngecEvents.DV.2022.txt
    ngecEvents.DV.2023.txt
    ngecEvents.DV.2024.txt
```

## Setup

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

The original analysis was run in a Conda environment with the scientific Python stack listed in `requirements.txt`. Package versions may need adjustment depending on platform support for `scikit-survival`.

## Core Workflow

Run the Cox model comparison and scale-selection step:

```bash
python core/run_stage1.py --output-dir result/stage1
```

Run the RSF/GBS comparison using stratified 5-fold cross-validation:

```bash
python core/run_stage2.py \
  --stage1-results-path result/stage1/results/stage1_final_selection.pkl \
  --output-dir result/stage2
```

Compute Cox-Snell residual fit metrics:

```bash
python core/residual_diagnostics.py \
  --stage2-results-path result/stage2/results/stage2_final_results.pkl \
  --output-dir result/residual_diagnostics
```

Run robustness analyses:

```bash
python core/actor_proxy_robustness.py \
  --stage1-selection result/stage1/results/stage1_final_selection.pkl \
  --output-dir result/actor_proxy_robustness

python core/dominant_country_robustness.py \
  --output-dir result/dominant_country_robustness
```

## Notes

- The primary validation design uses stratified 5-fold cross-validation.
- Generated outputs are written to `result/`, which is ignored by git.
- This release is limited to core analysis code; data files, figures, plotting scripts, manuscript files, and build artifacts are excluded.
