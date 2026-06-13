# Beyond Hotspots Replication Code

This repository contains the core analysis code for the manuscript:

**Beyond Hotspots: An Interpretable Modeling Framework for Risk to Humanitarian Aid**

The package is intentionally lightweight. It includes code for data construction, Cox model tournament, RSF/GBS survival-model comparison, residual diagnostics, and reviewer-requested robustness checks. It does **not** include raw event data, processed analytical datasets, model objects, result tables, figures, or plotting scripts.

## Repository Contents

```text
core/
  config.py                         # Region definitions, time windows, feature labels
  data_processing.py                # Aid/attack filtering and survival-dataset construction
  modeling.py                       # Cox, RSF, GBS, 5-fold CV, residual metrics
  run_stage1.py                     # Stage 1 Cox tournament and scale selection
  run_stage2.py                     # Stage 2 RSF/GBS comparison with stratified 5-fold CV
  residual_diagnostics.py           # Cox-Snell residual fit metrics for RSF and GBS
  actor_proxy_robustness.py         # ActorDiv/SectorComp robustness check
  dominant_country_robustness.py    # Dominant-country exclusion robustness check
  reconstruct_polecat_enhanced.py   # Raw-to-enhanced data audit helper
docs/
  DATA_AVAILABILITY.md              # Data source links and local file layout
requirements.txt
```

## Data

Data files are not included because they are large and should be obtained from the original data source. See `docs/DATA_AVAILABILITY.md` for download links and expected local paths.

Expected local structure after downloading/preparing data:

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

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The original analysis was run in a Conda environment with Python and the scientific stack listed in `requirements.txt`. Package versions may need adjustment depending on platform support for `scikit-survival`.

## Core Workflow

Run Stage 1 Cox tournament and scale selection:

```bash
python core/run_stage1.py --output-dir result/stage1
```

Run Stage 2 RSF/GBS comparison using stratified 5-fold cross-validation:

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

Run reviewer-requested robustness checks:

```bash
python core/actor_proxy_robustness.py \
  --stage1-selection result/stage1/results/stage1_final_selection.pkl \
  --output-dir result/actor_proxy_robustness

python core/dominant_country_robustness.py \
  --output-dir result/dominant_country_robustness
```

## Notes

- Stage 2 uses stratified 5-fold cross-validation as the primary validation design.
- Rolling temporal validation and plotting scripts are not included in this lightweight release.
- Generated results are written to `result/`, which is ignored by git.
- Figures and manuscript PDFs are not included in this repository.
