# Data Availability

This repository does not include raw or processed event data.

## Primary Event Data

The analysis uses POLECAT/PLOVER event data. Download the relevant annual NGEC event files from the source dataset:

- POLECAT Weekly Data: https://doi.org/10.7910/DVN/AJGVIT
- PLOVER/POLECAT documentation: https://doi.org/10.7910/DVN/LMFPIP

Place the downloaded annual files under:

```text
data/dataverse_files/
```

Expected filenames:

```text
ngecEvents.DV.2018.txt
ngecEvents.DV.2019.txt
ngecEvents.DV.2020.txt
ngecEvents.DV.2021.txt
ngecEvents.DV.2022.txt
ngecEvents.DV.2023.txt
ngecEvents.DV.2024.txt
```

## Analysis-Ready Dataset

The core scripts expect the processed event archive at:

```text
data/POLECAT_merged_cleaned_enhanced.parquet
```

This file is not included because it is large. The script `core/reconstruct_polecat_enhanced.py` documents and audits the raw-to-enhanced data consistency check used for the manuscript revision.

## Excluded Files

The following are intentionally excluded from the GitHub package:

- raw POLECAT text files
- processed parquet/csv event datasets
- trained model objects
- generated result tables
- figures and maps
- manuscript PDFs and LaTeX build artifacts
