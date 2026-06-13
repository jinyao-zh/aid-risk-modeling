# Data Availability

This repository does not include raw or processed event data.

## Primary Event Data

The analysis uses POLECAT/PLOVER event data. POLECAT is a machine-coded political event dataset based on the PLOVER ontology and distributed through Harvard Dataverse. Download the relevant annual NGEC event files from the source dataset:

- POLECAT Weekly Data: https://doi.org/10.7910/DVN/AJGVIT
- PLOVER/POLECAT documentation: https://doi.org/10.7910/DVN/LMFPIP

Recommended data citation:

> Scarborough, Grace I., Benjamin E. Bagozzi, Andreas Beger, John Berrie, Andrew Halterman, Philip A. Schrodt, and Jevon Spivey. 2023. "POLECAT Weekly Data." Harvard Dataverse. https://doi.org/10.7910/DVN/AJGVIT

Please also consult and cite the associated PLOVER/POLECAT documentation when using event-type definitions, ontology fields, or coding assumptions.

## License and Use Terms

This repository does not redistribute POLECAT/PLOVER data. Users should obtain the data directly from the Harvard Dataverse source page and follow the current license and data-use terms listed there. As of June 2026, the Dataverse page listed the dataset under CC0 1.0 and included additional notes on permitted use, source-text restrictions, warranty disclaimers, and liability limitations. Because upstream terms may change, the source page should be treated as authoritative.

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
data/processed/polecat_plover_event_archive_2018_2024.parquet
```

This file is not included because it is large. The script `core/audit_polecat_event_archive.py` documents and audits the raw-to-processed data preparation check used in the analysis.

## Excluded Files

The following are intentionally excluded from the GitHub package:

- raw POLECAT text files
- processed parquet/csv event datasets
- trained model objects
- generated result tables
- figures and maps
- manuscript files and LaTeX build artifacts
