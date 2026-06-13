"""
Reconstruct and audit the POLECAT merged-cleaned-enhanced event dataset.

This script documents the upstream preprocessing step from Dataverse NGEC
annual TSV files to the archived analysis-ready event dataset. It is designed
for reproducibility/audit tables rather than for changing downstream analyses.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "dataverse_files"
DEFAULT_ARCHIVE = PROJECT_ROOT / "data" / "POLECAT_merged_cleaned_enhanced.parquet"


def snake_case(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def geo_accuracy_level(feature_type: pd.Series) -> pd.Series:
    mapping = {"PCLI": 0, "ADM1": 1, "ADM2": 2, "ADM3": 3, "PPL": 4}
    return feature_type.fillna("Unknown").map(mapping).fillna(5).astype("int64")


def conflict_level(event_intensity: pd.Series) -> pd.Series:
    values = pd.to_numeric(event_intensity, errors="coerce")
    labels = np.select(
        [
            values <= -8,
            (values > -8) & (values <= -5),
            (values > -5) & (values < 0),
        ],
        [
            "High-Intensity Conflict",
            "Medium-Intensity Conflict",
            "Low-Intensity Conflict",
        ],
        default="Neutral/Cooperative",
    )
    return pd.Series(labels, index=event_intensity.index, dtype="object")


def transform_raw_chunk(chunk: pd.DataFrame, source_file: str) -> pd.DataFrame:
    chunk = chunk.rename(columns={c: snake_case(c) for c in chunk.columns}).copy()

    for date_col in ["event_date", "publication_date"]:
        if date_col in chunk.columns:
            chunk[date_col] = pd.to_datetime(chunk[date_col], errors="coerce")

    for numeric_col in [
        "event_intensity",
        "latitude",
        "longitude",
        "actor_cow",
        "recipient_cow",
    ]:
        if numeric_col in chunk.columns:
            chunk[numeric_col] = pd.to_numeric(chunk[numeric_col], errors="coerce")

    for cow_col in ["actor_cow", "recipient_cow"]:
        if cow_col in chunk.columns:
            chunk[cow_col] = chunk[cow_col].fillna(0).astype("int64")

    if "event_mode" in chunk.columns:
        chunk["event_mode"] = chunk["event_mode"].fillna("Unknown").replace({"None": "Unknown"})
    if "country" in chunk.columns:
        chunk["country"] = chunk["country"].replace({"None": pd.NA, "": pd.NA})
    if "feature_type" in chunk.columns:
        chunk["feature_type"] = chunk["feature_type"].fillna("Unknown").replace({"None": "Unknown"})

    for sector_col in ["primary_actor_sector", "primary_recipient_sector"]:
        if sector_col in chunk.columns:
            chunk[sector_col] = chunk[sector_col].fillna("UNKNOWN").replace({"None": "UNKNOWN"})

    chunk["source_file"] = source_file
    chunk["geo_accuracy_level"] = geo_accuracy_level(chunk.get("feature_type", pd.Series(index=chunk.index, dtype="object")))

    if {"publication_date", "event_date"}.issubset(chunk.columns):
        delay = (chunk["publication_date"] - chunk["event_date"]).dt.days
        chunk["reporting_delay"] = delay.fillna(0).astype("int64")
    else:
        chunk["reporting_delay"] = 0

    if "event_date" in chunk.columns:
        chunk["event_day_of_week"] = chunk["event_date"].dt.day_name()
        chunk["is_weekend"] = chunk["event_day_of_week"].isin(["Saturday", "Sunday"])
    else:
        chunk["event_day_of_week"] = pd.NA
        chunk["is_weekend"] = False

    actor = chunk.get("primary_actor_sector", pd.Series("UNKNOWN", index=chunk.index)).fillna("UNKNOWN")
    recipient = chunk.get("primary_recipient_sector", pd.Series("UNKNOWN", index=chunk.index)).fillna("UNKNOWN")
    chunk["interaction_type"] = actor.astype(str) + "_TO_" + recipient.astype(str)
    chunk["conflict_level"] = conflict_level(chunk.get("event_intensity", pd.Series(index=chunk.index, dtype="object")))
    return chunk


class Summary:
    def __init__(self) -> None:
        self.raw_input_rows = 0
        self.duplicate_event_id_rows_removed = 0
        self.rows = 0
        self.columns: set[str] = set()
        self.min_event_date = None
        self.max_event_date = None
        self.countries: set[str] = set()
        self.event_types: set[str] = set()
        self.complete_key_rows = 0
        self.source_file_counts: Counter[str] = Counter()
        self.event_type_counts: Counter[str] = Counter()
        self.country_counts: Counter[str] = Counter()

    def update(self, df: pd.DataFrame) -> None:
        self.rows += len(df)
        self.columns.update(df.columns)

        if "event_date" in df.columns:
            dates = df["event_date"].dropna()
            if not dates.empty:
                cur_min = dates.min()
                cur_max = dates.max()
                self.min_event_date = cur_min if self.min_event_date is None else min(self.min_event_date, cur_min)
                self.max_event_date = cur_max if self.max_event_date is None else max(self.max_event_date, cur_max)

        if "country" in df.columns:
            countries = df["country"].dropna().astype(str)
            self.countries.update(countries.unique())
            self.country_counts.update(countries)

        if "event_type" in df.columns:
            event_types = df["event_type"].dropna().astype(str)
            self.event_types.update(event_types.unique())
            self.event_type_counts.update(event_types)

        if "source_file" in df.columns:
            self.source_file_counts.update(df["source_file"].astype(str))

        key_cols = ["event_date", "country", "latitude", "longitude", "event_type"]
        if all(c in df.columns for c in key_cols):
            self.complete_key_rows += int(df[key_cols].notna().all(axis=1).sum())

    def as_metrics(self) -> dict[str, object]:
        return {
            "total_rows": self.rows,
            "total_columns": len(self.columns),
            "min_event_date": "" if self.min_event_date is None else str(pd.Timestamp(self.min_event_date).date()),
            "max_event_date": "" if self.max_event_date is None else str(pd.Timestamp(self.max_event_date).date()),
            "unique_countries": len(self.countries),
            "unique_event_types": len(self.event_types),
            "complete_event_date_country_lat_lon_event_type_rows": self.complete_key_rows,
        }


def summarize_raw(raw_dir: Path, chunksize: int) -> Summary:
    summary = Summary()
    seen_event_ids: set[str] = set()
    files = sorted(raw_dir.glob("ngecEvents.DV.*.txt"))
    if not files:
        raise FileNotFoundError(f"No ngecEvents.DV.*.txt files found in {raw_dir}")

    for path in files:
        for chunk in pd.read_csv(
            path,
            sep="\t",
            chunksize=chunksize,
            low_memory=False,
        ):
            transformed = transform_raw_chunk(chunk, path.name)
            summary.raw_input_rows += len(transformed)
            event_ids = transformed["event_id"].astype(str)
            duplicate_mask = event_ids.isin(seen_event_ids) | event_ids.duplicated(keep="first")
            if duplicate_mask.any():
                summary.duplicate_event_id_rows_removed += int(duplicate_mask.sum())
                transformed = transformed.loc[~duplicate_mask].copy()
                event_ids = transformed["event_id"].astype(str)
            seen_event_ids.update(event_ids.tolist())
            summary.update(transformed)
    return summary


def summarize_archive(archive_path: Path) -> Summary:
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    columns = [
        "event_date",
        "country",
        "latitude",
        "longitude",
        "event_type",
        "source_file",
    ]
    df = pd.read_parquet(archive_path, columns=columns)
    summary = Summary()
    summary.columns = set(pq.ParquetFile(archive_path).schema_arrow.names)
    summary.update(df)
    return summary


def compare_metrics(reconstructed: Summary, archived: Summary) -> pd.DataFrame:
    recon = reconstructed.as_metrics()
    arch = archived.as_metrics()
    rows = []
    for metric in recon:
        r_value = recon[metric]
        a_value = arch[metric]
        rows.append(
            {
                "metric": metric,
                "reconstructed_from_raw": r_value,
                "archived_enhanced_dataset": a_value,
                "match": r_value == a_value,
                "difference": "" if isinstance(r_value, str) else r_value - a_value,
            }
        )
    return pd.DataFrame(rows)


def counter_to_frame(counter: Counter[str], name: str) -> pd.DataFrame:
    return pd.DataFrame(counter.items(), columns=[name, "count"]).sort_values(name)


def write_markdown_report(output_dir: Path, comparison: pd.DataFrame) -> None:
    lines = [
        "# POLECAT Raw-to-Enhanced Dataset Audit",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This audit reconstructs the merged-cleaned-enhanced event-level dataset from the Dataverse annual NGEC files and compares the resulting core dataset-level metrics with the archived enhanced dataset used by the modeling scripts.",
        "",
        "## Summary",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Raw Input Deduplication",
        "",
        "The Dataverse annual files contain a small number of repeated `Event ID` rows. The archived enhanced dataset keeps one row per `event_id`; the reconstruction therefore applies the same deterministic de-duplication before comparison.",
        "",
        "## Interpretation",
        "",
        "Matching row counts, date ranges, source-file counts, and complete-key counts support the provenance claim that the archived enhanced dataset was constructed from the raw Dataverse annual POLECAT/NGEC event files with deterministic field harmonization and derived audit fields.",
        "",
    ]
    (output_dir / "raw_to_enhanced_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "result" / f"raw_to_enhanced_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    archive = Path(args.archive)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    reconstructed = summarize_raw(raw_dir, args.chunksize)
    archived = summarize_archive(archive)

    comparison = compare_metrics(reconstructed, archived)
    comparison.to_csv(tables_dir / "raw_to_enhanced_audit_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "raw_input_rows_before_deduplication": reconstructed.raw_input_rows,
                "duplicate_event_id_rows_removed": reconstructed.duplicate_event_id_rows_removed,
                "rows_after_deduplication": reconstructed.rows,
            }
        ]
    ).to_csv(tables_dir / "raw_event_id_deduplication_summary.csv", index=False, encoding="utf-8-sig")
    counter_to_frame(reconstructed.source_file_counts, "source_file").to_csv(
        tables_dir / "reconstructed_source_file_counts.csv", index=False, encoding="utf-8-sig"
    )
    counter_to_frame(archived.source_file_counts, "source_file").to_csv(
        tables_dir / "archived_source_file_counts.csv", index=False, encoding="utf-8-sig"
    )
    counter_to_frame(reconstructed.event_type_counts, "event_type").to_csv(
        tables_dir / "reconstructed_event_type_counts.csv", index=False, encoding="utf-8-sig"
    )
    counter_to_frame(reconstructed.country_counts, "country").to_csv(
        tables_dir / "reconstructed_country_counts.csv", index=False, encoding="utf-8-sig"
    )
    write_markdown_report(output_dir, comparison)

    print(f"Audit saved to: {output_dir}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
