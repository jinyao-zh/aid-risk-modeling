"""
Prepare the local POLECAT/PLOVER event archive used by the analysis scripts.

The script reads annual NGEC event files downloaded from Harvard Dataverse,
standardizes field names and core data types, adds a small set of derived
metadata fields, removes duplicate event IDs, and writes a local Parquet file.
The generated Parquet archive is ignored by git.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "dataverse_files"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "polecat_plover_event_archive_2018_2024.parquet"


def import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyarrow is required to write the processed Parquet archive. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return pa, pq


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
    return pd.Series(labels, index=event_intensity.index, dtype="string")


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
    chunk["geo_accuracy_level"] = geo_accuracy_level(
        chunk.get("feature_type", pd.Series(index=chunk.index, dtype="string"))
    )

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
    chunk["conflict_level"] = conflict_level(chunk.get("event_intensity", pd.Series(index=chunk.index)))

    for col in chunk.select_dtypes(include=["object"]).columns:
        chunk[col] = chunk[col].astype("string")

    return chunk


class PreparationSummary:
    def __init__(self) -> None:
        self.raw_input_rows = 0
        self.duplicate_or_missing_event_id_rows_removed = 0
        self.output_rows = 0
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
        self.output_rows += len(df)
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
            self.source_file_counts.update(df["source_file"].dropna().astype(str))

        key_cols = ["event_date", "country", "latitude", "longitude", "event_type"]
        if all(c in df.columns for c in key_cols):
            self.complete_key_rows += int(df[key_cols].notna().all(axis=1).sum())

    def as_row(self, output_path: Path) -> dict[str, object]:
        return {
            "output_path": str(output_path),
            "raw_input_rows": self.raw_input_rows,
            "duplicate_or_missing_event_id_rows_removed": self.duplicate_or_missing_event_id_rows_removed,
            "output_rows": self.output_rows,
            "total_columns": len(self.columns),
            "min_event_date": "" if self.min_event_date is None else str(pd.Timestamp(self.min_event_date).date()),
            "max_event_date": "" if self.max_event_date is None else str(pd.Timestamp(self.max_event_date).date()),
            "unique_countries": len(self.countries),
            "unique_event_types": len(self.event_types),
            "complete_event_date_country_lat_lon_event_type_rows": self.complete_key_rows,
        }


def get_raw_files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("ngecEvents.DV.*.txt"))
    if not files:
        raise FileNotFoundError(f"No ngecEvents.DV.*.txt files found in {raw_dir}")
    return files


def deduplicate_events(df: pd.DataFrame, seen_event_ids: set[str], summary: PreparationSummary) -> pd.DataFrame:
    if "event_id" not in df.columns:
        raise KeyError("Expected an `event_id` column after raw field-name normalization.")

    event_ids = df["event_id"].astype("string")
    missing_event_id = event_ids.isna() | (event_ids.str.len() == 0)
    duplicate_event_id = event_ids.isin(seen_event_ids) | event_ids.duplicated(keep="first")
    remove_mask = missing_event_id | duplicate_event_id

    if remove_mask.any():
        summary.duplicate_or_missing_event_id_rows_removed += int(remove_mask.sum())
        df = df.loc[~remove_mask].copy()
        event_ids = event_ids.loc[~remove_mask]

    seen_event_ids.update(event_ids.astype(str).tolist())
    return df


def table_from_frame(df: pd.DataFrame, schema: pa.Schema | None, columns: list[str] | None) -> pa.Table:
    pa, _ = import_pyarrow()
    if columns is not None:
        for col in columns:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[columns]

    table = pa.Table.from_pandas(df, preserve_index=False)
    if schema is not None:
        table = table.cast(schema, safe=False)
    return table


def write_counter(counter: Counter[str], path: Path, column_name: str) -> None:
    pd.DataFrame(counter.items(), columns=[column_name, "count"]).sort_values(column_name).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def write_preparation_outputs(summary: PreparationSummary, output_path: Path, summary_dir: Path) -> None:
    tables_dir = summary_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame([summary.as_row(output_path)])
    summary_df.to_csv(tables_dir / "event_archive_preparation_summary.csv", index=False, encoding="utf-8-sig")
    write_counter(summary.source_file_counts, tables_dir / "source_file_counts.csv", "source_file")
    write_counter(summary.event_type_counts, tables_dir / "event_type_counts.csv", "event_type")
    write_counter(summary.country_counts, tables_dir / "country_counts.csv", "country")

    row = summary.as_row(output_path)
    lines = [
        "# POLECAT/PLOVER Event Archive Preparation",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This report summarizes the local preparation of the processed POLECAT/PLOVER event archive.",
        "",
        "## Summary",
        "",
        f"- Output file: `{output_path}`",
        f"- Raw input rows: {row['raw_input_rows']}",
        f"- Removed duplicate or missing event IDs: {row['duplicate_or_missing_event_id_rows_removed']}",
        f"- Output rows: {row['output_rows']}",
        f"- Columns: {row['total_columns']}",
        f"- Date range: {row['min_event_date']} to {row['max_event_date']}",
        f"- Countries: {row['unique_countries']}",
        f"- Event types: {row['unique_event_types']}",
        "",
    ]
    (summary_dir / "event_archive_preparation_report.md").write_text("\n".join(lines), encoding="utf-8")


def prepare_event_archive(
    raw_dir: Path,
    output_path: Path,
    summary_dir: Path,
    chunksize: int,
    overwrite: bool,
) -> PreparationSummary:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    raw_files = get_raw_files(raw_dir)
    logging.info("Found %d raw files in %s", len(raw_files), raw_dir)

    summary = PreparationSummary()
    seen_event_ids: set[str] = set()
    _, pq = import_pyarrow()
    writer = None
    output_schema = None
    output_columns: list[str] | None = None

    try:
        for path in raw_files:
            logging.info("Processing %s", path.name)
            for chunk in pd.read_csv(path, sep="\t", chunksize=chunksize, low_memory=False):
                transformed = transform_raw_chunk(chunk, path.name)
                summary.raw_input_rows += len(transformed)
                transformed = deduplicate_events(transformed, seen_event_ids, summary)
                if transformed.empty:
                    continue

                if output_columns is None:
                    output_columns = list(transformed.columns)

                table = table_from_frame(transformed, output_schema, output_columns)
                if writer is None:
                    output_schema = table.schema
                    writer = pq.ParquetWriter(output_path, output_schema, compression="snappy")

                writer.write_table(table)
                summary.update(transformed)
    finally:
        if writer is not None:
            writer.close()

    write_preparation_outputs(summary, output_path, summary_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--summary-dir",
        default=str(PROJECT_ROOT / "result" / f"data_preparation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    summary = prepare_event_archive(
        raw_dir=Path(args.raw_dir),
        output_path=Path(args.output),
        summary_dir=Path(args.summary_dir),
        chunksize=args.chunksize,
        overwrite=args.overwrite,
    )
    print(f"Processed event archive saved to: {args.output}")
    print(pd.DataFrame([summary.as_row(Path(args.output))]).to_string(index=False))


if __name__ == "__main__":
    main()
