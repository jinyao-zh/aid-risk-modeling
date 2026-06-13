"""Actor-proxy sensitivity checks.

This script treats ActorDiv and SectorComp as event-data proxies for observed
actor heterogeneity and checks whether the main Cox results are robust when
ambiguous or unknown actor labels are excluded.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.integrate

if not hasattr(scipy.integrate, "trapz"):
    scipy.integrate.trapz = np.trapz

from config import AUTHORITATIVE_REGION_CONFIG, CONFIG
from data_processing import CoxDataProcessor
from modeling import COX_MODELS_CONFIG, scale_survival_data, train_final_cox_model


STAGE1_SELECTION = CONFIG["output_base_path"] / "stage1" / "results" / "stage1_final_selection.pkl"
DEFAULT_OUTPUT_DIR = CONFIG["output_base_path"] / "actor_proxy_robustness"

AMBIGUOUS_ACTOR_LABELS = {
    "",
    "none",
    "nan",
    "unknown",
    "unknown actor",
    "unknown recipient",
    "unknown country",
}
AMBIGUOUS_SECTOR_LABELS = {"", "none", "nan", "unknown", "unk"}


def non_ambiguous_unique_count(series: pd.Series, ambiguous: set[str]) -> int:
    if series.empty:
        return 0
    cleaned = series.dropna().astype(str).str.strip()
    cleaned = cleaned[~cleaned.str.lower().isin(ambiguous)]
    return int(cleaned.nunique())


class DualActorProxyProcessor(CoxDataProcessor):
    """Build baseline actor proxies and unknown-excluded alternatives in one pass."""

    def _create_actor_features(self, hist_attacks):  # noqa: D401 - keep parent naming convention
        if hist_attacks.empty:
            return {
                "actor_diversity": 0,
                "sector_complexity": 0,
                "actor_diversity_unknown_excluded": 0,
                "sector_complexity_unknown_excluded": 0,
            }

        actor_source_col = "actor_name"
        if actor_source_col not in hist_attacks.columns and "actor_name_raw" in hist_attacks.columns:
            actor_source_col = "actor_name_raw"

        actor_diversity = hist_attacks[actor_source_col].nunique() if actor_source_col in hist_attacks.columns else 0
        sector_complexity = hist_attacks["primary_actor_sector"].nunique() if "primary_actor_sector" in hist_attacks.columns else 0

        actor_diversity_clean = (
            non_ambiguous_unique_count(hist_attacks[actor_source_col], AMBIGUOUS_ACTOR_LABELS)
            if actor_source_col in hist_attacks.columns
            else 0
        )
        sector_complexity_clean = (
            non_ambiguous_unique_count(hist_attacks["primary_actor_sector"], AMBIGUOUS_SECTOR_LABELS)
            if "primary_actor_sector" in hist_attacks.columns
            else 0
        )
        return {
            "actor_diversity": int(actor_diversity),
            "sector_complexity": int(sector_complexity),
            "actor_diversity_unknown_excluded": actor_diversity_clean,
            "sector_complexity_unknown_excluded": sector_complexity_clean,
        }


def region_key_by_name() -> dict[str, str]:
    return {cfg["name"]: key for key, cfg in AUTHORITATIVE_REGION_CONFIG.items()}


def load_stage1_selection(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def build_survival_data(region_name: str, params: dict) -> pd.DataFrame:
    region_key = region_key_by_name()[region_name]
    region_config = AUTHORITATIVE_REGION_CONFIG[region_key]
    processor = DualActorProxyProcessor(buffer_days=3)
    region_df = processor.load_region_data(region_config)
    return processor.build_cox_survival_dataset(region_df, region_config, sensitivity_params=params)


def variant_survival_data(survival_data: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == "baseline_raw_labels":
        return survival_data
    if variant != "unknown_excluded":
        raise ValueError(f"Unknown actor proxy variant: {variant}")
    adjusted = survival_data.copy()
    adjusted["actor_diversity"] = adjusted["actor_diversity_unknown_excluded"]
    adjusted["sector_complexity"] = adjusted["sector_complexity_unknown_excluded"]
    return adjusted


def summarize_actor_features(region_name: str, variant: str, survival_data: pd.DataFrame) -> dict:
    return {
        "region": region_name,
        "actor_proxy_variant": variant,
        "n_samples": len(survival_data),
        "n_events": int(survival_data["event"].sum()),
        "event_rate": float(survival_data["event"].mean()),
        "actor_diversity_mean": float(survival_data["actor_diversity"].mean()),
        "actor_diversity_median": float(survival_data["actor_diversity"].median()),
        "actor_diversity_zero_share": float((survival_data["actor_diversity"] == 0).mean()),
        "sector_complexity_mean": float(survival_data["sector_complexity"].mean()),
        "sector_complexity_median": float(survival_data["sector_complexity"].median()),
        "sector_complexity_zero_share": float((survival_data["sector_complexity"] == 0).mean()),
    }


def fit_model_summary(region_name: str, variant: str, model_name: str, survival_data: pd.DataFrame) -> list[dict]:
    scaled = scale_survival_data(survival_data)
    features = [f for f in COX_MODELS_CONFIG[model_name]["features"] if f in scaled.columns]
    model_data = scaled[["duration", "event"] + features].dropna()
    model, metrics = train_final_cox_model(model_data, features, f"{region_name}_{variant}_{model_name}")
    records = []
    if model is None:
        return [
            {
                "region": region_name,
                "actor_proxy_variant": variant,
                "model_name": model_name,
                "term": "MODEL_FAILED",
                "coef": np.nan,
                "hazard_ratio": np.nan,
                "p_value": np.nan,
                "c_index": np.nan,
                "n_samples": len(model_data),
                "n_events": int(model_data["event"].sum()),
            }
        ]

    for term in ["actor_diversity", "sector_complexity"]:
        if term not in model.summary.index:
            continue
        row = model.summary.loc[term]
        records.append(
            {
                "region": region_name,
                "actor_proxy_variant": variant,
                "model_name": model_name,
                "term": term,
                "coef": float(row["coef"]),
                "hazard_ratio": float(row["exp(coef)"]),
                "p_value": float(row["p"]),
                "c_index": float(metrics.get("c_index", np.nan)),
                "n_samples": len(model_data),
                "n_events": int(model_data["event"].sum()),
            }
        )
    return records


def run(output_dir: Path, stage1_selection_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage1 = load_stage1_selection(stage1_selection_path)
    feature_records = []
    model_records = []
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)

    for region_name, region_result in stage1.items():
        params = dict(region_result["optimal_spatiotemporal_config"])
        params.setdefault("sensitivity_id", "selected")
        logging.info("Actor-proxy robustness: %s / %s", region_name, params["sensitivity_id"])

        survival_data_all = build_survival_data(region_name, params)

        for variant in ["baseline_raw_labels", "unknown_excluded"]:
            survival_data = variant_survival_data(survival_data_all, variant)
            feature_records.append(summarize_actor_features(region_name, variant, survival_data))
            for model_name in ["Cox_Actor", "Cox_Full"]:
                model_records.extend(fit_model_summary(region_name, variant, model_name, survival_data))

        pd.DataFrame(feature_records).to_csv(
            output_dir / "tables" / "actor_proxy_feature_robustness_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(model_records).to_csv(
            output_dir / "tables" / "actor_proxy_cox_robustness_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )

    tables_dir = output_dir / "tables"
    feature_df = pd.DataFrame(feature_records)
    model_df = pd.DataFrame(model_records)
    feature_df.to_csv(tables_dir / "actor_proxy_feature_robustness.csv", index=False, encoding="utf-8-sig")
    model_df.to_csv(tables_dir / "actor_proxy_cox_robustness.csv", index=False, encoding="utf-8-sig")
    return feature_df, model_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stage1-selection", default=str(STAGE1_SELECTION))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s - %(levelname)s - %(message)s")
    output_dir = Path(args.output_dir)
    feature_df, model_df = run(output_dir, Path(args.stage1_selection))

    report = output_dir / "actor_proxy_robustness_summary.md"
    report.write_text(
        "\n".join(
            [
                "# Actor Proxy Robustness Summary",
                "",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "This check compares the original ActorDiv/SectorComp operationalization with an alternative that excludes ambiguous labels such as Unknown, None, and Unknown Actor. The goal is not to claim that these variables directly observe organizational fragmentation, but to evaluate whether the actor-heterogeneity signal is robust to coding ambiguity in actor labels.",
                "",
                "## Feature Summary",
                "",
                feature_df.to_markdown(index=False),
                "",
                "## Cox Results",
                "",
                model_df.to_markdown(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved actor proxy robustness outputs to: {output_dir}")
    print(model_df.to_string(index=False))


if __name__ == "__main__":
    main()
