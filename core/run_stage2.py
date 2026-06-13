import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import AUTHORITATIVE_REGION_CONFIG, CONFIG
from data_processing import CoxDataProcessor
from modeling import scale_survival_data, train_ml_survival_models_cv


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def region_key_by_name() -> dict[str, str]:
    return {cfg["name"]: key for key, cfg in AUTHORITATIVE_REGION_CONFIG.items()}


def run_stage2(stage1_results_path: Path, output_dir: Path, n_splits: int = 5) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)

    stage1 = load_pickle(stage1_results_path)
    processor = CoxDataProcessor()
    name_to_key = region_key_by_name()

    summary_records = []
    serializable_results = {}

    for region_name, region_stage1 in stage1.items():
        logging.info("Stage 2 region: %s", region_name)
        region_key = name_to_key[region_name]
        region_config = AUTHORITATIVE_REGION_CONFIG[region_key]
        params = dict(region_stage1["optimal_spatiotemporal_config"])
        features = list(region_stage1["significant_features_for_stage2"])

        region_df = processor.load_region_data(region_config)
        survival_data = processor.build_cox_survival_dataset(region_df, region_config, sensitivity_params=params)
        scaled = scale_survival_data(survival_data)
        ml_results = train_ml_survival_models_cv(scaled, features, n_splits=n_splits)

        best_model = max(ml_results, key=lambda name: ml_results[name]["cv_mean"])
        for model_name, result in ml_results.items():
            summary_records.append(
                {
                    "region": region_name,
                    "model": model_name,
                    "c_index_cv_mean": result["cv_mean"],
                    "c_index_cv_std": result["cv_std"],
                    "n_samples": result["n_samples"],
                    "event_rate": result["event_rate"],
                    "optimal_config": params["sensitivity_id"],
                    "n_features": len(result["features"]),
                    "best_model": model_name == best_model,
                }
            )

        serializable_results[region_name] = {
            "optimal_spatiotemporal_config": params,
            "features": features,
            "ml_results": ml_results,
            "best_model": best_model,
        }

    pd.DataFrame(summary_records).to_csv(output_dir / "tables" / "stage2_performance_summary_cv.csv", index=False)
    save_pickle(serializable_results, output_dir / "results" / "stage2_final_results.pkl")
    logging.info("Stage 2 complete: %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2 RSF/GBS survival-model comparison with stratified 5-fold CV.")
    parser.add_argument("--stage1-results-path", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else CONFIG["output_base_path"] / f"stage2_{datetime.now():%Y%m%d_%H%M%S}"
    run_stage2(Path(args.stage1_results_path), output_dir, n_splits=args.n_splits)


if __name__ == "__main__":
    main()
