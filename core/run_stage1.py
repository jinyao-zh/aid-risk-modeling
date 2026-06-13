import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import AUTHORITATIVE_REGION_CONFIG, CONFIG, get_sensitivity_parameters
from data_processing import CoxDataProcessor
from modeling import select_best_cox_model, train_cox_model_competition


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def extract_significant_features(best_result: dict, alpha: float = 0.05) -> list[str]:
    summary = best_result["model"].summary
    if "p" not in summary.columns:
        return best_result.get("features", [])
    return summary[summary["p"] < alpha].index.tolist()


def run_stage1(output_dir: Path, sensitivity_mode: str = "refined") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)

    processor = CoxDataProcessor()
    performance_records = []
    stage1_selection = {}

    for region_key, region_config in AUTHORITATIVE_REGION_CONFIG.items():
        region_name = region_config["name"]
        logging.info("Stage 1 region: %s", region_name)
        region_df = processor.load_region_data(region_config)

        best_overall = None
        best_payload = None
        for params in get_sensitivity_parameters(mode=sensitivity_mode, region_key=region_key):
            survival_data = processor.build_cox_survival_dataset(region_df, region_config, sensitivity_params=params)
            if survival_data.empty:
                continue

            model_results = train_cox_model_competition(survival_data, f"{region_name}_{params['sensitivity_id']}")
            best_model_name, best_result = select_best_cox_model(model_results)
            if best_result is None:
                continue

            for model_name, result in model_results.items():
                performance_records.append(
                    {
                        "region": region_name,
                        "sensitivity_id": params["sensitivity_id"],
                        "model": model_name,
                        "theory": result["theory"],
                        "c_index": result["final_metrics"]["c_index"],
                        "cv_mean": result["cv_metrics"]["mean_c_index"],
                        "cv_std": result["cv_metrics"]["std_c_index"],
                        "n_samples": result["n_samples"],
                        "event_rate": result["event_rate"],
                    }
                )

            score = best_result["final_metrics"]["c_index"]
            if best_overall is None or score > best_overall["best_c_index"]:
                best_overall = {
                    "best_model_name": best_model_name,
                    "best_c_index": score,
                    "params": params,
                }
                best_payload = best_result

        if best_overall is None:
            logging.warning("No valid Stage 1 model for %s", region_name)
            continue

        significant_features = extract_significant_features(best_payload)
        stage1_selection[region_name] = {
            "optimal_spatiotemporal_config": best_overall["params"],
            "optimal_model_name": best_overall["best_model_name"],
            "optimal_c_index": best_overall["best_c_index"],
            "significant_features_for_stage2": significant_features,
        }

    pd.DataFrame(performance_records).to_csv(output_dir / "tables" / "global_performance_summary.csv", index=False)
    save_pickle(stage1_selection, output_dir / "results" / "stage1_final_selection.pkl")
    logging.info("Stage 1 complete: %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 1 Cox model comparison and scale selection.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sensitivity-mode", default="refined")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else CONFIG["output_base_path"] / f"stage1_{datetime.now():%Y%m%d_%H%M%S}"
    run_stage1(output_dir, sensitivity_mode=args.sensitivity_mode)


if __name__ == "__main__":
    main()
