import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import AUTHORITATIVE_REGION_CONFIG, CONFIG
from data_processing import CoxDataProcessor
from modeling import calculate_cox_snell_residuals, scale_survival_data, summarize_residual_fit


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def region_key_by_name() -> dict[str, str]:
    return {cfg["name"]: key for key, cfg in AUTHORITATIVE_REGION_CONFIG.items()}


def run_residual_diagnostics(stage2_results_path: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)

    stage2 = load_pickle(stage2_results_path)
    processor = CoxDataProcessor()
    name_to_key = region_key_by_name()
    records = []

    for region_name, region_result in stage2.items():
        logging.info("Residual diagnostics region: %s", region_name)
        region_config = AUTHORITATIVE_REGION_CONFIG[name_to_key[region_name]]
        params = dict(region_result["optimal_spatiotemporal_config"])
        region_df = processor.load_region_data(region_config)
        survival_data = processor.build_cox_survival_dataset(region_df, region_config, sensitivity_params=params)
        scaled = scale_survival_data(survival_data)

        for model_name, model_info in region_result["ml_results"].items():
            features = model_info["features"]
            model_data = scaled[["duration", "event"] + features].dropna()
            residuals = calculate_cox_snell_residuals(
                model_info["model"],
                model_data[features],
                model_data["duration"],
                model_data["event"],
            )
            metrics = summarize_residual_fit(residuals)
            records.append(
                {
                    "region": region_name,
                    "model": model_name,
                    "optimal_config": params["sensitivity_id"],
                    "n_samples": int(len(model_data)),
                    "n_events": int(model_data["event"].sum()),
                    **metrics,
                }
            )

    result = pd.DataFrame(records)
    result.to_csv(output_dir / "tables" / "cox_snell_residual_fit_metrics.csv", index=False)
    logging.info("Residual diagnostics complete: %s", output_dir)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Cox-Snell residual fit metrics for RSF and GBS.")
    parser.add_argument("--stage2-results-path", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else CONFIG["output_base_path"] / f"residuals_{datetime.now():%Y%m%d_%H%M%S}"
    run_residual_diagnostics(Path(args.stage2_results_path), output_dir)


if __name__ == "__main__":
    main()
