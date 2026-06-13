"""Dominant-country robustness checks.

This script removes the largest aid-observation country in each region and
re-estimates the Stage 1 Cox models under the region-specific optimal
spatiotemporal configuration. Eastern Europe also receives a Ukraine-excluded
diagnostic because Ukraine dominates the attack-event distribution.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.integrate
from sklearn.preprocessing import StandardScaler

if not hasattr(scipy.integrate, "trapz"):
    scipy.integrate.trapz = np.trapz

from config import AUTHORITATIVE_REGION_CONFIG, CONFIG  # noqa: E402
from data_processing import CoxDataProcessor  # noqa: E402
from modeling import COX_MODELS_CONFIG, train_cox_model_competition  # noqa: E402


OPTIMAL_CONFIGS = {
    "Sahel Region": {
        "pre_window_days": 45,
        "analysis_window_days": 45,
        "max_distance_km": 200,
        "sensitivity_id": "long_45d_200km",
    },
    "Middle East & Central Asia": {
        "pre_window_days": 30,
        "analysis_window_days": 30,
        "max_distance_km": 500,
        "sensitivity_id": "baseline_30d_500km",
    },
    "Eastern Europe": {
        "pre_window_days": 30,
        "analysis_window_days": 30,
        "max_distance_km": 10,
        "sensitivity_id": "baseline_30d_10km",
    },
}


SCENARIOS = [
    {
        "region": "Sahel Region",
        "scenario": "baseline",
        "excluded_country": "",
        "excluded_country_name": "",
        "exclusion_basis": "none",
    },
    {
        "region": "Sahel Region",
        "scenario": "exclude_NGA",
        "excluded_country": "NGA",
        "excluded_country_name": "Nigeria",
        "exclusion_basis": "largest_aid_observation_share",
    },
    {
        "region": "Middle East & Central Asia",
        "scenario": "baseline",
        "excluded_country": "",
        "excluded_country_name": "",
        "exclusion_basis": "none",
    },
    {
        "region": "Middle East & Central Asia",
        "scenario": "exclude_TUR",
        "excluded_country": "TUR",
        "excluded_country_name": "Turkey",
        "exclusion_basis": "largest_aid_observation_share",
    },
    {
        "region": "Eastern Europe",
        "scenario": "baseline",
        "excluded_country": "",
        "excluded_country_name": "",
        "exclusion_basis": "none",
    },
    {
        "region": "Eastern Europe",
        "scenario": "exclude_RUS",
        "excluded_country": "RUS",
        "excluded_country_name": "Russia",
        "exclusion_basis": "largest_aid_observation_share",
    },
    {
        "region": "Eastern Europe",
        "scenario": "exclude_UKR",
        "excluded_country": "UKR",
        "excluded_country_name": "Ukraine",
        "exclusion_basis": "largest_attack_event_share_diagnostic",
    },
]


def region_key_by_name() -> dict[str, str]:
    return {cfg["name"]: key for key, cfg in AUTHORITATIVE_REGION_CONFIG.items()}


def scale_survival_data(survival_data: pd.DataFrame) -> pd.DataFrame:
    numeric_features = survival_data.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["duration", "event", "aid_id", "predicted_risk"]
    features_to_scale = [col for col in numeric_features if col not in exclude_cols]
    scaled = survival_data.copy()
    if features_to_scale:
        scaler = StandardScaler()
        scaled[features_to_scale] = scaler.fit_transform(survival_data[features_to_scale])
    return scaled


def build_scenario_survival_data(
    processor: CoxDataProcessor,
    region_df: pd.DataFrame,
    region_config: dict,
    params: dict,
    excluded_country: str,
) -> pd.DataFrame:
    scenario_df = region_df
    if excluded_country:
        scenario_df = region_df[region_df["country"] != excluded_country].copy()
    return processor.build_cox_survival_dataset(scenario_df, region_config, sensitivity_params=params)


def summarize_region_events(region_df: pd.DataFrame, region_config: dict) -> dict:
    aid_count = int(region_df["event_type"].isin(region_config["aid_types"]).sum())
    attack_count = int(region_df["event_type"].isin(region_config["assault_events"]).sum())
    return {
        "region_events_after_country_exclusion": int(len(region_df)),
        "aid_observations_before_survival_build": aid_count,
        "attack_events_before_survival_build": attack_count,
        "n_countries_after_exclusion": int(region_df["country"].nunique()),
        "countries_after_exclusion": ";".join(sorted(region_df["country"].dropna().unique())),
    }


def collect_model_records(
    region: str,
    scenario: str,
    scenario_meta: dict,
    survival_data: pd.DataFrame,
    model_results: dict,
) -> tuple[list[dict], list[dict], list[dict]]:
    model_records: list[dict] = []
    full_term_records: list[dict] = []

    for model_name, result in model_results.items():
        final_metrics = result.get("final_metrics", {})
        cv_metrics = result.get("cv_metrics", {})
        ph_test = result.get("ph_test", {})
        model_records.append(
            {
                "region": region,
                "scenario": scenario,
                **scenario_meta,
                "model_name": model_name,
                "model_theory": result.get("theory", ""),
                "n_samples": int(result.get("n_samples", len(survival_data))),
                "n_events": int(survival_data["event"].sum()),
                "event_rate": float(survival_data["event"].mean()),
                "final_c_index": float(final_metrics.get("c_index", np.nan)),
                "cv_c_index_mean": float(cv_metrics.get("c_index_mean", np.nan)),
                "cv_c_index_std": float(cv_metrics.get("c_index_std", np.nan)),
                "ph_p_value": float(ph_test.get("p_value", np.nan)),
                "ph_assumption_violated": bool(ph_test.get("assumption_violated", False)),
                "features": ";".join(result.get("features", [])),
            }
        )

        if model_name == "Cox_Full" and result.get("model") is not None:
            model = result["model"]
            for term, row in model.summary.iterrows():
                full_term_records.append(
                    {
                        "region": region,
                        "scenario": scenario,
                        **scenario_meta,
                        "term": term,
                        "coef": float(row["coef"]),
                        "hazard_ratio": float(row["exp(coef)"]),
                        "ci_lower_95": float(row["exp(coef) lower 95%"]),
                        "ci_upper_95": float(row["exp(coef) upper 95%"]),
                        "se_coef": float(row["se(coef)"]),
                        "p_value": float(row["p"]),
                        "n_samples": int(result.get("n_samples", len(survival_data))),
                        "n_events": int(survival_data["event"].sum()),
                        "full_c_index": float(result.get("final_metrics", {}).get("c_index", np.nan)),
                    }
                )

    summary_records: list[dict] = []
    if model_records:
        perf = pd.DataFrame(model_records)
        top_row = perf.sort_values("final_c_index", ascending=False).iloc[0]
        full_row = perf[perf["model_name"] == "Cox_Full"].iloc[0]
        full_rank = int(
            perf["final_c_index"].rank(method="min", ascending=False)[perf["model_name"] == "Cox_Full"].iloc[0]
        )
        summary_records.append(
            {
                "region": region,
                "scenario": scenario,
                **scenario_meta,
                "n_samples": int(len(survival_data)),
                "n_events": int(survival_data["event"].sum()),
                "event_rate": float(survival_data["event"].mean()),
                "full_c_index": float(full_row["final_c_index"]),
                "full_cv_c_index_mean": float(full_row["cv_c_index_mean"]),
                "top_model_by_final_c_index": str(top_row["model_name"]),
                "top_model_final_c_index": float(top_row["final_c_index"]),
                "full_model_rank_by_final_c_index": full_rank,
            }
        )

    return summary_records, model_records, full_term_records


def make_direction_comparison(full_terms: pd.DataFrame) -> pd.DataFrame:
    records = []
    if full_terms.empty:
        return pd.DataFrame()

    for region in full_terms["region"].unique():
        base = full_terms[(full_terms["region"] == region) & (full_terms["scenario"] == "baseline")]
        if base.empty:
            continue
        base_lookup = base.set_index("term")
        for _, row in full_terms[(full_terms["region"] == region) & (full_terms["scenario"] != "baseline")].iterrows():
            term = row["term"]
            if term not in base_lookup.index:
                continue
            base_row = base_lookup.loc[term]
            base_direction = "positive" if float(base_row["hazard_ratio"]) > 1 else "negative"
            excluded_direction = "positive" if float(row["hazard_ratio"]) > 1 else "negative"
            records.append(
                {
                    "region": region,
                    "scenario": row["scenario"],
                    "excluded_country": row["excluded_country"],
                    "excluded_country_name": row["excluded_country_name"],
                    "term": term,
                    "baseline_hazard_ratio": float(base_row["hazard_ratio"]),
                    "excluded_hazard_ratio": float(row["hazard_ratio"]),
                    "baseline_p_value": float(base_row["p_value"]),
                    "excluded_p_value": float(row["p_value"]),
                    "baseline_direction": base_direction,
                    "excluded_direction": excluded_direction,
                    "direction_stable": base_direction == excluded_direction,
                }
            )
    return pd.DataFrame(records)


def write_markdown_summary(
    output_dir: Path,
    scenario_summary: pd.DataFrame,
    direction_comparison: pd.DataFrame,
) -> None:
    lines = [
        "# Dominant-Country Robustness Summary",
        "",
        "This check excludes the country contributing the largest share of aid-event observations in each regional analytical sample. Eastern Europe also includes a Ukraine-excluded diagnostic because Ukraine dominates the attack-event distribution.",
        "",
        "## Scenario Summary",
        "",
        scenario_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    if not direction_comparison.empty:
        core_terms = [
            "log_min_distance",
            "log_historical_frequency",
            "conflict_persistence",
            "temporal_clustering",
            "actor_diversity",
            "sector_complexity",
        ]
        core = direction_comparison[direction_comparison["term"].isin(core_terms)].copy()
        lines.extend(
            [
                "## Core-Term Direction Stability",
                "",
                core[
                    [
                        "region",
                        "scenario",
                        "term",
                        "baseline_hazard_ratio",
                        "excluded_hazard_ratio",
                        "baseline_direction",
                        "excluded_direction",
                        "direction_stable",
                    ]
                ].to_markdown(index=False, floatfmt=".3f"),
                "",
            ]
        )
    (output_dir / "dominant_country_robustness_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    processor = CoxDataProcessor(buffer_days=3)
    name_to_key = region_key_by_name()
    region_cache: dict[str, pd.DataFrame] = {}

    summary_records: list[dict] = []
    model_records: list[dict] = []
    full_term_records: list[dict] = []

    for scenario_cfg in SCENARIOS:
        region = scenario_cfg["region"]
        scenario = scenario_cfg["scenario"]
        excluded_country = scenario_cfg["excluded_country"]
        region_config = AUTHORITATIVE_REGION_CONFIG[name_to_key[region]]
        params = dict(OPTIMAL_CONFIGS[region])

        if region not in region_cache:
            region_cache[region] = processor.load_region_data(region_config)

        scenario_region_df = region_cache[region]
        if excluded_country:
            scenario_region_df = scenario_region_df[scenario_region_df["country"] != excluded_country].copy()

        event_meta = summarize_region_events(scenario_region_df, region_config)
        scenario_meta = {
            "excluded_country": excluded_country,
            "excluded_country_name": scenario_cfg["excluded_country_name"],
            "exclusion_basis": scenario_cfg["exclusion_basis"],
            "optimal_config": params["sensitivity_id"],
            **event_meta,
        }

        logging.info("Running %s / %s", region, scenario)
        survival_data = build_scenario_survival_data(
            processor=processor,
            region_df=scenario_region_df,
            region_config=region_config,
            params=params,
            excluded_country="",
        )
        scaled = scale_survival_data(survival_data)
        model_results = train_cox_model_competition(scaled, f"{region}_{scenario}_dominant_country")
        s_records, m_records, t_records = collect_model_records(
            region=region,
            scenario=scenario,
            scenario_meta=scenario_meta,
            survival_data=survival_data,
            model_results=model_results,
        )
        summary_records.extend(s_records)
        model_records.extend(m_records)
        full_term_records.extend(t_records)

        pd.DataFrame(summary_records).to_csv(tables_dir / "dominant_country_summary_partial.csv", index=False)
        pd.DataFrame(model_records).to_csv(tables_dir / "dominant_country_model_performance_partial.csv", index=False)
        pd.DataFrame(full_term_records).to_csv(tables_dir / "dominant_country_full_cox_terms_partial.csv", index=False)

    scenario_summary = pd.DataFrame(summary_records)
    model_performance = pd.DataFrame(model_records)
    full_terms = pd.DataFrame(full_term_records)
    direction_comparison = make_direction_comparison(full_terms)

    scenario_summary.to_csv(tables_dir / "dominant_country_summary.csv", index=False, encoding="utf-8-sig")
    model_performance.to_csv(tables_dir / "dominant_country_model_performance.csv", index=False, encoding="utf-8-sig")
    full_terms.to_csv(tables_dir / "dominant_country_full_cox_terms.csv", index=False, encoding="utf-8-sig")
    direction_comparison.to_csv(
        tables_dir / "dominant_country_full_cox_direction_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_markdown_summary(output_dir, scenario_summary, direction_comparison)

    return scenario_summary, model_performance, full_terms, direction_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CONFIG["output_base_path"] / "dominant_country_robustness",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    summary, _, _, direction = run(args.output_dir)
    print("\nScenario summary:")
    print(summary.to_string(index=False))
    if not direction.empty:
        core_terms = [
            "log_min_distance",
            "log_historical_frequency",
            "conflict_persistence",
            "temporal_clustering",
            "actor_diversity",
            "sector_complexity",
        ]
        print("\nCore direction comparison:")
        print(direction[direction["term"].isin(core_terms)].to_string(index=False))


if __name__ == "__main__":
    main()
