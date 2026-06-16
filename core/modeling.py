import logging
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

from config import CONFIG

warnings.filterwarnings("ignore")


COX_MODELS_CONFIG = {
    "Cox_Geospatial": {
        "features": ["log_min_distance", "spatial_dispersion", "log_robust_attack_density"],
        "theory": "Geographical Determinism Model",
    },
    "Cox_Historical": {
        "features": ["log_historical_frequency", "historical_intensity", "conflict_persistence"],
        "theory": "Historical Path Dependence Model",
    },
    "Cox_Battlefield": {
        "features": ["intensity_volatility", "temporal_clustering", "spatial_clustering"],
        "theory": "Battlefield Dynamics Model",
    },
    "Cox_Actor": {
        "features": ["actor_diversity", "sector_complexity"],
        "theory": "Actor Complexity Model",
    },
    "Cox_Full": {
        "features": [
            "log_min_distance",
            "spatial_dispersion",
            "log_robust_attack_density",
            "log_historical_frequency",
            "historical_intensity",
            "conflict_persistence",
            "intensity_volatility",
            "temporal_clustering",
            "spatial_clustering",
            "actor_diversity",
            "sector_complexity",
        ],
        "theory": "Full Feature Integration Model",
    },
}


def scale_survival_data(survival_data: pd.DataFrame) -> pd.DataFrame:
    """Standardize numeric covariates while leaving duration/event unchanged."""
    scaled = survival_data.copy()
    numeric_cols = scaled.select_dtypes(include=[np.number]).columns.tolist()
    covariates = [c for c in numeric_cols if c not in {"duration", "event"}]
    if covariates:
        scaled[covariates] = StandardScaler().fit_transform(scaled[covariates])
    return scaled


def _available_features(data: pd.DataFrame, features: list[str]) -> list[str]:
    return [feature for feature in features if feature in data.columns and data[feature].nunique(dropna=True) > 1]


def _clean_model_data(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    model_cols = ["duration", "event"] + features
    model_data = data[model_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    model_data = model_data[model_data["duration"] > 0]
    model_data["event"] = model_data["event"].astype(int)
    return model_data


def train_final_cox_model(data: pd.DataFrame, features: list[str], model_name: str = "Cox"):
    """Fit a penalized Cox model and return the fitted model plus basic metrics."""
    selected_features = _available_features(data, features)
    model_data = _clean_model_data(data, selected_features)

    if len(model_data) < 20 or model_data["event"].sum() < 2 or len(selected_features) == 0:
        raise ValueError(f"Insufficient data for {model_name}")

    cph = CoxPHFitter(penalizer=CONFIG.get("cox_penalizer", 0.1))
    cph.fit(model_data, duration_col="duration", event_col="event")
    partial_hazard = cph.predict_partial_hazard(model_data[selected_features])
    c_index = concordance_index(model_data["duration"], -partial_hazard, model_data["event"])

    return cph, {
        "c_index": float(c_index),
        "n_samples": int(len(model_data)),
        "n_events": int(model_data["event"].sum()),
        "features": selected_features,
    }


def perform_cox_cross_validation(data: pd.DataFrame, features: list[str], n_splits: int = 5) -> dict:
    """Run stratified K-fold C-index evaluation for one Cox specification."""
    selected_features = _available_features(data, features)
    model_data = _clean_model_data(data, selected_features)
    if len(model_data) < 20 or model_data["event"].sum() < n_splits or len(selected_features) == 0:
        return {"mean_c_index": np.nan, "std_c_index": np.nan, "fold_scores": []}

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in folds.split(model_data[selected_features], model_data["event"]):
        train = model_data.iloc[train_idx]
        test = model_data.iloc[test_idx]
        try:
            cph = CoxPHFitter(penalizer=CONFIG.get("cox_penalizer", 0.1))
            cph.fit(train, duration_col="duration", event_col="event")
            risk = cph.predict_partial_hazard(test[selected_features])
            scores.append(concordance_index(test["duration"], -risk, test["event"]))
        except Exception as exc:
            logging.warning("Cox CV fold failed: %s", exc)

    return {
        "mean_c_index": float(np.mean(scores)) if scores else np.nan,
        "std_c_index": float(np.std(scores)) if scores else np.nan,
        "fold_scores": [float(s) for s in scores],
    }


def test_proportional_hazards(model, model_data: pd.DataFrame) -> dict:
    try:
        results = proportional_hazard_test(model, model_data, time_transform="km")
        return {
            "min_p_value": float(results.summary["p"].min()),
            "passed": bool((results.summary["p"] > 0.05).all()),
        }
    except Exception as exc:
        logging.warning("PH test failed: %s", exc)
        return {"min_p_value": np.nan, "passed": None}


def train_cox_model_competition(survival_data: pd.DataFrame, region_name: str, n_splits: int = 5) -> dict:
    """Fit all theory-driven Cox specifications used in Stage 1."""
    scaled = scale_survival_data(survival_data)
    results = {}

    for model_name, config in COX_MODELS_CONFIG.items():
        selected_features = _available_features(scaled, config["features"])
        model_data = _clean_model_data(scaled, selected_features)
        if len(model_data) < 20 or model_data["event"].sum() < 2 or len(selected_features) == 0:
            continue

        try:
            final_model, final_metrics = train_final_cox_model(model_data, selected_features, f"{region_name}_{model_name}")
            cv_metrics = perform_cox_cross_validation(model_data, selected_features, n_splits=n_splits)
            ph_test = test_proportional_hazards(final_model, model_data[["duration", "event"] + selected_features])
            results[model_name] = {
                "model": final_model,
                "features": selected_features,
                "theory": config["theory"],
                "n_samples": int(len(model_data)),
                "event_rate": float(model_data["event"].mean()),
                "final_metrics": final_metrics,
                "cv_metrics": cv_metrics,
                "ph_test": ph_test,
            }
        except Exception as exc:
            logging.warning("%s failed in %s: %s", model_name, region_name, exc)

    return results


def select_best_cox_model(results: dict) -> tuple[str | None, dict | None]:
    valid = {
        name: result
        for name, result in results.items()
        if result.get("final_metrics", {}).get("c_index") is not None
    }
    if not valid:
        return None, None
    best_name = max(valid, key=lambda name: valid[name]["final_metrics"]["c_index"])
    return best_name, valid[best_name]


def run_cox_survival_analysis(survival_data: pd.DataFrame, region_name: str, output_dir=None):
    """Compatibility wrapper for the original Stage 1 script."""
    results = train_cox_model_competition(survival_data, region_name)
    best_name, best_result = select_best_cox_model(results)
    summary = pd.DataFrame(
        [
            {
                "model": name,
                "theory": result["theory"],
                "c_index": result["final_metrics"]["c_index"],
                "cv_mean": result["cv_metrics"]["mean_c_index"],
                "cv_std": result["cv_metrics"]["std_c_index"],
                "n_samples": result["n_samples"],
                "event_rate": result["event_rate"],
            }
            for name, result in results.items()
        ]
    )
    return results, best_result, summary


def _to_sksurv_y(duration: pd.Series, event: pd.Series):
    return np.array(list(zip(event.astype(bool), duration.astype(float))), dtype=[("event", "?"), ("time", "<f8")])


def train_ml_survival_models_cv(
    survival_data: pd.DataFrame,
    features: list[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Train and evaluate RSF and GBS with stratified 5-fold CV."""
    selected_features = _available_features(survival_data, features)
    model_data = _clean_model_data(survival_data, selected_features)
    X = model_data[selected_features]
    y = _to_sksurv_y(model_data["duration"], model_data["event"])

    model_specs = {
        "RandomSurvivalForest": (
            RandomSurvivalForest,
            {
                "n_estimators": 100,
                "max_features": "sqrt",
                "min_samples_split": 6,
                "min_samples_leaf": 3,
                "max_depth": None,
                "bootstrap": True,
                "random_state": random_state,
                "n_jobs": -1,
            },
        ),
        "GradientBoostingSurvival": (
            GradientBoostingSurvivalAnalysis,
            {
                "loss": "coxph",
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "subsample": 1.0,
                "min_samples_leaf": 1,
                "random_state": random_state,
            },
        ),
    }

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {}
    for model_name, (model_class, params) in model_specs.items():
        fold_scores = []
        for train_idx, test_idx in folds.split(X, model_data["event"]):
            model = model_class(**params)
            model.fit(X.iloc[train_idx], y[train_idx])
            risk = model.predict(X.iloc[test_idx])
            score = concordance_index_censored(
                y[test_idx]["event"],
                y[test_idx]["time"],
                risk,
            )[0]
            fold_scores.append(float(score))

        final_model = model_class(**params)
        final_model.fit(X, y)
        results[model_name] = {
            "model": final_model,
            "features": selected_features,
            "cv_scores": fold_scores,
            "cv_mean": float(np.mean(fold_scores)),
            "cv_std": float(np.std(fold_scores)),
            "n_samples": int(len(model_data)),
            "event_rate": float(model_data["event"].mean()),
        }

    return results


def calculate_cox_snell_residuals(model, X: pd.DataFrame, event_times: pd.Series, event_observed: pd.Series) -> np.ndarray:
    """Model-agnostic Cox-Snell style residual approximation for sksurv models."""
    try:
        survival_functions = model.predict_survival_function(X)
        residuals = []
        for surv_func, t in zip(survival_functions, event_times):
            times = surv_func.x
            probs = surv_func.y
            if t <= times[0]:
                survival_prob = probs[0]
            elif t >= times[-1]:
                survival_prob = probs[-1]
            else:
                survival_prob = np.interp(t, times, probs)
            residuals.append(-np.log(max(float(survival_prob), 1e-12)))
        return np.asarray(residuals)
    except Exception:
        if hasattr(model, "predict_cumulative_hazard_function"):
            chfs = model.predict_cumulative_hazard_function(X)
            return np.asarray([np.interp(t, chf.x, chf.y) for chf, t in zip(chfs, event_times)])
        raise


def summarize_residual_fit(residuals: np.ndarray) -> dict:
    """Compare residual empirical CDF with the Exp(1) reference."""
    residuals = np.asarray(residuals)
    residuals = residuals[np.isfinite(residuals)]
    residuals = np.sort(residuals[residuals >= 0])
    if len(residuals) == 0:
        return {"ks_distance": np.nan, "mean_absolute_cdf_deviation": np.nan, "integrated_absolute_deviation": np.nan}

    empirical_cdf = np.arange(1, len(residuals) + 1) / len(residuals)
    theoretical_cdf = 1 - np.exp(-residuals)
    abs_dev = np.abs(empirical_cdf - theoretical_cdf)
    return {
        "ks_distance": float(abs_dev.max()),
        "mean_absolute_cdf_deviation": float(abs_dev.mean()),
        "integrated_absolute_deviation": float(np.trapz(abs_dev, residuals)),
    }
