from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


ACADEMIC_PALETTE = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "tertiary": "#2ca02c",
    "quaternary": "#d62728",
    "text": "#212529",
}


CONFIG = {
    "output_base_path": PROJECT_ROOT / "result",
    "global_dataset": PROJECT_ROOT / "data" / "processed" / "polecat_plover_event_archive_2018_2024.parquet",
    "analysis_time_window": ("2018-01-01", "2024-06-30"),
    "analysis_window_days": 30,
    "max_distance_km": 500,
    "cox_penalizer": 0.1,
    "min_events_for_modeling": 20,
}


AUTHORITATIVE_REGION_CONFIG = {
    "sahel": {
        "name": "Sahel Region",
        "output_prefix": "sahel",
        "countries": ["MLI", "BFA", "NER", "NGA", "CMR", "TCD", "SDN"],
        "aid_types": ["AID", "FOOD_AID", "RELIEF_AID", "HUMANITARIAN_AID"],
        "assault_events": [
            "REBEL_ACTIVITY",
            "AMBUSH",
            "IED_EXPLOSION",
            "KIDNAPPING",
            "ARTILLERY_FIRE",
            "ASSAULT",
            "ATTACK",
            "BOMBING",
            "MILITARY_ENGAGEMENT",
            "SHELLING",
            "ARMED_CLASH",
            "AIRSTRIKE",
            "EXPLOSION",
        ],
    },
    "middle_east": {
        "name": "Middle East & Central Asia",
        "output_prefix": "middle_east",
        "countries": ["SYR", "YEM", "IRQ", "AFG", "LBN", "PSE", "PAK", "IRN", "ISR", "TUR"],
        "aid_types": ["AID", "HUMANITARIAN_AID", "FOOD_AID", "RELIEF_AID"],
        "assault_events": [
            "ARMED_CLASH",
            "MILITARY_ENGAGEMENT",
            "BOMBING",
            "ARTILLERY_FIRE",
            "SHELLING",
            "AIRSTRIKE",
            "ASSAULT",
            "ATTACK",
            "EXPLOSION",
            "REBEL_ACTIVITY",
            "IED_EXPLOSION",
            "KIDNAPPING",
            "AMBUSH",
        ],
    },
    "eastern_europe": {
        "name": "Eastern Europe",
        "output_prefix": "eastern_europe",
        "countries": ["UKR", "RUS", "BLR", "MDA", "GEO", "AZE"],
        "aid_types": ["AID", "HUMANITARIAN_AID", "FOOD_AID"],
        "assault_events": [
            "MILITARY_ENGAGEMENT",
            "ARTILLERY_FIRE",
            "ARMED_CLASH",
            "AIRSTRIKE",
            "ATTACK",
            "BOMBING",
            "ASSAULT",
            "AMBUSH",
            "KIDNAPPING",
            "SHELLING",
            "IED_EXPLOSION",
            "EXPLOSION",
        ],
    },
}

CONFIG["regions"] = AUTHORITATIVE_REGION_CONFIG


FEATURE_ABBREVIATIONS = {
    "log_historical_frequency": "LogFreq",
    "log_min_distance": "LogDist",
    "historical_intensity": "HistInt",
    "conflict_persistence": "Persist",
    "spatial_dispersion": "SpatDisp",
    "log_robust_attack_density": "LogDens",
    "intensity_volatility": "IntVol",
    "temporal_clustering": "TempCluster",
    "spatial_clustering": "SpCluster",
    "actor_diversity": "ActorDiv",
    "sector_complexity": "SectorComp",
}


COUNTRY_CODE_MAP = {
    "SYR": "Syria",
    "YEM": "Yemen",
    "IRQ": "Iraq",
    "AFG": "Afghanistan",
    "LBN": "Lebanon",
    "PSE": "Palestine",
    "PAK": "Pakistan",
    "IRN": "Iran",
    "ISR": "Israel",
    "TUR": "Turkey",
    "UKR": "Ukraine",
    "RUS": "Russia",
    "BLR": "Belarus",
    "MDA": "Moldova",
    "GEO": "Georgia",
    "AZE": "Azerbaijan",
    "MLI": "Mali",
    "BFA": "Burkina Faso",
    "NER": "Niger",
    "NGA": "Nigeria",
    "CMR": "Cameroon",
    "TCD": "Chad",
    "SDN": "Sudan",
}


def get_sensitivity_parameters(mode="refined", region_key=None):
    """Return the 12 spatiotemporal configurations used in Stage 1."""
    return [
        {"pre_window_days": 7, "analysis_window_days": 7, "max_distance_km": 50, "sensitivity_id": "rapid_7d_50km"},
        {"pre_window_days": 7, "analysis_window_days": 7, "max_distance_km": 100, "sensitivity_id": "rapid_7d_100km"},
        {"pre_window_days": 7, "analysis_window_days": 7, "max_distance_km": 200, "sensitivity_id": "rapid_7d_200km"},
        {"pre_window_days": 14, "analysis_window_days": 14, "max_distance_km": 50, "sensitivity_id": "short_14d_50km"},
        {"pre_window_days": 14, "analysis_window_days": 14, "max_distance_km": 100, "sensitivity_id": "short_14d_100km"},
        {"pre_window_days": 14, "analysis_window_days": 14, "max_distance_km": 200, "sensitivity_id": "short_14d_200km"},
        {"pre_window_days": 30, "analysis_window_days": 30, "max_distance_km": 10, "sensitivity_id": "baseline_30d_10km"},
        {"pre_window_days": 30, "analysis_window_days": 30, "max_distance_km": 100, "sensitivity_id": "baseline_30d_100km"},
        {"pre_window_days": 30, "analysis_window_days": 30, "max_distance_km": 500, "sensitivity_id": "baseline_30d_500km"},
        {"pre_window_days": 45, "analysis_window_days": 45, "max_distance_km": 100, "sensitivity_id": "long_45d_100km"},
        {"pre_window_days": 45, "analysis_window_days": 45, "max_distance_km": 200, "sensitivity_id": "long_45d_200km"},
        {"pre_window_days": 45, "analysis_window_days": 45, "max_distance_km": 500, "sensitivity_id": "long_45d_500km"},
    ]
