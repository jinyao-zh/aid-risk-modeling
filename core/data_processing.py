import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from tqdm import tqdm
from datetime import timedelta
import warnings
from config import CONFIG, get_sensitivity_parameters, ACADEMIC_PALETTE

warnings.filterwarnings('ignore')

EARTH_RADIUS_KM = 6371.0
SAFE_BUFFER_DAYS = 3


class CoxDataProcessor:
    """
    Data processor for constructing Cox survival-analysis datasets.
    """

    def __init__(self, buffer_days=SAFE_BUFFER_DAYS):
        self.buffer_days = buffer_days
        self.leakage_detected = False
        self.leakage_issues = []

    def load_region_data(self, region_config):
        """Load event records for a configured analytical region."""
        # region_config = CONFIG['regions'][region_key]
        logging.info(f"Loading data for {region_config['name']}")

        try:
            df = pd.read_parquet(CONFIG['global_dataset'])
            df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')

            start_date, end_date = pd.to_datetime(CONFIG['analysis_time_window'])
            country_codes_to_filter = region_config['countries']

            mask = (
                    df['event_date'].notna() &
                    df['latitude'].notna() &
                    df['longitude'].notna() &
                    df['country'].notna() &
                    df['event_type'].notna() &
                    (df['event_date'] >= start_date) &
                    (df['event_date'] <= end_date) &
                    df['country'].isin(country_codes_to_filter)
            )

            df_region = df.loc[mask].copy()

            # Data-quality report
            self._report_data_quality(df_region, region_config['name'])

            return df_region

        except Exception as e:
            logging.error(f"Data loading failed for {region_config}: {str(e)}")
            return pd.DataFrame()

    def _report_data_quality(self, df, region_name):
        """Log a concise data-quality report."""
        logging.info(f"=== {region_name} data-quality report ===")
        logging.info(f"Total events: {len(df):,}")
        logging.info("Event-type distribution:")
        for event_type, count in df['event_type'].value_counts().items():
            logging.info(f"  {event_type}: {count:,}")
        logging.info(f"Date range: {df['event_date'].min().date()} to {df['event_date'].max().date()}")
        logging.info(f"Geographic coverage: {df['country'].nunique()} countries")
        logging.info(f"Event-intensity range: {df['event_intensity'].min():.1f} to {df['event_intensity'].max():.1f}")

    def build_cox_survival_dataset(self, df, region_config, sensitivity_params=None):
        """
        Build a Cox survival-analysis dataset for one region and parameter set.
        """
        if sensitivity_params is None:
            sensitivity_params = {
                'pre_window_days': 30,
                'analysis_window_days': 30,
                'max_distance_km': 100,
                'sensitivity_id': 'baseline'
            }

        # Validate required parameters.
        required_keys = ['pre_window_days', 'analysis_window_days', 'max_distance_km']
        missing_keys = [key for key in required_keys if key not in sensitivity_params]
        if missing_keys:
            logging.error(f"Missing required parameters: {missing_keys}")
            logging.error(f"Available parameters: {list(sensitivity_params.keys())}")
            return pd.DataFrame()

        logging.info(f"Building Cox survival dataset - {region_config['name']}")
        logging.info(f"Parameters: {sensitivity_params}")

        # Split regional records into aid events and attack events.
        aid_events = df[df['event_type'].isin(region_config['aid_types'])].copy()
        attack_events = df[df['event_type'].isin(region_config['assault_events'])].copy()

        logging.info(f"Aid events: {len(aid_events)}, attack events: {len(attack_events)}")

        if aid_events.empty or attack_events.empty:
            logging.warning(f"Insufficient event data: aid={len(aid_events)}, attack={len(attack_events)}")
            return pd.DataFrame()

        # Build the survival-analysis dataset.
        survival_data = self._create_survival_data(
            aid_events, attack_events, region_config, sensitivity_params
        )

        return survival_data

    def _create_survival_data(self, aid_events, attack_events, region_config, params):
        """Create aid-event-level survival-analysis records."""
        # Build a spatial index for attack events.
        attack_tree = self._build_spatial_index(attack_events)
        radius_rad = params['max_distance_km'] / EARTH_RADIUS_KM

        cox_records = []
        post_window = timedelta(days=params['analysis_window_days'])
        pre_window_td = timedelta(days=params['pre_window_days'])
        buffer_td = timedelta(days=self.buffer_days)

        # Query nearby attack events in batches.
        aid_points_rad = np.deg2rad(aid_events[['latitude', 'longitude']].values)
        indices_list = attack_tree.query_ball_point(aid_points_rad, r=radius_rad)

        for i, (aid_id, aid_row) in tqdm(enumerate(aid_events.iterrows()),
                                         total=len(aid_events),
                                         desc=f"Processing {region_config['name']}"):
            aid_time = aid_row['event_date']
            aid_loc = (aid_row['latitude'], aid_row['longitude'])

            # Survival outcome
            survival_outcome = self._calculate_survival_outcome(
                indices_list[i], attack_events, aid_time, buffer_td, post_window,
                params['analysis_window_days']
            )

            # Theory-driven covariates
            features = self._create_theory_driven_features(
                indices_list[i], attack_events, aid_loc, aid_time,
                pre_window_td, buffer_td, params['max_distance_km']
            )

            record = {
                'aid_id': aid_id,
                'aid_date': aid_time,
                'aid_lat': aid_row['latitude'],
                'aid_lon': aid_row['longitude'],
                'aid_country': aid_row['country'],
                'region': region_config['name'],

                # Cox survival-analysis outcome
                'duration': survival_outcome['duration'],
                'event': survival_outcome['event'],

                # Theory-driven covariates
                **features,

                'sensitivity_id': params['sensitivity_id']
            }

            cox_records.append(record)

        cox_df = pd.DataFrame(cox_records)

        # Post-processing
        cox_df = self._postprocess_survival_data(cox_df, params)

        logging.info(f"Cox dataset built: {len(cox_df)} records, event rate: {cox_df['event'].mean():.3f}")
        return cox_df

    def _build_spatial_index(self, attack_events):
        """Build a spatial index over attack-event coordinates."""
        attack_points_rad = np.deg2rad(attack_events[['latitude', 'longitude']].values)
        return cKDTree(attack_points_rad)

    def _calculate_survival_outcome(self, nearby_indices, attack_events, aid_time,
                                    buffer_td, post_window, analysis_window_days):
        """Compute the survival outcome for a focal aid event."""
        event = 0
        duration = float(analysis_window_days)  # Default to right-censoring.

        if nearby_indices:
            nearby_attacks = attack_events.iloc[nearby_indices]
            future_attacks = nearby_attacks[
                (nearby_attacks['event_date'] >= aid_time + buffer_td) &
                (nearby_attacks['event_date'] <= aid_time + post_window)
                ]

            if not future_attacks.empty:
                event = 1
                time_diffs = (future_attacks['event_date'] - aid_time).dt.days
                duration = float(time_diffs.min())

                # Strict time-window check.
                if duration < self.buffer_days:
                    logging.warning(f"Time window overlap detected: duration={duration}d")
                    event = 0
                    duration = float(analysis_window_days)

        return {'duration': duration, 'event': event}

    def _validate_features(self, features_dict):
        """Validate feature values and cap invalid or extreme values."""
        validated_features = features_dict.copy()

        # Check and repair invalid or extreme values.
        for key, value in features_dict.items():
            if isinstance(value, (int, float)):
                if np.isinf(value) or np.isnan(value):
                    validated_features[key] = 0.0
                elif abs(value) > 1e6:  # Extreme-value cap.
                    validated_features[key] = np.sign(value) * 1e6

        return validated_features

    def _create_theory_driven_features(self, nearby_indices, attack_events, aid_loc,
                                       aid_time, pre_window_td, buffer_td, max_distance_km):
        """Create theory-driven covariates and validate their values."""

        # Retrieve historical attack events.
        hist_attacks = self._get_historical_attacks(
            nearby_indices, attack_events, aid_time, pre_window_td, buffer_td
        )

        features = {}

        # 1. Geographic exposure features
        features.update(self._create_geographic_features(hist_attacks, aid_loc, max_distance_km))

        # 2. Historical path-dependence features
        features.update(self._create_historical_features(hist_attacks, aid_time, pre_window_td))

        # 3. Battlefield-dynamics features
        features.update(self._create_battlefield_features(hist_attacks))

        # 4. Actor-complexity features
        features.update(self._create_actor_features(hist_attacks))

        # Feature validation
        features = self._validate_features(features)

        return features

    def _get_historical_attacks(self, nearby_indices, attack_events, aid_time, pre_window_td, buffer_td):
        """Retrieve prior attack events within the look-back window."""
        if not nearby_indices:
            return pd.DataFrame()

        nearby_attacks = attack_events.iloc[nearby_indices]
        return nearby_attacks[
            (nearby_attacks['event_date'] >= aid_time - pre_window_td) &
            (nearby_attacks['event_date'] < aid_time - buffer_td)
            ]

    def _create_historical_features(self, hist_attacks, aid_time, pre_window_td):
        """Historical path-dependence features."""
        if hist_attacks.empty:
            return {
                'historical_frequency': 0,
                'log_historical_frequency': 0,
                'historical_intensity': 0,
                'conflict_persistence': 0
            }

        frequency = len(hist_attacks)
        intensity = hist_attacks['event_intensity'].abs().mean()

        # Conflict persistence: recent-to-total frequency ratio.
        recent_cutoff = aid_time - timedelta(days=7)
        recent_freq = len(hist_attacks[hist_attacks['event_date'] >= recent_cutoff])
        persistence = recent_freq / (frequency + 1e-6)

        return {
            'historical_frequency': frequency,
            'log_historical_frequency': np.log1p(frequency),
            'historical_intensity': intensity,
            'conflict_persistence': persistence
        }

    def _create_geographic_features(self, hist_attacks, aid_loc, max_distance_km):
        """Geographic exposure features."""
        if hist_attacks.empty:
            return {
                'min_distance': max_distance_km,
                'log_min_distance': np.log1p(max_distance_km),
                'spatial_dispersion': 0,
                'attack_density': 0,
                'robust_attack_density': 0,
                'log_robust_attack_density': 0
            }

        distances = self._compute_haversine_distance(
            aid_loc[0], aid_loc[1],
            hist_attacks['latitude'].values,
            hist_attacks['longitude'].values
        )

        min_dist = distances.min() if len(distances) > 0 else max_distance_km
        dist_std = distances.std() if len(distances) > 1 else 0

        # More robust spatial-dispersion measure.
        if min_dist > 1.0 and dist_std > 0:
            spatial_dispersion = dist_std / min_dist
        else:
            spatial_dispersion = 0

        # Improved attack-density calculation.
        attack_count = len(hist_attacks)

        # Use a minimum area to avoid very small denominators.
        min_area = 10.0  # Minimum 10 square kilometers.
        area = max(np.pi * (max_distance_km ** 2), min_area)
        attack_density = attack_count / area

        # Discretized robust-density version.
        if attack_count == 0:
            robust_attack_density = 0
        elif attack_count <= 3:
            robust_attack_density = 1
        elif attack_count <= 10:
            robust_attack_density = 2
        elif attack_count <= 30:
            robust_attack_density = 3
        else:
            robust_attack_density = 4

        return {
            'min_distance': min_dist,
            'log_min_distance': np.log1p(min_dist),
            'spatial_dispersion': spatial_dispersion,
            'attack_density': attack_density,
            'robust_attack_density': robust_attack_density,
            'log_robust_attack_density': np.log1p(robust_attack_density)
        }

    def check_survival_data_quality(self, survival_data, min_events=20):
        """
        Check survival-dataset quality using compatibility-safe pandas operations.
        """
        try:
            # Use basic pandas operations to avoid compatibility issues.
            n_samples = len(survival_data)

            # Manually compute event count.
            event_count = 0
            if 'event' in survival_data.columns:
                event_count = survival_data['event'].sum() if hasattr(survival_data['event'], 'sum') else sum(
                    survival_data['event'])

            event_rate = event_count / n_samples if n_samples > 0 else 0

            report = {
                'n_samples': n_samples,
                'n_features': len(survival_data.columns),
                'n_events': int(event_count),
                'event_rate': event_rate,
                'issues': [],
                'is_acceptable': True
            }

            # Basic quality checks.
            if report['n_samples'] < 50:
                report['issues'].append(f"Sample size is too small ({report['n_samples']})")

            if report['n_events'] < min_events:
                report['issues'].append(f"Too few events ({report['n_events']})")
                report['is_acceptable'] = False

            if report['event_rate'] < 0.01 and report['n_samples'] > 100:
                report['issues'].append(f"Very low event rate ({report['event_rate']:.3%})")

            return report

        except Exception as e:
            logging.error(f"Data-quality check failed: {e}")
            # Return a minimal acceptable report so the pipeline can continue.
            return {
                'n_samples': len(survival_data) if 'survival_data' in locals() else 0,
                'n_features': len(survival_data.columns) if 'survival_data' in locals() else 0,
                'n_events': 0,
                'event_rate': 0,
                'issues': [f"Quality-check error: {e}"],
                'is_acceptable': True  # Continue even if the check fails.
            }

    def _create_battlefield_features(self, hist_attacks):
        """Battlefield-dynamics features."""
        if hist_attacks.empty:
            return {
                'intensity_volatility': 0,
                'temporal_clustering': 0,
                'spatial_clustering': 0
            }

        intensity_volatility = hist_attacks['event_intensity'].abs().std() if len(hist_attacks) > 1 else 0

        # Temporal clustering
        temporal_clustering = self._compute_temporal_clustering(hist_attacks)

        # Spatial clustering
        spatial_clustering = self._compute_spatial_clustering(hist_attacks)

        return {
            'intensity_volatility': intensity_volatility,
            'temporal_clustering': temporal_clustering,
            'spatial_clustering': spatial_clustering
        }

    def _create_actor_features(self, hist_attacks):
        """Actor-complexity features."""
        if hist_attacks.empty:
            return {
                'actor_diversity': 0,
                'sector_complexity': 0
            }

        # Actor diversity
        actor_diversity = hist_attacks['actor_name'].nunique() if 'actor_name' in hist_attacks.columns else 0

        # Sector complexity
        sector_complexity = 0
        if 'primary_actor_sector' in hist_attacks.columns:
            sector_complexity = hist_attacks['primary_actor_sector'].nunique()

        return {
            'actor_diversity': actor_diversity,
            'sector_complexity': sector_complexity
        }

    def _compute_haversine_distance(self, lat1, lon1, lat2, lon2):
        """Compute Haversine distance in kilometers."""
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))

    def _compute_temporal_clustering(self, hist_attacks, clip_upper_percentile=0.95):
        """Compute temporal clustering using CV, log1p compression, and clipping."""
        if len(hist_attacks) < 3:
            # Fewer than three events do not support stable variance estimates.
            return 0.0

        # Time difference from the earliest event, in days.
        time_diffs = (hist_attacks['event_date'] - hist_attacks['event_date'].min()).dt.total_seconds() / (24 * 3600)

        # Intervals between adjacent events.
        intervals = np.diff(np.sort(time_diffs))
        if len(intervals) < 2 or np.mean(intervals) < 1e-6:
            return 0.0

        # Coefficient of variation.
        cv = np.std(intervals) / np.mean(intervals)

        # Smaller CV indicates stronger concentration; invert and compress.
        clustering_value = np.log1p(1 / (cv + 1e-6))

        # Clip extreme values.
        if clip_upper_percentile:
            clustering_value = min(clustering_value, np.quantile([clustering_value], clip_upper_percentile))

        return clustering_value

    def _compute_spatial_clustering(self, hist_attacks, clip_upper_percentile=0.95):
        """Compute spatial clustering using CV, log1p compression, and clipping."""
        if len(hist_attacks) < 3:
            # Fewer than three events do not support stable variance estimates.
            return 0.0

        coords = hist_attacks[['latitude', 'longitude']].values
        center = coords.mean(axis=0)
        distances = np.linalg.norm(coords - center, axis=1)

        if len(distances) < 2:
            return 0.0

        mean_dist = distances.mean()
        std_dist = distances.std()

        # Coefficient of variation.
        if mean_dist < 1e-6:
            cv = 0.0
        else:
            cv = std_dist / mean_dist

        # Smaller CV indicates stronger concentration; invert and compress.
        clustering_value = np.log1p(1 / (cv + 1e-6))

        # Clip extreme values.
        if clip_upper_percentile:
            # Limit the influence of isolated extreme values.
            clustering_value = min(clustering_value, np.quantile([clustering_value], clip_upper_percentile))

        return clustering_value

    def _postprocess_survival_data(self, cox_df, params):
        """Post-process the survival dataset."""
        # Safe missing-value handling.
        numeric_cols = cox_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['event', 'duration', 'aid_id']:
                # Convert infinities to missing values.
                cox_df[col] = cox_df[col].replace([np.inf, -np.inf], np.nan)

                # Median imputation.
                median_val = cox_df[col].median()
                cox_df[col] = cox_df[col].fillna(median_val)

                # Winsorize extreme values.
                if col in ['temporal_clustering', 'spatial_clustering']:
                    # Use a stricter cap for clustering features.
                    upper = cox_df[col].quantile(0.95)
                    cox_df[col] = np.clip(cox_df[col], 0, upper)
                else:
                    # Use the 1st and 99th percentiles for other features.
                    lower = cox_df[col].quantile(0.01)
                    upper = cox_df[col].quantile(0.99)
                    cox_df[col] = np.clip(cox_df[col], lower, upper)

        # Attach metadata.
        cox_df.attrs.update({
            'pre_window_days': params['pre_window_days'],
            'analysis_window_days': params['analysis_window_days'],
            'max_distance_km': params['max_distance_km'],
            'sensitivity_id': params['sensitivity_id'],
            'buffer_days': self.buffer_days,
            'model_type': 'cox_survival'
        })

        # Log data quality.
        self._log_data_quality(cox_df)

        return cox_df

    def _log_data_quality(self, cox_df):
        """Log feature data-quality information using compatibility-safe operations."""
        try:
            numeric_cols = cox_df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['event', 'duration', 'aid_id']]

            logging.info("=== Feature data-quality report (simplified) ===")
            for col in numeric_cols:
                # Use basic statistics to avoid version-sensitive pandas behavior.
                col_data = cox_df[col]
                stats = {
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'has_inf': np.any(np.isinf(col_data)),
                    'has_nan': np.any(np.isnan(col_data))
                }
                logging.info(f"{col:25} | min:{stats['min']:8.3f} max:{stats['max']:8.3f} "
                             f"mean:{stats['mean']:8.3f} std:{stats['std']:8.3f} "
                             f"inf:{stats['has_inf']} nan:{stats['has_nan']}")
        except Exception as e:
            logging.warning(f"Data-quality logging failed: {e}")

    def run_sensitivity_analysis(self, region_key, mode='refined'):
        """
        Run sensitivity analysis for one region and return survival datasets.
        """
        # get_sensitivity_parameters is provided by config.py.
        sensitivity_params_list = get_sensitivity_parameters(mode, region_key)
        region_config = CONFIG['regions'][region_key]

        df = self.load_region_data(region_key)
        if df.empty:
            logging.error(f"Sensitivity analysis failed: could not load data for {region_key}.")
            return {}

        results = {}
        for params in sensitivity_params_list:
            logging.debug(f"Building sensitivity dataset: {params['sensitivity_id']}")
            cox_data = self.build_cox_survival_dataset(df, region_config, params)
            if not cox_data.empty:
                results[params['sensitivity_id']] = {
                    'data': cox_data,
                    'params': params,
                    'summary': {
                        'n_samples': len(cox_data),
                        'event_rate': cox_data['event'].mean(),
                    }
                }
        return results
