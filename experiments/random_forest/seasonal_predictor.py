"""
Enhanced Seasonal Pyracantha Predictor

This enhanced model specifically captures seasonal patterns and temporal dynamics:
1. Focuses on peak observation seasons (Autumn: Mar-May)
2. Uses weather patterns from observation periods as "ideal conditions"
3. Creates absence data from off-season periods and extreme weather
4. Includes phenological features (flowering/fruiting cycles)
5. Emphasizes recent observations as baseline for current habitat suitability

Key Insights from Data:
- 66% of observations occur in Autumn (Mar-May)
- Peak month is April (40% of all sightings)
- May is secondary peak (23% of sightings)
- Very low activity in Winter/Spring/Summer (34% combined)

This suggests Pyracantha is most visible during flowering/fruiting season.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import folium

class SeasonalPyracanthaPredictor:
    """
    Enhanced Random Forest predictor that captures seasonal patterns and temporal dynamics.
    
    Key Features:
    1. Seasonal Focus: Emphasizes autumn observation patterns
    2. Phenological Modeling: Includes flowering/fruiting cycles
    3. Recent Baseline: Uses recent observations as habitat baseline
    4. Temporal Dynamics: Models seasonal weather preferences
    5. Off-season Absence: Creates absence data from non-observation periods
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the seasonal predictor."""
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.feature_importance_ = None
        self.training_stats = {}
        self.seasonal_patterns = {}
        
    def analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze seasonal and temporal patterns in observations."""
        print("Analyzing seasonal patterns...")
        
        df_temp = df.copy()
        df_temp['time_observed_at'] = pd.to_datetime(df_temp['time_observed_at'])
        df_temp['obs_month'] = df_temp['time_observed_at'].dt.month
        df_temp['obs_year'] = df_temp['time_observed_at'].dt.year
        
        # Define seasons (Southern Hemisphere)
        season_map = {
            12: 'Summer', 1: 'Summer', 2: 'Summer',  # DJF
            3: 'Autumn', 4: 'Autumn', 5: 'Autumn',   # MAM - Peak season
            6: 'Winter', 7: 'Winter', 8: 'Winter',   # JJA
            9: 'Spring', 10: 'Spring', 11: 'Spring'  # SON
        }
        df_temp['obs_season'] = df_temp['obs_month'].map(season_map)
        
        # Unique observations by month/season
        obs_by_month = df_temp.groupby('obs_month')['inat_id'].nunique()
        obs_by_season = df_temp.groupby('obs_season')['inat_id'].nunique()
        
        # Recent observations (more weight for recent sightings)
        recent_cutoff = pd.Timestamp.now(tz='UTC') - pd.DateOffset(years=2)
        recent_obs = df_temp[df_temp['time_observed_at'] >= recent_cutoff]
        recent_by_season = recent_obs.groupby('obs_season')['inat_id'].nunique() if len(recent_obs) > 0 else obs_by_season
        
        # Peak season analysis
        peak_months = obs_by_month.nlargest(3).index.tolist()  # Top 3 months
        peak_season = obs_by_season.idxmax()
        
        patterns = {
            'obs_by_month': obs_by_month.to_dict(),
            'obs_by_season': obs_by_season.to_dict(),
            'recent_by_season': recent_by_season.to_dict(),
            'peak_months': peak_months,
            'peak_season': peak_season,
            'total_unique_obs': df_temp['inat_id'].nunique(),
            'peak_season_percentage': obs_by_season[peak_season] / df_temp['inat_id'].nunique() * 100
        }
        
        self.seasonal_patterns = patterns
        
        print(f"Peak observation season: {peak_season} ({patterns['peak_season_percentage']:.1f}% of observations)")
        print(f"Peak months: {peak_months}")
        print(f"Seasonal distribution: {dict(obs_by_season)}")
        
        return patterns
    
    def load_and_preprocess_data(self, data_path: str) -> pd.DataFrame:
        """Load and preprocess data with enhanced seasonal features."""
        print("Loading and preprocessing data with seasonal focus...")
        
        df = pd.read_csv(data_path)
        print(f"Loaded dataset: {df.shape[0]:,} records, {df.shape[1]} columns")
        
        # Convert dates
        df['date'] = pd.to_datetime(df['date'])
        df['time_observed_at'] = pd.to_datetime(df['time_observed_at'], utc=True)
        
        # Add comprehensive temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Southern Hemisphere seasons
        season_map = {
            12: 'Summer', 1: 'Summer', 2: 'Summer',
            3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
            6: 'Winter', 7: 'Winter', 8: 'Winter',
            9: 'Spring', 10: 'Spring', 11: 'Spring'
        }
        df['season'] = df['month'].map(season_map)
        
        # Phenological features based on observation patterns
        # Peak observation period (Autumn: Mar-May)
        df['is_peak_season'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_peak_month'] = df['month'].isin([4]).astype(int)  # April is strongest
        df['is_secondary_peak'] = df['month'].isin([5]).astype(int)  # May is secondary
        
        # Flowering/fruiting cycle approximation
        # Based on when species is most visible (reported)
        df['flowering_intensity'] = df['month'].map({
            1: 0.1, 2: 0.2, 3: 0.7, 4: 1.0, 5: 0.8, 6: 0.3,
            7: 0.1, 8: 0.1, 9: 0.2, 10: 0.3, 11: 0.4, 12: 0.2
        })
        
        # Distance from peak months (April = 0, decreasing with distance)
        df['distance_from_peak'] = df['month'].apply(
            lambda x: min(abs(x - 4), abs(x - 4 + 12), abs(x - 4 - 12))
        )
        
        # Seasonal weather preferences (based on observation periods)
        # These will be learned from actual observation weather
        df['optimal_season_weight'] = df['month'].map({
            1: 0.2, 2: 0.3, 3: 0.8, 4: 1.0, 5: 0.9, 6: 0.4,
            7: 0.2, 8: 0.2, 9: 0.3, 10: 0.4, 11: 0.5, 12: 0.3
        })
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Calculate temporal relationship to observations
        df_obs_dates = df.groupby('inat_id')['time_observed_at'].first().reset_index()
        df_obs_dates.columns = ['inat_id', 'observation_date']
        df = df.merge(df_obs_dates, on='inat_id')
        df['observation_date'] = df['observation_date'].dt.tz_localize(None)
        df['days_to_observation'] = (df['observation_date'] - df['date']).dt.days
        
        # Recent observation weight (emphasize recent sightings)
        current_year = datetime.now().year
        df['observation_recency'] = df['time_observed_at'].dt.year.apply(
            lambda x: max(0, 1 - (current_year - x) / 10)  # Linear decay over 10 years
        )
        
        return df
    
    def create_seasonal_training_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Create training dataset emphasizing seasonal patterns."""
        print("Creating seasonal training dataset...")
        
        # Analyze patterns first
        self.analyze_seasonal_patterns(df)
        
        # All records are potential presence, but weight by season and recency
        presence_data = df.copy()
        presence_data['target'] = 1
        
        # Calculate presence probability based on seasonal patterns and recency
        # Higher weight for peak season and recent observations
        presence_data['presence_weight'] = (
            presence_data['optimal_season_weight'] * 
            presence_data['observation_recency'] * 
            presence_data['flowering_intensity']
        )
        
        print(f"Total presence records: {len(presence_data):,}")
        
        # Create absence data with seasonal awareness
        absence_records = []
        
        # Strategy 1: Off-season periods (when species is rarely observed)
        off_season_data = df[df['optimal_season_weight'] < 0.5].copy()
        if len(off_season_data) > 0:
            n_off_season = min(len(off_season_data), len(presence_data) // 2)
            off_season_sample = off_season_data.sample(n=n_off_season, random_state=self.random_state)
            off_season_sample['target'] = 0
            off_season_sample['absence_reason'] = 'off_season'
            absence_records.append(off_season_sample)
            print(f"Off-season absence records: {n_off_season:,}")
        
        # Strategy 2: Pre-observation periods (before species was observed at location)
        pre_obs_data = df[df['days_to_observation'] >= 60].copy()  # At least 2 months before
        if len(pre_obs_data) > 0:
            n_pre_obs = min(len(pre_obs_data), len(presence_data) // 3)
            pre_obs_sample = pre_obs_data.sample(n=n_pre_obs, random_state=self.random_state)
            pre_obs_sample['target'] = 0
            pre_obs_sample['absence_reason'] = 'pre_observation'
            absence_records.append(pre_obs_sample)
            print(f"Pre-observation absence records: {n_pre_obs:,}")
        
        # Strategy 3: Extreme weather conditions during potential seasons
        # Use weather that's too extreme even for peak season
        peak_season_data = df[df['is_peak_season'] == 1].copy()
        
        if len(peak_season_data) > 0:
            # Temperature extremes
            temp_extreme = peak_season_data[
                (peak_season_data['T2M'] < peak_season_data['T2M'].quantile(0.05)) |
                (peak_season_data['T2M'] > peak_season_data['T2M'].quantile(0.95))
            ].copy()
            
            # Drought stress during peak season
            drought_stress = peak_season_data[
                (peak_season_data['PRECTOTCORR'] < peak_season_data['PRECTOTCORR'].quantile(0.05)) &
                (peak_season_data['RH2M'] < peak_season_data['RH2M'].quantile(0.1))
            ].copy()
            
            # Wind stress
            wind_extreme = peak_season_data[
                peak_season_data['WS2M'] > peak_season_data['WS2M'].quantile(0.95)
            ].copy()
            
            extreme_conditions = []
            for extreme_data in [temp_extreme, drought_stress, wind_extreme]:
                if len(extreme_data) > 0:
                    n_extreme = min(len(extreme_data), len(presence_data) // 8)
                    extreme_sample = extreme_data.sample(n=n_extreme, random_state=self.random_state)
                    extreme_sample['target'] = 0
                    extreme_sample['absence_reason'] = 'extreme_weather'
                    extreme_conditions.append(extreme_sample)
            
            if extreme_conditions:
                combined_extreme = pd.concat(extreme_conditions, ignore_index=True)
                absence_records.append(combined_extreme)
                print(f"Extreme weather absence records: {len(combined_extreme):,}")
        
        # Strategy 4: Geographic distance from known observations during peak season
        # Create absence points far from any known observation
        if len(presence_data) > 0:
            # Sample locations far from observations
            lat_range = (presence_data['latitude_x'].min() - 0.5, presence_data['latitude_x'].max() + 0.5)
            lon_range = (presence_data['longitude_x'].min() - 0.5, presence_data['longitude_x'].max() + 0.5)
            
            # Create synthetic locations
            n_synthetic = len(presence_data) // 4
            synthetic_data = presence_data.sample(n=n_synthetic, random_state=self.random_state).copy()
            
            # Randomly displace coordinates
            np.random.seed(self.random_state)
            synthetic_data['latitude_x'] += np.random.uniform(-0.3, 0.3, n_synthetic)
            synthetic_data['longitude_x'] += np.random.uniform(-0.3, 0.3, n_synthetic)
            
            # Keep within bounds
            synthetic_data['latitude_x'] = np.clip(synthetic_data['latitude_x'], lat_range[0], lat_range[1])
            synthetic_data['longitude_x'] = np.clip(synthetic_data['longitude_x'], lon_range[0], lon_range[1])
            
            synthetic_data['target'] = 0
            synthetic_data['absence_reason'] = 'geographic_distance'
            absence_records.append(synthetic_data)
            print(f"Geographic distance absence records: {n_synthetic:,}")
        
        # Combine all data
        if absence_records:
            all_absence = pd.concat(absence_records, ignore_index=True)
            combined_data = pd.concat([presence_data, all_absence], ignore_index=True)
        else:
            combined_data = presence_data
        
        print(f"Final dataset: {len(combined_data):,} records")
        print(f"  Presence: {len(combined_data[combined_data['target'] == 1]):,}")
        print(f"  Absence: {len(combined_data[combined_data['target'] == 0]):,}")
        
        # Prepare features and target - only numeric columns
        exclude_cols = [
            'target', 'inat_id', 'uuid', 'time_observed_at', 'created_at',
            'place_guess', 'scientific_name', 'common_name', 'quality_grade',
            'image_url', 'user_id', 'observation_date', 'season', 'id',
            'presence_weight', 'absence_reason', 'date'
        ]
        
        # Select only numeric columns that are not in exclude list
        numeric_cols = combined_data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        X = combined_data[feature_cols]
        y = combined_data['target']
        
        self.feature_columns = feature_cols
        print(f"Selected {len(feature_cols)} features for training")
        
        return X, y
    
    def train_seasonal_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train Random Forest with emphasis on seasonal patterns."""
        print("Training seasonal Random Forest model...")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest with optimized parameters for seasonal patterns
        self.model = RandomForestClassifier(
            n_estimators=500,  # More trees for complex seasonal patterns
            max_depth=15,      # Deeper trees to capture seasonal interactions
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'  # Handle imbalanced seasonal data
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions and evaluation
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance_ = feature_importance
        
        # Training statistics
        self.training_stats.update({
            'accuracy': accuracy,
            'auc_score': auc_score,
            'n_features': len(self.feature_columns),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'seasonal_patterns': self.seasonal_patterns
        })
        
        print(f"Model Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC Score: {auc_score:.4f}")
        print(f"  Features: {len(self.feature_columns)}")
        
        print(f"\\nTop 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self.training_stats
    
    def save_model(self, output_dir: Path):
        """Save the trained seasonal model."""
        output_dir.mkdir(exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance_,
            'training_stats': self.training_stats,
            'seasonal_patterns': self.seasonal_patterns,
            'model_type': 'seasonal_pyracantha_predictor',
            'training_date': datetime.now().isoformat()
        }
        
        model_path = output_dir / "seasonal_pyracantha_model.pkl"
        joblib.dump(model_data, model_path)
        
        # Save human-readable summary
        summary_path = output_dir / "seasonal_model_summary.json"
        summary = {
            'model_type': 'Seasonal Pyracantha Predictor',
            'training_date': datetime.now().isoformat(),
            'performance': {
                'accuracy': self.training_stats['accuracy'],
                'auc_score': self.training_stats['auc_score']
            },
            'seasonal_insights': self.seasonal_patterns,
            'top_features': self.feature_importance_.head(15).to_dict('records'),
            'feature_count': len(self.feature_columns)
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Seasonal model saved to: {model_path}")
        print(f"Model summary saved to: {summary_path}")
        
        return model_path

def main():
    """Run the enhanced seasonal analysis and training."""
    print("=" * 70)
    print("ENHANCED SEASONAL PYRACANTHA PREDICTOR")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Initialize predictor
        predictor = SeasonalPyracanthaPredictor(random_state=42)
        
        # Load and preprocess data
        data_path = "../../data/dataset.csv"
        df = predictor.load_and_preprocess_data(data_path)
        
        # Create seasonal training dataset
        X, y = predictor.create_seasonal_training_dataset(df)
        
        # Train seasonal model
        stats = predictor.train_seasonal_model(X, y)
        
        # Save model
        output_dir = Path("outputs")
        model_path = predictor.save_model(output_dir)
        
        print()
        print("=" * 70)
        print("SEASONAL MODEL TRAINING COMPLETED!")
        print(f"Model saved: {model_path}")
        print(f"Peak observation season: {stats['seasonal_patterns']['peak_season']}")
        print(f"Peak season coverage: {stats['seasonal_patterns']['peak_season_percentage']:.1f}%")
        print(f"Model accuracy: {stats['accuracy']:.4f}")
        print(f"AUC Score: {stats['auc_score']:.4f}")
        print("=" * 70)
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()
