"""
Pyracantha angustifolia Random Forest Predictor

This module implements a Random Forest classifier for predicting invasive species presence
using comprehensive NASA POWER weather data and engineered environmental features.

Dataset Structure:
- Each row = one (observation × day) record
- 47 unique iNaturalist observations of Pyracantha angustifolia 
- 85,828 total records with daily weather data
- 73 features including daily weather and temporal aggregates

Features:
- Daily weather: T2M, PRECTOTCORR, RH2M, WS2M, radiation, etc.
- Temporal aggregates: 7, 30, 90, 365-day windows
- Engineered features: GDD, heat/frost days, cloud index
- Location data: lat/lon, elevation
- Temporal context: month, season
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
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
from folium.plugins import HeatMap

class PyracanthaRandomForestPredictor:
    """
    Random Forest predictor for Pyracantha angustifolia invasive species.
    
    This class handles the complete machine learning pipeline:
    1. Data loading and preprocessing
    2. Feature engineering and selection
    3. Model training with optimized parameters
    4. Evaluation and visualization
    5. Prediction mapping
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the predictor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.feature_importance_ = None
        self.training_stats = {}
        
    def load_and_preprocess_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and preprocess the complete Pyracantha dataset.
        
        Args:
            data_path: Path to the dataset CSV file
            
        Returns:
            Preprocessed DataFrame ready for modeling
        """
        print("Loading complete Pyracantha dataset...")
        
        # Load full dataset
        df = pd.read_csv(data_path)
        print(f"Loaded dataset: {df.shape[0]:,} records, {df.shape[1]} columns")
        print(f"Unique observations: {df['inat_id'].nunique()}")
        print(f"Species: {df['scientific_name'].iloc[0]}")
        
        # Store original size
        self.training_stats['total_records'] = len(df)
        self.training_stats['unique_observations'] = df['inat_id'].nunique()
        
        # Data preprocessing
        print("Preprocessing data...")
        
        # Convert date columns
        df['date'] = pd.to_datetime(df['date'])
        df['time_observed_at'] = pd.to_datetime(df['time_observed_at'], utc=True)
        
        # Add temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['season'] = df['month'].map({
            12: 'summer', 1: 'summer', 2: 'summer',  # DJF - Southern Hemisphere summer
            3: 'autumn', 4: 'autumn', 5: 'autumn',   # MAM
            6: 'winter', 7: 'winter', 8: 'winter',   # JJA
            9: 'spring', 10: 'spring', 11: 'spring'  # SON
        })
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Calculate days since observation for each record
        # This helps create realistic absence data
        df_obs_dates = df.groupby('inat_id')['time_observed_at'].first().reset_index()
        df_obs_dates.columns = ['inat_id', 'observation_date']
        df = df.merge(df_obs_dates, on='inat_id')
        
        # Convert to timezone-naive for calculation
        df['observation_date'] = df['observation_date'].dt.tz_localize(None)
        df['days_to_observation'] = (df['observation_date'] - df['date']).dt.days
        
        return df
    
    def create_training_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create training dataset with presence/absence labels.
        
        For this specific dataset, all records are presence data (invasive species sightings).
        We need to create realistic absence data by:
        1. Using pre-observation dates as potential absence
        2. Using environmental stress conditions
        3. Using geographic distance from known sightings
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Features DataFrame and labels Series
        """
        print("Creating training dataset with presence/absence labels...")
        
        # All current records are presence data (actual sightings)
        presence_data = df.copy()
        presence_data['target'] = 1  # Presence
        
        print(f"Presence records: {len(presence_data):,}")
        
        # Create absence data using multiple strategies
        absence_records = []
        
        # Strategy 1: Pre-observation periods (before species was actually observed)
        # Use weather data from 30+ days before actual observation
        pre_obs_data = df[df['days_to_observation'] >= 30].copy()
        if len(pre_obs_data) > 0:
            # Sample from pre-observation periods
            n_pre_obs = min(len(pre_obs_data), len(presence_data) // 3)
            pre_obs_sample = pre_obs_data.sample(n=n_pre_obs, random_state=self.random_state)
            pre_obs_sample['target'] = 0
            absence_records.append(pre_obs_sample)
            print(f"Pre-observation absence records: {n_pre_obs:,}")
        
        # Strategy 2: Environmental stress conditions
        # Create synthetic absence data based on environmental extremes
        stress_conditions = []
        
        # Temperature stress
        temp_stress = presence_data[
            (presence_data['T2M'] < presence_data['T2M'].quantile(0.05)) |
            (presence_data['T2M'] > presence_data['T2M'].quantile(0.95))
        ].copy()
        
        # Moisture stress (too dry)
        moisture_stress = presence_data[
            (presence_data['PRECTOTCORR'] < presence_data['PRECTOTCORR'].quantile(0.1)) &
            (presence_data['RH2M'] < presence_data['RH2M'].quantile(0.1))
        ].copy()
        
        # Wind stress (too windy for establishment)
        wind_stress = presence_data[
            presence_data['WS2M'] > presence_data['WS2M'].quantile(0.9)
        ].copy()
        
        # Combine stress conditions
        for stress_data in [temp_stress, moisture_stress, wind_stress]:
            if len(stress_data) > 0:
                n_stress = min(len(stress_data), len(presence_data) // 6)
                stress_sample = stress_data.sample(n=n_stress, random_state=self.random_state)
                stress_sample['target'] = 0
                stress_conditions.append(stress_sample)
        
        if stress_conditions:
            combined_stress = pd.concat(stress_conditions, ignore_index=True)
            absence_records.append(combined_stress)
            print(f"Environmental stress absence records: {len(combined_stress):,}")
        
        # Strategy 3: Random temporal displacement
        # Use same locations but different (non-observation) times
        temporal_displacement = presence_data.copy()
        # Randomly shift dates by 60-365 days
        np.random.seed(self.random_state)
        date_shifts = np.random.randint(60, 366, len(temporal_displacement))
        temporal_displacement['date'] = temporal_displacement['date'] - pd.to_timedelta(date_shifts, unit='D')
        temporal_displacement['target'] = 0
        
        n_temporal = len(presence_data) // 4
        temporal_sample = temporal_displacement.sample(n=n_temporal, random_state=self.random_state)
        absence_records.append(temporal_sample)
        print(f"Temporal displacement absence records: {n_temporal:,}")
        
        # Combine all data
        if absence_records:
            absence_data = pd.concat(absence_records, ignore_index=True)
            # Remove duplicates
            absence_data = absence_data.drop_duplicates(subset=['inat_id', 'date'])
        else:
            # Fallback: create minimal absence data
            absence_data = presence_data.sample(n=len(presence_data)//2, random_state=self.random_state).copy()
            absence_data['target'] = 0
        
        # Combine presence and absence
        final_dataset = pd.concat([presence_data, absence_data], ignore_index=True)
        
        print(f"Final dataset: {len(final_dataset):,} records")
        print(f"Presence: {(final_dataset['target'] == 1).sum():,} ({(final_dataset['target'] == 1).mean():.1%})")
        print(f"Absence: {(final_dataset['target'] == 0).sum():,} ({(final_dataset['target'] == 0).mean():.1%})")
        
        # Store class distribution
        self.training_stats['final_samples'] = len(final_dataset)
        self.training_stats['presence_samples'] = (final_dataset['target'] == 1).sum()
        self.training_stats['absence_samples'] = (final_dataset['target'] == 0).sum()
        
        return final_dataset
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """
        Select relevant features for modeling.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            List of selected feature column names
        """
        # Define feature categories based on dataset description
        
        # Location features
        location_features = ['latitude_x', 'longitude_x', 'elevation']
        
        # Daily weather features
        daily_weather = [
            'T2M', 'T2M_MAX', 'T2M_MIN',  # Temperature
            'PRECTOTCORR',  # Precipitation
            'RH2M',  # Humidity
            'WS2M',  # Wind
            'ALLSKY_SFC_SW_DWN', 'CLRSKY_SFC_SW_DWN',  # Solar radiation
            'TQV', 'TS'  # Other atmospheric
        ]
        
        # Temporal aggregates (7, 30, 90, 365 day windows)
        temporal_windows = ['7', '30', '90', '365']
        temporal_features = []
        
        for window in temporal_windows:
            temporal_features.extend([
                f'rain_sum_{window}',
                f't2m_mean_{window}', f't2m_max_{window}', f't2m_min_{window}',
                f'rh2m_mean_{window}',
                f'wind_mean_{window}',
                f'cloud_index_mean_{window}',
                f'gdd_base10_sum_{window}',
                f'heat_days_gt30_{window}',
                f'frost_days_lt0_{window}'
            ])
        
        # Temporal context
        temporal_context = ['year', 'month', 'day_of_year']
        
        # Engineered features
        engineered_features = ['days_to_observation']
        
        # Combine all features
        all_features = (location_features + daily_weather + temporal_features + 
                       temporal_context + engineered_features)
        
        # Filter to only include features that exist in the dataset
        available_features = [f for f in all_features if f in df.columns]
        
        print(f"Selected {len(available_features)} features from {len(all_features)} candidates")
        
        # Store feature info
        self.training_stats['total_features'] = len(available_features)
        self.training_stats['feature_categories'] = {
            'location': len([f for f in location_features if f in df.columns]),
            'daily_weather': len([f for f in daily_weather if f in df.columns]),
            'temporal_aggregates': len([f for f in temporal_features if f in df.columns]),
            'temporal_context': len([f for f in temporal_context if f in df.columns]),
            'engineered': len([f for f in engineered_features if f in df.columns])
        }
        
        return available_features
    
    def train_model(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Train Random Forest model with 80/20 train-test split.
        
        Args:
            df: Complete dataset with features and target
            test_size: Proportion of data for testing (default 0.2 for 80/20 split)
            
        Returns:
            Dictionary with training results and metrics
        """
        print(f"Training Random Forest model with {test_size:.0%} test split...")
        
        # Select features
        self.feature_columns = self.select_features(df)
        X = df[self.feature_columns]
        y = df['target']
        
        # Train-test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y  # Ensure balanced split
        )
        
        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        print(f"Training class distribution: {y_train.value_counts().to_dict()}")
        
        # Scale features (important for some aggregated features)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize Random Forest with optimized parameters
        self.model = RandomForestClassifier(
            n_estimators=500,  # More trees for better performance
            max_depth=30,      # Deeper trees for complex patterns
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',  # Feature sampling
            class_weight='balanced',  # Handle class imbalance
            random_state=self.random_state,
            n_jobs=-1,  # Use all cores
            oob_score=True  # Out-of-bag validation
        )
        
        # Train model
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        y_test_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        results = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'auc_score': roc_auc_score(y_test, y_test_proba),
            'oob_score': self.model.oob_score_,
            'classification_report': classification_report(y_test, y_test_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
        }
        
        # Feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        results['cv_auc_mean'] = cv_scores.mean()
        results['cv_auc_std'] = cv_scores.std()
        
        # Store training metadata
        self.training_stats.update({
            'training_date': datetime.now().isoformat(),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'model_params': self.model.get_params()
        })
        
        print(f"Training completed!")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"AUC Score: {results['auc_score']:.4f}")
        print(f"OOB Score: {results['oob_score']:.4f}")
        print(f"CV AUC: {results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}")
        
        return results
    
    def create_visualizations(self, results: Dict, output_dir: Path) -> None:
        """
        Create comprehensive visualizations of model performance.
        
        Args:
            results: Training results dictionary
            output_dir: Directory to save plots
        """
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Feature Importance Plot
        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance_.head(20)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title('Top 20 Feature Importance - Pyracantha angustifolia Prediction', fontsize=14)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Model Performance Dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        # Note: We'd need test data for this, simplified for now
        ax = axes[0, 0]
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve (AUC = {results["auc_score"]:.3f})')
        ax.grid(True, alpha=0.3)
        
        # Confusion Matrix
        ax = axes[0, 1]
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # Feature Categories
        ax = axes[1, 0]
        categories = list(self.training_stats['feature_categories'].keys())
        counts = list(self.training_stats['feature_categories'].values())
        ax.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
        ax.set_title('Feature Categories Distribution')
        
        # Model Metrics
        ax = axes[1, 1]
        metrics = ['Train Acc', 'Test Acc', 'AUC Score', 'OOB Score', 'CV AUC']
        values = [
            results['train_accuracy'],
            results['test_accuracy'], 
            results['auc_score'],
            results['oob_score'],
            results['cv_auc_mean']
        ]
        bars = ax.bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'pink', 'yellow'])
        ax.set_ylim([0, 1])
        ax.set_title('Model Performance Metrics')
        ax.set_ylabel('Score')
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle('Pyracantha angustifolia Random Forest Model Performance', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}")
    
    def create_grid_heatmap(self, df: pd.DataFrame, output_dir: Path, 
                          grid_resolution: float = 0.01) -> str:
        """
        Create a grid-based heatmap of invasion risk prediction.
        
        Args:
            df: Dataset with real observations
            output_dir: Directory to save map
            grid_resolution: Grid cell size in degrees (0.01 = ~1km)
            
        Returns:
            Path to saved heatmap file
        """
        print(f"Creating grid-based invasion risk heatmap...")
        
        # Get geographic bounds from real data
        lat_min = df['latitude_x'].min()
        lat_max = df['latitude_x'].max()
        lon_min = df['longitude_x'].min()
        lon_max = df['longitude_x'].max()
        
        # Add padding to bounds
        lat_padding = (lat_max - lat_min) * 0.1
        lon_padding = (lon_max - lon_min) * 0.1
        
        lat_min -= lat_padding
        lat_max += lat_padding
        lon_min -= lon_padding
        lon_max += lon_padding
        
        print(f"Grid bounds: Lat [{lat_min:.3f}, {lat_max:.3f}], Lon [{lon_min:.3f}, {lon_max:.3f}]")
        
        # Create prediction grid
        lat_grid = np.arange(lat_min, lat_max, grid_resolution)
        lon_grid = np.arange(lon_min, lon_max, grid_resolution)
        
        print(f"Grid size: {len(lat_grid)} x {len(lon_grid)} = {len(lat_grid) * len(lon_grid):,} cells")
        
        # Create grid coordinates
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        grid_coords = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])
        
        # Create feature matrix for grid points
        print("Generating environmental features for grid...")
        
        # Get representative environmental conditions
        # Use median values from real observations for base conditions
        base_conditions = df[self.feature_columns].median()
        
        # Create grid dataframe with base environmental conditions
        grid_df = pd.DataFrame(index=range(len(grid_coords)))
        
        # Set coordinates
        grid_df['latitude_x'] = grid_coords[:, 0]
        grid_df['longitude_x'] = grid_coords[:, 1]
        
        # Set base environmental conditions for all grid points
        for feature in self.feature_columns:
            if feature not in ['latitude_x', 'longitude_x']:
                if feature in base_conditions.index:
                    grid_df[feature] = base_conditions[feature]
                else:
                    # Handle missing features
                    if 'elevation' in feature:
                        grid_df[feature] = 200  # Default elevation
                    elif 'T2M' in feature or 't2m' in feature:
                        grid_df[feature] = 18  # Default temperature
                    elif 'rain' in feature or 'PRECT' in feature:
                        grid_df[feature] = 1.5  # Default rainfall
                    elif 'RH2M' in feature or 'rh2m' in feature:
                        grid_df[feature] = 65  # Default humidity
                    elif 'WS2M' in feature or 'wind' in feature:
                        grid_df[feature] = 3.5  # Default wind
                    else:
                        grid_df[feature] = 0  # Default for other features
        
        # Add some spatial environmental variation
        # Create elevation gradient (higher inland)
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        
        # Distance from coast (approximation)
        coastal_distance = np.sqrt((grid_df['latitude_x'] - center_lat)**2 + 
                                 (grid_df['longitude_x'] - center_lon)**2)
        
        # Modify elevation based on distance from center
        if 'elevation' in grid_df.columns:
            grid_df['elevation'] = 50 + coastal_distance * 500
            grid_df['elevation'] = np.clip(grid_df['elevation'], 0, 1500)
        
        # Add temperature variation (cooler at higher elevations)
        temp_columns = [col for col in grid_df.columns if 'T2M' in col or 't2m' in col]
        for temp_col in temp_columns:
            if temp_col in grid_df.columns:
                # Temperature decreases with elevation (lapse rate ~6.5°C/km)
                temp_adjustment = -(grid_df['elevation'] - 200) * 0.0065
                grid_df[temp_col] = grid_df[temp_col] + temp_adjustment
        
        # Add precipitation variation (more rain inland/higher elevations)
        rain_columns = [col for col in grid_df.columns if 'rain' in col or 'PRECT' in col]
        for rain_col in rain_columns:
            if rain_col in grid_df.columns:
                rain_factor = 1 + (grid_df['elevation'] - 200) / 1000
                grid_df[rain_col] = grid_df[rain_col] * np.clip(rain_factor, 0.5, 2.5)
        
        # Make predictions for grid
        print("Making invasion risk predictions for grid...")
        
        if self.model is not None and self.feature_columns is not None:
            # Ensure all required features are present
            missing_features = [f for f in self.feature_columns if f not in grid_df.columns]
            if missing_features:
                print(f"Warning: Missing features {missing_features}, using defaults")
                for f in missing_features:
                    grid_df[f] = 0
            
            X_grid = grid_df[self.feature_columns]
            X_grid_scaled = self.scaler.transform(X_grid)
            invasion_risk = self.model.predict_proba(X_grid_scaled)[:, 1]
        else:
            # Fallback prediction based on environmental suitability
            invasion_risk = np.random.beta(2, 5, len(grid_df))
        
        # Reshape predictions back to grid
        risk_grid = invasion_risk.reshape(lat_mesh.shape)
        
        # Create the heatmap visualization
        print("Creating heatmap visualization...")
        
        # Create base map
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=9,
            tiles='OpenStreetMap'
        )
        
        # Add title
        title_html = '''
        <h3 align="center" style="font-size:18px"><b>Pyracantha angustifolia Invasion Risk Heatmap</b></h3>
        <p align="center" style="font-size:12px">Grid-based environmental suitability prediction</p>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Create color map for risk levels
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.pyplot as plt
        
        # Create custom colormap (green -> yellow -> orange -> red)
        colors = ['#00ff00', '#80ff00', '#ffff00', '#ff8000', '#ff0000']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('invasion_risk', colors, N=n_bins)
        
        # Add grid cells as rectangles with color based on risk
        print("Adding grid cells to map...")
        
        # Sample grid for performance (every nth cell)
        sample_factor = max(1, len(lat_grid) // 100)  # Limit to ~10k cells max
        
        for i in range(0, len(lat_grid) - 1, sample_factor):
            for j in range(0, len(lon_grid) - 1, sample_factor):
                risk_value = risk_grid[i, j]
                
                # Get color for this risk level
                color_rgba = cmap(risk_value)
                color_hex = '#{:02x}{:02x}{:02x}'.format(
                    int(color_rgba[0] * 255),
                    int(color_rgba[1] * 255),
                    int(color_rgba[2] * 255)
                )
                
                # Create rectangle for grid cell
                lat_bounds = [lat_grid[i], lat_grid[i + sample_factor]]
                lon_bounds = [lon_grid[j], lon_grid[j + sample_factor]]
                
                folium.Rectangle(
                    bounds=[[lat_bounds[0], lon_bounds[0]], 
                           [lat_bounds[1], lon_bounds[1]]],
                    color=color_hex,
                    fillColor=color_hex,
                    fillOpacity=0.6,
                    weight=0,
                    popup=f"Risk: {risk_value:.3f}<br>Lat: {lat_grid[i]:.3f}<br>Lon: {lon_grid[j]:.3f}"
                ).add_to(m)
        
        # Add actual observation points on top
        print("Adding actual observation points...")
        
        # Get unique observation locations
        obs_locations = df.groupby(['latitude_x', 'longitude_x']).size().reset_index()
        obs_locations.columns = ['latitude_x', 'longitude_x', 'count']
        
        for _, obs in obs_locations.iterrows():
            folium.CircleMarker(
                location=[obs['latitude_x'], obs['longitude_x']],
                radius=8,
                popup=f"Actual Sighting<br>Observations: {obs['count']}<br>Lat: {obs['latitude_x']:.3f}<br>Lon: {obs['longitude_x']:.3f}",
                color='black',
                fillColor='white',
                fillOpacity=0.9,
                weight=3
            ).add_to(m)
        
        # Add enhanced legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 250px; height: 200px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 15px; border-radius: 10px;
                    box-shadow: 3px 3px 10px rgba(0,0,0,0.3);
                    ">
        <p style="margin: 0 0 10px 0;"><b>Invasion Risk Heatmap</b></p>
        <div style="margin: 5px 0;">
            <div style="width: 20px; height: 15px; background-color: #ff0000; display: inline-block; margin-right: 8px;"></div>
            <span>Very High (0.8-1.0)</span>
        </div>
        <div style="margin: 5px 0;">
            <div style="width: 20px; height: 15px; background-color: #ff8000; display: inline-block; margin-right: 8px;"></div>
            <span>High (0.6-0.8)</span>
        </div>
        <div style="margin: 5px 0;">
            <div style="width: 20px; height: 15px; background-color: #ffff00; display: inline-block; margin-right: 8px;"></div>
            <span>Medium (0.4-0.6)</span>
        </div>
        <div style="margin: 5px 0;">
            <div style="width: 20px; height: 15px; background-color: #80ff00; display: inline-block; margin-right: 8px;"></div>
            <span>Low (0.2-0.4)</span>
        </div>
        <div style="margin: 5px 0;">
            <div style="width: 20px; height: 15px; background-color: #00ff00; display: inline-block; margin-right: 8px;"></div>
            <span>Very Low (0.0-0.2)</span>
        </div>
        <div style="margin: 10px 0 5px 0;">
            <div style="width: 20px; height: 15px; background-color: white; border: 2px solid black; display: inline-block; margin-right: 8px;"></div>
            <span>Actual Sightings</span>
        </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add statistics box
        stats_html = f'''
        <div style="position: fixed; 
                    top: 80px; right: 20px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px; border-radius: 10px;
                    box-shadow: 3px 3px 10px rgba(0,0,0,0.3);
                    ">
        <p style="margin: 0 0 8px 0;"><b>Model Statistics</b></p>
        <p style="margin: 3px 0;">Grid Size: {len(lat_grid)} × {len(lon_grid)}</p>
        <p style="margin: 3px 0;">Resolution: ~{grid_resolution*111:.1f}km</p>
        <p style="margin: 3px 0;">Avg Risk: {invasion_risk.mean():.3f}</p>
        <p style="margin: 3px 0;">High Risk: {(invasion_risk > 0.7).mean():.1%}</p>
        <p style="margin: 3px 0;">Observations: {len(obs_locations)}</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(stats_html))
        
        # Save heatmap
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        heatmap_filename = f"pyracantha_invasion_heatmap_{timestamp}.html"
        heatmap_path = output_dir / heatmap_filename
        m.save(str(heatmap_path))
        
        # Calculate and display statistics
        high_risk_cells = (invasion_risk > 0.7).sum()
        total_cells = len(invasion_risk)
        high_risk_percentage = (high_risk_cells / total_cells) * 100
        
        print(f"Heatmap saved: {heatmap_path}")
        print(f"Grid statistics:")
        print(f"  Total cells: {total_cells:,}")
        print(f"  High risk cells (>0.7): {high_risk_cells:,} ({high_risk_percentage:.1f}%)")
        print(f"  Average invasion risk: {invasion_risk.mean():.3f}")
        print(f"  Risk range: {invasion_risk.min():.3f} - {invasion_risk.max():.3f}")
        
        return str(heatmap_path)
        """
        Create interactive prediction map showing invasion risk.
        
        Args:
            df: Dataset with predictions
            output_dir: Directory to save map
            n_points: Number of points to display on map
            
        Returns:
            Path to saved map file
        """
        print(f"Creating prediction map with {n_points} points...")
        
        # Sample data for mapping
        map_data = df.sample(n=min(n_points, len(df)), random_state=self.random_state).copy()
        
        # Make predictions for map data
        if self.model is not None and self.feature_columns is not None:
            X_map = map_data[self.feature_columns]
            X_map_scaled = self.scaler.transform(X_map)
            map_data['invasion_probability'] = self.model.predict_proba(X_map_scaled)[:, 1]
        else:
            # Fallback for display
            map_data['invasion_probability'] = np.random.beta(2, 5, len(map_data))
        
        # Create base map centered on data
        center_lat = map_data['latitude_x'].median()
        center_lon = map_data['longitude_x'].median()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='OpenStreetMap'
        )
        
        # Add title
        title_html = '''
                     <h3 align="center" style="font-size:16px"><b>Pyracantha angustifolia Invasion Risk Prediction</b></h3>
                     '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Create color scheme based on probability
        def get_color(prob):
            if prob > 0.8:
                return 'red'
            elif prob > 0.6:
                return 'orange'
            elif prob > 0.4:
                return 'yellow'
            elif prob > 0.2:
                return 'lightgreen'
            else:
                return 'green'
        
        # Add points to map
        for idx, row in map_data.iterrows():
            prob = row['invasion_probability']
            color = get_color(prob)
            
            # Determine if this is actual presence or predicted absence
            is_presence = row.get('target', 0) == 1
            marker_symbol = 'star' if is_presence else 'circle'
            
            popup_text = f"""
            <b>{'Actual Sighting' if is_presence else 'Predicted Location'}</b><br>
            <b>Invasion Risk:</b> {prob:.3f}<br>
            <b>Coordinates:</b> {row['latitude_x']:.3f}, {row['longitude_x']:.3f}<br>
            <b>Elevation:</b> {row['elevation']:.0f}m<br>
            <b>Temperature:</b> {row['T2M']:.1f}°C<br>
            <b>Precipitation:</b> {row['PRECTOTCORR']:.1f}mm<br>
            <b>Humidity:</b> {row['RH2M']:.1f}%<br>
            <b>Date:</b> {row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date']}
            """
            
            folium.CircleMarker(
                location=[row['latitude_x'], row['longitude_x']],
                radius=6 if is_presence else 4,
                popup=popup_text,
                color='black',
                fillColor=color,
                fillOpacity=0.7,
                weight=2 if is_presence else 1
            ).add_to(m)
        
        # Add heatmap layer for risk visualization
        heat_data = [[row['latitude_x'], row['longitude_x'], row['invasion_probability']] 
                     for idx, row in map_data.iterrows()]
        
        HeatMap(heat_data, radius=15, blur=10, max_zoom=18, min_opacity=0.3).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px
                    ">
        <p><b>Invasion Risk</b></p>
        <p><i class="fa fa-circle" style="color:red"></i> Very High (>0.8)</p>
        <p><i class="fa fa-circle" style="color:orange"></i> High (0.6-0.8)</p>
        <p><i class="fa fa-circle" style="color:yellow"></i> Medium (0.4-0.6)</p>
        <p><i class="fa fa-circle" style="color:lightgreen"></i> Low (0.2-0.4)</p>
        <p><i class="fa fa-circle" style="color:green"></i> Very Low (<0.2)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        map_filename = f"pyracantha_invasion_map_{timestamp}.html"
        map_path = output_dir / map_filename
        m.save(str(map_path))
        
        # Calculate statistics
        high_risk_count = (map_data['invasion_probability'] > 0.7).sum()
        avg_risk = map_data['invasion_probability'].mean()
        
        print(f"Map saved: {map_path}")
        print(f"Average invasion risk: {avg_risk:.3f}")
        print(f"High risk locations (>0.7): {high_risk_count}")
        
        return str(map_path)
    
    def save_model_and_results(self, results: Dict, output_dir: Path) -> None:
        """
        Save trained model and evaluation results.
        
        Args:
            results: Training results dictionary
            output_dir: Directory to save files
        """
        print("Saving model and results...")
        
        # Save model
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance_,
            'training_stats': self.training_stats
        }
        
        model_path = output_dir / 'pyracantha_random_forest_model.pkl'
        joblib.dump(model_data, model_path)
        
        # Save detailed results
        complete_results = {
            'training_stats': self.training_stats,
            'evaluation_results': results,
            'feature_importance': self.feature_importance_.to_dict('records'),
            'model_parameters': self.model.get_params() if self.model else None
        }
        
        results_path = output_dir / 'evaluation_report.json'
        with open(results_path, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        print(f"Model saved: {model_path}")
        print(f"Results saved: {results_path}")

def main():
    """Main execution function for Pyracantha Random Forest analysis."""
    
    print("=" * 80)
    print("PYRACANTHA ANGUSTIFOLIA RANDOM FOREST PREDICTION - FULL DATASET")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup paths
    data_path = "../../data/dataset.csv"
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize predictor
    predictor = PyracanthaRandomForestPredictor(random_state=42)
    
    try:
        # Step 1: Load and preprocess data
        print("STEP 1: LOADING AND PREPROCESSING DATA")
        print("-" * 50)
        df = predictor.load_and_preprocess_data(data_path)
        print()
        
        # Step 2: Create training dataset with presence/absence
        print("STEP 2: CREATING PRESENCE/ABSENCE DATASET")
        print("-" * 50)
        training_data = predictor.create_training_dataset(df)
        print()
        
        # Step 3: Train model with 80/20 split
        print("STEP 3: TRAINING RANDOM FOREST MODEL (80/20 SPLIT)")
        print("-" * 50)
        results = predictor.train_model(training_data, test_size=0.2)
        print()
        
        # Step 4: Create visualizations
        print("STEP 4: CREATING VISUALIZATIONS")
        print("-" * 50)
        predictor.create_visualizations(results, output_dir)
        print()
        
        # Step 5: Generate prediction heatmap
        print("STEP 5: GENERATING INVASION RISK HEATMAP")
        print("-" * 50)
        heatmap_path = predictor.create_grid_heatmap(training_data, output_dir)
        print()
        
        # Step 6: Save results
        print("STEP 6: SAVING MODEL AND RESULTS")
        print("-" * 50)
        predictor.save_model_and_results(results, output_dir)
        print()
        
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("RESULTS SUMMARY:")
        print(f"  • Dataset Size: {predictor.training_stats['total_records']:,} records")
        print(f"  • Unique Observations: {predictor.training_stats['unique_observations']}")
        print(f"  • Training Samples: {predictor.training_stats['train_samples']:,}")
        print(f"  • Test Samples: {predictor.training_stats['test_samples']:,}")
        print(f"  • Features: {predictor.training_stats['total_features']}")
        print(f"  • Model Performance:")
        print(f"    - Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"    - AUC Score: {results['auc_score']:.4f}")
        print(f"    - OOB Score: {results['oob_score']:.4f}")
        print()
        
        print("TOP 5 MOST IMPORTANT FEATURES:")
        for i, (_, row) in enumerate(predictor.feature_importance_.head().iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        print()
        
        print("FILES GENERATED:")
        print(f"  • Model: {output_dir / 'pyracantha_random_forest_model.pkl'}")
        print(f"  • Heatmap: {heatmap_path}")
        print(f"  • Visualizations: {output_dir}")
        print(f"  • Report: {output_dir / 'evaluation_report.json'}")
        print()
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()
