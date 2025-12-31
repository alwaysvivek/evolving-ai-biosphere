"""
analytics.py
ML-powered analytics module for the AI Ecosphere simulation.
Uses Pandas for data logging and Scikit-learn for behavior analysis.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from collections import defaultdict
import json
import os
from datetime import datetime


class EcosystemAnalytics:
    """
    Comprehensive analytics system for tracking and analyzing ecosystem dynamics.
    Uses ML techniques to find patterns and predict trends.
    """
    
    def __init__(self, output_dir="analytics_output"):
        """Initialize the analytics system."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # DataFrames for time-series data
        self.generation_data = pd.DataFrame()
        self.behavior_data = []
        self.extinction_events = []
        
        # Tracking metrics
        self.metrics_history = {
            'generation': [],
            'total_population': [],
            'plant_count': [],
            'herbivore_count': [],
            'predator_count': [],
            'diversity_score': [],
            'temperature': [],
            'light': [],
            'births': [],
            'deaths': [],
            'hive_experiences': []
        }
        
    def log_generation(self, generation, cells, stats, temperature, light, 
                      diversity_score, hive_exp_count):
        """
        Log data for a single generation.
        
        Args:
            generation: Current generation number
            cells: Dictionary of cells in the ecosystem
            stats: Statistics dictionary
            temperature: Current temperature
            light: Current light level
            diversity_score: Shannon diversity score
            hive_exp_count: Number of hive experiences
        """
        # Count species
        species_count = defaultdict(int)
        for cell in cells.values():
            species_count[cell.type] += 1
        
        # Add to metrics history
        self.metrics_history['generation'].append(generation)
        self.metrics_history['total_population'].append(len(cells))
        self.metrics_history['plant_count'].append(species_count[0])
        self.metrics_history['herbivore_count'].append(species_count[1])
        self.metrics_history['predator_count'].append(species_count[2])
        self.metrics_history['diversity_score'].append(diversity_score)
        self.metrics_history['temperature'].append(temperature)
        self.metrics_history['light'].append(light)
        self.metrics_history['births'].append(stats.get('total_births', 0))
        self.metrics_history['deaths'].append(stats.get('total_deaths', 0))
        self.metrics_history['hive_experiences'].append(hive_exp_count)
        
    def log_behavior(self, cell_type, action, energy, age, reward=0.0):
        """
        Log individual organism behavior for clustering analysis.
        
        Args:
            cell_type: Type of cell (0=plant, 1=herbivore, 2=predator)
            action: Action taken (0-3)
            energy: Energy level
            age: Age of organism
            reward: Reward received (for learning organisms)
        """
        self.behavior_data.append({
            'cell_type': cell_type,
            'action': action,
            'energy': energy,
            'age': age,
            'reward': reward
        })
        
    def log_extinction(self, generation, species_name, count):
        """Log an extinction event."""
        self.extinction_events.append({
            'generation': generation,
            'species': species_name,
            'count': count
        })
        
    def compute_statistics(self):
        """
        Compute comprehensive statistical analysis of the ecosystem.
        Returns a dictionary of statistics.
        """
        if not self.metrics_history['generation']:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        
        stats = {
            'summary': {
                'total_generations': len(df),
                'max_population': df['total_population'].max(),
                'min_population': df['total_population'].min(),
                'avg_population': df['total_population'].mean(),
                'std_population': df['total_population'].std(),
                'avg_diversity': df['diversity_score'].mean(),
                'total_births': df['births'].iloc[-1] if len(df) > 0 else 0,
                'total_deaths': df['deaths'].iloc[-1] if len(df) > 0 else 0,
            },
            'species_stats': {
                'plants': {
                    'max': df['plant_count'].max(),
                    'avg': df['plant_count'].mean(),
                    'current': df['plant_count'].iloc[-1] if len(df) > 0 else 0,
                },
                'herbivores': {
                    'max': df['herbivore_count'].max(),
                    'avg': df['herbivore_count'].mean(),
                    'current': df['herbivore_count'].iloc[-1] if len(df) > 0 else 0,
                },
                'predators': {
                    'max': df['predator_count'].max(),
                    'avg': df['predator_count'].mean(),
                    'current': df['predator_count'].iloc[-1] if len(df) > 0 else 0,
                }
            },
            'environmental_stats': {
                'avg_temperature': df['temperature'].mean(),
                'avg_light': df['light'].mean(),
                'temp_variance': df['temperature'].var(),
                'light_variance': df['light'].var(),
            }
        }
        
        # Calculate correlation between species populations
        if len(df) > 10:
            stats['correlations'] = {
                'plant_herbivore': df[['plant_count', 'herbivore_count']].corr().iloc[0, 1],
                'herbivore_predator': df[['herbivore_count', 'predator_count']].corr().iloc[0, 1],
                'plant_predator': df[['plant_count', 'predator_count']].corr().iloc[0, 1],
            }
        
        return stats
    
    def analyze_population_trends(self, window_size=20):
        """
        Analyze population trends using moving averages and trend detection.
        
        Args:
            window_size: Window size for moving average
            
        Returns:
            Dictionary with trend analysis
        """
        if len(self.metrics_history['generation']) < window_size:
            return {'status': 'insufficient_data'}
        
        df = pd.DataFrame(self.metrics_history)
        
        # Calculate moving averages
        df['plant_ma'] = df['plant_count'].rolling(window=window_size).mean()
        df['herbivore_ma'] = df['herbivore_count'].rolling(window=window_size).mean()
        df['predator_ma'] = df['predator_count'].rolling(window=window_size).mean()
        df['total_ma'] = df['total_population'].rolling(window=window_size).mean()
        
        # Detect trends (increasing, decreasing, stable)
        recent_data = df.tail(window_size)
        
        def detect_trend(series):
            if len(series) < 2:
                return 'stable'
            recent_slope = (series.iloc[-1] - series.iloc[0]) / len(series)
            if recent_slope > 1.0:
                return 'increasing'
            elif recent_slope < -1.0:
                return 'decreasing'
            else:
                return 'stable'
        
        trends = {
            'plants': detect_trend(recent_data['plant_count']),
            'herbivores': detect_trend(recent_data['herbivore_count']),
            'predators': detect_trend(recent_data['predator_count']),
            'total': detect_trend(recent_data['total_population']),
            'diversity': detect_trend(recent_data['diversity_score']),
        }
        
        # Detect boom-bust cycles
        def detect_cycles(series, threshold=0.3):
            """Detect oscillatory behavior in population."""
            if len(series) < 10:
                return False
            normalized = (series - series.mean()) / (series.std() + 1e-9)
            # Simple cycle detection: check for sign changes
            sign_changes = np.sum(np.diff(np.sign(normalized)) != 0)
            return sign_changes > len(series) * threshold
        
        cycles = {
            'plants': detect_cycles(df['plant_count'].tail(50)),
            'herbivores': detect_cycles(df['herbivore_count'].tail(50)),
            'predators': detect_cycles(df['predator_count'].tail(50)),
        }
        
        return {
            'trends': trends,
            'cycles_detected': cycles,
            'moving_averages': {
                'plants': recent_data['plant_ma'].iloc[-1] if len(recent_data) > 0 else 0,
                'herbivores': recent_data['herbivore_ma'].iloc[-1] if len(recent_data) > 0 else 0,
                'predators': recent_data['predator_ma'].iloc[-1] if len(recent_data) > 0 else 0,
            }
        }
    
    def cluster_behaviors(self, n_clusters=3):
        """
        Use K-means clustering to identify distinct behavior patterns.
        
        Args:
            n_clusters: Number of behavior clusters to identify
            
        Returns:
            Dictionary with clustering results and metrics
        """
        if len(self.behavior_data) < n_clusters * 2:
            return {'status': 'insufficient_data', 'samples': len(self.behavior_data)}
        
        # Convert behavior data to DataFrame
        df = pd.DataFrame(self.behavior_data)
        
        # Prepare features for clustering
        features = df[['cell_type', 'action', 'energy', 'age', 'reward']].values
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(features_scaled)
        
        # Calculate silhouette score (measure of cluster quality)
        silhouette = silhouette_score(features_scaled, df['cluster'])
        
        # Analyze each cluster
        cluster_profiles = []
        for i in range(n_clusters):
            cluster_data = df[df['cluster'] == i]
            profile = {
                'cluster_id': i,
                'size': len(cluster_data),
                'dominant_type': cluster_data['cell_type'].mode()[0] if len(cluster_data) > 0 else -1,
                'common_action': cluster_data['action'].mode()[0] if len(cluster_data) > 0 else -1,
                'avg_energy': cluster_data['energy'].mean(),
                'avg_age': cluster_data['age'].mean(),
                'avg_reward': cluster_data['reward'].mean(),
            }
            cluster_profiles.append(profile)
        
        return {
            'status': 'success',
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'cluster_profiles': cluster_profiles,
            'total_behaviors': len(df)
        }
    
    def predict_population_crash(self, lookback=30, lookahead=10):
        """
        Use linear regression to predict if a population crash is likely.
        
        Args:
            lookback: Number of generations to use for training
            lookahead: Number of generations to predict ahead
            
        Returns:
            Dictionary with predictions and model metrics
        """
        if len(self.metrics_history['generation']) < lookback + lookahead:
            return {'status': 'insufficient_data'}
        
        df = pd.DataFrame(self.metrics_history)
        
        # Use recent data for prediction
        recent_df = df.tail(lookback + lookahead)
        
        predictions = {}
        
        for species in ['plant_count', 'herbivore_count', 'predator_count']:
            # Prepare training data (use past to predict future)
            X = recent_df['generation'].values[:-lookahead].reshape(-1, 1)
            y = recent_df[species].values[:-lookahead]
            
            # Fit linear regression model
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict future
            X_future = recent_df['generation'].values[-lookahead:].reshape(-1, 1)
            y_pred = model.predict(X_future)
            
            # Check if crash is predicted (population drops below threshold)
            current_pop = recent_df[species].iloc[-lookahead]
            predicted_pop = y_pred[-1]
            crash_threshold = 0.3  # 70% decrease
            
            crash_predicted = predicted_pop < (current_pop * crash_threshold)
            
            # Calculate model metrics on training data
            y_train_pred = model.predict(X)
            r2 = r2_score(y, y_train_pred)
            
            predictions[species] = {
                'current': float(current_pop),
                'predicted': float(predicted_pop),
                'crash_predicted': crash_predicted,
                'trend_slope': float(model.coef_[0]),
                'r2_score': float(r2),
            }
        
        return {
            'status': 'success',
            'predictions': predictions,
            'lookback': lookback,
            'lookahead': lookahead
        }
    
    def export_to_csv(self, filename=None):
        """
        Export all collected data to CSV files.
        
        Args:
            filename: Optional base filename (defaults to timestamp)
        """
        if filename is None:
            filename = f"ecosystem_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Export generation data
        df = pd.DataFrame(self.metrics_history)
        csv_path = os.path.join(self.output_dir, f"{filename}_generations.csv")
        df.to_csv(csv_path, index=False)
        print(f"Exported generation data to {csv_path}")
        
        # Export behavior data if available
        if self.behavior_data:
            behavior_df = pd.DataFrame(self.behavior_data)
            behavior_path = os.path.join(self.output_dir, f"{filename}_behaviors.csv")
            behavior_df.to_csv(behavior_path, index=False)
            print(f"Exported behavior data to {behavior_path}")
        
        # Export extinction events if available
        if self.extinction_events:
            extinction_df = pd.DataFrame(self.extinction_events)
            extinction_path = os.path.join(self.output_dir, f"{filename}_extinctions.csv")
            extinction_df.to_csv(extinction_path, index=False)
            print(f"Exported extinction data to {extinction_path}")
        
        return csv_path
    
    def generate_report(self):
        """
        Generate a comprehensive analytics report.
        Returns formatted string report.
        """
        stats = self.compute_statistics()
        trends = self.analyze_population_trends()
        
        report = []
        report.append("=" * 80)
        report.append("ECOSYSTEM ANALYTICS REPORT")
        report.append("=" * 80)
        
        if stats:
            report.append("\n### SUMMARY STATISTICS ###")
            report.append(f"Total Generations: {stats['summary']['total_generations']}")
            report.append(f"Population - Max: {stats['summary']['max_population']:.0f}, "
                         f"Avg: {stats['summary']['avg_population']:.1f}, "
                         f"Min: {stats['summary']['min_population']:.0f}")
            report.append(f"Average Diversity Score: {stats['summary']['avg_diversity']:.3f}")
            report.append(f"Total Births: {stats['summary']['total_births']}")
            report.append(f"Total Deaths: {stats['summary']['total_deaths']}")
            
            report.append("\n### SPECIES STATISTICS ###")
            for species, data in stats['species_stats'].items():
                report.append(f"{species.capitalize()}: "
                            f"Current={data['current']:.0f}, "
                            f"Max={data['max']:.0f}, "
                            f"Avg={data['avg']:.1f}")
            
            if 'correlations' in stats:
                report.append("\n### POPULATION CORRELATIONS ###")
                report.append(f"Plant-Herbivore: {stats['correlations']['plant_herbivore']:.3f}")
                report.append(f"Herbivore-Predator: {stats['correlations']['herbivore_predator']:.3f}")
                report.append(f"Plant-Predator: {stats['correlations']['plant_predator']:.3f}")
        
        if trends.get('status') != 'insufficient_data':
            report.append("\n### POPULATION TRENDS ###")
            for species, trend in trends['trends'].items():
                report.append(f"{species.capitalize()}: {trend}")
            
            report.append("\n### BOOM-BUST CYCLES DETECTED ###")
            for species, has_cycle in trends['cycles_detected'].items():
                report.append(f"{species.capitalize()}: {'YES' if has_cycle else 'NO'}")
        
        # Clustering analysis
        clustering = self.cluster_behaviors(n_clusters=3)
        if clustering.get('status') == 'success':
            report.append("\n### BEHAVIOR CLUSTERING ###")
            report.append(f"Total Behaviors Analyzed: {clustering['total_behaviors']}")
            report.append(f"Silhouette Score: {clustering['silhouette_score']:.3f}")
            report.append("Cluster Profiles:")
            for profile in clustering['cluster_profiles']:
                type_names = {0: "Plant", 1: "Herbivore", 2: "Predator"}
                action_names = {0: "Reproduce", 1: "Move", 2: "Rest", 3: "Hunt/Eat"}
                report.append(f"  Cluster {profile['cluster_id']}: "
                            f"Size={profile['size']}, "
                            f"Type={type_names.get(profile['dominant_type'], 'Unknown')}, "
                            f"Action={action_names.get(profile['common_action'], 'Unknown')}, "
                            f"AvgEnergy={profile['avg_energy']:.1f}")
        
        # Crash prediction
        crash_pred = self.predict_population_crash()
        if crash_pred.get('status') == 'success':
            report.append("\n### POPULATION CRASH PREDICTIONS ###")
            for species, pred in crash_pred['predictions'].items():
                species_name = species.replace('_count', '').capitalize()
                status = "⚠️ CRASH LIKELY" if pred['crash_predicted'] else "✓ Stable"
                report.append(f"{species_name}: {status}")
                report.append(f"  Current: {pred['current']:.0f}, "
                            f"Predicted: {pred['predicted']:.1f}, "
                            f"Trend: {pred['trend_slope']:.2f}/gen")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
