"""
analytics.py
ML-powered analytics module for the AI Ecosphere simulation.
Uses Pandas for data logging and Scikit-learn for behavior analysis.
Includes vector database for behavior embeddings and RAGAS-style evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, silhouette_score
from collections import defaultdict
import os
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Vector database features disabled.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using simple embeddings.")


class BehaviorVectorDB:
    """
    Vector database for behavior embedding similarity search.
    Stores organism behaviors as embeddings and enables semantic similarity search.
    """
    
    def __init__(self, embedding_dim=128):
        """Initialize the vector database."""
        self.embedding_dim = embedding_dim
        self.behaviors = []
        self.metadata = []
        
        if FAISS_AVAILABLE:
            # Initialize FAISS index for fast similarity search
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            self.index = None
            
        # Simple embedding fallback (when sentence-transformers not available)
        self.scaler = StandardScaler()
        
    def _create_embedding(self, behavior_dict):
        """
        Create an embedding vector from a behavior dictionary.
        
        Args:
            behavior_dict: Dictionary with keys: cell_type, action, energy, age, reward
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        # Create a feature vector from behavior attributes
        features = np.array([
            behavior_dict.get('cell_type', 0),
            behavior_dict.get('action', 0),
            behavior_dict.get('energy', 0),
            behavior_dict.get('age', 0),
            behavior_dict.get('reward', 0),
        ], dtype=np.float32)
        
        # Normalize features
        features = features / (np.linalg.norm(features) + 1e-9)
        
        # Expand to embedding_dim using random projection (simple approach)
        # In production, use learned embeddings or sentence-transformers
        np.random.seed(int(features.sum() * 1000) % 2**32)  # Deterministic per unique input
        projection_matrix = np.random.randn(5, self.embedding_dim).astype(np.float32)
        embedding = features @ projection_matrix
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        
        return embedding
        
    def add_behavior(self, behavior_dict):
        """
        Add a behavior to the vector database.
        
        Args:
            behavior_dict: Dictionary containing behavior information
        """
        embedding = self._create_embedding(behavior_dict)
        self.behaviors.append(embedding)
        self.metadata.append(behavior_dict.copy())
        
        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(embedding.reshape(1, -1))
            
    def search_similar(self, query_behavior, k=5):
        """
        Search for k most similar behaviors.
        
        Args:
            query_behavior: Dictionary or embedding to search for
            k: Number of similar behaviors to return
            
        Returns:
            List of (distance, metadata) tuples for k nearest neighbors
        """
        if isinstance(query_behavior, dict):
            query_embedding = self._create_embedding(query_behavior)
        else:
            query_embedding = query_behavior
            
        query_embedding = query_embedding.reshape(1, -1)
        
        if FAISS_AVAILABLE and self.index is not None and len(self.behaviors) > 0:
            # Use FAISS for fast search
            distances, indices = self.index.search(query_embedding, min(k, len(self.behaviors)))
            results = [(distances[0][i], self.metadata[indices[0][i]]) 
                      for i in range(len(indices[0]))]
        else:
            # Fallback: manual distance computation
            if len(self.behaviors) == 0:
                return []
                
            behaviors_array = np.array(self.behaviors)
            distances = np.linalg.norm(behaviors_array - query_embedding, axis=1)
            top_k_indices = np.argsort(distances)[:k]
            results = [(distances[i], self.metadata[i]) for i in top_k_indices]
            
        return results
        
    def get_statistics(self):
        """Get statistics about the vector database."""
        return {
            'total_behaviors': len(self.behaviors),
            'embedding_dim': self.embedding_dim,
            'faiss_enabled': FAISS_AVAILABLE and self.index is not None,
        }


class RAGASEvaluator:
    """
    RAGAS-style automated evaluation framework for ecosystem quality.
    Evaluates ecosystem health, diversity, behavioral quality, and stability.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.evaluation_history = []
        
    def evaluate_ecosystem(self, generation_data, behavior_data, extinction_events):
        """
        Perform comprehensive ecosystem evaluation.
        
        Args:
            generation_data: DataFrame or dict of generation metrics
            behavior_data: List of behavior dictionaries
            extinction_events: List of extinction event dictionaries
            
        Returns:
            Dictionary with evaluation scores and metrics
        """
        scores = {}
        
        # Convert to DataFrame if dict
        if isinstance(generation_data, dict):
            gen_df = pd.DataFrame(generation_data)
        else:
            gen_df = generation_data
            
        if len(gen_df) == 0:
            return {'status': 'insufficient_data'}
        
        # 1. Ecosystem Health Score (0-100)
        scores['health'] = self._evaluate_health(gen_df)
        
        # 2. Diversity Score (0-100)
        scores['diversity'] = self._evaluate_diversity(gen_df)
        
        # 3. Stability Score (0-100)
        scores['stability'] = self._evaluate_stability(gen_df)
        
        # 4. Behavioral Quality Score (0-100)
        scores['behavioral_quality'] = self._evaluate_behavioral_quality(behavior_data)
        
        # 5. Resilience Score (0-100)
        scores['resilience'] = self._evaluate_resilience(gen_df, extinction_events)
        
        # Overall Score (weighted average)
        scores['overall'] = (
            scores['health'] * 0.25 +
            scores['diversity'] * 0.20 +
            scores['stability'] * 0.25 +
            scores['behavioral_quality'] * 0.15 +
            scores['resilience'] * 0.15
        )
        
        # Add timestamp and store
        scores['timestamp'] = datetime.now()
        scores['generation'] = gen_df['generation'].iloc[-1] if len(gen_df) > 0 else 0
        self.evaluation_history.append(scores.copy())
        
        return scores
        
    def _evaluate_health(self, gen_df):
        """Evaluate ecosystem health based on population metrics."""
        if len(gen_df) == 0:
            return 0.0
            
        recent = gen_df.tail(20)
        
        # Check if all species are present
        has_all_species = (
            recent['plant_count'].mean() > 0 and
            recent['herbivore_count'].mean() > 0 and
            recent['predator_count'].mean() > 0
        )
        
        # Population balance (ideal is roughly equal)
        total_pop = recent['total_population'].mean()
        if total_pop > 0:
            plant_ratio = recent['plant_count'].mean() / total_pop
            herb_ratio = recent['herbivore_count'].mean() / total_pop
            pred_ratio = recent['predator_count'].mean() / total_pop
            
            # Penalize extreme imbalances
            balance_score = 1.0 - np.std([plant_ratio, herb_ratio, pred_ratio])
        else:
            balance_score = 0.0
            
        # Energy flow (births should exceed deaths or be close)
        birth_rate = recent['births'].diff().mean()
        death_rate = recent['deaths'].diff().mean()
        vitality = min(1.0, max(0.0, birth_rate / (death_rate + 1)))
        
        health = (
            (50 if has_all_species else 10) +
            balance_score * 30 +
            vitality * 20
        )
        
        return min(100, max(0, health))
        
    def _evaluate_diversity(self, gen_df):
        """Evaluate ecosystem diversity."""
        if len(gen_df) == 0:
            return 0.0
            
        recent = gen_df.tail(20)
        avg_diversity = recent['diversity_score'].mean()
        
        # Shannon diversity is already 0-1, scale to 0-100
        return avg_diversity * 100
        
    def _evaluate_stability(self, gen_df):
        """Evaluate ecosystem stability (low variance = high stability)."""
        if len(gen_df) < 10:
            return 50.0  # Neutral score for insufficient data
            
        recent = gen_df.tail(30)
        
        # Calculate coefficient of variation (CV) for populations
        cvs = []
        for col in ['plant_count', 'herbivore_count', 'predator_count']:
            mean_val = recent[col].mean()
            if mean_val > 0:
                cv = recent[col].std() / mean_val
                cvs.append(cv)
                
        if len(cvs) == 0:
            return 0.0
            
        avg_cv = np.mean(cvs)
        
        # Lower CV = higher stability
        # CV > 1.0 is very unstable, CV < 0.3 is very stable
        stability = max(0, 100 - (avg_cv * 50))
        
        return min(100, max(0, stability))
        
    def _evaluate_behavioral_quality(self, behavior_data):
        """Evaluate quality of organism behaviors."""
        if len(behavior_data) < 10:
            return 50.0  # Neutral score
            
        behavior_df = pd.DataFrame(behavior_data)
        
        # Metrics:
        # 1. Reward efficiency (higher rewards = better behaviors)
        avg_reward = behavior_df['reward'].mean()
        reward_score = min(50, max(0, avg_reward * 10 + 25))
        
        # 2. Energy management (organisms maintaining good energy levels)
        avg_energy = behavior_df['energy'].mean()
        energy_score = min(30, max(0, avg_energy / 4))
        
        # 3. Longevity (older organisms indicate successful behavior)
        avg_age = behavior_df['age'].mean()
        longevity_score = min(20, max(0, avg_age / 10))
        
        quality = reward_score + energy_score + longevity_score
        return min(100, max(0, quality))
        
    def _evaluate_resilience(self, gen_df, extinction_events):
        """Evaluate ecosystem resilience to disturbances."""
        if len(gen_df) < 20:
            return 50.0
            
        # Count extinction events
        num_extinctions = len(extinction_events)
        
        # Penalize extinctions
        extinction_penalty = min(40, num_extinctions * 10)
        
        # Check recovery (population growth after dips)
        recent = gen_df.tail(50)
        min_pop = recent['total_population'].min()
        max_pop = recent['total_population'].max()
        current_pop = recent['total_population'].iloc[-1]
        
        if max_pop > min_pop:
            recovery_ratio = (current_pop - min_pop) / (max_pop - min_pop)
        else:
            recovery_ratio = 1.0
            
        recovery_score = recovery_ratio * 40
        
        resilience = 60 - extinction_penalty + recovery_score
        return min(100, max(0, resilience))
        
    def get_evaluation_report(self):
        """Generate a formatted evaluation report."""
        if not self.evaluation_history:
            return "No evaluations performed yet."
            
        latest = self.evaluation_history[-1]
        
        report = []
        report.append("\n" + "=" * 80)
        report.append("RAGAS-STYLE AUTOMATED EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Generation: {latest['generation']}")
        report.append(f"Timestamp: {latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("### ECOSYSTEM QUALITY SCORES ###")
        report.append(f"Overall Score:        {latest['overall']:.1f}/100 {self._get_grade(latest['overall'])}")
        report.append(f"  Health:             {latest['health']:.1f}/100 {self._get_grade(latest['health'])}")
        report.append(f"  Diversity:          {latest['diversity']:.1f}/100 {self._get_grade(latest['diversity'])}")
        report.append(f"  Stability:          {latest['stability']:.1f}/100 {self._get_grade(latest['stability'])}")
        report.append(f"  Behavioral Quality: {latest['behavioral_quality']:.1f}/100 {self._get_grade(latest['behavioral_quality'])}")
        report.append(f"  Resilience:         {latest['resilience']:.1f}/100 {self._get_grade(latest['resilience'])}")
        report.append("")
        
        # Recommendations
        report.append("### RECOMMENDATIONS ###")
        if latest['health'] < 50:
            report.append("⚠️  Low health detected. Consider species reintroduction or environmental adjustment.")
        if latest['diversity'] < 40:
            report.append("⚠️  Low diversity. Ecosystem at risk of collapse.")
        if latest['stability'] < 40:
            report.append("⚠️  High instability. Expect boom-bust cycles.")
        if latest['behavioral_quality'] < 50:
            report.append("⚠️  Poor behavioral quality. Increase training or mutation rates.")
        if latest['resilience'] < 40:
            report.append("⚠️  Low resilience. System vulnerable to extinction events.")
            
        if latest['overall'] >= 75:
            report.append("✅ Ecosystem performing excellently!")
        elif latest['overall'] >= 60:
            report.append("✅ Ecosystem in good condition.")
        elif latest['overall'] >= 40:
            report.append("⚠️  Ecosystem needs attention.")
        else:
            report.append("❌ Critical ecosystem condition!")
            
        report.append("=" * 80)
        
        return "\n".join(report)
        
    def _get_grade(self, score):
        """Convert score to letter grade."""
        if score >= 90:
            return "(A)"
        elif score >= 80:
            return "(B)"
        elif score >= 70:
            return "(C)"
        elif score >= 60:
            return "(D)"
        else:
            return "(F)"


class EcosystemAnalytics:
    """
    Comprehensive analytics system for tracking and analyzing ecosystem dynamics.
    Uses ML techniques to find patterns and predict trends.
    Includes vector database for behavior similarity search and automated evaluation.
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
        
        # NEW: Vector database for behavior embeddings
        self.vector_db = BehaviorVectorDB(embedding_dim=128)
        
        # NEW: RAGAS-style evaluator
        self.evaluator = RAGASEvaluator()
        
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
        behavior_dict = {
            'cell_type': cell_type,
            'action': action,
            'energy': energy,
            'age': age,
            'reward': reward
        }
        self.behavior_data.append(behavior_dict)
        
        # NEW: Also add to vector database for similarity search
        self.vector_db.add_behavior(behavior_dict)
        
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
        
        # NEW: Vector database statistics
        vdb_stats = self.vector_db.get_statistics()
        report.append("\n### VECTOR DATABASE STATISTICS ###")
        report.append(f"Total Behaviors Stored: {vdb_stats['total_behaviors']}")
        report.append(f"Embedding Dimension: {vdb_stats['embedding_dim']}")
        report.append(f"FAISS Enabled: {'Yes' if vdb_stats['faiss_enabled'] else 'No'}")
        
        # NEW: Behavior similarity example (find similar to most recent)
        if len(self.behavior_data) > 0:
            query = self.behavior_data[-1]
            similar = self.vector_db.search_similar(query, k=3)
            if similar:
                report.append("\n### BEHAVIOR SIMILARITY SEARCH (Recent Behavior) ###")
                type_names = {0: "Plant", 1: "Herbivore", 2: "Predator"}
                action_names = {0: "Reproduce", 1: "Move", 2: "Rest", 3: "Hunt/Eat"}
                report.append(f"Query: {type_names.get(query['cell_type'], 'Unknown')} - "
                            f"{action_names.get(query['action'], 'Unknown')} (Energy: {query['energy']:.1f})")
                for i, (dist, meta) in enumerate(similar[:3]):
                    report.append(f"  Similar {i+1} (dist={dist:.3f}): "
                                f"{type_names.get(meta['cell_type'], 'Unknown')} - "
                                f"{action_names.get(meta['action'], 'Unknown')} "
                                f"(Energy: {meta['energy']:.1f}, Age: {meta['age']}, Reward: {meta['reward']:.2f})")
        
        # NEW: RAGAS-style automated evaluation
        evaluation = self.evaluator.evaluate_ecosystem(
            self.metrics_history,
            self.behavior_data,
            self.extinction_events
        )
        if evaluation.get('status') != 'insufficient_data':
            report.append("\n### AUTOMATED ECOSYSTEM EVALUATION ###")
            report.append(f"Overall Score: {evaluation['overall']:.1f}/100 {self.evaluator._get_grade(evaluation['overall'])}")
            report.append(f"  Health:             {evaluation['health']:.1f}/100")
            report.append(f"  Diversity:          {evaluation['diversity']:.1f}/100")
            report.append(f"  Stability:          {evaluation['stability']:.1f}/100")
            report.append(f"  Behavioral Quality: {evaluation['behavioral_quality']:.1f}/100")
            report.append(f"  Resilience:         {evaluation['resilience']:.1f}/100")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
