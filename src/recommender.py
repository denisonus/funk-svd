import numpy as np
import pandas as pd

from src.config import GAMES_DATA_PATH
from src.funk_svd import FunkSVD


class GameRecommender:
    def __init__(self, model_path=None, train_data=None, games_data_path=None):
        self.model = FunkSVD()
        self.train_data = train_data
        self.games_data = None

        # Load games data if path provided
        if games_data_path:
            self.load_games_data(games_data_path)

        if model_path:
            self.model.load(model_path)

    def get_predictions(self, user_id, item_ids=None):
        """Get predictions for a user for specific items"""
        return self.model.predict_for_user(user_id, item_ids)

    def load_games_data(self, games_data_path=None):
        """Load game information from CSV file"""
        path = games_data_path or GAMES_DATA_PATH
        games_df = pd.read_csv(path)
        # Create dictionary for fast lookup by BggId
        self.games_data = {int(row['BGGId']): row.to_dict() for _, row in games_df.iterrows()}
        return self

    def get_recommendations(self, user_id, n=10, attributes=None):
        """Get top N recommendations for a user with specified game attributes."""
        if self.train_data is None or len(self.train_data) == 0:
            raise ValueError("Train data is required for filtering recommendations")

        # Get all predictions
        predictions = self.model.predict_for_user(user_id)

        # Filter out already rated items
        rated_items = set(item['BggId'] for item in self.train_data if item['UserId'] == user_id)
        predictions = {item_id: rating for item_id, rating in predictions.items()
                      if item_id not in rated_items}

        # Sort and get top N
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]
        
        # Build recommendation list
        recommendations = []
        for item_id, rating in sorted_predictions:
            item_id = int(item_id)
            rec = {
                'BggId': item_id,
                'PredictedRating': rating
            }
            
            # Add requested game attributes if available
            if attributes and self.games_data:
                if item_id in self.games_data:
                    game_info = self.games_data[item_id]
                    for attr in attributes:
                        if attr in game_info:
                            rec[attr] = game_info[attr]
            
            recommendations.append(rec)
            
        return recommendations

    def train(self, train_data, test_data=None, **kwargs):
        """Train the model with optional parameters"""
        self.train_data = train_data  # Store for later use
        self.model = FunkSVD(**kwargs)
        self.model.fit(train_data, test_data)
        return self

    def save(self, model_path):
        """Save the trained recommender"""
        if self.model:
            # Update save path in model if different
            old_path = self.model.save_path
            self.model.save_path = model_path
            self.model.save(self.model.n_factors - 1, True)  # Save as final model
            self.model.save_path = old_path  # Restore original path
        return self

    @staticmethod
    def get_popular_recommendations(train_data, n=10):
        """
        Get top N popular recommendations based on average ratings and number of ratings.
        """
        # Group by item and calculate average rating and count
        item_stats = {}
        for item in train_data:
            item_id = item['BggId']
            rating = item['Rating']

            if item_id not in item_stats:
                item_stats[item_id] = {'sum': 0, 'count': 0}

            item_stats[item_id]['sum'] += rating
            item_stats[item_id]['count'] += 1

        # Calculate popularity score (average rating weighted by log of count)
        item_scores = {}
        for item_id, stats in item_stats.items():
            avg_rating = stats['sum'] / stats['count']
            # Use log to prevent extremely popular items from dominating
            popularity = avg_rating * np.log1p(stats['count'])
            item_scores[item_id] = popularity

        # Sort by score and return top N
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]
