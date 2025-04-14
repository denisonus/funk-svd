import numpy as np
import pandas as pd

from src.config import GAMES_DATA_PATH
from src.funk_svd import FunkSVD
from src.utils.persistence import load_model, save_model


class GameRecommender:
    def __init__(self, model_path=None, train_data=None, games_data_path=None):
        self.model = FunkSVD()
        # Convert train_data to DataFrame if needed
        self.train_data = self._ensure_dataframe(train_data) if train_data is not None else None
        self.games_data = None

        # Load games data if path provided
        if games_data_path:
            self.load_games_data(games_data_path)

        if model_path:
            load_model(self.model, model_path)

    def _ensure_dataframe(self, data):
        """Convert data to pandas DataFrame if it's not already"""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, list) or isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        else:
            raise ValueError("train_data must be a DataFrame, list, or numpy array")

    def get_predictions(self, user_id, item_ids=None):
        """Get predictions for a user for specific items"""
        return self.model.predict_for_user(user_id, item_ids)

    def load_games_data(self, games_data_path=None):
        """Load game information from CSV file"""
        path = games_data_path or GAMES_DATA_PATH
        games_df = pd.read_csv(path)
        # Create dictionary for fast lookup by BGGId
        self.games_data = {int(row['BGGId']): row.to_dict() for _, row in games_df.iterrows()}
        return self

    def get_recommendations(self, user_id, n=10, attributes=None):
        """Get top N recommendations for a user with specified game attributes."""
        if self.train_data is None or len(self.train_data) == 0:
            raise ValueError("Train data is required for filtering recommendations")

        # Get all predictions
        predictions = self.model.predict_for_user(user_id)

        # Filter out already rated items using DataFrame operations
        rated_items = set(self.train_data[self.train_data['UserId'] == user_id]['BGGId'].unique())
        predictions = {item_id: rating for item_id, rating in predictions.items() if item_id not in rated_items}

        # Sort and get top N
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]

        # Build recommendation list
        recommendations = []
        for item_id, rating in sorted_predictions:
            item_id = int(item_id)
            rec = {'BGGId': item_id, 'PredictedRating': rating}

            # Add requested game attributes if available
            if attributes and self.games_data and item_id in self.games_data:
                game_info = self.games_data[item_id]
                for attr in attributes:
                    if attr in game_info:
                        rec[attr] = game_info[attr]

            recommendations.append(rec)

        return recommendations

    def train(self, train_data, test_data=None, **kwargs):
        """Train the model with optional parameters"""
        # Convert to DataFrame if not already
        self.train_data = self._ensure_dataframe(train_data)
        test_data_df = self._ensure_dataframe(test_data) if test_data is not None else None

        # Initialize model with parameters and train directly with DataFrames
        self.model = FunkSVD(**kwargs)
        self.model.fit(self.train_data, test_data_df)
        return self

    def save(self, model_path):
        """Save the trained recommender"""
        if self.model:
            save_model(self.model, model_path, self.model.n_factors - 1, True)
        return self

    def _update_train_data(self, ratings_df, user_id):
        """Helper method to update training data with new ratings"""
        if self.train_data is None:
            self.train_data = pd.DataFrame(columns=['UserId', 'BGGId', 'Rating', 'Username'])

        # Add user_id column if it doesn't exist
        if 'UserId' not in ratings_df.columns:
            ratings_df = ratings_df.copy()
            ratings_df['UserId'] = user_id

        # Add Username column if it doesn't exist
        if 'Username' not in ratings_df.columns:
            ratings_df['Username'] = f'user_{user_id}'

        # Append to DataFrame
        self.train_data = pd.concat([self.train_data, ratings_df], ignore_index=True)

    def add_user_ratings(self, ratings, user_id=None):
        """Add ratings from a user to the model."""
        if not self.model:
            raise ValueError("Model must be trained or loaded before adding ratings")

        # Convert ratings to DataFrame if needed
        ratings_df = pd.DataFrame(ratings) if not isinstance(ratings, pd.DataFrame) else ratings

        # Add ratings to the model
        success, user_id, is_new_user = self.model.add_ratings(ratings_df, user_id)

        # Update internal training data if successful
        if success:
            self._update_train_data(ratings_df, user_id)

        return success, user_id, is_new_user

    @staticmethod
    def get_popular_recommendations(train_data, n=10):
        """Get top N popular recommendations based on average ratings and counts."""
        # Convert to DataFrame if not already
        if not isinstance(train_data, pd.DataFrame):
            train_data = pd.DataFrame(train_data)

        # Group by item and calculate statistics
        item_stats = train_data.groupby('BGGId').agg(
            avg_rating=('Rating', 'mean'),
            count=('Rating', 'count')
        )

        # Calculate popularity score
        item_stats['popularity'] = item_stats['avg_rating'] * np.log1p(item_stats['count'])

        # Sort by score and return top N
        top_items = item_stats.sort_values('popularity', ascending=False).head(n)
        return [(int(idx), float(row['popularity'])) for idx, row in top_items.iterrows()]
