from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd

from src.funk_svd import FunkSVD


class GameRecommender:
    def __init__(self, train_data: pd.DataFrame, games_data_path: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the game recommender system.
        
        Args:
            train_data: DataFrame with user ratings
            games_data_path: Path to games data CSV file. Required for attribute-based recommendations.
        """
        self.model: FunkSVD = FunkSVD()
        # Convert train_data to DataFrame if needed
        self.train_data: pd.DataFrame = train_data
        self.games_data: Optional[Dict[int, Dict[str, Any]]] = None
        if games_data_path:
            self.load_games_data(games_data_path)

    def get_predictions(self, user_id: int, item_ids: Optional[List[int]] = None) -> Dict[int, float]:
        """Get predictions for a user for specific items"""
        return self.model.predict_for_user(user_id, item_ids)

    def load_games_data(self, games_data_path: Union[str, Path]) -> None:
        """Load game information from CSV file"""
        games_df = pd.read_csv(games_data_path)
        # Create dictionary for fast lookup by BGGId
        self.games_data = {int(row['BGGId']): row.to_dict() for _, row in games_df.iterrows()}

    def get_recommendations(self, user_id: int, n: int = 10, attributes: Optional[List[str]] = None) -> List[
        Dict[str, Any]]:
        rated_items = set(self.train_data[self.train_data['UserId'] == user_id]['BGGId'].unique())
        candidate_items = [item_id for item_id in self.model.item_ids if item_id not in rated_items]

        predictions = self.model.predict_for_user(user_id, candidate_items)
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

    def train(self, test_data: Optional[pd.DataFrame] = None, **kwargs) -> None:
        """Train the model with optional parameters"""
        self.model = FunkSVD(**kwargs)
        self.model.fit(self.train_data, test_data)

    def save(self, save_path: str) -> None:
        """Save the trained recommender"""
        self.model.save_model(save_path)

    def load(self, model_path: str) -> None:
        """Load a trained recommender model"""
        self.model.load_model(model_path)

    def add_user_ratings(self, ratings: List[Dict[str, Any]], user_id: Optional[int] = None) -> \
            Tuple[bool, Optional[int], bool]:
        """
        Add or update user ratings with simplified latent factor learning.

        Args:
            ratings: List of rating dictionaries with 'BGGId' and 'Rating' keys
            user_id: Optional user ID. If None, creates a new user.

        Returns:
            tuple: (success, user_id, is_new_user)
        """
        # Convert and validate ratings
        ratings_df = self._prepare_ratings(ratings)
        if ratings_df.empty:
            return False, None, False

        # Let the model handle user creation/retrieval
        user_id, user_idx, is_new_user = self.model.add_or_get_user(user_id)

        # Let the model update user factors
        item_indices = [self.model.item_id_to_idx[game_id] for game_id in ratings_df['BGGId']]
        actual_ratings = ratings_df['Rating'].values
        success = self.model.update_user_factors(user_idx, item_indices, actual_ratings)

        # Update training data (stays in GameRecommender)
        self._update_training_data(user_id, ratings_df)

        return success, user_id, is_new_user

    def _prepare_ratings(self, ratings: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare and validate ratings data."""
        # Convert to DataFrame
        ratings_df = pd.DataFrame(ratings)

        # Filter for valid game IDs that exist in our model
        valid_game_ids = set(self.model.item_id_to_idx.keys())
        return ratings_df[ratings_df['BGGId'].isin(valid_game_ids)]

    def _update_training_data(self, user_id: int, ratings_df: pd.DataFrame) -> None:
        """Update internal training data with new ratings."""
        ratings_to_add = ratings_df.copy()
        ratings_to_add['UserId'] = user_id
        if 'Username' not in ratings_to_add.columns:
            ratings_to_add['Username'] = f'user_{user_id}'

        # Ensure train_data exists
        if self.train_data is None:
            self.train_data = pd.DataFrame(columns=['UserId', 'BGGId', 'Rating', 'Username'])

        # Remove existing ratings for this user-item pair before adding new ones
        if not self.train_data.empty:
            # Create a multi-index for faster filtering
            user_items = set(zip(ratings_to_add['UserId'], ratings_to_add['BGGId']))
            # Filter out existing ratings that will be replaced
            mask = ~pd.Series([
                (row['UserId'], row['BGGId']) in user_items
                for _, row in self.train_data.iterrows()
            ])
            self.train_data = pd.concat([self.train_data[mask], ratings_to_add],
                                        ignore_index=True)
        else:
            self.train_data = ratings_to_add

    @staticmethod
    def get_popular_recommendations(train_data: pd.DataFrame, n: int = 10) -> List[Tuple[int, float]]:
        """Get top N popular recommendations based on average ratings and counts."""
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
