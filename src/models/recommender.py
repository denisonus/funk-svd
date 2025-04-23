from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from loguru import logger

from src.models.funk_svd import FunkSVD
from src.evaluation.metrics import evaluate_recommendations


class GameRecommender:
    def __init__(self, train_data: pd.DataFrame, games_data: Optional[pd.DataFrame] = None) -> None:
        self.model = FunkSVD()
        self.train_data = train_data
        self.games_data = None
        if games_data is not None:
            self.games_data = {int(row['BGGId']): row.to_dict() for _, row in games_data.iterrows()}

    def get_predictions(self, user_id: int, item_ids: Optional[List[int]] = None) -> Dict[int, float]:
        return self.model.predict_for_user(user_id, item_ids)

    def get_recommendations(self, user_id: int, n: int = 10, attributes: Optional[List[str]] = None) -> List[
        Dict[str, Any]]:
        rated_items = set(self.train_data[self.train_data['UserId'] == user_id]['BGGId'].unique())
        candidate_items = [i for i in self.model.item_ids if i not in rated_items]
        predictions = self.model.predict_for_user(user_id, candidate_items)
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]

        results = []
        for item_id, rating in sorted_preds:
            rec = {'BGGId': item_id, 'PredictedRating': rating}
            if attributes and self.games_data and item_id in self.games_data:
                game_info = self.games_data[item_id]
                rec.update({attr: game_info[attr] for attr in attributes if attr in game_info})
            results.append(rec)
        return results

    def train(self, test_data: Optional[pd.DataFrame] = None, **kwargs) -> None:
        self.model = FunkSVD(**kwargs)
        self.model.fit(self.train_data, test_data)

    def save(self, save_path: str) -> None:
        self.model.save_model(save_path)

    def load(self, model_path: str) -> None:
        self.model.load_model(model_path)

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, Dict[int, float]]:
        return evaluate_recommendations(model=self.model, test_data=test_data)

    def add_ratings(self, ratings: List[Dict[str, Any]], user_id: Optional[int] = None) -> Tuple[
        bool, Optional[int], bool]:
        """
        Adds new ratings and triggers an incremental update of the models.
        """
        if not ratings:
            logger.warning("add_ratings called with empty ratings list.")
            return False, user_id, False

        processed_user_id, _, is_new_user = self.model.add_or_get_user(user_id)

        ratings_df = self._prepare_ratings_df(ratings, processed_user_id)
        if ratings_df.empty:
            logger.warning(f"No valid ratings to process for user {processed_user_id}.")
            return False, processed_user_id, is_new_user

        self._update_internal_train_data(ratings_df)

        success = self.model.update_with_ratings(ratings_df)

        return success, processed_user_id, is_new_user

    def _prepare_ratings_df(self, ratings: List[Dict[str, Any]], user_id: int) -> pd.DataFrame:
        """Creates a DataFrame from the ratings list, assigns user ID, and filters invalid items."""
        ratings_df = pd.DataFrame(ratings)
        if ratings_df.empty or 'BGGId' not in ratings_df.columns or 'Rating' not in ratings_df.columns:
             logger.error("Invalid ratings format provided.")
             return pd.DataFrame()

        ratings_df['UserId'] = user_id
        ratings_df['BGGId'] = ratings_df['BGGId'].astype(int)
        ratings_df['Rating'] = ratings_df['Rating'].astype(float)

        return ratings_df[['UserId', 'BGGId', 'Rating']]

    def _update_internal_train_data(self, new_ratings_df: pd.DataFrame) -> None:
        """Appends new ratings to the recommender's training data, handling duplicates."""
        if new_ratings_df.empty:
            return

        combined = pd.concat([self.train_data, new_ratings_df], ignore_index=True)
        self.train_data = combined.drop_duplicates(subset=['UserId', 'BGGId'], keep='last')
        logger.debug(f"Updated internal training data. New size: {len(self.train_data)}")

    @staticmethod
    def get_popular_recommendations(train_data: pd.DataFrame, n: int = 10) -> List[Tuple[int, float]]:
        stats = train_data.groupby('BGGId').agg(avg_rating=('Rating', 'mean'), count=('Rating', 'count'))
        stats['popularity'] = stats['avg_rating'] * np.log1p(stats['count'])
        top_items = stats.sort_values('popularity', ascending=False).head(n)
        return [(int(idx), float(row['popularity'])) for idx, row in top_items.iterrows()]

