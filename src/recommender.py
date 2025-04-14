from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd

from src.funk_svd import FunkSVD


class GameRecommender:
    def __init__(self, train_data: pd.DataFrame, games_data_path: Optional[Union[str, Path]] = None) -> None:
        self.model = FunkSVD()
        self.train_data = train_data
        self.games_data = None
        if games_data_path:
            self.load_games_data(games_data_path)

    def load_games_data(self, games_data_path: Union[str, Path]) -> None:
        games_df = pd.read_csv(games_data_path)
        self.games_data = {int(row['BGGId']): row.to_dict() for _, row in games_df.iterrows()}

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

    def add_ratings(self, ratings: List[Dict[str, Any]], user_id: Optional[int] = None) -> Tuple[
        bool, Optional[int], bool]:
        ratings_df = self._process_ratings(ratings)
        if ratings_df.empty:
            return False, None, False

        user_id, user_idx, is_new_user = self.model.add_or_get_user(user_id)
        item_indices = [self.model.item_id_to_idx[game_id] for game_id in ratings_df['BGGId']]
        if not item_indices:
            return False, user_id, is_new_user

        success = self.model.update_user_factors(user_idx, item_indices, ratings_df['Rating'].values)
        return success, user_id, is_new_user

    def _process_ratings(self, ratings: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process ratings and update training data, avoiding duplicates."""
        ratings_df = pd.DataFrame(ratings)
        if not self.model.item_id_to_idx or ratings_df.empty:
            return pd.DataFrame()

        valid_ids = set(self.model.item_id_to_idx.keys())
        ratings_df = ratings_df[ratings_df['BGGId'].isin(valid_ids)]
        if ratings_df.empty:
            return ratings_df

        combined = pd.concat([self.train_data, ratings_df], ignore_index=True)
        self.train_data = combined.drop_duplicates(subset=['UserId', 'BGGId'], keep='last')

        return ratings_df

    @staticmethod
    def get_popular_recommendations(train_data: pd.DataFrame, n: int = 10) -> List[Tuple[int, float]]:
        stats = train_data.groupby('BGGId').agg(avg_rating=('Rating', 'mean'), count=('Rating', 'count'))
        stats['popularity'] = stats['avg_rating'] * np.log1p(stats['count'])
        top_items = stats.sort_values('popularity', ascending=False).head(n)
        return [(int(idx), float(row['popularity'])) for idx, row in top_items.iterrows()]
