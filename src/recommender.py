import numpy as np

from src.funk_svd import FunkSVD


class GameRecommender:
    def __init__(self, model_path=None, train_data=None):
        self.model = FunkSVD()
        self.train_data = train_data  # Store for filtering recommendations

        if model_path:
            self.model.load(model_path)

    def get_predictions(self, user_id, item_ids=None):
        """Get predictions for specific user and items"""
        return self.model.predict_for_user(user_id, item_ids)

    def get_top_n_recommendations(self, user_id, n=10, include_scores=True):
        """Get top N recommendations for a user without requiring train_data parameter"""
        if not self.train_data:
            raise ValueError("Train data is required for filtering recommendations")

        # Get all predictions
        predictions = self.model.predict_for_user(user_id)

        # Filter out already rated items
        rated_items = set(item['BggId'] for item in self.train_data if item['UserId'] == user_id)
        predictions = {item_id: rating for item_id, rating in predictions.items()
                      if item_id not in rated_items}

        # Sort and return top N
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]

        # Return with or without scores
        if include_scores:
            return sorted_predictions
        else:
            return [item_id for item_id, _ in sorted_predictions]

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
