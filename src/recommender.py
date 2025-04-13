import numpy as np
import pandas as pd

from src.config import GAMES_DATA_PATH
from src.funk_svd import FunkSVD


class GameRecommender:
    def __init__(self, model_path=None, train_data=None, games_data_path=None):
        self.model = FunkSVD()
        # Convert train_data to list if it's a numpy array
        self.train_data = list(train_data)
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
        # Convert train_data to list if it's a numpy array
        self.train_data = list(train_data) if isinstance(train_data, np.ndarray) else train_data
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

    def add_new_user(self, ratings, user_id=None):
        """
        Add a new user to the model with their ratings.
        
        Parameters:
        -----------
        ratings : list of dict
            List of ratings from the user, each containing 'BggId' and 'Rating' keys
        user_id : int, optional
            The ID of the new user. If None, a new ID will be automatically generated.
            
        Returns:
        --------
        tuple
            (success, user_id) where success is True if user was added, 
            False if user already existed, and user_id is the ID of the user
        """
        if not self.model:
            raise ValueError("Model must be trained or loaded before adding new users")
            
        # Add user to the model
        success, user_id = self.model.add_new_user(ratings, user_id)
        
        # Update internal training data if successful
        if success and self.train_data is not None:
            # Ensure train_data is a list
            if isinstance(self.train_data, np.ndarray):
                self.train_data = list(self.train_data)
                
            # Create proper format for each rating
            for rating_data in ratings:
                # Ensure we have all required fields
                new_rating = {
                    'UserId': user_id,
                    'BggId': rating_data['BggId'],
                    'Rating': rating_data['Rating'],
                    'Username': rating_data.get('Username', f'user_{user_id}')  # Default username if not provided
                }

                self.train_data.append(new_rating)
        
        return success, user_id
        
    def add_user_ratings(self, ratings, user_id=None):
        """
        Add ratings from either a new or existing user to the model.
        
        Parameters:
        -----------
        ratings : list of dict
            List of ratings from the user, each containing 'BggId' and 'Rating' keys
        user_id : int, optional
            The ID of the user. If None, a new ID will be automatically generated.
            
        Returns:
        --------
        tuple
            (success, user_id, is_new_user) where:
            - success is True if ratings were successfully added
            - user_id is the ID of the user
            - is_new_user is True if a new user was created, False if updating existing user
        """
        if not self.model:
            raise ValueError("Model must be trained or loaded before adding ratings")
            
        # Add ratings to the model
        success, user_id, is_new_user = self.model.add_ratings(ratings, user_id)
        
        # Update internal training data if successful
        if success and self.train_data is not None:
            # Ensure train_data is a list
            if isinstance(self.train_data, np.ndarray):
                self.train_data = list(self.train_data)
                
            # Create proper format for each rating
            for rating_data in ratings:
                # Ensure we have all required fields
                new_rating = {
                    'UserId': user_id,
                    'BggId': rating_data['BggId'],
                    'Rating': rating_data['Rating'],
                    'Username': rating_data.get('Username', f'user_{user_id}')  # Default username if not provided
                }

                self.train_data.append(new_rating)
        
        return success, user_id, is_new_user

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
