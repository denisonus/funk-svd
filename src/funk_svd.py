import random

import numpy as np
import pandas as pd
from loguru import logger

from src.config import FUNK_SVD_CONFIG
from src.utils.persistence import save_model
from src.utils.utils import ensure_dir


class FunkSVD:
    def __init__(self, **kwargs):
        params = FUNK_SVD_CONFIG.copy()
        params.update(kwargs)

        self.save_path = params['save_path']
        self.n_factors = params['n_factors']
        self.max_iterations = params['max_iterations']
        self.stop_threshold = params['stop_threshold']
        self.learn_rate = params['learn_rate']
        self.bias_learn_rate = params['bias_learn_rate']
        self.regularization = params['regularization']
        self.bias_reg = params['bias_reg']

        # Model state
        self.global_mean = 0
        self.user_bias = None
        self.item_bias = None
        self.user_factors = None
        self.item_factors = None
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}
        self.user_ids = []
        self.item_ids = []

        # For reproducibility
        random.seed(42)
        if self.save_path:
            ensure_dir(self.save_path)

    def initialize_factors(self, data):
        """Initialize model parameters directly from DataFrame"""
        # Extract unique IDs directly from DataFrame
        self.user_ids = list(data['UserId'].unique())
        self.item_ids = list(data['BGGId'].unique())

        # Create mappings
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item_id_to_idx = {mid: idx for idx, mid in enumerate(self.item_ids)}

        # Initialize factors and bias terms
        n_users = len(self.user_id_to_idx)
        n_items = len(self.item_id_to_idx)
        self.user_factors = np.full((n_users, self.n_factors), 0.1)
        self.item_factors = np.full((n_items, self.n_factors), 0.1)
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        # Calculate global mean directly from DataFrame
        self.global_mean = data['Rating'].mean()

    def predict(self, user_idx, item_idx, factors_to_use=None):
        """Predict rating for user-item pair"""
        factors_to_use = self.n_factors if factors_to_use is None else min(factors_to_use, self.n_factors)

        # Calculate prediction
        pq = np.dot(self.user_factors[user_idx][:factors_to_use], self.item_factors[item_idx][:factors_to_use].T)
        prediction = self.global_mean + self.user_bias[user_idx] + self.item_bias[item_idx] + pq

        return np.clip(prediction, 1, 10)

    def predict_for_user(self, user_id, item_ids=None):
        """Predict ratings for a user"""
        if user_id not in self.user_id_to_idx:
            logger.warning(f"User {user_id} not in training data")
            return {}

        if item_ids is None:
            item_ids = self.item_ids

        user_idx = self.user_id_to_idx[user_id]
        predictions = {}

        for item_id in item_ids:
            if item_id in self.item_id_to_idx:
                item_idx = self.item_id_to_idx[item_id]
                predictions[item_id] = self.predict(user_idx, item_idx)

        return predictions

    def fit(self, train_data, test_data=None):
        """Train the model using DataFrame inputs directly"""
        # Initialize factors from DataFrame
        self.initialize_factors(train_data)

        # Convert to optimized format for computation-intensive parts only
        # This maintains the DataFrame interface while optimizing for the training loop
        train_ratings = list(train_data[['UserId', 'BGGId', 'Rating']].itertuples(index=False, name=None))

        test_ratings = None
        if test_data is not None:
            test_ratings = list(test_data[['UserId', 'BGGId', 'Rating']].itertuples(index=False, name=None))

        # Train each factor
        for factor in range(self.n_factors):
            iterations = 0
            last_train_rmse = float('inf')
            last_test_err = float('inf')
            test_err = float('inf')
            finished = False

            # Shuffle indices for SGD
            indices = random.sample(range(len(train_ratings)), len(train_ratings))

            while not finished:
                # Update model
                train_rmse = self.update_factor(factor, indices, train_ratings)

                # Calculate test error if available
                if test_ratings:
                    test_err = self.calculate_rmse(test_ratings, factor)
                    logger.info(
                        f"Epoch {iterations}, factor={factor}, Train RMSE={train_rmse:.4f}, Test RMSE={test_err:.4f}")
                else:
                    logger.info(f"Epoch {iterations}, factor={factor}, Train RMSE={train_rmse:.4f}")

                iterations += 1

                # Check stopping conditions
                finished = self.finished(iterations, last_train_rmse, train_rmse, last_test_err, test_err)
                last_train_rmse = train_rmse
                last_test_err = test_err

            save_model(self, factor, finished)

        # Save final model
        save_model(self, self.n_factors - 1, True)
        return self

    def update_factor(self, factor_idx, indices, ratings):
        """Update a factor using stochastic gradient descent"""
        for idx in indices:
            user_id, item_id, rating = ratings[idx]

            u = self.user_id_to_idx[user_id]
            i = self.item_id_to_idx[item_id]

            # Calculate error
            err = rating - self.predict(u, i, factor_idx + 1)

            # Update bias terms
            self.user_bias[u] += self.bias_learn_rate * (err - self.bias_reg * self.user_bias[u])
            self.item_bias[i] += self.bias_learn_rate * (err - self.bias_reg * self.item_bias[i])

            # Update factor values
            u_factor = self.user_factors[u][factor_idx]
            i_factor = self.item_factors[i][factor_idx]

            self.user_factors[u][factor_idx] += self.learn_rate * (err * i_factor - self.regularization * u_factor)
            self.item_factors[i][factor_idx] += self.learn_rate * (err * u_factor - self.regularization * i_factor)

        return self.calculate_rmse(ratings, factor_idx)

    def finished(self, iterations, last_err, current_err, last_test_err=float('inf'), test_err=float('inf')):
        """Determine if training should stop based on convergence or test error increase"""

        # Check max iterations
        if iterations >= self.max_iterations:
            logger.info(f'Finished training: reached max iterations ({iterations})')
            return True

        # Check convergence
        if abs(last_err - current_err) < self.stop_threshold:
            logger.info(f'Finished training: converged (improvement={last_err - current_err:.6f})')
            return True

        # Check for overfitting
        if test_err > last_test_err and iterations > 1:
            logger.info(f'Early stopping: Test RMSE increased from {last_test_err:.6f} to {test_err:.6f}')
            return True

        return False

    def calculate_rmse(self, ratings, factor_idx):
        """Calculate RMSE for given data"""
        squared_sum = 0
        count = len(ratings)

        for user_id, movie_id, rating in ratings:
            u = self.user_id_to_idx[user_id]
            i = self.item_id_to_idx[movie_id]
            pred = self.predict(u, i, factor_idx + 1)
            squared_sum += (pred - rating) ** 2

        return np.sqrt(squared_sum / count)

    def _learn_user_factors(self, user_idx, ratings):
        """Learn latent factors for a user based on their ratings"""
        # Filter ratings for items in the model
        valid_item_ids = set(self.item_id_to_idx.keys())
        valid_ratings = ratings[ratings['BGGId'].isin(valid_item_ids)]

        if valid_ratings.empty:
            logger.warning("No valid ratings to learn from for this user")
            return

        # Use simplified SGD to learn user factors
        epochs = 20
        learn_rate = self.learn_rate * 2

        for epoch in range(epochs):
            # Shuffle ratings for SGD
            valid_ratings = valid_ratings.sample(frac=1).reset_index(drop=True)

            # Track error for convergence checking
            error_sum = 0

            for _, row in valid_ratings.iterrows():
                item_idx = self.item_id_to_idx[row['BGGId']]
                rating = row['Rating']

                # Calculate prediction error
                pred = self.predict(user_idx, item_idx)
                err = rating - pred
                error_sum += err ** 2

                # Update user bias
                self.user_bias[user_idx] += self.bias_learn_rate * (err - self.bias_reg * self.user_bias[user_idx])

                # Update user factors (keeping item factors fixed)
                for f in range(self.n_factors):
                    self.user_factors[user_idx, f] += learn_rate * (
                            err * self.item_factors[item_idx, f] -
                            self.regularization * self.user_factors[user_idx, f]
                    )

            # Check for convergence
            rmse = np.sqrt(error_sum / len(valid_ratings))
            if epoch > 0 and rmse < 0.01:
                logger.debug(f"User factors converged after {epoch + 1} epochs with RMSE: {rmse:.4f}")
                break

        logger.debug(f"Finished learning factors for user with final RMSE: {rmse:.4f}")

    def add_ratings(self, ratings, user_id=None):
        """Add ratings from a user to the model"""
        # Convert ratings to DataFrame if not already
        if not isinstance(ratings, pd.DataFrame):
            ratings = pd.DataFrame(ratings)

        # Check if it's a new or existing user
        is_new_user = user_id is None or user_id not in self.user_id_to_idx

        if is_new_user:
            # Generate new user ID if not provided
            if user_id is None:
                user_id = max(self.user_ids) + 1 if self.user_ids else 1
                logger.info(f"Generated new user ID: {user_id}")

            # Add user to mappings
            user_idx = len(self.user_ids)
            self.user_ids.append(user_id)
            self.user_id_to_idx[user_id] = user_idx

            # Extend user factors and bias arrays
            self.user_factors = np.vstack([self.user_factors, np.full((1, self.n_factors), 0.1)])
            self.user_bias = np.append(self.user_bias, 0)

            logger.info(f"Added new user {user_id} with {len(ratings)} ratings")
        else:
            # Get existing user index
            user_idx = self.user_id_to_idx[user_id]
            logger.info(f"Updating existing user {user_id} with {len(ratings)} new ratings")

        # Add user_id to ratings if not present
        if 'UserId' not in ratings.columns:
            ratings['UserId'] = user_id

        # Learn/update factors for the user based on their ratings
        self._learn_user_factors(user_idx, ratings)

        return True, user_id, is_new_user
