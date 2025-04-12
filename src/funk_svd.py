import random
from pathlib import Path

import numpy as np
from loguru import logger

from src.config import FUNK_SVD_CONFIG
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
        """Initialize model parameters"""
        # Extract unique IDs
        self.user_ids = list(np.unique(data['UserId']))
        self.item_ids = list(np.unique(data['BggId']))

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

        # Calculate global mean
        self.global_mean = np.mean(data['Rating'])

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
        """Train the model"""
        self.initialize_factors(train_data)

        # Convert to tuples for faster processing
        train_ratings = [(row['UserId'], row['BggId'], row['Rating']) for row in train_data]
        test_ratings = None
        if test_data is not None:
            test_ratings = [(row['UserId'], row['BggId'], row['Rating']) for row in test_data]

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

            self.save(factor, finished)

        # Save final model
        self.save(self.n_factors - 1, True)
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

    def save(self, factor_idx, finished):
        """Save model to disk, unless save_path is None"""
        if self.save_path is None:
            return

        save_path = Path(self.save_path) / 'model'
        if finished:
            save_path = save_path / 'final'
        else:
            save_path = save_path / str(factor_idx)

        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {save_path}")

        # Save essential model data
        np.save(save_path / 'user_factors.npy', self.user_factors)
        np.save(save_path / 'item_factors.npy', self.item_factors)
        np.save(save_path / 'user_bias.npy', self.user_bias)
        np.save(save_path / 'item_bias.npy', self.item_bias)

        # Save minimal metadata in one file
        metadata = {'user_id_to_idx': self.user_id_to_idx, 'item_id_to_idx': self.item_id_to_idx,
                    'global_mean': self.global_mean, 'n_factors': self.n_factors, 'current_factor': factor_idx,
                    'user_ids': self.user_ids, 'item_ids': self.item_ids}
        np.save(save_path / 'metadata.npy', metadata)

    def load(self, model_path):
        """Load model from disk"""
        # Convert string path to Path object if needed
        model_path = Path(model_path)
        logger.info(f"Loading model from {model_path}")

        # Load essential model data
        self.user_factors = np.load(model_path / 'user_factors.npy')
        self.item_factors = np.load(model_path / 'item_factors.npy')
        self.user_bias = np.load(model_path / 'user_bias.npy')
        self.item_bias = np.load(model_path / 'item_bias.npy')

        # Load metadata
        metadata = np.load(model_path / 'metadata.npy', allow_pickle=True).item()
        self.user_id_to_idx = metadata['user_id_to_idx']
        self.item_id_to_idx = metadata['item_id_to_idx']
        self.global_mean = metadata['global_mean']
        self.n_factors = metadata['n_factors']
        self.user_ids = metadata['user_ids']
        self.item_ids = metadata['item_ids']

        return self
