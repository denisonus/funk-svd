import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import FUNK_SVD_CONFIG


class FunkSVD:
    def __init__(self, **kwargs: Any) -> None:
        params = FUNK_SVD_CONFIG.copy()
        params.update(kwargs)

        self.n_factors: int = params['n_factors']
        self.max_iterations: int = params['max_iterations']
        self.stop_threshold: float = params['stop_threshold']
        self.learn_rate: float = params['learn_rate']
        self.bias_learn_rate: float = params['bias_learn_rate']
        self.regularization: float = params['regularization']
        self.bias_reg: float = params['bias_reg']

        # Model state
        self.global_mean: float = 0.0
        self.user_bias: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_id_to_idx: Dict[int, int] = {}
        self.item_id_to_idx: Dict[int, int] = {}
        self.user_ids: List[int] = []
        self.item_ids: List[int] = []

        # For reproducibility
        random.seed(42)

    def initialize_factors(self, data: pd.DataFrame) -> None:
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

    def predict(self, user_idx: int, item_idx: int, factors_to_use: Optional[int] = None) -> float:
        """Predict rating for user-item pair"""
        factors_to_use = self.n_factors if factors_to_use is None else min(factors_to_use, self.n_factors)

        # Calculate prediction
        pq = np.dot(self.user_factors[user_idx][:factors_to_use], self.item_factors[item_idx][:factors_to_use].T)
        prediction = self.global_mean + self.user_bias[user_idx] + self.item_bias[item_idx] + pq

        return np.clip(prediction, 1, 10)

    def predict_for_user(self, user_id: int, item_ids: Optional[List[int]] = None) -> Dict[int, float]:
        """Predict ratings for a user"""
        if user_id not in self.user_id_to_idx:
            logger.warning(f"User {user_id} not in training data")
            return {}

        if item_ids is None:
            item_ids = self.item_ids

        user_idx = self.user_id_to_idx[user_id]
        predictions: Dict[int, float] = {}

        for item_id in item_ids:
            if item_id in self.item_id_to_idx:
                item_idx = self.item_id_to_idx[item_id]
                predictions[item_id] = self.predict(user_idx, item_idx)

        return predictions

    def fit(self, train_data: pd.DataFrame, test_data: Optional[pd.DataFrame] = None) -> None:
        """Train the model using DataFrame inputs"""
        # Initialize factors from DataFrame
        self.initialize_factors(train_data)

        # Convert to optimized format for training
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

    def update_factor(self, factor_idx: int, indices: List[int], ratings: List[Tuple]) -> float:
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

    def finished(self, iterations: int, last_err: float, current_err: float,
                 last_test_err: float = float('inf'), test_err: float = float('inf')) -> bool:
        # Maximum iterations check
        if iterations >= self.max_iterations:
            logger.debug(f'Finished training: reached max iterations ({iterations})')
            return True

        # Convergence check
        if abs(last_err - current_err) < self.stop_threshold:
            logger.debug(f'Finished training: converged (improvement={last_err - current_err:.6f})')
            return True

        # Check for overfitting
        if test_err > last_test_err * 1.005 and iterations > 5:
            logger.info(f'Early stopping: Test RMSE increased from {last_test_err:.6f} to {test_err:.6f}')
            return True

        return False

    def calculate_rmse(self, ratings: List[Tuple], factor_idx: int) -> float:
        """Calculate RMSE for given data"""
        squared_sum = 0
        count = len(ratings)

        for user_id, movie_id, rating in ratings:
            u = self.user_id_to_idx[user_id]
            i = self.item_id_to_idx[movie_id]
            pred = self.predict(u, i, factor_idx + 1)
            squared_sum += (pred - rating) ** 2

        return np.sqrt(squared_sum / count)

    def calculate_mae(self, ratings: List[Tuple], factor_idx: int) -> float:
        """Calculate MAE for given data"""
        abs_sum = 0
        count = len(ratings)

        for user_id, movie_id, rating in ratings:
            u = self.user_id_to_idx[user_id]
            i = self.item_id_to_idx[movie_id]
            pred = self.predict(u, i, factor_idx + 1)
            abs_sum += abs(pred - rating)

        return abs_sum / count

    def add_or_get_user(self, user_id: Optional[int] = None) -> Tuple[int, int, bool]:
        """Add a new user or get existing user index."""
        if user_id is None:
            user_id = max(self.user_ids) + 1 if self.user_ids else 1
            is_new_user = True
        else:
            is_new_user = user_id not in self.user_id_to_idx

        if is_new_user:
            # Add user to model mappings
            user_idx = len(self.user_ids)
            self.user_ids.append(user_id)
            self.user_id_to_idx[user_id] = user_idx

            # Extend user factors and bias arrays
            self.user_factors = np.vstack([
                self.user_factors,
                np.full((1, self.n_factors), 0.1)
            ])
            self.user_bias = np.append(self.user_bias, 0)
        else:
            user_idx = self.user_id_to_idx[user_id]

        return user_id, user_idx, is_new_user

    def update_user_factors(self, user_idx: int, item_indices: List[int], ratings: np.ndarray) -> bool:
        """Update user latent factors based on their ratings."""
        item_biases = self.item_bias[item_indices]
        bias_adjusted_ratings = ratings - self.global_mean - item_biases

        # Get item factors for rated items
        item_factors_subset = self.item_factors[item_indices]
        reg_term = self.regularization * np.eye(self.n_factors)

        try:
            # Matrix solution: (X^T X + λI)^-1 X^T y
            user_factors = np.linalg.solve(
                item_factors_subset.T @ item_factors_subset + reg_term,
                item_factors_subset.T @ bias_adjusted_ratings
            )
            self.user_factors[user_idx] = user_factors

            # Update user bias with average error
            predictions = np.sum(user_factors * item_factors_subset, axis=1)
            errors = ratings - (self.global_mean + predictions + item_biases)
            self.user_bias[user_idx] = np.mean(errors)
            return True
        except np.linalg.LinAlgError:
            # Fallback to simple averaging if matrix solution fails
            self.user_factors[user_idx] = np.mean(item_factors_subset, axis=0)
            return False

    def save_model(self, save_path: Union[str, Path]) -> None:
        """
        Save model to disk, unless save_path is None
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Saving model to {save_path}")

        np.save(save_path / 'user_factors.npy', self.user_factors)
        np.save(save_path / 'item_factors.npy', self.item_factors)
        np.save(save_path / 'user_bias.npy', self.user_bias)
        np.save(save_path / 'item_bias.npy', self.item_bias)

        metadata = {
            'user_id_to_idx': self.user_id_to_idx,
            'item_id_to_idx': self.item_id_to_idx,
            'global_mean': self.global_mean,
            'n_factors': self.n_factors,
            'user_ids': self.user_ids,
            'item_ids': self.item_ids
        }
        np.save(save_path / 'metadata.npy', metadata)

    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load model data into an existing FunkSVD instance
        """
        model_path = Path(model_path)
        logger.debug(f"Loading model from {model_path}")

        self.user_factors = np.load(model_path / 'user_factors.npy')
        self.item_factors = np.load(model_path / 'item_factors.npy')
        self.user_bias = np.load(model_path / 'user_bias.npy')
        self.item_bias = np.load(model_path / 'item_bias.npy')
        metadata = np.load(model_path / 'metadata.npy', allow_pickle=True).item()

        self.user_id_to_idx = metadata['user_id_to_idx']
        self.item_id_to_idx = metadata['item_id_to_idx']
        self.global_mean = metadata['global_mean']
        self.n_factors = metadata['n_factors']
        self.user_ids = metadata['user_ids']
        self.item_ids = metadata['item_ids']

