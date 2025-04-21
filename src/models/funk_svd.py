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
        self.max_iterations: int = params['max_iterations'] # Max iterations for full fit
        self.stop_threshold: float = params['stop_threshold']
        self.learn_rate: float = params['learn_rate']
        self.bias_learn_rate: float = params['bias_learn_rate']
        self.regularization: float = params['regularization']
        self.bias_reg: float = params['bias_reg']
        # Add a parameter for incremental updates
        self.incremental_learn_rate: float = params.get('incremental_learn_rate', self.learn_rate * 0.5) # Use a smaller LR for updates
        self.incremental_bias_learn_rate: float = params.get('incremental_bias_learn_rate', self.bias_learn_rate * 0.5)
        self.incremental_iterations: int = params.get('incremental_iterations', 5) # Number of passes over new data

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
                train_rmse = self.update_factor(factor, indices, train_ratings, self.learn_rate, self.bias_learn_rate)

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

    def update_factor(self, factor_idx: int, indices: List[int], ratings: List[Tuple], learn_rate: float, bias_learn_rate: float) -> float:
        """Update a factor using stochastic gradient descent"""
        for idx in indices:
            user_id, item_id, rating = ratings[idx]

            # Check if user/item exists in the current model state before proceeding
            if user_id not in self.user_id_to_idx or item_id not in self.item_id_to_idx:
                continue # Skip if user or item is unknown (e.g., during incremental update with new items not yet handled)

            u = self.user_id_to_idx[user_id]
            i = self.item_id_to_idx[item_id]

            # Calculate error
            err = rating - self.predict(u, i, factor_idx + 1)

            # Update bias terms
            self.user_bias[u] += bias_learn_rate * (err - self.bias_reg * self.user_bias[u])
            self.item_bias[i] += bias_learn_rate * (err - self.bias_reg * self.item_bias[i])

            # Update factor values
            u_factor = self.user_factors[u][factor_idx]
            i_factor = self.item_factors[i][factor_idx]

            self.user_factors[u][factor_idx] += learn_rate * (err * i_factor - self.regularization * u_factor)
            self.item_factors[i][factor_idx] += learn_rate * (err * u_factor - self.regularization * i_factor)

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
        """Add a new user or get existing user index. Initializes new users."""
        if user_id is None:
            user_id = max(self.user_ids) + 1 if self.user_ids else 1
            is_new_user = True
        else:
            is_new_user = user_id not in self.user_id_to_idx

        if is_new_user:
            user_idx = len(self.user_ids)
            self.user_ids.append(user_id)
            self.user_id_to_idx[user_id] = user_idx

            # Initialize new user: Use average bias/factors if available, else defaults
            new_bias = np.mean(self.user_bias) if self.user_bias is not None and len(self.user_bias) > 0 else 0.0
            new_factors = np.mean(self.user_factors, axis=0) if self.user_factors is not None and len(self.user_factors) > 0 else np.full((1, self.n_factors), 0.1)

            self.user_factors = np.vstack([self.user_factors, new_factors]) if self.user_factors is not None else new_factors.reshape(1, -1)
            self.user_bias = np.append(self.user_bias, new_bias) if self.user_bias is not None else np.array([new_bias])

            logger.info(f"Added new user {user_id} with index {user_idx}. Initial bias: {new_bias:.4f}")
        else:
            user_idx = self.user_id_to_idx[user_id]

        return user_id, user_idx, is_new_user

    def add_or_get_item(self, item_id: int) -> Tuple[int, int, bool]:
        """Add a new item or get existing item index. Initializes new items."""
        is_new_item = item_id not in self.item_id_to_idx

        if is_new_item:
            item_idx = len(self.item_ids)
            self.item_ids.append(item_id)
            self.item_id_to_idx[item_id] = item_idx

            # Initialize new item: Use average bias/factors if available, else defaults
            new_bias = np.mean(self.item_bias) if self.item_bias is not None and len(self.item_bias) > 0 else 0.0
            # Initialize factors similar to users, using mean of existing item factors
            new_factors = np.mean(self.item_factors, axis=0) if self.item_factors is not None and len(self.item_factors) > 0 else np.full((1, self.n_factors), 0.1)

            self.item_factors = np.vstack([self.item_factors, new_factors]) if self.item_factors is not None else new_factors.reshape(1, -1)
            self.item_bias = np.append(self.item_bias, new_bias) if self.item_bias is not None else np.array([new_bias])

            logger.info(f"Added new item {item_id} with index {item_idx}. Initial bias: {new_bias:.4f}")
        else:
            item_idx = self.item_id_to_idx[item_id]

        return item_id, item_idx, is_new_item

    def update_with_ratings(self, new_ratings_df: pd.DataFrame) -> bool:
        """
        Update model factors and biases using incremental SGD based on new ratings.
        Handles adding new users and new items.
        """
        if new_ratings_df.empty:
            logger.warning("Received empty DataFrame for incremental update.")
            return False

        # 1. Ensure all users exist, add if necessary
        for user_id in new_ratings_df['UserId'].unique():
            self.add_or_get_user(user_id)

        # 2. Ensure all items exist, add if necessary
        for item_id in new_ratings_df['BGGId'].unique():
            self.add_or_get_item(item_id)

        # 3. Convert new ratings to the list format used in training
        # No need to filter items anymore as they are added above
        new_ratings_tuples = list(new_ratings_df[['UserId', 'BGGId', 'Rating']].itertuples(index=False, name=None))

        if not new_ratings_tuples:
             logger.warning("No ratings remaining after processing for incremental update.")
             return False # Could happen if initial df was empty or only had invalid data types

        logger.info(f"Performing incremental SGD update with {len(new_ratings_tuples)} ratings (including potential new users/items)...")

        # 4. Perform SGD updates for a fixed number of iterations on the new data
        for factor in range(self.n_factors):
            # Shuffle indices once per factor update pass
            indices = random.sample(range(len(new_ratings_tuples)), len(new_ratings_tuples))
            for iteration in range(self.incremental_iterations):
                rmse = self.update_factor(
                    factor_idx=factor,
                    indices=indices, # Use the same shuffled indices for inner iterations
                    ratings=new_ratings_tuples,
                    learn_rate=self.incremental_learn_rate,
                    bias_learn_rate=self.incremental_bias_learn_rate
                )
                # Optional: Log progress less frequently or based on condition
                # if (iteration + 1) % self.incremental_iterations == 0: # Log only at the end of iterations for this factor
                #    logger.debug(f"Incremental Update: Factor {factor}, Iteration {iteration+1}, Final Batch RMSE: {rmse:.4f}")

        logger.info(f"Incremental SGD update completed. Final RMSE on batch for last factor: {rmse:.4f}") # Log final RMSE
        return True

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

