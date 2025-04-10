import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


class MatrixFactorization:
    Regularization = 0.02
    LearnRate = 0.01
    BiasLearnRate = 0.005
    BiasReg = 0.002

    def __init__(self, save_path='./models/funk_svd/', n_factors=20, max_iterations=15, stop_threshold=0.005):
        self.logger = logging.getLogger('funkSVD')
        self.save_path = save_path
        self.n_factors = n_factors
        self.MAX_ITERATIONS = max_iterations
        self.stop_threshold = stop_threshold

        # Initialize to None - will be set during fit
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.all_movies_mean = 0.0
        self.user_id_to_index = None
        self.item_id_to_index = None
        self.user_ids = None
        self.movie_ids = None
        self.iterations = 0

        # Set random seed for reproducibility
        random.seed(42)
        ensure_dir(save_path)

    def initialize_factors(self, data):

        """Initialize factor matrices and bias terms"""
        # Extract unique IDs
        self.user_ids = set(np.unique(data['userId']))
        self.movie_ids = set(np.unique(data['bggId']))

        # Create mappings for efficient index access
        self.user_id_to_index = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item_id_to_index = {mid: idx for idx, mid in enumerate(self.movie_ids)}

        # Initialize latent factors with proper scaling
        self.item_factors = np.full((len(self.item_id_to_index), self.n_factors), 0.1)
        self.user_factors = np.full((len(self.user_id_to_index), self.n_factors), 0.1)

        # Initialize bias terms
        self.user_bias = {u: 0 for u in self.user_id_to_index.values()}
        self.item_bias = {i: 0 for i in self.item_id_to_index.values()}

        # Calculate global mean
        self.all_movies_mean = np.mean(data['rating'])

    def predict(self, user, item, factors_to_use=None):
        """Predict a single rating using the model"""
        if factors_to_use is None:
            factors_to_use = self.n_factors

        # Ensure we don't exceed available factors
        factors_to_use = min(factors_to_use, self.n_factors)

        # Dot product of user and item factors (up to the specified number of factors)
        pq = np.dot(self.user_factors[user][:factors_to_use], self.item_factors[item][:factors_to_use].T)

        # Add bias terms
        b_ui = self.all_movies_mean + self.user_bias[user] + self.item_bias[item]
        prediction = b_ui + pq

        # Clip prediction to valid range
        if prediction > 10:
            prediction = 10
        elif prediction < 1:
            prediction = 1

        return prediction

    def fit(self, train_data, test_data=None):
        """Train the model on the given data array and evaluate on test data if provided"""
        self.initialize_factors(train_data)

        # Convert data to a list for faster processing
        train_ratings = [(row['userId'], row['bggId'], row['rating']) for row in train_data]

        # Convert test data if provided
        test_ratings = None
        if test_data is not None:
            test_ratings = [(row['userId'], row['bggId'], row['rating']) for row in test_data]

        # Randomly shuffle the indices for SGD
        index_randomized = random.sample(range(len(train_ratings)), len(train_ratings))

        # Train one factor at a time
        for factor in range(self.n_factors):
            iterations = 0
            last_err = float('inf')
            iteration_err = float('inf')
            last_test_err = float('inf')
            test_err = float('inf')
            finished = False

            while not finished:
                iteration_err = self.stocastic_gradient_descent(factor, index_randomized, train_ratings)

                # Calculate test error if test data is provided
                if test_ratings:
                    test_err = self.calculate_rmse(test_ratings, factor)
                    self.logger.info(
                        f"Epoch {iterations}, factor={factor}, Train RMSE={iteration_err:.4f}, Test RMSE={test_err:.4f}")
                else:
                    self.logger.info(f"Epoch {iterations}, factor={factor}, Train RMSE={iteration_err:.4f}")

                iterations += 1

                # Check if training should stop
                finished = self.finished(iterations, last_err, iteration_err, last_test_err, test_err)
                last_err = iteration_err
                last_test_err = test_err

            self.save(factor, finished)
            if test_ratings:
                self.logger.info(
                    f"Completed factor {factor} after {iterations} iterations with Train RMSE {iteration_err:.4f}, Test RMSE {test_err:.4f}")
            else:
                self.logger.info(
                    f"Completed factor {factor} after {iterations} iterations with RMSE {iteration_err:.4f}")

        # Save final model
        self.save(self.n_factors - 1, True)
        return self

    def stocastic_gradient_descent(self, factor, index_randomized, ratings):
        """Update factors using stochastic gradient descent"""
        lr = self.LearnRate
        b_lr = self.BiasLearnRate
        r = self.Regularization
        bias_r = self.BiasReg

        for idx in index_randomized:
            user_id, movie_id, rating = ratings[idx]

            # Get mapped indices
            u = self.user_id_to_index[user_id]
            i = self.item_id_to_index[movie_id]

            # Calculate prediction error
            err = rating - self.predict(u, i, factor + 1)

            # Update bias terms
            self.user_bias[u] += b_lr * (err - bias_r * self.user_bias[u])
            self.item_bias[i] += b_lr * (err - bias_r * self.item_bias[i])

            # Get current factor values
            user_fac = self.user_factors[u][factor]
            item_fac = self.item_factors[i][factor]

            # Update latent factors for this dimension
            self.user_factors[u][factor] += lr * (err * item_fac - r * user_fac)
            self.item_factors[i][factor] += lr * (err * user_fac - r * item_fac)

        # Calculate RMSE for current iteration
        return self.calculate_rmse(ratings, factor)

    def finished(self, iterations, last_err, current_err, last_test_err=float('inf'), test_err=float('inf')):
        """Determine if training should stop based on convergence or test error increase"""
        if iterations >= self.MAX_ITERATIONS or abs(last_err - current_err) < self.stop_threshold:
            self.logger.info(f'Finished training: iterations={iterations}, improvement={last_err - current_err:.6f}')
            return True
        elif test_err > last_test_err and iterations > 1:  # Stop if test error starts increasing (overfitting)
            self.logger.info(f'Early stopping: Test RMSE increased from {last_test_err:.6f} to {test_err:.6f}')
            return True
        else:
            self.iterations += 1
            return False

    def calculate_rmse(self, ratings, factor):
        """Calculate RMSE for given data"""
        squared_errors = 0
        count = 0

        for user_id, movie_id, rating in ratings:
            u = self.user_id_to_index[user_id]
            i = self.item_id_to_index[movie_id]

            prediction = self.predict(u, i, factor + 1)
            squared_errors += (prediction - rating) ** 2
            count += 1

        return np.sqrt(squared_errors / count)

    def save(self, factor, finished):
        """Save the model to disk"""
        pass  # save_path = self.save_path + '/model/'  # if not finished:  #     save_path += str(factor) + '/'  # else:  #     save_path += 'final/'  # Store final model in a separate directory  #  # ensure_dir(save_path)  #  # self.logger.info(f"Saving factors to {save_path}")  #  # # Convert indices back to original IDs for saving  # user_bias = {uid: self.user_bias[self.user_id_to_index[uid]] for uid in self.user_ids}  # item_bias = {iid: self.item_bias[self.item_id_to_index[iid]] for iid in self.movie_ids}  #  # # Save user factors, item factors and bias terms  # with open(save_path + 'user_factors.pkl', 'wb') as uf_file:  #     pickle.dump(self.user_factors, uf_file)  # with open(save_path + 'item_factors.pkl', 'wb') as if_file:  #     pickle.dump(self.item_factors, if_file)  # with open(save_path + 'user_bias.pkl', 'wb') as ub_file:  #     pickle.dump(user_bias, ub_file)  # with open(save_path + 'item_bias.pkl', 'wb') as ib_file:  #     pickle.dump(item_bias, ib_file)  # with open(save_path + 'metadata.pkl', 'wb') as meta_file:  #     metadata = {'u_inx': self.user_id_to_index, 'i_inx': self.item_id_to_index,  #                 'all_movies_mean': self.all_movies_mean, 'n_factors': self.n_factors, 'current_factor': factor}  #     pickle.dump(metadata, meta_file)


def load_data(file_path):
    """Load data from CSV file"""
    # Load the data using numpy's efficient CSV loading
    data = np.loadtxt(file_path, delimiter=',', skiprows=1,
                      dtype={'names': ('bggId', 'rating', 'userId'), 'formats': ('i4', 'f4', 'i4')})
    return data


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('funkSVD')
    logger.info("[BEGIN] Calculating matrix factorization")

    # Load train and test data from separate files
    train_data = load_data('../data/processed/user_ratings_train_500K.csv')
    test_data = load_data('../data/processed/user_ratings_test_500K.csv')

    logger.info(
        f"Loaded {len(train_data)} training samples from {len(np.unique(train_data['userId']))} users on {len(np.unique(train_data['bggId']))} items")
    logger.info(
        f"Loaded {len(test_data)} test samples from {len(np.unique(test_data['userId']))} users on {len(np.unique(test_data['bggId']))} items")

    # Create and train model
    model = MatrixFactorization(n_factors=20, max_iterations=15, save_path='./models/funk_svd/')

    # Train model with train data and validate on test data
    model.fit(train_data, test_data)

    # Calculate final RMSE on test data
    final_test_rmse = model.calculate_rmse([(row['userId'], row['bggId'], row['rating']) for row in test_data],
                                           model.n_factors - 1)
    logger.info(f"Final Test RMSE: {final_test_rmse:.4f}")

    logger.info("[DONE] Calculating matrix factorization")


if __name__ == "__main__":
    main()
