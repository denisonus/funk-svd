import numpy as np
from loguru import logger

from src.funk_svd import FunkSVD
from src.utils.utils import load_data


def main():

    # Load data
    train_data = load_data('../data/processed/user_ratings_train_500K.csv')
    test_data = load_data('../data/processed/user_ratings_test_500K.csv')

    logger.info(f"Training: {len(train_data)} samples from {len(np.unique(train_data['userId']))} users")
    logger.info(f"Testing: {len(test_data)} samples from {len(np.unique(test_data['userId']))} users")

    # Train model
    model = FunkSVD(save_path=None)
    model.fit(train_data, test_data)

    # Final evaluation
    test_tuples = [(row['userId'], row['bggId'], row['rating']) for row in test_data]
    final_rmse = model.calculate_rmse(test_tuples, model.n_factors - 1)
    logger.info(f"Final Test RMSE: {final_rmse:.4f}")


if __name__ == "__main__":
    main()