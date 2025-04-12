import numpy as np
from loguru import logger

from src.funk_svd import FunkSVD
from src.utils.utils import load_data


def main():

    # Load data
    train_data = load_data('../data/processed/user_ratings_train_100K.csv')
    test_data = load_data('../data/processed/user_ratings_test_100K.csv')

    logger.info(f"Training: {len(train_data)} samples from {len(np.unique(train_data['UserId']))} users")
    logger.info(f"Testing: {len(test_data)} samples from {len(np.unique(test_data['UserId']))} users")

    # Train model
    model = FunkSVD(load_existing_model=True)
    model.fit(train_data, test_data)

    # Final evaluation
    test_tuples = [(row['UserId'], row['BggId'], row['Rating']) for row in test_data]
    final_rmse = model.calculate_rmse(test_tuples, model.n_factors - 1)
    logger.info(f"Final Test RMSE: {final_rmse:.4f}")


if __name__ == "__main__":
    main()