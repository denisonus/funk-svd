import numpy as np
from loguru import logger

from src.config import FUNK_SVD_CONFIG
from src.recommender import GameRecommender
from src.utils.utils import get_test_data, get_train_data


def main():
    # Load data
    train_data = get_train_data()
    test_data = get_test_data()

    logger.info(f"Training: {len(train_data)} samples from {len(np.unique(train_data['UserId']))} users")
    logger.info(f"Testing: {len(test_data)} samples from {len(np.unique(test_data['UserId']))} users")

    # Load model through recommender
    model_path = FUNK_SVD_CONFIG['save_path'] / 'model' / 'final'
    recommender = GameRecommender(model_path=model_path, train_data=train_data)
    recommender.train(train_data=train_data, test_data=test_data)


if __name__ == "__main__":
    main()