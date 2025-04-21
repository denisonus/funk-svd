from pprint import pprint

import numpy as np

from src.config import FUNK_SVD_MODEL_DIR
from src.data.load_dataset import get_train_data, get_test_data, get_games_data
from src.recommender import GameRecommender

train_data = get_train_data()
test_data = get_test_data()
games_data = get_games_data()
print(f"Training: {len(train_data)} samples from {len(np.unique(train_data['UserId']))} users")
print(f"Testing: {len(test_data)} samples from {len(np.unique(test_data['UserId']))} users")

recommender = GameRecommender(train_data, games_data)
recommender.train(test_data)
recommender.save(FUNK_SVD_MODEL_DIR)
pprint(recommender.evaluate(test_data))
