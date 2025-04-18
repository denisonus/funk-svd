import numpy as np

from src.config import GAMES_DATA_PATH
from src.data.load_dataset import get_train_data, get_test_data
from src.recommender import GameRecommender

train_data = get_train_data()
test_data = get_test_data()
print(f"Training: {len(train_data)} samples from {len(np.unique(train_data['UserId']))} users")
print(f"Testing: {len(test_data)} samples from {len(np.unique(test_data['UserId']))} users")

recommender = GameRecommender(train_data, GAMES_DATA_PATH)
recommender.train(test_data)