from functools import lru_cache

import pandas as pd

from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH, GAMES_DATA_PATH


@lru_cache(maxsize=1)
def get_train_data() -> pd.DataFrame:
    dtypes = {'game_id': 'int32', 'rating': 'float32', 'username': 'category', 'user_id': 'int32'}

    return pd.read_csv(TRAIN_DATA_PATH, dtype=dtypes)


@lru_cache(maxsize=1)
def get_test_data() -> pd.DataFrame:
    dtypes = {'game_id': 'int32', 'rating': 'float32', 'username': 'category', 'user_id': 'int32'}

    return pd.read_csv(TEST_DATA_PATH, dtype=dtypes)


@lru_cache(maxsize=1)
def get_games_data() -> pd.DataFrame:
    return pd.read_csv(GAMES_DATA_PATH)
