import os
from functools import lru_cache
from pathlib import Path
from typing import Union

import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent

TRAIN_DATA_PATH = BASE_DIR / 'data' / 'processed' / 'user_ratings_train_100K.csv'
TEST_DATA_PATH = BASE_DIR / 'data' / 'processed' / 'user_ratings_test_100K.csv'


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load data from CSV file using pandas"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    return pd.read_csv(file_path, dtype={
        'game_id': 'int32',
        'rating': 'float32',
        'username': 'category',
        'user_id': 'int32'
    })


@lru_cache(maxsize=1)
def get_train_data() -> pd.DataFrame:
    return load_data(TRAIN_DATA_PATH)


@lru_cache(maxsize=1)
def get_test_data() -> pd.DataFrame:
    return load_data(TEST_DATA_PATH)
