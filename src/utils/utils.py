from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# Using pathlib for path operations
TRAIN_DATA_PATH = Path('..') / 'data' / 'processed' / 'user_ratings_train_100K.csv'
TEST_DATA_PATH = Path('..') / 'data' / 'processed' / 'user_ratings_test_100K.csv'


def load_data(file_path):
    """Load data from CSV file using pandas, then convert to structured numpy array"""
    df = pd.read_csv(file_path)
    data = np.array(list(zip(df['BGGId'], df['Rating'], df['Username'], df['UserId'])),
                    dtype=np.dtype([('BggId', 'i4'), ('Rating', 'f4'), ('Username', 'U50'), ('UserId', 'i4')]))

    return data


@lru_cache(maxsize=1)
def get_train_data():
    return load_data(TRAIN_DATA_PATH)


@lru_cache(maxsize=1)
def get_test_data():
    return load_data(TEST_DATA_PATH)


def ensure_dir(file_path):
    directory = Path(file_path).parent
    directory.mkdir(parents=True, exist_ok=True)
