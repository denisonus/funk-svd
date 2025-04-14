from functools import lru_cache
from pathlib import Path
import os

import numpy as np
import pandas as pd

# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

# Define absolute paths to data files
TRAIN_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'user_ratings_train_100K.csv'
TEST_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'user_ratings_test_100K.csv'


def load_data(file_path):
    """Load data from CSV file using pandas"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return pd.read_csv(file_path)


@lru_cache(maxsize=1)
def get_train_data():
    return load_data(TRAIN_DATA_PATH)


@lru_cache(maxsize=1)
def get_test_data():
    return load_data(TEST_DATA_PATH)


def ensure_dir(file_path):
    directory = Path(file_path).parent
    directory.mkdir(parents=True, exist_ok=True)
