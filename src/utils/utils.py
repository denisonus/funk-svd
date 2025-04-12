import os

import pandas as pd
import numpy as np


def load_data(file_path):
    """Load data from CSV file using pandas, then convert to structured numpy array"""
    df = pd.read_csv(file_path)
    data = np.array(
        list(zip(df['BGGId'], df['Rating'], df['Username'], df['UserId'])),
        dtype=np.dtype([
            ('BggId', 'i4'),
            ('Rating', 'f4'),
            ('Username', 'U50'),
            ('UserId', 'i4')
        ])
    )

    return data


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
