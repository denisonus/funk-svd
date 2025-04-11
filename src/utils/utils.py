import os

import numpy as np


def load_data(file_path):
    """Load data from CSV file"""
    data = np.loadtxt(file_path, delimiter=',', skiprows=1,
                      dtype={'names': ('bggId', 'rating', 'userId'), 'formats': ('i4', 'f4', 'i4')})
    return data


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
