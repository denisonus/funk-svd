from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Model paths (as relative paths)
MODELS_DIR = BASE_DIR / 'models'
FUNK_SVD_MODEL_DIR = MODELS_DIR / 'funk_svd'
GRID_SEARCH_DIR = MODELS_DIR / 'grid_search'
GAMES_DATA_PATH = BASE_DIR / 'data' / 'raw' / 'games.csv'

# Default FunkSVD parameters
FUNK_SVD_CONFIG = {
    'n_factors': 20,
    'max_iterations': 10,
    'stop_threshold': 0.005,
    'learn_rate': 0.002,
    'bias_learn_rate': 0.005,
    'regularization': 0.002,
    'bias_reg': 0.002
}

# Grid search parameters
GRID_SEARCH_CONFIG = {
    'save_path': GRID_SEARCH_DIR,
    'param_grid': {
        'n_factors': [10, 20],
        'learn_rate': [0.005, 0.01],
        'bias_learn_rate': [0.01, 0.02],
        'regularization': [0.01, 0.05],
        'bias_reg': [0.02, 0.1],
        'save_path': [None]
    }
}
