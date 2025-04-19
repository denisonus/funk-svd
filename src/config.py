from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Model paths (as relative paths)
MODELS_DIR = BASE_DIR / 'data' / 'models'
FUNK_SVD_MODEL_DIR = MODELS_DIR / 'funk_svd'
GRID_SEARCH_DIR = MODELS_DIR / 'grid_search'
GAMES_DATA_PATH = BASE_DIR / 'data' / 'raw' / 'games.csv'

# Default FunkSVD parameters
FUNK_SVD_CONFIG = {
    'n_factors': 20,
    'max_iterations': 50,
    'stop_threshold': 0.001,
    'learn_rate': 0.002,
    'bias_learn_rate': 0.005,
    'regularization': 0.002,
    'bias_reg': 0.002
}

# Grid search parameters
GRID_SEARCH_CONFIG = {
    'save_path': GRID_SEARCH_DIR,
    'param_grid': {
        'n_factors': [15, 20, 30],
        'learn_rate': [0.002, 0.005],
        'bias_learn_rate': [0.002, 0.005],
        'regularization': [0.002, 0.005],
        'bias_reg': [0.005, 0.01],
    }
}

# "best_rmse": {
#     "value": 1.1644047926090164,
#     "params": {
#       "n_factors": 15,
#       "learn_rate": 0.002,
#       "bias_learn_rate": 0.005,
#       "regularization": 0.002,
#       "bias_reg": 0.01
#     }
# "value": 1.1640805557088096,
#     "params": {
#       "n_factors": 15,
#       "learn_rate": 0.005,
#       "bias_learn_rate": 0.005,
#       "regularization": 0.005,
#       "bias_reg": 0.01
#     }
# 'param_grid': {
#         'n_factors': [6, 9, 12],
#         'learn_rate': [0.01, 0.02],
#         'bias_learn_rate': [0.01, 0.02],
#         'regularization': [0.01, 0.02],
#         'bias_reg': [0.01, 0.02],
#     }
# Best parameters: {'n_factors': 12, 'learn_rate': 0.01, 'bias_learn_rate': 0.01, 'regularization': 0.01, 'bias_reg': 0.01}

# 'param_grid': {
#         'n_factors': [10, 20],
#         'learn_rate': [0.005, 0.01],
#         'bias_learn_rate': [0.01, 0.02],
#         'regularization': [0.01, 0.05],
#         'bias_reg': [0.02, 0.1],
#     }
# Best parameters: {'n_factors': 10, 'learn_rate': 0.01, 'bias_learn_rate': 0.01, 'regularization': 0.01, 'bias_reg': 0.02}
# Params from the book: {'n_factors': 25, 'learn_rate': 0.002, 'bias_learn_rate': 0.005, 'regularization': 0.002, 'bias_reg': 0.002}
# Params from GitHub: {'n_factors': 20, 'learn_rate': 0.01, 'bias_learn_rate': 0.01, 'regularization': 0.02, 'bias_reg': 0.02}