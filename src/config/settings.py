from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

# Model paths
MODELS_DIR = BASE_DIR / 'data' / 'models'
FUNK_SVD_MODEL_DIR = MODELS_DIR / 'funk_svd'
GRID_SEARCH_DIR = MODELS_DIR / 'grid_search'

TRAIN_DATA_PATH = BASE_DIR / 'data' / 'processed' / 'user_ratings_train_100K.csv'
TEST_DATA_PATH = BASE_DIR / 'data' / 'processed' / 'user_ratings_test_100K.csv'
GAMES_DATA_PATH = BASE_DIR / 'data' / 'raw' / 'games.csv'

# Default FunkSVD parameters
FUNK_SVD_CONFIG = {
    'n_factors': 5,
    'max_iterations': 100,
    'stop_threshold': 0.001,
    'learn_rate': 0.002,
    'bias_learn_rate': 0.002,
    'regularization': 0.002,
    'bias_reg': 0.002
}

GRID_SEARCH_CONFIG = {
    'save_path': GRID_SEARCH_DIR,
    'load_results': False,
    'evaluation_k_values': [10, 20],
    'primary_metric': 'ndcg@20',
    'param_grid': {
        'n_factors': [5, 10],
        'learn_rate': [0.002, 0.005],
        'bias_learn_rate': [0.002, 0.005],
        'regularization': [0.002, 0.005],
        'bias_reg': [0.002, 0.005],
    }
}

EVALUATION_CONFIG = {
    'relevance_threshold': 7.0,  # Minimum rating to be considered relevant
    'k_values': [10, 20],
}
