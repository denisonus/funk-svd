import itertools
import json
import time
from pathlib import Path

from loguru import logger

from src.funk_svd import FunkSVD
from src.utils.utils import ensure_dir


class FunkSVDGridSearch:
    def __init__(self, param_grid, save_path='./models/grid_search/'):
        """
        param_grid: Dictionary mapping parameter names to lists of values to try
        save_path: Directory to save results
        """
        self.param_grid = param_grid
        self.save_path = save_path
        self.results = []
        ensure_dir(save_path)

    def generate_parameter_combinations(self):
        """Generate all combinations of parameters to test"""
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())

        combinations = list(itertools.product(*param_values))
        return [dict(zip(param_names, combo)) for combo in combinations]

    def fit(self, train_data, test_data):
        """Run grid search and find the best hyperparameters with resume capability"""
        # Try to load existing results
        self._load_results()

        combinations = self.generate_parameter_combinations()
        logger.info(f"Testing {len(combinations)} hyperparameter combinations")

        # Find already tested parameter combinations
        tested_params = [result['params'] for result in self.results]
        remaining_combinations = [params for params in combinations if params not in tested_params]

        if tested_params:
            logger.info(f"Resuming grid search: {len(tested_params)} combinations already tested, "
                       f"{len(remaining_combinations)} remaining")
            # Find best RMSE from previous runs
            best_result = self.get_best_result()
            best_rmse = best_result['rmse'] if best_result else float('inf')
            best_params = best_result['params'] if best_result else None
        else:
            logger.info(f"Starting new grid search with {len(combinations)} combinations")
            best_rmse = float('inf')
            best_params = None

        for i, params in enumerate(remaining_combinations):
            run_id = f"run_{len(tested_params) + i + 1}"
            logger.info(f"[{run_id}] Starting training with parameters: {params}")

            # Train model with current parameters
            start_time = time.time()
            model = FunkSVD(**params)
            model.fit(train_data, test_data)

            # Evaluate model
            test_tuples = [(row['userId'], row['bggId'], row['rating']) for row in test_data]
            final_rmse = model.calculate_rmse(test_tuples, model.n_factors - 1)
            train_time = time.time() - start_time

            # Save results
            result = {
                'run_id': run_id,
                'params': params,
                'rmse': final_rmse,
                'training_time': train_time
            }
            self.results.append(result)

            logger.info(f"[{run_id}] Completed training: RMSE={final_rmse:.4f}, Time={train_time:.2f}s")

            # Update best parameters
            if final_rmse < best_rmse:
                best_rmse = final_rmse
                best_params = params
                logger.info(f"New best parameters found! RMSE={best_rmse:.4f}")

            # Save current results after each run
            self._save_results()

        logger.info(f"Grid search complete. Best RMSE: {best_rmse:.4f}")
        logger.info(f"Best parameters: {best_params}")
        return best_params

    def _load_results(self):
        """Load existing results if available"""
        results_path = Path(self.save_path) / "grid_search_results.json"
        if results_path.exists():
            try:
                with open(results_path, 'r') as f:
                    data = json.load(f)
                    self.results = data.get('results', [])
                logger.info(f"Loaded {len(self.results)} existing results from {results_path}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}")
        return False

    def _save_results(self):
        """Save current results to disk"""
        results_path = Path(self.save_path) / "grid_search_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'results': self.results,
                'best_result': self.get_best_result()
            }, f, indent=2)

    def get_best_result(self):
        """Get the best result based on RMSE"""
        if not self.results:
            return None
        return min(self.results, key=lambda x: x['rmse'])


def run_grid_search():
    """Run a grid search to find optimal hyperparameters"""
    # Load data
    from src.utils.utils import load_data
    train_data = load_data('../data/processed/user_ratings_train_100K.csv')
    test_data = load_data('../data/processed/user_ratings_test_100K.csv')

    # Define parameter grid
    param_grid = {
        'n_factors': [10, 20],
        'learn_rate': [0.005, 0.01],
        'bias_learn_rate': [0.01, 0.02],
        'regularization': [0.01, 0.05],
        'bias_reg': [0.02, 0.1],
        'save_path': [None]
    }

    # Run grid search
    grid_search = FunkSVDGridSearch(param_grid)
    best_params = grid_search.fit(train_data, test_data)

    return best_params


if __name__ == "__main__":
    best_params = run_grid_search()
    logger.info(f"Best parameters: {best_params}")
