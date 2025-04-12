import itertools
import json
import time
from pathlib import Path

from loguru import logger

from src.funk_svd import FunkSVD
from src.utils.utils import ensure_dir, load_data


class FunkSVDGridSearch:
    def __init__(self, param_grid, save_path='./models/grid_search/', load_results=True):
        self.param_grid = param_grid
        self.save_path = Path(save_path)
        self.results = []
        ensure_dir(save_path)
        self.results_path = self.save_path / "grid_search_results.json"

        if load_results and self.results_path.exists():
            try:
                with open(self.results_path, 'r') as f:
                    self.results = json.load(f).get('results', [])
                logger.info(f"Loaded {len(self.results)} existing results")
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}")


    def fit(self, train_data, test_data):
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        all_combinations = [dict(zip(param_names, combo)) for combo in itertools.product(*param_values)]

        # Skip already tested combinations
        tested_params = [result['params'] for result in self.results]
        remaining_combinations = [params for params in all_combinations if params not in tested_params]

        logger.info(f"Testing {len(remaining_combinations)} of {len(all_combinations)} combinations")

        # Track best parameters
        best_result = min(self.results, key=lambda x: x['rmse']) if self.results else None
        best_rmse = best_result['rmse'] if best_result else float('inf')
        best_params = best_result['params'] if best_result else None

        # Run remaining combinations
        for i, params in enumerate(remaining_combinations):
            run_id = f"run_{len(tested_params) + i + 1}"
            logger.info(f"[{run_id}] Testing parameters: {params}")

            # Train and evaluate model
            start_time = time.time()
            model = FunkSVD(**params)
            model.fit(train_data, test_data)

            test_tuples = [(row['UserId'], row['BggId'], row['Rating']) for row in test_data]
            final_rmse = model.calculate_rmse(test_tuples, model.n_factors - 1)
            train_time = time.time() - start_time

            # Record result
            result = {
                'run_id': run_id,
                'params': params,
                'rmse': final_rmse,
                'training_time': train_time
            }
            self.results.append(result)

            # Update the best parameters if improved
            if final_rmse < best_rmse:
                best_rmse = final_rmse
                best_params = params
                logger.info(f"New best: RMSE={best_rmse:.4f}")

            # Save progress after each run
            with open(self.results_path, 'w') as f:
                json.dump({'results': self.results}, f, indent=2)

        logger.info(f"Grid search complete. Best RMSE: {best_rmse:.4f}")
        logger.info(f"Best parameters: {best_params}")
        return best_params


def run_grid_search(load_previous_results=True):
    # Load data
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
    grid_search = FunkSVDGridSearch(param_grid, load_results=load_previous_results)
    return grid_search.fit(train_data, test_data)


if __name__ == "__main__":
    grid_search_best_params = run_grid_search(load_previous_results=False)
    logger.info(f"Best parameters: {grid_search_best_params}")
