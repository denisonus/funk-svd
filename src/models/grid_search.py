import itertools
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Union

from loguru import logger

from src.config import GRID_SEARCH_CONFIG
from src.models.funk_svd import FunkSVD
from src.data.load_dataset import get_train_data, get_test_data


class FunkSVDGridSearch:
    def __init__(self, param_grid: Dict[str, List[Any]], save_path: Union[str, Path], load_results: bool) -> None:
        self.param_grid = param_grid
        self.save_path = Path(save_path)
        self.results: List[Dict[str, Any]] = []
        self.results_path = self.save_path / "grid_search_results.json"
        self.best_model_path = self.save_path / "best_model"

        # Create save directory if it doesn't exist
        self.save_path.mkdir(parents=True, exist_ok=True)

        if load_results and self.results_path.exists():
            try:
                with open(self.results_path, 'r') as f:
                    self.results = json.load(f).get('results', [])
                logger.info(f"Loaded {len(self.results)} existing results")
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}")

    def fit(self, train_data: Any, test_data: Any) -> Dict[str, Any]:
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
        best_model = None

        # Convert test data to list of tuples for RMSE calculation
        test_tuples = list(test_data[['UserId', 'BGGId', 'Rating']].itertuples(index=False, name=None))

        # Run remaining combinations
        for i, params in enumerate(remaining_combinations):
            run_id = f"run_{len(tested_params) + i + 1}"
            logger.info(f"[{run_id}] Testing parameters: {params}")

            # Train and evaluate model
            start_time = time.time()
            model = FunkSVD(**params)
            model.fit(train_data, test_data)

            # Use the correct format for calculate_rmse (list of tuples)
            final_rmse = model.calculate_rmse(test_tuples, model.n_factors - 1)
            train_time = time.time() - start_time

            # Record result
            result = {'run_id': run_id, 'params': params, 'rmse': final_rmse, 'training_time': train_time}
            self.results.append(result)

            # Update the best parameters if improved
            if final_rmse < best_rmse:
                best_rmse = final_rmse
                best_params = params
                best_model = model
                logger.info(f"New best: RMSE={best_rmse:.4f}")

            # Save progress after each run
            with open(self.results_path, 'w') as f:
                json.dump({'results': self.results}, f, indent=2)

        # Save the best model if found
        if best_model:
            logger.info(f"Saving best model with RMSE: {best_rmse:.4f}")
            best_model.save_model(self.best_model_path)

        logger.info(f"Grid search complete. Best RMSE: {best_rmse:.4f}")
        logger.info(f"Best parameters: {best_params}")
        return best_params


def run_grid_search(load_previous_results: bool) -> Dict[str, Any]:
    # Load data
    train_data = get_train_data()
    test_data = get_test_data()

    # Use parameters from config
    param_grid = GRID_SEARCH_CONFIG['param_grid']
    save_path = GRID_SEARCH_CONFIG['save_path']

    # Run grid search
    grid_search = FunkSVDGridSearch(param_grid, save_path=save_path, load_results=load_previous_results)
    return grid_search.fit(train_data, test_data)


if __name__ == "__main__":
    grid_search_best_params = run_grid_search(load_previous_results=True)
    logger.info(f"Best parameters: {grid_search_best_params}")
