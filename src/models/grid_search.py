import concurrent.futures
import itertools
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Any, Union

from loguru import logger

from src.config import GRID_SEARCH_CONFIG
from src.data.load_dataset import get_train_data, get_test_data
from src.models.funk_svd import FunkSVD


def _evaluate_params_worker(params: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    """Worker function to evaluate a single parameter combination in a separate process"""

    logger.info(f"[{run_id}] Testing parameters: {params}")

    # Load training data and train model
    train_data = get_train_data()
    test_data = get_test_data()
    test_tuples = list(test_data[['UserId', 'BGGId', 'Rating']].itertuples(index=False, name=None))

    start_time = time.time()
    model = FunkSVD(**params)
    model.fit(train_data, None)

    # Evaluate on test data
    test_rmse = model.calculate_rmse(test_tuples, model.n_factors - 1)
    train_time = time.time() - start_time

    # Save model to a temp file
    model_path = f"temp_model_{run_id}"
    model.save_model(model_path)

    return {'run_id': run_id, 'params': params, 'test_rmse': test_rmse, 'training_time': train_time,
            'model_path': model_path}


class FunkSVDGridSearch:
    def __init__(self, param_grid: Dict[str, List[Any]], save_path: Union[str, Path], load_results: bool) -> None:
        self.param_grid = param_grid
        self.save_path = Path(save_path)
        self.results = []
        self.results_path = self.save_path / "grid_search_results.json"
        self.best_model_path = self.save_path / "best_model"
        self.n_jobs = os.cpu_count() - 1

        # Create save directory and load existing results if needed
        self.save_path.mkdir(parents=True, exist_ok=True)

        if load_results and self.results_path.exists():
            try:
                with open(self.results_path, 'r') as f:
                    self.results = json.load(f).get('results', [])
                if self.results:
                    logger.info(f"Loaded {len(self.results)} existing results")
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}")

    def run(self) -> Dict[str, Any]:
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        all_combinations = [dict(zip(param_names, combo)) for combo in itertools.product(*param_values)]

        # Skip already tested combinations
        tested_params = [result['params'] for result in self.results]
        remaining_combinations = [params for params in all_combinations if params not in tested_params]

        logger.info(
            f"Testing {len(remaining_combinations)} of {len(all_combinations)} combinations using {self.n_jobs} processes")

        # Find best result from previous runs
        best_rmse = float('inf')
        best_params = None

        if self.results:
            best_result = min(self.results, key=lambda x: x.get('test_rmse', float('inf')))
            best_rmse = best_result['test_rmse']
            best_params = best_result['params']

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all tasks
            future_to_params = {}
            for i, params in enumerate(remaining_combinations):
                run_id = f"run_{len(tested_params) + i + 1}"
                future = executor.submit(_evaluate_params_worker, params, run_id)
                future_to_params[future] = params

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_params):
                    model_path = None
                    try:
                        result = future.result()
                        model_path = result.pop('model_path', None)
                        self.results.append(result)

                        # Update the best parameters if improved
                        if result['test_rmse'] < best_rmse:
                            best_rmse = result['test_rmse']
                            best_params = result['params']

                            # Save best model
                            if model_path and os.path.exists(model_path):
                                best_model = FunkSVD()
                                best_model.load_model(model_path)
                                best_model.save_model(self.best_model_path)
                                logger.info(f"New best: Test RMSE={best_rmse:.4f}")
                    except Exception as e:
                        logger.error(f"Error evaluating parameters {future_to_params[future]}: {e}")
                    finally:
                        # Clean up temporary model file
                        if model_path and os.path.exists(model_path):
                            shutil.rmtree(model_path)

                    # Save progress after each run
                    with open(self.results_path, 'w') as f:
                        json.dump({'results': self.results}, f, indent=2)

        logger.info(f"Grid search complete. Best Test RMSE: {best_rmse:.4f}")
        return best_params


if __name__ == "__main__":
    grid_search = FunkSVDGridSearch(param_grid=GRID_SEARCH_CONFIG['param_grid'],
                                    save_path=GRID_SEARCH_CONFIG['save_path'], load_results=False)
    grid_search_best_params = grid_search.run()
    logger.info(f"Best parameters: {grid_search_best_params}")

