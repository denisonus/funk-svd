import concurrent.futures
import itertools
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple

from loguru import logger

from src.config import GRID_SEARCH_CONFIG
from src.data.load_dataset import get_train_data, get_test_data
from src.models.evaluation import evaluate_recommendations
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
    model.fit(train_data, test_data)

    # Traditional error metrics for rating prediction
    test_rmse = model.calculate_rmse(test_tuples, model.n_factors - 1)
    test_mae = model.calculate_mae(test_tuples, model.n_factors - 1)
    train_time = time.time() - start_time

    # Calculate recommendation metrics (Precision, Recall, NDCG, Coverage)
    recommendation_metrics = evaluate_recommendations(model, test_data)

    # Save model to a temp file
    model_path = f"temp_model_{run_id}"
    model.save_model(model_path)

    # Merge all metrics together for result
    result = {'run_id': run_id, 'params': params, 'test_rmse': test_rmse, 'test_mae': test_mae,
              'training_time': train_time, 'model_path': model_path}

    # Add recommendation metrics to result
    for metric_name, values_by_k in recommendation_metrics.items():
        for k, value in values_by_k.items():
            result[f'{metric_name}@{k}'] = value

    return result


class FunkSVDGridSearch:
    def __init__(self, param_grid: Dict[str, List[Any]], save_path: Union[str, Path], load_results: bool,
                 evaluation_k_values: List[int], primary_metric: str) -> None:
        """Initialize grid search for FunkSVD model"""
        self.param_grid = param_grid
        self.save_path = Path(save_path)
        self.results = []
        self.results_path = self.save_path / "grid_search_results.json"
        self.best_model_path = self.save_path / "best_model"
        self.n_jobs = os.cpu_count() - 1
        self.evaluation_k_values = evaluation_k_values
        self.primary_metric = primary_metric

        # Set up metrics to track
        self.metrics_to_track = ['test_rmse', 'test_mae'] + [f'{m}@{k}' for m in
                                                             ['precision', 'recall', 'ndcg', 'coverage'] for k in
                                                             self.evaluation_k_values]

        # Initialize best results dictionary
        self.best_results = self._initialize_best_results()

        # Validate primary metric
        valid_metrics = self.metrics_to_track
        if self.primary_metric not in valid_metrics:
            logger.warning(f"Invalid primary metric: {primary_metric}. Using 'test_rmse'")
            self.primary_metric = 'test_rmse'

        # Create save directory and load existing results if needed
        self.save_path.mkdir(parents=True, exist_ok=True)

        if load_results and self.results_path.exists():
            self._load_previous_results()

    def _initialize_best_results(self) -> Dict[str, Dict[str, Any]]:
        """Initialize the dictionary to track best results for each metric"""
        return {
            metric: {'value': float('inf') if 'rmse' in metric or 'mae' in metric else float('-inf'), 'params': None}
            for metric in self.metrics_to_track}

    def _load_previous_results(self) -> None:
        """Load previous grid search results if available"""
        try:
            with open(self.results_path, 'r') as f:
                self.results = json.load(f).get('results', [])
            if self.results:
                logger.info(f"Loaded {len(self.results)} existing results")
                self._update_best_results_from_previous_runs()
        except Exception as e:
            logger.warning(f"Failed to load existing results: {e}")

    def _update_best_results_from_previous_runs(self) -> None:
        """Update best values for all metrics from previous results"""
        for metric in self.metrics_to_track:
            if any(metric in result for result in self.results):
                self._update_best_for_metric(metric, self.results)

    def _is_better_value(self, metric: str, current_value: float, best_value: float) -> bool:
        """Check if current metric value is better than previous best"""
        if 'rmse' in metric or 'mae' in metric:  # Lower is better
            return current_value < best_value
        else:  # Higher is better
            return current_value > best_value

    def _update_best_for_metric(self, metric: str, results: List[Dict[str, Any]]) -> None:
        """Update best value for a specific metric"""
        relevant_results = [r for r in results if metric in r]
        if not relevant_results:
            return

        if 'rmse' in metric or 'mae' in metric:  # Lower is better
            best_result = min(relevant_results, key=lambda x: x.get(metric, float('inf')))
        else:  # Higher is better
            best_result = max(relevant_results, key=lambda x: x.get(metric, float('-inf')))

        self.best_results[metric] = {'value': best_result[metric], 'params': best_result['params']}

    def _update_best_from_result(self, result: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Update best metrics from a single result
        Returns: (is_primary_improved, best_primary_value)
        """
        is_primary_improved = False
        primary_value = self.best_results[self.primary_metric]['value']

        # Check if this result improves the primary metric
        if self.primary_metric in result:
            current_value = result[self.primary_metric]
            if self._is_better_value(self.primary_metric, current_value, primary_value):
                self.best_results[self.primary_metric]['value'] = current_value
                self.best_results[self.primary_metric]['params'] = result['params']
                primary_value = current_value
                is_primary_improved = True

        # Update best results for all other metrics
        for metric in self.metrics_to_track:
            if metric == self.primary_metric or metric not in result:
                continue

            current_value = result[metric]
            best_value = self.best_results[metric]['value']

            if self._is_better_value(metric, current_value, best_value):
                self.best_results[metric] = {'value': current_value, 'params': result['params']}

        return is_primary_improved, primary_value

    def _save_results(self) -> None:
        """Save current results to disk"""
        with open(self.results_path, 'w') as f:
            json.dump({'results': self.results}, f, indent=2)

        # Save the best results summary
        summary_metrics = {metric: {'value': values['value'], 'params': values['params']} for metric, values in
                           self.best_results.items()}

        with open(self.save_path / "best_results.json", 'w') as f:
            json.dump(summary_metrics, f, indent=2)

    def _log_best_metrics(self, result: Dict[str, Any]) -> None:
        """Log the metrics for the current best model"""
        log_metrics = [f"RMSE: {result.get('test_rmse', 'N/A'):.4f}", f"MAE: {result.get('test_mae', 'N/A'):.4f}"]
        for k in self.evaluation_k_values:
            log_metrics.extend(
                [f"P@{k}: {result.get(f'precision@{k}', 'N/A'):.4f}", f"R@{k}: {result.get(f'recall@{k}', 'N/A'):.4f}",
                 f"NDCG@{k}: {result.get(f'ndcg@{k}', 'N/A'):.4f}"])
        logger.info(", ".join(log_metrics))

    def run(self) -> Dict[str, Any]:
        """Run grid search and return best parameters"""
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        all_combinations = [dict(zip(param_names, combo)) for combo in itertools.product(*param_values)]

        # Skip already tested combinations
        tested_params = [result['params'] for result in self.results]
        remaining_combinations = [params for params in all_combinations if params not in tested_params]

        logger.info(
            f"Testing {len(remaining_combinations)} of {len(all_combinations)} combinations using {self.n_jobs} processes")
        logger.info(f"Primary optimization metric: {self.primary_metric}")

        if not remaining_combinations:
            logger.info("No new parameter combinations to test")
            return self.best_results[self.primary_metric]['params']

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all tasks
            all_futures = [executor.submit(_evaluate_params_worker, params, f"run_{len(tested_params) + i + 1}") for
                           i, params in enumerate(remaining_combinations)]

            for future in concurrent.futures.as_completed(all_futures):
                model_path = None
                try:
                    result = future.result()
                    model_path = result.pop('model_path', None)
                    self.results.append(result)

                    # Update the best metrics and check if primary metric improved
                    is_primary_improved, best_primary_value = self._update_best_from_result(result)

                    # Save best model if primary metric improved
                    if is_primary_improved and model_path and os.path.exists(model_path):
                        best_model = FunkSVD()
                        best_model.load_model(model_path)
                        best_model.save_model(self.best_model_path)
                        logger.info(f"New best {self.primary_metric}: {best_primary_value:.4f}")
                        self._log_best_metrics(result)

                except Exception as e:
                    logger.error(f"Error evaluating parameters: {e}")
                finally:
                    # Clean up temporary model file
                    if model_path and os.path.exists(model_path):
                        shutil.rmtree(model_path)

                # Save progress after each run
                self._save_results()

        logger.info(
            f"Grid search complete. Primary metric ({self.primary_metric}): {self.best_results[self.primary_metric]['value']:.4f}")

        # Return best parameters based on primary metric
        return self.best_results[self.primary_metric]['params']


if __name__ == "__main__":
    grid_search = FunkSVDGridSearch(param_grid=GRID_SEARCH_CONFIG['param_grid'],
                                    save_path=GRID_SEARCH_CONFIG['save_path'],
                                    load_results=GRID_SEARCH_CONFIG['load_results'],
                                    evaluation_k_values=GRID_SEARCH_CONFIG['evaluation_k_values'],
                                    primary_metric=GRID_SEARCH_CONFIG['primary_metric'])
    grid_search_best_params = grid_search.run()
    logger.info(f"Best parameters: {grid_search_best_params}")
