import time
from typing import Dict, List

from surprise.accuracy import mae, rmse, mse, fcp


class MetricsCalculator:
    @staticmethod
    def calculate_accuracy_metrics(predictions) -> Dict[str, float]:
        return {
            "rmse": rmse(predictions, verbose=False),
            "mae": mae(predictions, verbose=False),
            "mse": mse(predictions, verbose=False),
            "fcp": fcp(predictions, verbose=False)
        }

    @staticmethod
    def calculate_timing_metrics(fit_times: List[float], test_times: List[float]) -> Dict[str, float]:
        return {
            "mean_fit_time": sum(fit_times) / len(fit_times),
            "mean_test_time": sum(test_times) / len(test_times),
            "total_time": sum(fit_times) + sum(test_times)
        }

    @staticmethod
    def aggregate_cv_results(fold_results: List[Dict]) -> Dict[str, float]:
        metrics = ["rmse", "mae", "mse", "fcp", "fit_time", "test_time"]
        aggregated = {}
        
        for metric in metrics:
            values = [fold[metric] for fold in fold_results]
            aggregated[f"mean_{metric}"] = sum(values) / len(values)
            aggregated[f"std_{metric}"] = (sum((x - aggregated[f"mean_{metric}"]) ** 2 for x in values) / len(values)) ** 0.5
        
        return aggregated
