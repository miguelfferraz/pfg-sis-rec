import time
from typing import Dict, List

from surprise import Dataset
from surprise.model_selection import KFold

from ..base_model import BaseModel
from .metrics import MetricsCalculator


class ModelEvaluator:
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.metrics_calc = MetricsCalculator()

    def evaluate_model(self, model: BaseModel, dataset: Dataset) -> Dict:
        kf = KFold(n_splits=self.cv_folds, random_state=self.random_state)
        fold_results = []

        for fold, (trainset, testset) in enumerate(kf.split(dataset)):
            fold_result = self._evaluate_single_fold(model, trainset, testset, fold)
            fold_results.append(fold_result)

        return self._compile_results(model, fold_results)

    def _evaluate_single_fold(self, model: BaseModel, trainset, testset, fold_num: int) -> Dict:
        start_time = time.time()
        model.fit(trainset)
        fit_time = time.time() - start_time

        start_time = time.time()
        predictions = model.test(testset)
        test_time = time.time() - start_time

        accuracy_metrics = self.metrics_calc.calculate_accuracy_metrics(predictions)

        return {
            "fold": fold_num,
            "rmse": accuracy_metrics["rmse"],
            "mae": accuracy_metrics["mae"],
            "mse": accuracy_metrics["mse"],
            "fcp": accuracy_metrics["fcp"],
            "fit_time": fit_time,
            "test_time": test_time,
            "n_predictions": len(predictions)
        }

    def _compile_results(self, model: BaseModel, fold_results: List[Dict]) -> Dict:
        aggregated_metrics = self.metrics_calc.aggregate_cv_results(fold_results)

        return {
            "model_name": model.get_name(),
            "model_params": model.get_params(),
            "cv_config": {
                "n_folds": self.cv_folds,
                "random_state": self.random_state
            },
            "fold_results": fold_results,
            "aggregated_metrics": aggregated_metrics,
            "summary": {
                "rmse": aggregated_metrics["mean_rmse"],
                "mae": aggregated_metrics["mean_mae"],
                "mse": aggregated_metrics["mean_mse"],
                "fcp": aggregated_metrics["mean_fcp"],
                "total_time": sum(fold["fit_time"] + fold["test_time"] for fold in fold_results)
            }
        }
