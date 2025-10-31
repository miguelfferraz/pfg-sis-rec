import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split

from src.models.model_factory import ModelFactory
from src.stages.base_stage import BaseStage


class TrainingStage(BaseStage):
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        logger = context.get("logger")
        ratings_df = context["ratings_df"]
        training_config = self.config.get("training", {})
        output_config = self.config.get("output", {})

        rating_scale = context.get("rating_scale", (1.0, 5.0))
        reader = Reader(rating_scale=rating_scale)
        surprise_data = ratings_df[["user_id", "item_id", "rating"]].copy()
        dataset = Dataset.load_from_df(surprise_data, reader)

        models_config = training_config.get("models", [])
        validation_config = training_config.get("validation", {})

        if logger:
            logger.info(f"Training {len(models_config)} model(s)")
            logger.info(f"Validation type: {validation_config.get('type', 'simple')}")

        all_predictions = []
        training_results = {}

        for model_config in models_config:
            model_name = model_config["name"]
            model_params = model_config.get("params", {})

            if logger:
                logger.info(f"Training model: {model_name}")

            model = ModelFactory.create_model(model_name, **model_params)

            start_time = time.time()
            
            if validation_config.get("type") == "cross_validate":
                predictions, results = self._cross_validate(model, dataset, validation_config, model_name, logger)
            elif validation_config.get("type") == "train_test_split":
                predictions, results = self._train_test_split(model, dataset, validation_config, model_name, logger)
            else:
                predictions, results = self._simple_train(model, dataset, model_name, logger)

            training_time = time.time() - start_time
            results["training_time"] = training_time

            all_predictions.extend(predictions)
            training_results[model_name] = results

            if logger:
                logger.info(f"Model {model_name} completed in {training_time:.2f}s")

        predictions_df = pd.DataFrame(all_predictions)
        context["predictions_df"] = predictions_df
        context["training_results"] = training_results

        save_path = output_config.get("save_path")
        if save_path:
            self._save_results(save_path, training_results, predictions_df)

        return context

    def _cross_validate(
        self, model, dataset, validation_config: Dict[str, Any], model_name: str, logger=None
    ) -> tuple[List[Dict], Dict]:
        cv = validation_config.get("cv", 5)
        metrics = validation_config.get("metrics", ["rmse", "mae"])

        if logger:
            logger.info(f"Running {cv}-fold cross-validation")

        cv_results = cross_validate(model.algorithm, dataset, measures=metrics, cv=cv, return_train_measures=True)

        predictions = []
        fold = 0
        from surprise.model_selection import KFold

        kf = KFold(n_splits=cv)
        for trainset, testset in kf.split(dataset):
            fold_start = time.time()
            
            model_instance = ModelFactory.create_model(model_name, **model.get_params())
            model_instance.fit(trainset)
            fold_predictions = model_instance.test(testset)

            for pred in fold_predictions:
                predictions.append(
                    {
                        "user_id": pred.uid,
                        "item_id": pred.iid,
                        "rating_true": pred.r_ui,
                        "rating_pred": pred.est,
                        "fold": fold,
                        "model_name": model_name,
                    }
                )
            
            if logger:
                fold_time = time.time() - fold_start
                logger.info(f"  Fold {fold + 1}/{cv} completed in {fold_time:.2f}s")
            
            fold += 1

        results = {
            "validation_type": "cross_validate",
            "cv": cv,
            "metrics": {metric: cv_results[f"test_{metric}"].tolist() for metric in metrics},
            "mean_metrics": {metric: float(cv_results[f"test_{metric}"].mean()) for metric in metrics},
        }

        return predictions, results

    def _train_test_split(
        self, model, dataset, validation_config: Dict[str, Any], model_name: str, logger=None
    ) -> tuple[List[Dict], Dict]:
        test_size = validation_config.get("test_size", 0.2)
        random_state = validation_config.get("random_state", 42)
        metrics = validation_config.get("metrics", ["rmse", "mae"])

        trainset, testset = train_test_split(dataset, test_size=test_size, random_state=random_state)

        model.fit(trainset)
        test_predictions = model.test(testset)

        predictions = []
        for pred in test_predictions:
            predictions.append(
                {
                    "user_id": pred.uid,
                    "item_id": pred.iid,
                    "rating_true": pred.r_ui,
                    "rating_pred": pred.est,
                    "fold": 0,
                    "model_name": model_name,
                }
            )

        results = {"validation_type": "train_test_split", "test_size": test_size, "metrics": {}}

        for metric_name in metrics:
            if metric_name.lower() == "rmse":
                results["metrics"]["rmse"] = accuracy.rmse(test_predictions, verbose=False)
            elif metric_name.lower() == "mae":
                results["metrics"]["mae"] = accuracy.mae(test_predictions, verbose=False)
            elif metric_name.lower() == "fcp":
                results["metrics"]["fcp"] = accuracy.fcp(test_predictions, verbose=False)

        return predictions, results

    def _simple_train(self, model, dataset, model_name: str, logger=None) -> tuple[List[Dict], Dict]:
        trainset = dataset.build_full_trainset()
        testset = trainset.build_testset()

        model.fit(trainset)
        test_predictions = model.test(testset)

        predictions = []
        for pred in test_predictions:
            predictions.append(
                {
                    "user_id": pred.uid,
                    "item_id": pred.iid,
                    "rating_true": pred.r_ui,
                    "rating_pred": pred.est,
                    "fold": 0,
                    "model_name": model_name,
                }
            )

        rmse = accuracy.rmse(test_predictions, verbose=False)
        mae = accuracy.mae(test_predictions, verbose=False)
        fcp_value = accuracy.fcp(test_predictions, verbose=False)

        results = {"validation_type": "simple_train", "metrics": {"rmse": rmse, "mae": mae, "fcp": fcp_value}}

        return predictions, results

    def _save_results(self, save_path: str, training_results: Dict, predictions_df: pd.DataFrame):
        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_results_path = output_dir / "training_results.json"
        with open(training_results_path, "w") as f:
            json.dump(training_results, f, indent=2)

        predictions_path = output_dir / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
