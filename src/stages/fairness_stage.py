import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.fairness.fairness_calculator import FairnessCalculator
from src.fairness.group_builder import GroupBuilder
from src.stages.base_stage import BaseStage


class FairnessStage(BaseStage):
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        fairness_config = self.config.get("fairness")
        if not fairness_config:
            return context

        predictions_df = context.get("predictions_df")
        users_df = context.get("users_df")
        items_df = context.get("items_df")
        loader = context.get("loader")
        output_config = self.config.get("output", {})

        if predictions_df is None or predictions_df.empty:
            return context

        positive_threshold = fairness_config.get("positive_threshold", 4.0)
        calculator = FairnessCalculator(positive_threshold=positive_threshold)

        fairness_results = {}

        model_names = predictions_df["model_name"].unique()

        for model_name in model_names:
            model_predictions = predictions_df[predictions_df["model_name"] == model_name]

            model_results = {
                "overall_metrics": self._calculate_overall_metrics(model_predictions),
                "by_fold": {},
                "aggregated": {"user_fairness": {}, "item_fairness": {}},
            }

            user_attributes = fairness_config.get("user_attributes", [])
            for attr_config in user_attributes:
                attribute_name = attr_config["name"]

                group_builder = GroupBuilder(attr_config, loader, entity_type="user")
                predictions_with_groups = self._merge_groups(model_predictions, users_df, group_builder, "user_id")

                if predictions_with_groups.empty:
                    continue

                group_column = f"group_{attribute_name}"
                group_labels = group_builder.get_group_labels(predictions_with_groups)

                fold_results = []
                for fold_id in sorted(predictions_with_groups["fold"].unique()):
                    fold_predictions = predictions_with_groups[predictions_with_groups["fold"] == fold_id]
                    fold_result = calculator.calculate_for_fold(fold_predictions, group_column, group_labels)

                    if str(fold_id) not in model_results["by_fold"]:
                        model_results["by_fold"][str(fold_id)] = {"user_fairness": {}, "item_fairness": {}}

                    model_results["by_fold"][str(fold_id)]["user_fairness"][attribute_name] = fold_result
                    fold_results.append(fold_result)

                aggregated_result = calculator.aggregate_folds(fold_results)
                model_results["aggregated"]["user_fairness"][attribute_name] = aggregated_result

            item_attributes = fairness_config.get("item_attributes", [])
            for attr_config in item_attributes:
                attribute_name = attr_config["name"]

                group_builder = GroupBuilder(attr_config, loader, entity_type="item")
                predictions_with_groups = self._merge_groups(model_predictions, items_df, group_builder, "item_id")

                if predictions_with_groups.empty:
                    continue

                group_column = f"group_{attribute_name}"
                group_labels = group_builder.get_group_labels(predictions_with_groups)

                fold_results = []
                for fold_id in sorted(predictions_with_groups["fold"].unique()):
                    fold_predictions = predictions_with_groups[predictions_with_groups["fold"] == fold_id]
                    fold_result = calculator.calculate_for_fold(fold_predictions, group_column, group_labels)

                    if str(fold_id) not in model_results["by_fold"]:
                        model_results["by_fold"][str(fold_id)] = {"user_fairness": {}, "item_fairness": {}}

                    model_results["by_fold"][str(fold_id)]["item_fairness"][attribute_name] = fold_result
                    fold_results.append(fold_result)

                aggregated_result = calculator.aggregate_folds(fold_results)
                model_results["aggregated"]["item_fairness"][attribute_name] = aggregated_result

            fairness_results[model_name] = model_results

        context["fairness_results"] = fairness_results

        save_path = output_config.get("save_path")
        if save_path:
            self._save_results(save_path, fairness_results)

        return context

    def _merge_groups(
        self, predictions_df: pd.DataFrame, entity_df: pd.DataFrame, group_builder: GroupBuilder, id_column: str
    ) -> pd.DataFrame:
        entity_with_groups = group_builder.build_groups(entity_df)
        group_column = f"group_{group_builder.attribute_name}"

        if group_column not in entity_with_groups.columns:
            return pd.DataFrame()

        merge_columns = [id_column, group_column]

        predictions_copy = predictions_df.copy()
        entity_copy = entity_with_groups[merge_columns].copy()

        predictions_copy[id_column] = predictions_copy[id_column].astype(str)
        entity_copy[id_column] = entity_copy[id_column].astype(str)

        predictions_with_groups = predictions_copy.merge(entity_copy, on=id_column, how="left")

        return predictions_with_groups

    def _calculate_overall_metrics(self, predictions_df: pd.DataFrame) -> Dict[str, float]:
        y_true = predictions_df["rating_true"].values
        y_pred = predictions_df["rating_pred"].values

        mse = ((y_true - y_pred) ** 2).mean()
        rmse = mse**0.5
        mae = abs(y_true - y_pred).mean()

        return {"rmse": float(rmse), "mae": float(mae)}

    def _save_results(self, save_path: str, fairness_results: Dict):
        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        fairness_path = output_dir / "fairness_results.json"
        with open(fairness_path, "w") as f:
            json.dump(fairness_results, f, indent=2)
