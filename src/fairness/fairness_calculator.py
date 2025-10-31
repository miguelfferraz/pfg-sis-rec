from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equal_opportunity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    true_negative_rate,
)
from sklearn.metrics import recall_score


class FairnessCalculator:
    def __init__(self, positive_threshold: float = 4.0):
        self.positive_threshold = positive_threshold

    def calculate_for_fold(
        self, predictions_df: pd.DataFrame, group_column: str, group_labels: List[str]
    ) -> Dict[str, Any]:
        predictions_df = predictions_df.dropna(subset=[group_column])

        if len(predictions_df) == 0:
            return self._empty_results(group_labels)

        y_true = predictions_df["rating_true"].values
        y_pred = predictions_df["rating_pred"].values
        y_true_binary = (y_true >= self.positive_threshold).astype(int)
        y_pred_binary = (y_pred >= self.positive_threshold).astype(int)
        sensitive_features = predictions_df[group_column].values

        metric_frame = self._build_metric_frame(y_true_binary, y_pred_binary, sensitive_features)

        results = {"groups": {}, "fairness_metrics": {}}

        for group_label in group_labels:
            group_mask = sensitive_features == group_label
            group_size = group_mask.sum()

            if group_size == 0:
                results["groups"][group_label] = self._empty_group_result()
                continue

            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]

            results["groups"][group_label] = {
                "count": int(group_size),
                "accuracy_metrics": self._calculate_accuracy_metrics(y_true_group, y_pred_group),
                "classification_metrics": self._extract_classification_metrics_for_group(metric_frame, group_label),
            }

        results["fairness_metrics"] = self._calculate_fairness_metrics(
            y_true_binary, y_pred_binary, sensitive_features, metric_frame
        )
        results["disparities"] = self._calculate_disparities(results["groups"])

        return results

    def _calculate_accuracy_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))

        return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae)}

    def _build_metric_frame(
        self, y_true_binary: np.ndarray, y_pred_binary: np.ndarray, sensitive_features: np.ndarray
    ) -> MetricFrame:
        return MetricFrame(
            metrics={
                "tpr": recall_score,
                "tnr": true_negative_rate,
                "fpr": false_positive_rate,
                "fnr": false_negative_rate,
                "selection_rate": selection_rate,
            },
            y_true=y_true_binary,
            y_pred=y_pred_binary,
            sensitive_features=sensitive_features,
        )

    def _extract_classification_metrics_for_group(
        self, metric_frame: MetricFrame, group_label: str
    ) -> Dict[str, float]:
        try:
            by_group = metric_frame.by_group
            return {
                "tpr": float(by_group.loc[group_label, "tpr"]),
                "tnr": float(by_group.loc[group_label, "tnr"]),
                "fpr": float(by_group.loc[group_label, "fpr"]),
                "fnr": float(by_group.loc[group_label, "fnr"]),
            }
        except (KeyError, ValueError):
            return {"tpr": None, "tnr": None, "fpr": None, "fnr": None}

    def _calculate_fairness_metrics(
        self,
        y_true_binary: np.ndarray,
        y_pred_binary: np.ndarray,
        sensitive_features: np.ndarray,
        metric_frame: MetricFrame,
    ) -> Dict[str, Any]:
        try:
            by_group = metric_frame.by_group

            eq_opp_ratio = equal_opportunity_ratio(y_true_binary, y_pred_binary, sensitive_features=sensitive_features)
            eq_opp_diff = equal_opportunity_difference(
                y_true_binary, y_pred_binary, sensitive_features=sensitive_features
            )

            eq_odds_ratio = equalized_odds_ratio(y_true_binary, y_pred_binary, sensitive_features=sensitive_features)
            eq_odds_diff = equalized_odds_difference(
                y_true_binary, y_pred_binary, sensitive_features=sensitive_features
            )

            dem_parity_ratio = demographic_parity_ratio(
                y_true_binary, y_pred_binary, sensitive_features=sensitive_features
            )
            dem_parity_diff = demographic_parity_difference(
                y_true_binary, y_pred_binary, sensitive_features=sensitive_features
            )

            fairness_metrics = {
                "equal_opportunity": {
                    "ratio": float(eq_opp_ratio) if eq_opp_ratio is not None else None,
                    "difference": float(eq_opp_diff) if eq_opp_diff is not None else None,
                },
                "equalized_odds": {
                    "ratio": float(eq_odds_ratio) if eq_odds_ratio is not None else None,
                    "difference": float(eq_odds_diff) if eq_odds_diff is not None else None,
                    "tpr_ratio": self._calculate_ratio(by_group["tpr"]),
                    "tpr_difference": self._calculate_difference(by_group["tpr"]),
                    "fpr_ratio": self._calculate_ratio(by_group["fpr"]),
                    "fpr_difference": self._calculate_difference(by_group["fpr"]),
                },
                "demographic_parity": {
                    "ratio": float(dem_parity_ratio) if dem_parity_ratio is not None else None,
                    "difference": float(dem_parity_diff) if dem_parity_diff is not None else None,
                },
            }

            return fairness_metrics

        except Exception:
            return {
                "equal_opportunity": {"ratio": None, "difference": None},
                "equalized_odds": {
                    "ratio": None,
                    "difference": None,
                    "tpr_ratio": None,
                    "tpr_difference": None,
                    "fpr_ratio": None,
                    "fpr_difference": None,
                },
                "demographic_parity": {"ratio": None, "difference": None},
            }

    def _calculate_ratio(self, values: pd.Series) -> float:
        values_clean = values.dropna()
        if len(values_clean) < 2 or values_clean.max() == 0:
            return None
        return float(values_clean.min() / values_clean.max())

    def _calculate_difference(self, values: pd.Series) -> float:
        values_clean = values.dropna()
        if len(values_clean) < 2:
            return None
        return float(values_clean.max() - values_clean.min())

    def _calculate_disparities(self, groups: Dict[str, Any]) -> Dict[str, Any]:
        disparities = {}

        metrics_to_check = ["mse", "rmse", "mae"]
        for metric in metrics_to_check:
            values = []
            for group_data in groups.values():
                if group_data["accuracy_metrics"] is not None:
                    values.append(group_data["accuracy_metrics"][metric])

            if len(values) >= 2:
                disparities[metric] = {"ratio": max(values) / min(values), "difference": max(values) - min(values)}
            else:
                disparities[metric] = {"ratio": None, "difference": None}

        return disparities

    def _empty_group_result(self) -> Dict[str, Any]:
        return {"count": 0, "accuracy_metrics": None, "classification_metrics": None}

    def _empty_results(self, group_labels: List[str]) -> Dict[str, Any]:
        return {
            "groups": {label: self._empty_group_result() for label in group_labels},
            "fairness_metrics": {
                "equal_opportunity": {"ratio": None, "difference": None},
                "equalized_odds": {
                    "ratio": None,
                    "difference": None,
                    "tpr_ratio": None,
                    "tpr_difference": None,
                    "fpr_ratio": None,
                    "fpr_difference": None,
                },
                "demographic_parity": {"ratio": None, "difference": None},
            },
            "disparities": {},
        }

    def aggregate_folds(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not fold_results:
            return {}

        group_labels = list(fold_results[0]["groups"].keys())

        aggregated = {"groups": {}, "fairness_metrics": {}, "disparities": {}}

        for group_label in group_labels:
            aggregated["groups"][group_label] = self._aggregate_group_metrics(fold_results, group_label)

        aggregated["fairness_metrics"] = self._aggregate_fairness_metrics(fold_results)
        aggregated["disparities"] = self._aggregate_disparities(fold_results)

        return aggregated

    def _aggregate_group_metrics(self, fold_results: List[Dict[str, Any]], group_label: str) -> Dict[str, Any]:
        counts = []
        accuracy_metrics = {"mse": [], "rmse": [], "mae": []}
        classification_metrics = {"tpr": [], "tnr": [], "fpr": [], "fnr": []}

        for fold_result in fold_results:
            group_data = fold_result["groups"].get(group_label, {})
            if group_data.get("count", 0) > 0:
                counts.append(group_data["count"])

                if group_data["accuracy_metrics"]:
                    for metric in accuracy_metrics:
                        accuracy_metrics[metric].append(group_data["accuracy_metrics"][metric])

                if group_data["classification_metrics"]:
                    for metric in classification_metrics:
                        value = group_data["classification_metrics"][metric]
                        if value is not None:
                            classification_metrics[metric].append(value)

        return {
            "count": int(np.mean(counts)) if counts else 0,
            "accuracy_metrics": (
                {
                    metric: {"mean": float(np.mean(values)), "std": float(np.std(values))}
                    for metric, values in accuracy_metrics.items()
                    if values
                }
                if any(accuracy_metrics.values())
                else None
            ),
            "classification_metrics": (
                {
                    metric: {"mean": float(np.mean(values)), "std": float(np.std(values))}
                    for metric, values in classification_metrics.items()
                    if values
                }
                if any(classification_metrics.values())
                else None
            ),
        }

    def _aggregate_fairness_metrics(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        aggregated = {}

        for metric_name in ["equal_opportunity", "demographic_parity"]:
            ratios = []
            differences = []

            for fold_result in fold_results:
                metric_data = fold_result["fairness_metrics"].get(metric_name, {})
                if metric_data.get("ratio") is not None:
                    ratios.append(metric_data["ratio"])
                if metric_data.get("difference") is not None:
                    differences.append(metric_data["difference"])

            aggregated[metric_name] = {
                "ratio": {"mean": float(np.mean(ratios)), "std": float(np.std(ratios))} if ratios else None,
                "difference": (
                    {"mean": float(np.mean(differences)), "std": float(np.std(differences))} if differences else None
                ),
            }

        eq_odds_data = {
            "ratio": [],
            "difference": [],
            "tpr_ratio": [],
            "tpr_difference": [],
            "fpr_ratio": [],
            "fpr_difference": [],
        }
        for fold_result in fold_results:
            eq_odds = fold_result["fairness_metrics"].get("equalized_odds", {})
            for key in eq_odds_data:
                if eq_odds.get(key) is not None:
                    eq_odds_data[key].append(eq_odds[key])

        aggregated["equalized_odds"] = {
            key: {"mean": float(np.mean(values)), "std": float(np.std(values))} if values else None
            for key, values in eq_odds_data.items()
        }

        return aggregated

    def _aggregate_disparities(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        aggregated = {}

        metrics = ["mse", "rmse", "mae"]
        for metric in metrics:
            ratios = []
            differences = []

            for fold_result in fold_results:
                disparity_data = fold_result["disparities"].get(metric, {})
                if disparity_data.get("ratio") is not None:
                    ratios.append(disparity_data["ratio"])
                if disparity_data.get("difference") is not None:
                    differences.append(disparity_data["difference"])

            aggregated[metric] = {
                "ratio": {"mean": float(np.mean(ratios)), "std": float(np.std(ratios))} if ratios else None,
                "difference": (
                    {"mean": float(np.mean(differences)), "std": float(np.std(differences))} if differences else None
                ),
            }

        return aggregated
