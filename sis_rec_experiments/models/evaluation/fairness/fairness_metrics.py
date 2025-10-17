from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    selection_rate,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score


class FairnessMetricsCalculator:
    def __init__(self, rating_threshold: float = 4.0):
        self.rating_threshold = rating_threshold

    def _convert_to_binary(self, ratings: np.ndarray) -> np.ndarray:
        return (ratings >= self.rating_threshold).astype(int)

    def calculate_demographic_parity(
        self, y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: pd.Series
    ) -> Dict[str, float]:
        y_pred_binary = self._convert_to_binary(y_pred)

        return {
            "demographic_parity_difference": demographic_parity_difference(
                y_true, y_pred_binary, sensitive_features=sensitive_features
            ),
            "demographic_parity_ratio": demographic_parity_ratio(
                y_true, y_pred_binary, sensitive_features=sensitive_features
            ),
            "selection_rate_by_group": self._calculate_selection_rate_by_group(y_pred_binary, sensitive_features),
        }

    def calculate_equal_opportunity(
        self, y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: pd.Series
    ) -> Dict[str, float]:
        y_true_binary = self._convert_to_binary(y_true)
        y_pred_binary = self._convert_to_binary(y_pred)

        return {
            "equalized_odds_difference": equalized_odds_difference(
                y_true_binary, y_pred_binary, sensitive_features=sensitive_features
            ),
            "equalized_odds_ratio": equalized_odds_ratio(
                y_true_binary, y_pred_binary, sensitive_features=sensitive_features
            ),
            "true_positive_rate_by_group": self._calculate_tpr_by_group(
                y_true_binary, y_pred_binary, sensitive_features
            ),
        }

    def calculate_calibration_fairness(
        self, y_true: np.ndarray, y_pred: np.ndarray, sensitive_features: pd.Series
    ) -> Dict[str, float]:
        results = {}

        for group in sensitive_features.unique():
            mask = sensitive_features == group
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]

            # Calcula RMSE por grupo
            rmse = np.sqrt(np.mean((group_y_true - group_y_pred) ** 2))
            mae = np.mean(np.abs(group_y_true - group_y_pred))

            results[f"rmse_{group}"] = rmse
            results[f"mae_{group}"] = mae

        # Calcula diferença máxima entre grupos
        rmse_values = [v for k, v in results.items() if k.startswith("rmse_")]
        mae_values = [v for k, v in results.items() if k.startswith("mae_")]

        results["max_rmse_difference"] = max(rmse_values) - min(rmse_values)
        results["max_mae_difference"] = max(mae_values) - min(mae_values)

        return results

    def calculate_coverage_fairness(
        self, recommendations: Dict[int, List[int]], user_demographics: pd.DataFrame, item_metadata: pd.DataFrame = None
    ) -> Dict[str, Any]:
        results = {}

        # Calcula diversidade de itens recomendados por grupo
        for feature in ["gender", "age_group", "occupation_category"]:
            if feature in user_demographics.columns:
                group_diversity = {}

                for group in user_demographics[feature].unique():
                    group_users = user_demographics[user_demographics[feature] == group]["user_id"].tolist()

                    # Coleta todos os itens recomendados para este grupo
                    group_items = set()
                    for user_id in group_users:
                        if user_id in recommendations:
                            group_items.update(recommendations[user_id])

                    group_diversity[group] = len(group_items)

                results[f"item_diversity_{feature}"] = group_diversity

                # Calcula diferença de diversidade entre grupos
                diversity_values = list(group_diversity.values())
                if diversity_values:
                    results[f"diversity_difference_{feature}"] = max(diversity_values) - min(diversity_values)

        return results

    def _calculate_selection_rate_by_group(
        self, y_pred_binary: np.ndarray, sensitive_features: pd.Series
    ) -> Dict[str, float]:
        rates = {}
        for group in sensitive_features.unique():
            mask = sensitive_features == group
            group_predictions = y_pred_binary[mask]
            rates[str(group)] = np.mean(group_predictions)
        return rates

    def _calculate_tpr_by_group(
        self, y_true_binary: np.ndarray, y_pred_binary: np.ndarray, sensitive_features: pd.Series
    ) -> Dict[str, float]:
        tpr_rates = {}
        for group in sensitive_features.unique():
            mask = sensitive_features == group
            group_y_true = y_true_binary[mask]
            group_y_pred = y_pred_binary[mask]

            # Calcula TPR (recall) para o grupo
            if np.sum(group_y_true) > 0:  # Evita divisão por zero
                tpr = recall_score(group_y_true, group_y_pred, zero_division=0)
                tpr_rates[str(group)] = tpr
            else:
                tpr_rates[str(group)] = 0.0

        return tpr_rates

    def calculate_comprehensive_fairness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame,
        recommendations: Dict[int, List[int]] = None,
        user_demographics: pd.DataFrame = None,
    ) -> Dict[str, Any]:
        results = {}

        # Para cada feature sensível
        for feature in sensitive_features.columns:
            if feature != "user_id":
                feature_series = sensitive_features[feature]

                # Paridade demográfica
                demo_parity = self.calculate_demographic_parity(y_true, y_pred, feature_series)
                results[f"demographic_parity_{feature}"] = demo_parity

                # Igualdade de oportunidades
                equal_opp = self.calculate_equal_opportunity(y_true, y_pred, feature_series)
                results[f"equal_opportunity_{feature}"] = equal_opp

                # Calibração
                calibration = self.calculate_calibration_fairness(y_true, y_pred, feature_series)
                results[f"calibration_{feature}"] = calibration

        # Cobertura (se dados disponíveis)
        if recommendations is not None and user_demographics is not None:
            coverage = self.calculate_coverage_fairness(recommendations, user_demographics)
            results["coverage_fairness"] = coverage

        return results
