from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .demographic_analyzer import DemographicAnalyzer
from .fairness_metrics import FairnessMetricsCalculator


class FairnessEvaluator:
    def __init__(self, sensitive_features: List[str], rating_threshold: float = 4.0):
        self.sensitive_features = sensitive_features
        self.rating_threshold = rating_threshold

        self.demographic_analyzer = DemographicAnalyzer()
        self.metrics_calculator = FairnessMetricsCalculator(rating_threshold)

    def evaluate_model_fairness(
        self, predictions: List, user_demographics: pd.DataFrame, recommendations: Optional[Dict[int, List[int]]] = None
    ) -> Dict[str, Any]:
        processed_demographics = self.demographic_analyzer.process_demographics(user_demographics)

        prediction_data = self._extract_prediction_data(predictions)

        aligned_data = self._align_demographic_data(prediction_data, processed_demographics)

        fairness_metrics = self._calculate_fairness_metrics(aligned_data, recommendations)

        demographic_stats = self.demographic_analyzer.get_group_statistics(processed_demographics)

        minority_groups = self.demographic_analyzer.identify_minority_groups(processed_demographics)

        results = {
            "fairness_metrics": fairness_metrics,
            "demographic_statistics": demographic_stats,
            "minority_groups": minority_groups,
            "evaluation_config": {
                "sensitive_features": self.sensitive_features,
                "rating_threshold": self.rating_threshold,
                "total_predictions": len(predictions),
                "total_users": len(processed_demographics),
            },
            "data_alignment": {
                "users_with_demographics": len(aligned_data),
                "coverage_percentage": len(aligned_data) / len(predictions) * 100,
            },
        }

        results["executive_summary"] = self._generate_executive_summary(results)

        return results

    def _extract_prediction_data(self, predictions: List) -> pd.DataFrame:
        data = []
        for pred in predictions:
            data.append({"user_id": pred.uid, "item_id": pred.iid, "rating_true": pred.r_ui, "rating_pred": pred.est})

        return pd.DataFrame(data)

    def _align_demographic_data(self, prediction_data: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame:
        prediction_data["user_id"] = prediction_data["user_id"].astype(str)
        demographics["user_id"] = demographics["user_id"].astype(str)

        aligned = prediction_data.merge(demographics[["user_id"] + self.sensitive_features], on="user_id", how="inner")

        return aligned

    def _calculate_fairness_metrics(
        self, aligned_data: pd.DataFrame, recommendations: Optional[Dict[int, List[int]]]
    ) -> Dict[str, Any]:
        y_true = aligned_data["rating_true"].values
        y_pred = aligned_data["rating_pred"].values

        # Cria DataFrame de features sensíveis
        sensitive_features_df = aligned_data[["user_id"] + self.sensitive_features].copy()

        # Calcula métricas abrangentes
        fairness_metrics = self.metrics_calculator.calculate_comprehensive_fairness(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features_df,
            recommendations=recommendations,
            user_demographics=aligned_data if recommendations else None,
        )

        return fairness_metrics

    def _generate_executive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        summary = {
            "overall_fairness_score": 0.0,
            "fairness_issues_detected": [],
            "recommendations": [],
            "key_findings": {},
        }

        fairness_metrics = results["fairness_metrics"]

        for feature in self.sensitive_features:
            feature_issues = []

            demo_parity_key = f"demographic_parity_{feature}"
            if demo_parity_key in fairness_metrics:
                demo_diff = fairness_metrics[demo_parity_key].get("demographic_parity_difference", 0)
                if abs(demo_diff) > 0.1:  # Threshold de 10%
                    feature_issues.append(f"Diferença significativa na paridade demográfica: {demo_diff:.3f}")

            equal_opp_key = f"equal_opportunity_{feature}"
            if equal_opp_key in fairness_metrics:
                eq_odds_diff = fairness_metrics[equal_opp_key].get("equalized_odds_difference", 0)
                if abs(eq_odds_diff) > 0.1:  # Threshold de 10%
                    feature_issues.append(f"Diferença significativa na igualdade de oportunidades: {eq_odds_diff:.3f}")

            calibration_key = f"calibration_{feature}"
            if calibration_key in fairness_metrics:
                max_rmse_diff = fairness_metrics[calibration_key].get("max_rmse_difference", 0)
                if max_rmse_diff > 0.2:  # Threshold de 0.2 RMSE
                    feature_issues.append(f"Diferença significativa na calibração (RMSE): {max_rmse_diff:.3f}")

            if feature_issues:
                summary["fairness_issues_detected"].extend([f"{feature}: {issue}" for issue in feature_issues])

            summary["key_findings"][feature] = {"issues_count": len(feature_issues), "issues": feature_issues}

        total_issues = len(summary["fairness_issues_detected"])
        max_possible_issues = len(self.sensitive_features) * 3
        summary["overall_fairness_score"] = max(0, 1 - (total_issues / max_possible_issues))

        if total_issues == 0:
            summary["recommendations"].append("Modelo apresenta boa equidade geral")
        else:
            summary["recommendations"].append("Considere técnicas de mitigação de viés")
            summary["recommendations"].append("Analise distribuição de dados de treinamento")
            if total_issues > max_possible_issues * 0.5:
                summary["recommendations"].append("Considere re-treinamento com técnicas de fairness")

        return summary

    def generate_fairness_report(self, results: Dict[str, Any], output_path: Optional[Path] = None) -> str:
        report_lines = []

        report_lines.append("=" * 80)
        report_lines.append("RELATÓRIO DE AVALIAÇÃO DE FAIRNESS")
        report_lines.append("=" * 80)
        report_lines.append("")

        config = results["evaluation_config"]
        report_lines.append("CONFIGURAÇÃO:")
        report_lines.append(f"  Features Sensíveis: {', '.join(config['sensitive_features'])}")
        report_lines.append(f"  Threshold de Rating: {config['rating_threshold']}")
        report_lines.append(f"  Total de Predições: {config['total_predictions']}")
        report_lines.append(f"  Total de Usuários: {config['total_users']}")
        report_lines.append("")

        summary = results["executive_summary"]
        report_lines.append("RESUMO EXECUTIVO:")
        report_lines.append(f"  Score Geral de Fairness: {summary['overall_fairness_score']:.3f}")
        report_lines.append(f"  Problemas Detectados: {len(summary['fairness_issues_detected'])}")
        report_lines.append("")

        if summary["fairness_issues_detected"]:
            report_lines.append("PROBLEMAS DE FAIRNESS IDENTIFICADOS:")
            for issue in summary["fairness_issues_detected"]:
                report_lines.append(f"  • {issue}")
            report_lines.append("")

        report_lines.append("RECOMENDAÇÕES:")
        for rec in summary["recommendations"]:
            report_lines.append(f"  • {rec}")
        report_lines.append("")

        demo_stats = results["demographic_statistics"]
        report_lines.append("ESTATÍSTICAS DEMOGRÁFICAS:")
        for feature, stats in demo_stats.items():
            if isinstance(stats, dict) and not feature.endswith("_cross"):
                report_lines.append(f"  {feature.upper()}:")
                for group, count in stats.items():
                    percentage = count / config["total_users"] * 100
                    report_lines.append(f"    {group}: {count} ({percentage:.1f}%)")
                report_lines.append("")

        report_text = "\n".join(report_lines)

        if output_path:
            output_path.write_text(report_text, encoding="utf-8")

        return report_text
