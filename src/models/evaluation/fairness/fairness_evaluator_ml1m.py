from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .demographic_analyzer import DemographicAnalyzer
from .demographic_analyzer_ml1m import DemographicAnalyzerML1M
from .fairness_metrics import FairnessMetricsCalculator


class FairnessEvaluatorML1M:
    """
    Avaliador de fairness específico para MovieLens 1M.
    Usa o DemographicAnalyzerML1M para processar dados demográficos.
    """
    
    def __init__(self, sensitive_features: List[str], rating_threshold: float = 4.0, use_ml1m_analyzer: bool = True):
        self.sensitive_features = sensitive_features
        self.rating_threshold = rating_threshold
        self.use_ml1m_analyzer = use_ml1m_analyzer

        # Escolhe o analisador apropriado
        if use_ml1m_analyzer:
            self.demographic_analyzer = DemographicAnalyzerML1M()
        else:
            self.demographic_analyzer = DemographicAnalyzer()
            
        self.metrics_calculator = FairnessMetricsCalculator(rating_threshold)

    def evaluate_model_fairness(
        self, predictions: List, user_demographics: pd.DataFrame, recommendations: Optional[Dict[int, List[int]]] = None
    ) -> Dict[str, Any]:
        """Avalia fairness do modelo usando dados específicos do ML-1M"""
        
        # Processa demografia usando analisador apropriado
        if self.use_ml1m_analyzer:
            processed_demographics = self.demographic_analyzer.process_demographics_ml1m(user_demographics)
        else:
            processed_demographics = self.demographic_analyzer.process_demographics(user_demographics)

        # Extrai dados das predições
        prediction_data = self._extract_prediction_data(predictions)

        # Alinha dados demográficos com predições
        aligned_data = self._align_demographic_data(prediction_data, processed_demographics)

        # Calcula métricas de fairness
        fairness_metrics = self._calculate_fairness_metrics(aligned_data, recommendations)

        # Estatísticas demográficas
        demographic_stats = self.demographic_analyzer.get_group_statistics(processed_demographics)

        # Identifica grupos minoritários
        minority_groups = self.demographic_analyzer.identify_minority_groups(processed_demographics)

        # Gera resumo executivo
        executive_summary = self._generate_executive_summary(fairness_metrics, minority_groups)

        return {
            "fairness_metrics": fairness_metrics,
            "demographic_statistics": demographic_stats,
            "minority_groups": minority_groups,
            "executive_summary": executive_summary,
            "evaluation_config": {
                "sensitive_features": self.sensitive_features,
                "rating_threshold": self.rating_threshold,
                "total_predictions": len(predictions),
                "total_users": len(processed_demographics),
                "use_ml1m_analyzer": self.use_ml1m_analyzer
            },
            "data_alignment": {
                "users_with_demographics": len(aligned_data),
                "coverage_percentage": (len(aligned_data) / len(prediction_data)) * 100 if len(prediction_data) > 0 else 0
            },
            "evaluation_status": "completed_full",
            "note": "Avaliação completa com métricas de fairness detalhadas usando analisador ML-1M"
        }

    def _extract_prediction_data(self, predictions: List) -> pd.DataFrame:
        """Extrai dados das predições em formato DataFrame"""
        
        prediction_records = []
        for pred in predictions:
            prediction_records.append({
                "user_id": pred.uid,
                "item_id": pred.iid,
                "true_rating": pred.r_ui,
                "predicted_rating": pred.est
            })

        return pd.DataFrame(prediction_records)

    def _align_demographic_data(self, prediction_data: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame:
        """Alinha dados de predição com dados demográficos"""
        
        # Converte user_id para int se necessário
        prediction_data["user_id"] = prediction_data["user_id"].astype(int)
        demographics["user_id"] = demographics["user_id"].astype(int)

        # Merge dos dados
        aligned_data = prediction_data.merge(demographics, on="user_id", how="inner")

        return aligned_data

    def _calculate_fairness_metrics(self, aligned_data: pd.DataFrame, recommendations: Optional[Dict[int, List[int]]] = None) -> Dict[str, Any]:
        """Calcula métricas de fairness para features sensíveis"""
        
        y_true = aligned_data["true_rating"].values
        y_pred = aligned_data["predicted_rating"].values

        # Cria DataFrame com features sensíveis
        sensitive_features_df = aligned_data[["user_id"] + self.sensitive_features].copy()

        # Calcula métricas abrangentes
        fairness_metrics = self.metrics_calculator.calculate_comprehensive_fairness(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features_df,
            recommendations=recommendations,
            user_demographics=aligned_data if recommendations else None
        )

        return fairness_metrics

    def _generate_executive_summary(self, fairness_metrics: Dict[str, Any], minority_groups: Dict[str, List[str]]) -> Dict[str, Any]:
        """Gera resumo executivo da avaliação de fairness"""
        
        fairness_issues = []
        fairness_threshold = 0.1

        # Analisa cada feature sensível
        for feature in self.sensitive_features:
            feature_issues = []

            # Demographic Parity
            dp_key = f"demographic_parity_{feature}"
            if dp_key in fairness_metrics:
                dp_diff = abs(fairness_metrics[dp_key].get("demographic_parity_difference", 0))
                if dp_diff >= fairness_threshold:
                    feature_issues.append(f"Demographic parity violation: {dp_diff:.3f}")

            # Equal Opportunity
            eo_key = f"equal_opportunity_{feature}"
            if eo_key in fairness_metrics:
                eo_diff = abs(fairness_metrics[eo_key].get("equalized_odds_difference", 0))
                if eo_diff >= fairness_threshold:
                    feature_issues.append(f"Equal opportunity violation: {eo_diff:.3f}")

            if feature_issues:
                fairness_issues.append({
                    "feature": feature,
                    "issues": feature_issues
                })

        # Calcula score geral de fairness
        total_violations = sum(len(issue["issues"]) for issue in fairness_issues)
        max_possible_violations = len(self.sensitive_features) * 2  # DP + EO para cada feature
        fairness_score = max(0, 1 - (total_violations / max_possible_violations))

        # Gera recomendações
        recommendations = []
        if fairness_score == 1.0:
            recommendations.append("Modelo apresenta boa equidade geral")
        elif fairness_score >= 0.8:
            recommendations.append("Modelo apresenta equidade aceitável com pequenos ajustes necessários")
        elif fairness_score >= 0.6:
            recommendations.append("Modelo requer melhorias significativas de equidade")
        else:
            recommendations.append("Modelo apresenta sérios problemas de equidade - revisão completa necessária")

        if minority_groups:
            recommendations.append("Atenção especial necessária para grupos minoritários identificados")

        return {
            "overall_fairness_score": fairness_score,
            "fairness_issues_detected": fairness_issues,
            "recommendations": recommendations,
            "detailed_analysis": {
                feature: {
                    "issues_count": len([issue for issue in fairness_issues if issue["feature"] == feature]),
                    "issues": [issue["issues"] for issue in fairness_issues if issue["feature"] == feature]
                } for feature in self.sensitive_features
            }
        }

    def generate_fairness_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Gera relatório textual de fairness"""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("RELATÓRIO DE FAIRNESS - MOVIELENS 1M")
        report_lines.append("=" * 80)

        # Configuração
        config = evaluation_results.get("evaluation_config", {})
        report_lines.append(f"\nConfiguração:")
        report_lines.append(f"  Features Sensíveis: {', '.join(config.get('sensitive_features', []))}")
        report_lines.append(f"  Threshold de Rating: {config.get('rating_threshold', 'N/A')}")
        report_lines.append(f"  Total de Predições: {config.get('total_predictions', 'N/A'):,}")
        report_lines.append(f"  Analisador ML-1M: {'Sim' if config.get('use_ml1m_analyzer', False) else 'Não'}")

        # Resumo executivo
        summary = evaluation_results.get("executive_summary", {})
        report_lines.append(f"\nResumo Executivo:")
        report_lines.append(f"  Score de Fairness: {summary.get('overall_fairness_score', 'N/A'):.3f}")
        report_lines.append(f"  Issues Detectados: {len(summary.get('fairness_issues_detected', []))}")

        # Métricas detalhadas
        metrics = evaluation_results.get("fairness_metrics", {})
        report_lines.append(f"\nMétricas Detalhadas:")
        
        for feature in config.get('sensitive_features', []):
            report_lines.append(f"\n  {feature.upper()}:")
            
            dp_key = f"demographic_parity_{feature}"
            if dp_key in metrics:
                dp_diff = metrics[dp_key].get("demographic_parity_difference", 0)
                status = "✅" if abs(dp_diff) < 0.1 else "❌"
                report_lines.append(f"    Demographic Parity: {dp_diff:.4f} {status}")
            
            eo_key = f"equal_opportunity_{feature}"
            if eo_key in metrics:
                eo_diff = metrics[eo_key].get("equalized_odds_difference", 0)
                status = "✅" if abs(eo_diff) < 0.1 else "❌"
                report_lines.append(f"    Equalized Odds: {eo_diff:.4f} {status}")

        # Grupos minoritários
        minority_groups = evaluation_results.get("minority_groups", {})
        if minority_groups:
            report_lines.append(f"\nGrupos Minoritários Identificados:")
            for feature, groups in minority_groups.items():
                report_lines.append(f"  {feature}: {', '.join(groups)}")

        # Recomendações
        recommendations = summary.get("recommendations", [])
        if recommendations:
            report_lines.append(f"\nRecomendações:")
            for rec in recommendations:
                report_lines.append(f"  • {rec}")

        report_lines.append("\n" + "=" * 80)

        return "\n".join(report_lines)

