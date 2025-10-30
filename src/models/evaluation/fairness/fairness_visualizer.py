from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class FairnessVisualizer:
    def __init__(self, style: str = "whitegrid", palette: str = "Set2"):
        sns.set_style(style)
        self.palette = palette
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 10

    def plot_demographic_distribution(
        self, demographics: pd.DataFrame, output_path: Optional[Path] = None
    ) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Distribuição Demográfica dos Usuários", fontsize=16, fontweight="bold")

        if "gender" in demographics.columns:
            gender_counts = demographics["gender"].value_counts()
            axes[0, 0].pie(
                gender_counts.values,
                labels=gender_counts.index,
                autopct="%1.1f%%",
                colors=sns.color_palette(self.palette, len(gender_counts)),
            )
            axes[0, 0].set_title("Distribuição por Gênero")

        if "age_group" in demographics.columns:
            sns.countplot(data=demographics, x="age_group", ax=axes[0, 1], palette=self.palette)
            axes[0, 1].set_title("Distribuição por Faixa Etária")
            axes[0, 1].tick_params(axis="x", rotation=45)

        if "occupation_category" in demographics.columns:
            sns.countplot(data=demographics, y="occupation_category", ax=axes[1, 0], palette=self.palette)
            axes[1, 0].set_title("Distribuição por Categoria Profissional")

        if "age" in demographics.columns:
            axes[1, 1].hist(demographics["age"], bins=20, alpha=0.7, color=sns.color_palette(self.palette)[0])
            axes[1, 1].set_title("Distribuição de Idade")
            axes[1, 1].set_xlabel("Idade")
            axes[1, 1].set_ylabel("Frequência")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_fairness_metrics_comparison(
        self, fairness_results: Dict[str, Any], output_path: Optional[Path] = None
    ) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Comparação de Métricas de Fairness", fontsize=16, fontweight="bold")

        metrics_data = []

        for key, value in fairness_results.items():
            if isinstance(value, dict):
                if "demographic_parity_difference" in value:
                    metrics_data.append(
                        {
                            "Feature": key.replace("demographic_parity_", ""),
                            "Metric": "Demographic Parity Diff",
                            "Value": value["demographic_parity_difference"],
                        }
                    )

                if "equalized_odds_difference" in value:
                    metrics_data.append(
                        {
                            "Feature": key.replace("equal_opportunity_", ""),
                            "Metric": "Equalized Odds Diff",
                            "Value": value["equalized_odds_difference"],
                        }
                    )

                if "max_rmse_difference" in value:
                    metrics_data.append(
                        {
                            "Feature": key.replace("calibration_", ""),
                            "Metric": "Max RMSE Diff",
                            "Value": value["max_rmse_difference"],
                        }
                    )

        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)

            sns.barplot(data=df_metrics, x="Feature", y="Value", hue="Metric", ax=axes[0, 0], palette=self.palette)
            axes[0, 0].set_title("Diferenças de Fairness por Feature")
            axes[0, 0].tick_params(axis="x", rotation=45)
            axes[0, 0].axhline(y=0, color="black", linestyle="--", alpha=0.5)

        self._plot_fairness_heatmap(fairness_results, axes[0, 1])
        self._plot_fairness_radar(fairness_results, axes[1, 0])
        self._plot_performance_distribution(fairness_results, axes[1, 1])

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_bias_detection_dashboard(
        self, fairness_results: Dict[str, Any], demographics: pd.DataFrame, output_path: Optional[Path] = None
    ) -> plt.Figure:
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        fig.suptitle("Dashboard de Detecção de Viés", fontsize=20, fontweight="bold")

        ax1 = fig.add_subplot(gs[0, :])
        self._plot_demographic_overview(demographics, ax1)

        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[1, 2])

        self._plot_parity_metrics(fairness_results, ax2)
        self._plot_opportunity_metrics(fairness_results, ax3)
        self._plot_calibration_metrics(fairness_results, ax4)

        ax5 = fig.add_subplot(gs[2, :2])
        ax6 = fig.add_subplot(gs[2, 2])

        self._plot_group_performance_comparison(fairness_results, ax5)
        self._plot_fairness_score_summary(fairness_results, ax6)

        ax7 = fig.add_subplot(gs[3, :])
        self._plot_recommendations_text(fairness_results, ax7)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        return fig

    def _plot_fairness_heatmap(self, fairness_results: Dict[str, Any], ax: plt.Axes):
        heatmap_data = {}
        features = []
        metrics = ["Demographic Parity", "Equal Opportunity", "Calibration"]

        for key, value in fairness_results.items():
            if isinstance(value, dict):
                if "demographic_parity" in key:
                    feature = key.replace("demographic_parity_", "")
                    if feature not in features:
                        features.append(feature)
                    heatmap_data[(feature, "Demographic Parity")] = abs(value.get("demographic_parity_difference", 0))
                elif "equal_opportunity" in key:
                    feature = key.replace("equal_opportunity_", "")
                    heatmap_data[(feature, "Equal Opportunity")] = abs(value.get("equalized_odds_difference", 0))
                elif "calibration" in key:
                    feature = key.replace("calibration_", "")
                    heatmap_data[(feature, "Calibration")] = value.get("max_rmse_difference", 0)

        if heatmap_data and features:
            matrix = np.zeros((len(features), len(metrics)))
            for i, feature in enumerate(features):
                for j, metric in enumerate(metrics):
                    matrix[i, j] = heatmap_data.get((feature, metric), 0)

            sns.heatmap(matrix, xticklabels=metrics, yticklabels=features, annot=True, fmt=".3f", cmap="Reds", ax=ax)
            ax.set_title("Heatmap de Problemas de Fairness")
        else:
            ax.text(0.5, 0.5, "Dados insuficientes\npara heatmap", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Heatmap de Fairness")

    def _plot_fairness_radar(self, fairness_results: Dict[str, Any], ax: plt.Axes):
        ax.text(0.5, 0.5, "Gráfico Radar\n(Em desenvolvimento)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Radar de Fairness")

    def _plot_performance_distribution(self, fairness_results: Dict[str, Any], ax: plt.Axes):
        calibration_data = []

        for key, value in fairness_results.items():
            if "calibration" in key and isinstance(value, dict):
                for metric_key, metric_value in value.items():
                    if metric_key.startswith("rmse_") and not metric_key.startswith("max_"):
                        group = metric_key.replace("rmse_", "")
                        calibration_data.append({"Group": group, "RMSE": metric_value})

        if calibration_data:
            df_cal = pd.DataFrame(calibration_data)
            sns.barplot(data=df_cal, x="Group", y="RMSE", ax=ax, palette=self.palette)
            ax.set_title("RMSE por Grupo Demográfico")
            ax.tick_params(axis="x", rotation=45)
        else:
            ax.text(0.5, 0.5, "Dados de performance\nnão disponíveis", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Performance por Grupo")

    def _plot_demographic_overview(self, demographics: pd.DataFrame, ax: plt.Axes):
        ax.text(
            0.5,
            0.5,
            f"Total de Usuários: {len(demographics)}\n" f"Features Disponíveis: {list(demographics.columns)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Overview Demográfico")
        ax.axis("off")

    def _plot_parity_metrics(self, fairness_results: Dict[str, Any], ax: plt.Axes):
        ax.text(0.5, 0.5, "Métricas de\nParidade Demográfica", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Paridade Demográfica")

    def _plot_opportunity_metrics(self, fairness_results: Dict[str, Any], ax: plt.Axes):
        ax.text(0.5, 0.5, "Métricas de\nIgualdade de Oportunidades", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Igualdade de Oportunidades")

    def _plot_calibration_metrics(self, fairness_results: Dict[str, Any], ax: plt.Axes):
        ax.text(0.5, 0.5, "Métricas de\nCalibração", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Calibração")

    def _plot_group_performance_comparison(self, fairness_results: Dict[str, Any], ax: plt.Axes):
        ax.text(
            0.5,
            0.5,
            "Comparação de Performance\nentre Grupos Demográficos",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Comparação de Performance")

    def _plot_fairness_score_summary(self, fairness_results: Dict[str, Any], ax: plt.Axes):
        ax.text(0.5, 0.5, "Score Geral\nde Fairness", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Score de Fairness")

    def _plot_recommendations_text(self, fairness_results: Dict[str, Any], ax: plt.Axes):
        ax.text(0.5, 0.5, "Recomendações para\nMelhoria de Fairness", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Recomendações")
        ax.axis("off")
