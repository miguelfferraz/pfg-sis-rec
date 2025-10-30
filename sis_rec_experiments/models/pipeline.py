import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..loaders.builder import create_loader
from ..preprocessing import ColdStartFilter, StatsAnalyzer
from .evaluation.evaluator import ModelEvaluator
from .evaluation.fairness import FairnessEvaluator, FairnessVisualizer
from .evaluation.fairness.fairness_evaluator_ml1m import FairnessEvaluatorML1M
from .model_factory import ModelFactory


class ModelPipeline:
    def __init__(
        self,
        output_dir: str = "sis_rec_experiments/models/results",
        apply_preprocessing: bool = False,
        preprocessing_params: Dict = None,
        evaluate_fairness: bool = False,
        fairness_config: Optional[Dict] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Configura evaluator para armazenar predições se fairness estiver habilitado
        self.evaluator = ModelEvaluator(store_predictions=evaluate_fairness)
        self.apply_preprocessing = apply_preprocessing
        self.evaluate_fairness = evaluate_fairness

        if preprocessing_params is None:
            preprocessing_params = {"min_user_ratings": 20, "min_item_ratings": 10}
        self.cold_start_filter = ColdStartFilter(**preprocessing_params)
        self.stats_analyzer = StatsAnalyzer()

        # Configuração de fairness
        if self.evaluate_fairness:
            if fairness_config is None:
                fairness_config = {
                    "sensitive_features": ["gender", "age_group", "occupation_category"],
                    "rating_threshold": 4.0,
                }
            self.fairness_config = fairness_config
            
            # Escolhe o avaliador apropriado baseado na configuração
            if fairness_config.get("use_ml1m_analyzer", False):
                self.fairness_evaluator = FairnessEvaluatorML1M(
                    sensitive_features=fairness_config["sensitive_features"],
                    rating_threshold=fairness_config.get("rating_threshold", 4.0),
                    use_ml1m_analyzer=True
                )
            else:
                self.fairness_evaluator = FairnessEvaluator(
                    sensitive_features=fairness_config["sensitive_features"],
                    rating_threshold=fairness_config.get("rating_threshold", 4.0),
                )
            self.fairness_visualizer = FairnessVisualizer()

    def run_experiment(self, dataset_name: str, model_name: str, model_params: Dict = None) -> Dict:
        if model_params is None:
            model_params = {}

        loader = create_loader(dataset_name)
        loader.load_ratings()

        original_df = loader.ratings_df.copy()

        if self.apply_preprocessing:
            filtered_df, filtering_report = self.cold_start_filter.filter_dataset(loader.ratings_df)
            loader.ratings_df = filtered_df

            print(self.stats_analyzer.generate_comparison_report(filtering_report))

            preprocessing_plot_path = self.output_dir / f"{dataset_name}_preprocessing_analysis.png"
            self.stats_analyzer.plot_filtering_impact(
                original_df, filtered_df, filtering_report, str(preprocessing_plot_path)
            )
        else:
            filtering_report = None

        surprise_dataset = loader.to_surprise_dataset()

        model = ModelFactory.create_model(model_name, **model_params)
        results = self.evaluator.evaluate_model(model, surprise_dataset)

        results["dataset_name"] = dataset_name
        results["experiment_timestamp"] = datetime.now().isoformat()
        results["preprocessing_applied"] = self.apply_preprocessing
        results["fairness_evaluated"] = self.evaluate_fairness

        if filtering_report:
            results["preprocessing_report"] = filtering_report

        # Avaliação de fairness (se habilitada)
        if self.evaluate_fairness:
            fairness_results = self._evaluate_fairness(results, dataset_name, loader)
            results["fairness_evaluation"] = fairness_results

        self._save_results(results)
        self._generate_plots(results)

        return results

    def _save_results(self, results: Dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = results["dataset_name"]
        model_name = results["model_name"]

        filename = f"{dataset_name}_{model_name}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved: {filepath}")

    def _generate_plots(self, results: Dict):
        self._plot_cv_metrics(results)

    def _plot_cv_metrics(self, results: Dict):
        fold_results = results["fold_results"]
        folds = [f"Fold {r['fold']+1}" for r in fold_results]
        rmse_values = [r["rmse"] for r in fold_results]
        mae_values = [r["mae"] for r in fold_results]
        mse_values = [r["mse"] for r in fold_results]
        fcp_values = [r["fcp"] for r in fold_results]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        x = np.arange(len(folds))
        width = 0.6

        ax1.bar(x, rmse_values, width, label="RMSE", color="skyblue", alpha=0.7)
        ax1.set_xlabel("Fold")
        ax1.set_ylabel("RMSE")
        ax1.set_title(f'RMSE per Fold - {results["model_name"]}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(folds)
        ax1.grid(True, alpha=0.3)

        mean_rmse = results["aggregated_metrics"]["mean_rmse"]
        ax1.axhline(y=mean_rmse, color="red", linestyle="--", label=f"Mean: {mean_rmse:.4f}")
        ax1.legend()

        ax2.bar(x, mae_values, width, label="MAE", color="lightcoral", alpha=0.7)
        ax2.set_xlabel("Fold")
        ax2.set_ylabel("MAE")
        ax2.set_title(f'MAE per Fold - {results["model_name"]}')
        ax2.set_xticks(x)
        ax2.set_xticklabels(folds)
        ax2.grid(True, alpha=0.3)

        mean_mae = results["aggregated_metrics"]["mean_mae"]
        ax2.axhline(y=mean_mae, color="red", linestyle="--", label=f"Mean: {mean_mae:.4f}")
        ax2.legend()

        ax3.bar(x, mse_values, width, label="MSE", color="lightgreen", alpha=0.7)
        ax3.set_xlabel("Fold")
        ax3.set_ylabel("MSE")
        ax3.set_title(f'MSE per Fold - {results["model_name"]}')
        ax3.set_xticks(x)
        ax3.set_xticklabels(folds)
        ax3.grid(True, alpha=0.3)

        mean_mse = results["aggregated_metrics"]["mean_mse"]
        ax3.axhline(y=mean_mse, color="red", linestyle="--", label=f"Mean: {mean_mse:.4f}")
        ax3.legend()

        ax4.bar(x, fcp_values, width, label="FCP", color="orange", alpha=0.7)
        ax4.set_xlabel("Fold")
        ax4.set_ylabel("FCP")
        ax4.set_title(f'FCP per Fold - {results["model_name"]}')
        ax4.set_xticks(x)
        ax4.set_xticklabels(folds)
        ax4.grid(True, alpha=0.3)

        mean_fcp = results["aggregated_metrics"]["mean_fcp"]
        ax4.axhline(y=mean_fcp, color="red", linestyle="--", label=f"Mean: {mean_fcp:.4f}")
        ax4.legend()

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = results["dataset_name"]
        model_name = results["model_name"]

        plot_filename = f"{dataset_name}_{model_name}_{timestamp}_cv_metrics.png"
        plot_filepath = self.output_dir / plot_filename

        plt.savefig(plot_filepath, dpi=300, bbox_inches="tight")
        print(f"Plot saved: {plot_filepath}")
        plt.close()  # Fecha o plot sem exibir

    def _evaluate_fairness(self, results: Dict, dataset_name: str, loader) -> Dict:
        try:
            # Carrega dados demográficos (suporta MovieLens 100k e 1M)
            if hasattr(loader, "load_user_demographics"):
                user_demographics = loader.load_user_demographics()
            elif hasattr(loader, "load_users"):
                user_demographics = loader.load_users()
            else:
                user_demographics = None

            if user_demographics is not None:
                # Verifica se temos predições armazenadas
                if "all_predictions" in results and results["all_predictions"]:
                    # Avaliação completa de fairness com predições
                    fairness_results = self.fairness_evaluator.evaluate_model_fairness(
                        predictions=results["all_predictions"],
                        user_demographics=user_demographics
                    )
                    fairness_results["evaluation_status"] = "completed_full"
                    fairness_results["note"] = "Avaliação completa com métricas de fairness detalhadas"
                else:
                    # Avaliação básica sem predições
                    fairness_results = {
                        "demographic_statistics": self._get_demographic_stats(user_demographics),
                        "fairness_config": self.fairness_config,
                        "evaluation_status": "completed_basic",
                        "note": "Avaliação básica - predições não disponíveis para métricas avançadas",
                    }

                self._generate_fairness_plots(user_demographics, dataset_name, results["model_name"])

                return fairness_results
            else:
                return {"evaluation_status": "skipped", "reason": "Dataset não suporta dados demográficos"}

        except Exception as e:
            return {"evaluation_status": "error", "error_message": str(e)}

    def _get_demographic_stats(self, demographics: pd.DataFrame) -> Dict:
        """Calcula estatísticas demográficas básicas."""
        from .evaluation.fairness import DemographicAnalyzer

        analyzer = DemographicAnalyzer()
        processed_demographics = analyzer.process_demographics(demographics)
        stats = analyzer.get_group_statistics(processed_demographics)
        minority_groups = analyzer.identify_minority_groups(processed_demographics)

        return {"group_statistics": stats, "minority_groups": minority_groups, "total_users": len(demographics)}

    def _generate_fairness_plots(self, demographics: pd.DataFrame, dataset_name: str, model_name: str):
        """Gera plots de fairness."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Plot de distribuição demográfica
            demo_plot_path = self.output_dir / f"{dataset_name}_{model_name}_{timestamp}_demographics.png"
            fig = self.fairness_visualizer.plot_demographic_distribution(demographics, demo_plot_path)
            plt.close(fig)

            print(f"Fairness demographic plot saved: {demo_plot_path}")

        except Exception as e:
            print(f"Erro ao gerar plots de fairness: {e}")
