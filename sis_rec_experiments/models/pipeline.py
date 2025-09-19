import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from ..loaders.builder import create_loader
from .evaluation.evaluator import ModelEvaluator
from .model_factory import ModelFactory


class ModelPipeline:
    def __init__(self, output_dir: str = "sis_rec_experiments/models/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = ModelEvaluator()

    def run_experiment(self, dataset_name: str, model_name: str, model_params: Dict = None) -> Dict:
        if model_params is None:
            model_params = {}

        loader = create_loader(dataset_name)
        loader.load_ratings()
        surprise_dataset = loader.to_surprise_dataset()

        model = ModelFactory.create_model(model_name, **model_params)
        results = self.evaluator.evaluate_model(model, surprise_dataset)

        results["dataset_name"] = dataset_name
        results["experiment_timestamp"] = datetime.now().isoformat()

        self._save_results(results)
        self._generate_plots(results)

        return results

    def _save_results(self, results: Dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = results["dataset_name"]
        model_name = results["model_name"]
        
        filename = f"{dataset_name}_{model_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved: {filepath}")

    def _generate_plots(self, results: Dict):
        self._plot_cv_metrics(results)

    def _plot_cv_metrics(self, results: Dict):
        fold_results = results["fold_results"]
        folds = [f"Fold {r['fold']+1}" for r in fold_results]
        rmse_values = [r["rmse"] for r in fold_results]
        mae_values = [r["mae"] for r in fold_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        x = np.arange(len(folds))
        width = 0.35

        ax1.bar(x, rmse_values, width, label='RMSE', color='skyblue', alpha=0.7)
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('RMSE')
        ax1.set_title(f'RMSE per Fold - {results["model_name"]}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(folds)
        ax1.grid(True, alpha=0.3)

        mean_rmse = results["aggregated_metrics"]["mean_rmse"]
        ax1.axhline(y=mean_rmse, color='red', linestyle='--', 
                   label=f'Mean: {mean_rmse:.4f}')
        ax1.legend()

        ax2.bar(x, mae_values, width, label='MAE', color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('MAE')
        ax2.set_title(f'MAE per Fold - {results["model_name"]}')
        ax2.set_xticks(x)
        ax2.set_xticklabels(folds)
        ax2.grid(True, alpha=0.3)

        mean_mae = results["aggregated_metrics"]["mean_mae"]
        ax2.axhline(y=mean_mae, color='red', linestyle='--',
                   label=f'Mean: {mean_mae:.4f}')
        ax2.legend()

        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = results["dataset_name"]
        model_name = results["model_name"]
        
        plot_filename = f"{dataset_name}_{model_name}_{timestamp}_cv_metrics.png"
        plot_filepath = self.output_dir / plot_filename
        
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_filepath}")
        plt.show()
