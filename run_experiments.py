#!/usr/bin/env python3
"""
Script para executar experimentos com múltiplos modelos de recomendação
no dataset MovieLens 1M com pré-processamento e cross-validation.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Adiciona o diretório do projeto ao path
sys.path.append(str(Path(__file__).parent))

from sis_rec_experiments.models.pipeline import ModelPipeline


def create_output_directory():
    """Cria diretório de saída com data de hoje"""
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = f"sis_rec_experiments/models/results/experiments_{today}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def print_experiment_header():
    """Imprime cabeçalho do experimento"""
    print("=" * 80)
    print("EXPERIMENTOS DE SISTEMAS DE RECOMENDAÇÃO")
    print("=" * 80)
    print(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Dataset: MovieLens 1M")
    print(f"Pré-processamento: Habilitado")
    print(f"Cross-Validation: 5 folds")
    print("=" * 80)
    print()


def print_model_results(results, model_name):
    """Imprime resultados de um modelo de forma organizada"""
    print(f"📊 RESULTADOS - {model_name.upper()}")
    print("-" * 50)

    summary = results["summary"]
    print(f"RMSE:             {summary['rmse']:.4f}")
    print(f"MAE:              {summary['mae']:.4f}")
    print(f"FCP:              {summary['fcp']:.4f}")
    print(f"Tempo Execução:   {summary['total_time']:.2f}s")

    # Detalhes adicionais
    agg_metrics = results["aggregated_metrics"]
    print(f"RMSE (±std):      {agg_metrics['mean_rmse']:.4f} (±{agg_metrics['std_rmse']:.4f})")
    print(f"MAE (±std):       {agg_metrics['mean_mae']:.4f} (±{agg_metrics['std_mae']:.4f})")
    print(f"FCP (±std):       {agg_metrics['mean_fcp']:.4f} (±{agg_metrics['std_fcp']:.4f})")
    print()


def print_final_summary(all_results):
    """Imprime resumo final comparativo"""
    print("=" * 80)
    print("RESUMO COMPARATIVO FINAL")
    print("=" * 80)
    print(f"{'Modelo':<15} {'RMSE':<8} {'MAE':<8} {'FCP':<8} {'Tempo(s)':<10}")
    print("-" * 60)

    for model_name, results in all_results.items():
        summary = results["summary"]
        print(
            f"{model_name:<15} {summary['rmse']:<8.4f} {summary['mae']:<8.4f} "
            f"{summary['fcp']:<8.4f} {summary['total_time']:<10.2f}"
        )

    # Encontra o melhor modelo por métrica
    best_rmse = min(all_results.items(), key=lambda x: x[1]["summary"]["rmse"])
    best_mae = min(all_results.items(), key=lambda x: x[1]["summary"]["mae"])
    best_fcp = max(all_results.items(), key=lambda x: x[1]["summary"]["fcp"])
    best_time = min(all_results.items(), key=lambda x: x[1]["summary"]["total_time"])

    print()
    print("🏆 MELHORES RESULTADOS:")
    print(f"Melhor RMSE:      {best_rmse[0]} ({best_rmse[1]['summary']['rmse']:.4f})")
    print(f"Melhor MAE:       {best_mae[0]} ({best_mae[1]['summary']['mae']:.4f})")
    print(f"Melhor FCP:       {best_fcp[0]} ({best_fcp[1]['summary']['fcp']:.4f})")
    print(f"Mais Rápido:      {best_time[0]} ({best_time[1]['summary']['total_time']:.2f}s)")
    print("=" * 80)


def main():
    """Função principal para executar todos os experimentos"""

    # Configuração
    output_dir = create_output_directory()
    dataset_name = "movielens1m"

    # Modelos a serem testados
    models_to_test = [
        ("baseline", "Baseline", {}),
        ("svd", "SVD", {}),
        ("svdpp", "SVD++", {}),
        ("knn_basic", "KNN Basic", {}),
        ("knn_baseline", "KNN Baseline", {}),
    ]

    # Parâmetros de pré-processamento
    preprocessing_params = {"min_user_ratings": 10, "min_item_ratings": 10}

    print_experiment_header()

    # Inicializa pipeline com pré-processamento habilitado
    pipeline = ModelPipeline(output_dir=output_dir, apply_preprocessing=True, preprocessing_params=preprocessing_params)

    all_results = {}

    # Executa experimentos para cada modelo
    for i, (model_key, model_name, model_params) in enumerate(models_to_test, 1):
        print(f"🚀 Executando experimento {i}/{len(models_to_test)}: {model_name}")
        print("-" * 50)

        try:
            # Executa o experimento
            results = pipeline.run_experiment(
                dataset_name=dataset_name, model_name=model_key, model_params=model_params
            )

            # Armazena resultados
            all_results[model_name] = results

            # Imprime resultados do modelo
            print_model_results(results, model_name)

        except Exception as e:
            print(f"❌ Erro ao executar {model_name}: {str(e)}")
            print()
            continue

    # Imprime resumo final
    if all_results:
        print_final_summary(all_results)
        print(f"\n📁 Todos os resultados e gráficos foram salvos em: {output_dir}")
    else:
        print("❌ Nenhum experimento foi executado com sucesso.")


if __name__ == "__main__":
    main()
