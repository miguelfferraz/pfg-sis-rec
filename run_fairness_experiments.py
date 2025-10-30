#!/usr/bin/env python3
"""
Script para executar experimentos com avaliação de fairness
no dataset MovieLens com modelos de recomendação.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Adiciona o diretório do projeto ao path
sys.path.append(str(Path(__file__).parent))

from sis_rec_experiments.models.pipeline import ModelPipeline


def create_fairness_output_directory():
    """Cria diretório de saída para experimentos de fairness"""
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = f"sis_rec_experiments/models/results/fairness_experiments_{today}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def print_fairness_experiment_header():
    """Imprime cabeçalho do experimento de fairness"""
    print("=" * 80)
    print("EXPERIMENTOS DE FAIRNESS EM SISTEMAS DE RECOMENDAÇÃO")
    print("=" * 80)
    print(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Dataset: MovieLens 100k")
    print(f"Avaliação de Fairness: Habilitada")
    print(f"Features Sensíveis: Gênero, Faixa Etária, Categoria Profissional")
    print(f"Cross-Validation: 5 folds")
    print("=" * 80)
    print()


def print_fairness_results(results, model_name):
    """Imprime resultados de fairness de forma organizada"""
    print(f"📊 RESULTADOS DE FAIRNESS - {model_name.upper()}")
    print("-" * 50)

    # Resultados básicos de performance
    summary = results["summary"]
    print(f"RMSE:             {summary['rmse']:.4f}")
    print(f"MAE:              {summary['mae']:.4f}")
    print(f"FCP:              {summary['fcp']:.4f}")

    # Resultados de fairness
    if "fairness_evaluation" in results:
        fairness = results["fairness_evaluation"]
        print(f"\nFAIRNESS EVALUATION:")
        print(f"Status:           {fairness.get('evaluation_status', 'N/A')}")

        if "demographic_statistics" in fairness:
            demo_stats = fairness["demographic_statistics"]
            print(f"Total de Usuários: {demo_stats.get('total_users', 'N/A')}")

            # Estatísticas por gênero
            if "group_statistics" in demo_stats and "gender" in demo_stats["group_statistics"]:
                gender_stats = demo_stats["group_statistics"]["gender"]
                print(f"Distribuição por Gênero:")
                for gender, count in gender_stats.items():
                    percentage = count / demo_stats["total_users"] * 100
                    print(f"  {gender}: {count} ({percentage:.1f}%)")

            # Grupos minoritários
            if "minority_groups" in demo_stats:
                minority = demo_stats["minority_groups"]
                if any(minority.values()):
                    print(f"Grupos Minoritários Detectados:")
                    for feature, groups in minority.items():
                        if groups:
                            print(f"  {feature}: {', '.join(groups)}")

        # Métricas detalhadas de fairness (se disponíveis)
        if "fairness_metrics" in fairness:
            fairness_metrics = fairness["fairness_metrics"]
            print(f"\nMÉTRICAS DE FAIRNESS DETALHADAS:")
            
            for metric_key, metric_value in fairness_metrics.items():
                if isinstance(metric_value, dict):
                    print(f"  {metric_key.upper()}:")
                    for sub_key, sub_value in metric_value.items():
                        if isinstance(sub_value, (int, float)):
                            print(f"    {sub_key}: {sub_value:.4f}")
                        elif isinstance(sub_value, dict):
                            print(f"    {sub_key}:")
                            for group, value in sub_value.items():
                                print(f"      {group}: {value:.4f}")

        if "note" in fairness:
            print(f"\nNota: {fairness['note']}")

    print()


def main():
    """Função principal para executar experimentos de fairness"""

    # Configuração
    output_dir = create_fairness_output_directory()
    dataset_name = "movielens"  # Usando MovieLens 100k

    # Modelos a serem testados com fairness
    models_to_test = [
        ("baseline", "Baseline", {}),
        ("svd", "SVD", {"n_factors": 50}),
        ("knn_basic", "KNN Basic", {"k": 40}),
    ]

    # Configuração de fairness
    fairness_config = {"sensitive_features": ["gender", "age_group", "occupation_category"], "rating_threshold": 4.0}

    # Parâmetros de pré-processamento (mais leves para teste)
    preprocessing_params = {"min_user_ratings": 5, "min_item_ratings": 5}

    print_fairness_experiment_header()

    # Inicializa pipeline com avaliação de fairness habilitada
    pipeline = ModelPipeline(
        output_dir=output_dir,
        apply_preprocessing=True,
        preprocessing_params=preprocessing_params,
        evaluate_fairness=True,
        fairness_config=fairness_config,
    )

    all_results = {}

    # Executa experimentos para cada modelo
    for i, (model_key, model_name, model_params) in enumerate(models_to_test, 1):
        print(f"🚀 Executando experimento de fairness {i}/{len(models_to_test)}: {model_name}")
        print("-" * 50)

        try:
            # Executa o experimento com avaliação de fairness
            results = pipeline.run_experiment(
                dataset_name=dataset_name, model_name=model_key, model_params=model_params
            )

            # Armazena resultados
            all_results[model_name] = results

            # Imprime resultados do modelo
            print_fairness_results(results, model_name)

        except Exception as e:
            print(f"❌ Erro ao executar {model_name}: {str(e)}")
            import traceback

            traceback.print_exc()
            print()
            continue

    # Resumo final
    if all_results:
        print("=" * 80)
        print("RESUMO DOS EXPERIMENTOS DE FAIRNESS")
        print("=" * 80)
        print(f"{'Modelo':<15} {'RMSE':<8} {'MAE':<8} {'Fairness':<15}")
        print("-" * 50)

        for model_name, results in all_results.items():
            summary = results["summary"]
            fairness_status = "N/A"
            if "fairness_evaluation" in results:
                fairness_status = results["fairness_evaluation"].get("evaluation_status", "N/A")

            print(f"{model_name:<15} {summary['rmse']:<8.4f} {summary['mae']:<8.4f} {fairness_status:<15}")

        print(f"\n📁 Todos os resultados e gráficos foram salvos em: {output_dir}")
        print("\n🎯 PRÓXIMOS PASSOS:")
        print("  • Analise os gráficos demográficos gerados")
        print("  • Verifique os arquivos JSON para detalhes completos")
        print("  • Para avaliação completa de fairness, integre predições no evaluator")
    else:
        print("❌ Nenhum experimento foi executado com sucesso.")


if __name__ == "__main__":
    main()
