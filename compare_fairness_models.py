#!/usr/bin/env python3
"""
Script para comparar fairness entre diferentes modelos de recomendação.

Modelos analisados:
- Baseline
- KNN
- KNN Baseline  
- SVD
- SVD++

Métricas de fairness:
- Demographic Parity
- Equal Opportunity
- Equalized Odds
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Adiciona o diretório do projeto ao path
sys.path.append(str(Path(__file__).parent))

from sis_rec_experiments.models.pipeline import ModelPipeline


def create_fairness_comparison_output_directory():
    """Cria diretório de saída para comparação de fairness"""
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = f"sis_rec_experiments/models/results/fairness_comparison_{today}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def print_fairness_comparison_header():
    """Imprime cabeçalho da comparação de fairness"""
    print("=" * 80)
    print("COMPARAÇÃO DE FAIRNESS ENTRE MODELOS DE RECOMENDAÇÃO")
    print("=" * 80)
    print(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Dataset: MovieLens 100k")
    print(f"Modelos: Baseline, KNN, KNN Baseline, SVD, SVD++")
    print(f"Métricas: Demographic Parity, Equal Opportunity, Equalized Odds")
    print(f"Cross-Validation: 5 folds")
    print("=" * 80)
    print()


def print_detailed_fairness_results(results, model_name):
    """Imprime resultados detalhados de fairness"""
    print(f"📊 FAIRNESS ANALYSIS - {model_name.upper()}")
    print("-" * 60)
    
    # Performance básica
    summary = results["summary"]
    print(f"Performance Metrics:")
    print(f"  RMSE: {summary['rmse']:.4f}")
    print(f"  MAE:  {summary['mae']:.4f}")
    print(f"  FCP:  {summary['fcp']:.4f}")
    print()
    
    # Análise de fairness
    if "fairness_evaluation" in results:
        fairness = results["fairness_evaluation"]
        print(f"Fairness Evaluation Status: {fairness.get('evaluation_status', 'N/A')}")
        
        # Estatísticas demográficas
        if "demographic_statistics" in fairness:
            demo_stats = fairness["demographic_statistics"]
            print(f"\nDemographic Statistics:")
            print(f"  Total Users: {demo_stats.get('total_users', 'N/A')}")
            
            if "group_statistics" in demo_stats:
                group_stats = demo_stats["group_statistics"]
                
                # Distribuição por gênero
                if "gender" in group_stats:
                    print(f"  Gender Distribution:")
                    for gender, count in group_stats["gender"].items():
                        percentage = count / demo_stats["total_users"] * 100
                        print(f"    {gender}: {count} ({percentage:.1f}%)")
        
        # Métricas de fairness detalhadas
        if "fairness_metrics" in fairness:
            fairness_metrics = fairness["fairness_metrics"]
            print(f"\nDetailed Fairness Metrics:")
            
            for metric_key, metric_value in fairness_metrics.items():
                if isinstance(metric_value, dict):
                    print(f"  {metric_key.upper()}:")
                    for sub_key, sub_value in metric_value.items():
                        if isinstance(sub_value, (int, float)):
                            print(f"    {sub_key}: {sub_value:.4f}")
                        elif isinstance(sub_value, dict):
                            print(f"    {sub_key}:")
                            for group, value in sub_value.items():
                                if isinstance(value, (int, float)):
                                    print(f"      {group}: {value:.4f}")
        
        # Resumo executivo
        if "executive_summary" in fairness:
            summary_data = fairness["executive_summary"]
            print(f"\nExecutive Summary:")
            print(f"  Overall Fairness Score: {summary_data.get('overall_fairness_score', 'N/A'):.3f}")
            print(f"  Issues Detected: {len(summary_data.get('fairness_issues_detected', []))}")
            
            if summary_data.get('fairness_issues_detected'):
                print(f"  Fairness Issues:")
                for issue in summary_data['fairness_issues_detected'][:3]:  # Mostra apenas os 3 primeiros
                    print(f"    • {issue}")
    
    print("\n" + "="*60 + "\n")


def extract_fairness_metrics_for_comparison(all_results):
    """Extrai métricas de fairness para comparação"""
    comparison_data = []
    
    for model_name, results in all_results.items():
        row = {
            'Model': model_name,
            'RMSE': results['summary']['rmse'],
            'MAE': results['summary']['mae'],
            'FCP': results['summary']['fcp']
        }
        
        # Extrai métricas de fairness se disponíveis
        if 'fairness_evaluation' in results:
            fairness = results['fairness_evaluation']
            
            # Score geral de fairness
            if 'executive_summary' in fairness:
                exec_summary = fairness['executive_summary']
                row['Fairness_Score'] = exec_summary.get('overall_fairness_score', 0)
                row['Issues_Count'] = len(exec_summary.get('fairness_issues_detected', []))
            
            # Métricas específicas
            if 'fairness_metrics' in fairness:
                metrics = fairness['fairness_metrics']
                
                # Demographic Parity para gênero
                if 'demographic_parity_gender' in metrics:
                    demo_parity = metrics['demographic_parity_gender']
                    row['Demo_Parity_Diff'] = demo_parity.get('demographic_parity_difference', 0)
                    row['Demo_Parity_Ratio'] = demo_parity.get('demographic_parity_ratio', 1)
                
                # Equal Opportunity para gênero
                if 'equal_opportunity_gender' in metrics:
                    equal_opp = metrics['equal_opportunity_gender']
                    row['Equal_Opp_Diff'] = equal_opp.get('equalized_odds_difference', 0)
                    row['Equal_Opp_Ratio'] = equal_opp.get('equalized_odds_ratio', 1)
                
                # Calibração para gênero
                if 'calibration_gender' in metrics:
                    calibration = metrics['calibration_gender']
                    row['Max_RMSE_Diff'] = calibration.get('max_rmse_difference', 0)
                    row['Max_MAE_Diff'] = calibration.get('max_mae_difference', 0)
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


def create_fairness_comparison_plots(comparison_df, output_dir):
    """Cria gráficos de comparação de fairness"""
    
    # Configuração do estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Gráfico de Performance vs Fairness
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparação de Fairness entre Modelos', fontsize=16, fontweight='bold')
    
    # Performance Metrics
    models = comparison_df['Model']
    x_pos = np.arange(len(models))
    
    axes[0, 0].bar(x_pos, comparison_df['RMSE'], alpha=0.7, color='skyblue')
    axes[0, 0].set_title('RMSE por Modelo')
    axes[0, 0].set_xlabel('Modelo')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(models, rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Fairness Score (se disponível)
    if 'Fairness_Score' in comparison_df.columns:
        axes[0, 1].bar(x_pos, comparison_df['Fairness_Score'], alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Score de Fairness por Modelo')
        axes[0, 1].set_xlabel('Modelo')
        axes[0, 1].set_ylabel('Fairness Score')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(models, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
    
    # Demographic Parity Difference (se disponível)
    if 'Demo_Parity_Diff' in comparison_df.columns:
        axes[1, 0].bar(x_pos, comparison_df['Demo_Parity_Diff'].abs(), alpha=0.7, color='coral')
        axes[1, 0].set_title('Diferença de Paridade Demográfica (|valor|)')
        axes[1, 0].set_xlabel('Modelo')
        axes[1, 0].set_ylabel('|Demographic Parity Difference|')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(models, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Threshold (0.1)')
        axes[1, 0].legend()
    
    # Equal Opportunity Difference (se disponível)
    if 'Equal_Opp_Diff' in comparison_df.columns:
        axes[1, 1].bar(x_pos, comparison_df['Equal_Opp_Diff'].abs(), alpha=0.7, color='gold')
        axes[1, 1].set_title('Diferença de Igualdade de Oportunidades (|valor|)')
        axes[1, 1].set_xlabel('Modelo')
        axes[1, 1].set_ylabel('|Equal Opportunity Difference|')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(models, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Threshold (0.1)')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Salva o gráfico
    plot_path = Path(output_dir) / f"fairness_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico de comparação salvo: {plot_path}")
    plt.close()
    
    return plot_path


def main():
    """Função principal para comparar fairness entre modelos"""
    
    # Configuração
    output_dir = create_fairness_comparison_output_directory()
    dataset_name = "movielens"
    
    # Modelos a serem comparados
    models_to_compare = [
        ("baseline", "Baseline", {}),
        ("knn_basic", "KNN", {}),
        ("knn_baseline", "KNN Baseline", {}),
        ("svd", "SVD", {"n_factors": 50}),
        ("svdpp", "SVD++", {"n_factors": 50})
    ]
    
    # Configuração de fairness
    fairness_config = {
        "sensitive_features": ["gender", "age_group", "occupation_category"],
        "rating_threshold": 4.0
    }
    
    # Parâmetros de pré-processamento
    preprocessing_params = {
        "min_user_ratings": 5,
        "min_item_ratings": 5
    }
    
    print_fairness_comparison_header()
    
    # Inicializa pipeline com avaliação de fairness completa
    pipeline = ModelPipeline(
        output_dir=output_dir,
        apply_preprocessing=True,
        preprocessing_params=preprocessing_params,
        evaluate_fairness=True,
        fairness_config=fairness_config
    )
    
    all_results = {}
    
    # Executa experimentos para cada modelo
    for i, (model_key, model_name, model_params) in enumerate(models_to_compare, 1):
        print(f"🚀 Executando modelo {i}/{len(models_to_compare)}: {model_name}")
        print("-" * 60)
        
        try:
            # Executa o experimento
            results = pipeline.run_experiment(
                dataset_name=dataset_name,
                model_name=model_key,
                model_params=model_params
            )
            
            # Armazena resultados
            all_results[model_name] = results
            
            # Imprime resultados detalhados
            print_detailed_fairness_results(results, model_name)
            
        except Exception as e:
            print(f"❌ Erro ao executar {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            print()
            continue
    
    # Análise comparativa final
    if all_results:
        print("=" * 80)
        print("ANÁLISE COMPARATIVA FINAL")
        print("=" * 80)
        
        # Extrai dados para comparação
        comparison_df = extract_fairness_metrics_for_comparison(all_results)
        
        # Mostra tabela comparativa
        print("\n📊 TABELA COMPARATIVA:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Cria gráficos de comparação
        plot_path = create_fairness_comparison_plots(comparison_df, output_dir)
        
        # Salva dados de comparação
        comparison_path = Path(output_dir) / f"fairness_comparison_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"📁 Dados de comparação salvos: {comparison_path}")
        
        # Análise final
        print(f"\n🎯 ANÁLISE FINAL:")
        
        # Melhor modelo por performance
        best_rmse_model = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']
        print(f"  • Melhor Performance (RMSE): {best_rmse_model}")
        
        # Melhor modelo por fairness (se disponível)
        if 'Fairness_Score' in comparison_df.columns:
            best_fairness_model = comparison_df.loc[comparison_df['Fairness_Score'].idxmax(), 'Model']
            print(f"  • Melhor Fairness Score: {best_fairness_model}")
        
        # Modelo com menos problemas de fairness
        if 'Issues_Count' in comparison_df.columns:
            least_issues_model = comparison_df.loc[comparison_df['Issues_Count'].idxmin(), 'Model']
            print(f"  • Menos Problemas de Fairness: {least_issues_model}")
        
        print(f"\n📁 Todos os resultados salvos em: {output_dir}")
        
    else:
        print("❌ Nenhum experimento foi executado com sucesso.")


if __name__ == "__main__":
    main()
