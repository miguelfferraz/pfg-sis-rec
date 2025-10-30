#!/usr/bin/env python3
"""
Experimentos de fairness para MovieLens 1M com os modelos solicitados.
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from sis_rec_experiments.loaders.builder import create_loader
from sis_rec_experiments.models.pipeline import ModelPipeline
from sis_rec_experiments.models.evaluation.fairness.demographic_analyzer_ml1m import DemographicAnalyzerML1M


def setup_ml1m_fairness_config():
    """Configura parâmetros específicos para fairness no ML-1M"""
    
    return {
        'sensitive_features': ['gender', 'age_group'],  # Removemos occupation_category para focar
        'rating_threshold': 4.0,  # Mesmo threshold do ML-100k
        'use_ml1m_analyzer': True
    }


def create_results_directory():
    """Cria diretório para salvar resultados"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"sis_rec_experiments/models/results/ml1m_fairness_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    return results_dir


def run_model_experiment(model_config: Dict[str, Any], dataset_name: str, fairness_config: Dict[str, Any]) -> Dict[str, Any]:
    """Executa experimento para um modelo específico"""
    
    print(f"\n🔄 Executando {model_config['name']}...")
    
    # Configura pipeline com fairness habilitada
    pipeline = ModelPipeline(
        apply_preprocessing=True,  # Aplica pré-processamento padrão
        evaluate_fairness=True,
        fairness_config=fairness_config
    )
    
    # Executa experimento
    results = pipeline.run_experiment(
        dataset_name=dataset_name,
        model_name=model_config['algorithm'],
        model_params=model_config.get('params', {})
    )
    
    # Adiciona informações do modelo
    results['model_config'] = model_config
    results['experiment_timestamp'] = datetime.now().isoformat()
    
    return results


def save_results(results: Dict[str, Any], results_dir: Path, model_name: str):
    """Salva resultados em arquivo JSON"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ml1m_{model_name}_{timestamp}.json"
    filepath = results_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Resultados salvos: {filepath}")
    return filepath


def print_fairness_summary(results: Dict[str, Any], model_name: str):
    """Imprime resumo das métricas de fairness"""
    
    print(f"\n📊 RESUMO FAIRNESS - {model_name}")
    print("-" * 50)
    
    if 'fairness_evaluation' in results:
        fairness = results['fairness_evaluation']
        
        # Performance geral
        if 'average_metrics' in results:
            metrics = results['average_metrics']
            print(f"RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            print(f"MAE:  {metrics.get('mae', 'N/A'):.4f}")
        
        # Métricas de fairness
        if 'fairness_metrics' in fairness:
            fm = fairness['fairness_metrics']
            
            print("\nFAIRNESS METRICS:")
            
            # Gênero
            if 'demographic_parity_gender' in fm:
                dp_gender = fm['demographic_parity_gender']['demographic_parity_difference']
                print(f"  DP Gender: {dp_gender:.4f}")
            
            if 'equal_opportunity_gender' in fm:
                eo_gender = fm['equal_opportunity_gender']['equalized_odds_difference']
                print(f"  EO Gender: {eo_gender:.4f}")
            
            # Idade
            if 'demographic_parity_age_group' in fm:
                dp_age = fm['demographic_parity_age_group']['demographic_parity_difference']
                print(f"  DP Age:    {dp_age:.4f}")
            
            if 'equal_opportunity_age_group' in fm:
                eo_age = fm['equal_opportunity_age_group']['equalized_odds_difference']
                print(f"  EO Age:    {eo_age:.4f}")
        
        # Status geral
        if 'executive_summary' in fairness:
            summary = fairness['executive_summary']
            fairness_score = summary.get('overall_fairness_score', 0)
            issues_count = len(summary.get('fairness_issues_detected', []))
            
            print(f"\nFairness Score: {fairness_score:.3f}")
            print(f"Issues Detected: {issues_count}")


def analyze_ml1m_demographics():
    """Analisa distribuição demográfica do ML-1M"""
    
    print("🔍 ANÁLISE DEMOGRÁFICA - MOVIELENS 1M")
    print("=" * 60)
    
    # Carrega dados
    loader = create_loader('movielens1m')
    users_df = loader.load_users()
    
    # Processa com analisador específico
    analyzer = DemographicAnalyzerML1M()
    analysis = analyzer.analyze_ml1m_distribution(users_df)
    
    print(f"\nTotal de usuários: {analysis['total_users']:,}")
    
    print("\n📊 DISTRIBUIÇÃO POR GRUPOS ETÁRIOS:")
    for group, count in analysis['grouped_ages'].items():
        percentage = (count / analysis['total_users']) * 100
        print(f"  {group:<8}: {count:>5} usuários ({percentage:>5.1f}%)")
    
    print("\n🚹🚺 DISTRIBUIÇÃO POR GÊNERO:")
    for gender, count in analysis['gender_distribution'].items():
        percentage = (count / analysis['total_users']) * 100
        print(f"  {gender:<8}: {count:>5} usuários ({percentage:>5.1f}%)")
    
    print("\n💼 DISTRIBUIÇÃO POR CATEGORIA PROFISSIONAL:")
    for category, count in analysis['occupation_categories'].items():
        percentage = (count / analysis['total_users']) * 100
        print(f"  {category:<12}: {count:>5} usuários ({percentage:>5.1f}%)")
    
    return analysis


def main():
    """Função principal"""
    
    print("🎯 EXPERIMENTOS DE FAIRNESS - MOVIELENS 1M")
    print("=" * 60)
    
    # Análise demográfica inicial
    demo_analysis = analyze_ml1m_demographics()
    
    # Configuração dos modelos (nomes corretos do sistema)
    models_config = [
        {
            'name': 'Baseline',
            'algorithm': 'baseline',
            'params': {}
        },
        {
            'name': 'KNN',
            'algorithm': 'knn_basic',
            'params': {'sim_options': {'name': 'cosine', 'user_based': True}}
        },
        {
            'name': 'KNN_Baseline',
            'algorithm': 'knn_baseline',
            'params': {'sim_options': {'name': 'cosine', 'user_based': True}}
        },
        {
            'name': 'SVD',
            'algorithm': 'svd',
            'params': {'n_factors': 50, 'n_epochs': 20}
        },
        {
            'name': 'SVD++',
            'algorithm': 'svdpp',
            'params': {'n_factors': 50, 'n_epochs': 20}
        }
    ]
    
    # Configuração de fairness
    fairness_config = setup_ml1m_fairness_config()
    
    # Cria diretório de resultados
    results_dir = create_results_directory()
    print(f"\n📁 Resultados serão salvos em: {results_dir}")
    
    # Salva análise demográfica
    demo_file = results_dir / "demographic_analysis.json"
    with open(demo_file, 'w', encoding='utf-8') as f:
        json.dump(demo_analysis, f, indent=2, ensure_ascii=False)
    
    # Executa experimentos
    all_results = {}
    
    for model_config in models_config:
        try:
            # Executa experimento
            results = run_model_experiment(
                model_config=model_config,
                dataset_name='movielens1m',
                fairness_config=fairness_config
            )
            
            # Salva resultados
            filepath = save_results(results, results_dir, model_config['name'])
            
            # Imprime resumo
            print_fairness_summary(results, model_config['name'])
            
            # Armazena para análise final
            all_results[model_config['name']] = results
            
        except Exception as e:
            print(f"❌ Erro ao executar {model_config['name']}: {e}")
            continue
    
    # Salva resumo consolidado
    summary_file = results_dir / "experiment_summary.json"
    summary_data = {
        'experiment_info': {
            'dataset': 'MovieLens 1M',
            'total_models': len(models_config),
            'successful_runs': len(all_results),
            'fairness_config': fairness_config,
            'demographic_analysis': demo_analysis
        },
        'models_results': {name: {
            'rmse': results.get('average_metrics', {}).get('rmse', None),
            'mae': results.get('average_metrics', {}).get('mae', None),
            'fairness_score': results.get('fairness_evaluation', {}).get('executive_summary', {}).get('overall_fairness_score', None)
        } for name, results in all_results.items()}
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ EXPERIMENTOS CONCLUÍDOS!")
    print(f"📊 {len(all_results)} modelos executados com sucesso")
    print(f"📁 Resultados salvos em: {results_dir}")
    
    return results_dir, all_results


if __name__ == "__main__":
    main()
