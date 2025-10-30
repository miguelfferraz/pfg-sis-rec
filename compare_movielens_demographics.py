#!/usr/bin/env python3
"""
Comparação demográfica entre MovieLens 100k e MovieLens 1M.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sis_rec_experiments.loaders.builder import create_loader
from sis_rec_experiments.models.evaluation.fairness.demographic_analyzer import DemographicAnalyzer
from sis_rec_experiments.models.evaluation.fairness.demographic_analyzer_ml1m import DemographicAnalyzerML1M


def analyze_ml100k_demographics():
    """Analisa demografia do MovieLens 100k"""
    
    print("🔍 ANALISANDO MOVIELENS 100K")
    print("-" * 40)
    
    # Carrega dados
    loader = create_loader('movielens')
    users_df = loader.load_user_demographics()
    
    # Processa com analisador padrão
    analyzer = DemographicAnalyzer()
    processed_df = analyzer.process_demographics(users_df)
    
    # Estatísticas básicas
    total_users = len(processed_df)
    
    print(f"Total de usuários: {total_users:,}")
    
    # Distribuição por gênero
    gender_dist = processed_df['gender'].value_counts()
    print(f"\n🚹🚺 DISTRIBUIÇÃO POR GÊNERO:")
    for gender, count in gender_dist.items():
        percentage = (count / total_users) * 100
        print(f"  {gender:<8}: {count:>4} usuários ({percentage:>5.1f}%)")
    
    # Distribuição por faixa etária
    age_dist = processed_df['age_group'].value_counts()
    print(f"\n🎂 DISTRIBUIÇÃO POR FAIXA ETÁRIA:")
    for age_group, count in age_dist.items():
        percentage = (count / total_users) * 100
        print(f"  {age_group:<11}: {count:>4} usuários ({percentage:>5.1f}%)")
    
    # Estatísticas de idade original
    age_stats = users_df['age'].describe()
    print(f"\n📊 ESTATÍSTICAS DE IDADE:")
    print(f"  Mínima: {age_stats['min']:.0f} anos")
    print(f"  Máxima: {age_stats['max']:.0f} anos")
    print(f"  Média:  {age_stats['mean']:.1f} anos")
    print(f"  Mediana: {age_stats['50%']:.1f} anos")
    
    # Cross-tabulation
    cross_tab = pd.crosstab(processed_df['gender'], processed_df['age_group'])
    print(f"\n📋 CROSS-TABULATION (Gênero × Idade):")
    print(cross_tab)
    
    return {
        'dataset': 'MovieLens 100k',
        'total_users': total_users,
        'gender_distribution': gender_dist.to_dict(),
        'age_distribution': age_dist.to_dict(),
        'age_stats': age_stats.to_dict(),
        'cross_tabulation': cross_tab,
        'processed_df': processed_df
    }


def analyze_ml1m_demographics():
    """Analisa demografia do MovieLens 1M"""
    
    print("\n\n🔍 ANALISANDO MOVIELENS 1M")
    print("-" * 40)
    
    # Carrega dados
    loader = create_loader('movielens1m')
    users_df = loader.load_users()
    
    # Processa com analisador específico ML-1M
    analyzer = DemographicAnalyzerML1M()
    processed_df = analyzer.process_demographics_ml1m(users_df)
    
    # Estatísticas básicas
    total_users = len(processed_df)
    
    print(f"Total de usuários: {total_users:,}")
    
    # Distribuição por gênero
    gender_dist = processed_df['gender'].value_counts()
    print(f"\n🚹🚺 DISTRIBUIÇÃO POR GÊNERO:")
    for gender, count in gender_dist.items():
        percentage = (count / total_users) * 100
        print(f"  {gender:<8}: {count:>4} usuários ({percentage:>5.1f}%)")
    
    # Distribuição por faixa etária
    age_dist = processed_df['age_group'].value_counts()
    print(f"\n🎂 DISTRIBUIÇÃO POR FAIXA ETÁRIA:")
    for age_group, count in age_dist.items():
        percentage = (count / total_users) * 100
        print(f"  {age_group:<11}: {count:>4} usuários ({percentage:>5.1f}%)")
    
    # Mostra mapeamento de códigos de idade
    print(f"\n📊 CÓDIGOS DE IDADE ORIGINAIS:")
    age_codes = users_df['age'].value_counts().sort_index()
    age_mapping = analyzer.get_age_mapping_info()
    
    for code, count in age_codes.items():
        group = analyzer.categorize_age_ml1m(code)
        percentage = (count / total_users) * 100
        print(f"  Código {code:>2}: {count:>4} usuários ({percentage:>5.1f}%) → {group}")
    
    # Cross-tabulation
    cross_tab = pd.crosstab(processed_df['gender'], processed_df['age_group'])
    print(f"\n📋 CROSS-TABULATION (Gênero × Idade):")
    print(cross_tab)
    
    return {
        'dataset': 'MovieLens 1M',
        'total_users': total_users,
        'gender_distribution': gender_dist.to_dict(),
        'age_distribution': age_dist.to_dict(),
        'age_codes': age_codes.to_dict(),
        'age_mapping': age_mapping,
        'cross_tabulation': cross_tab,
        'processed_df': processed_df
    }


def compare_datasets(ml100k_data, ml1m_data):
    """Compara os dois datasets"""
    
    print("\n\n" + "=" * 80)
    print("📊 COMPARAÇÃO ENTRE DATASETS")
    print("=" * 80)
    
    # Comparação básica
    print(f"\n📈 ESTATÍSTICAS GERAIS:")
    print(f"{'Dataset':<15} | {'Usuários':<10} | {'Proporção':<10}")
    print("-" * 40)
    print(f"{'ML-100k':<15} | {ml100k_data['total_users']:>8,} | {'1.0x':<10}")
    print(f"{'ML-1M':<15} | {ml1m_data['total_users']:>8,} | {ml1m_data['total_users']/ml100k_data['total_users']:>8.1f}x")
    
    # Comparação por gênero
    print(f"\n🚹🚺 COMPARAÇÃO POR GÊNERO:")
    print(f"{'Gênero':<10} | {'ML-100k':<15} | {'ML-1M':<15} | {'Diferença':<10}")
    print("-" * 60)
    
    for gender in ['Male', 'Female']:
        count_100k = ml100k_data['gender_distribution'].get(gender, 0)
        count_1m = ml1m_data['gender_distribution'].get(gender, 0)
        
        pct_100k = (count_100k / ml100k_data['total_users']) * 100
        pct_1m = (count_1m / ml1m_data['total_users']) * 100
        
        diff = pct_1m - pct_100k
        
        print(f"{gender:<10} | {count_100k:>4} ({pct_100k:>5.1f}%) | {count_1m:>4} ({pct_1m:>5.1f}%) | {diff:>+6.1f}pp")
    
    # Comparação por faixa etária
    print(f"\n🎂 COMPARAÇÃO POR FAIXA ETÁRIA:")
    print(f"{'Faixa Etária':<12} | {'ML-100k':<15} | {'ML-1M':<15} | {'Diferença':<10}")
    print("-" * 65)
    
    for age_group in ['Young', 'Adult', 'Middle_Age', 'Senior']:
        count_100k = ml100k_data['age_distribution'].get(age_group, 0)
        count_1m = ml1m_data['age_distribution'].get(age_group, 0)
        
        pct_100k = (count_100k / ml100k_data['total_users']) * 100 if count_100k > 0 else 0
        pct_1m = (count_1m / ml1m_data['total_users']) * 100 if count_1m > 0 else 0
        
        if count_100k > 0 or count_1m > 0:
            diff = pct_1m - pct_100k
            print(f"{age_group:<12} | {count_100k:>4} ({pct_100k:>5.1f}%) | {count_1m:>4} ({pct_1m:>5.1f}%) | {diff:>+6.1f}pp")
    
    # Nota sobre agrupamento
    print(f"\n⚠️ NOTA: No ML-1M, Adult inclui Middle_Age (conforme solicitado)")
    
    # Cross-tabulations lado a lado
    print(f"\n📋 CROSS-TABULATIONS:")
    print(f"\nML-100k:")
    print(ml100k_data['cross_tabulation'])
    
    print(f"\nML-1M:")
    print(ml1m_data['cross_tabulation'])


def generate_summary_insights(ml100k_data, ml1m_data):
    """Gera insights resumidos"""
    
    print(f"\n\n💡 INSIGHTS PRINCIPAIS:")
    print("-" * 30)
    
    # Proporção de gênero
    male_pct_100k = (ml100k_data['gender_distribution']['Male'] / ml100k_data['total_users']) * 100
    male_pct_1m = (ml1m_data['gender_distribution']['Male'] / ml1m_data['total_users']) * 100
    
    print(f"🚹 GÊNERO:")
    print(f"  • Ambos datasets têm maioria masculina")
    print(f"  • ML-100k: {male_pct_100k:.1f}% homens")
    print(f"  • ML-1M: {male_pct_1m:.1f}% homens")
    print(f"  • Diferença: {abs(male_pct_1m - male_pct_100k):.1f} pontos percentuais")
    
    # Faixa etária
    adult_pct_100k = (ml100k_data['age_distribution'].get('Adult', 0) / ml100k_data['total_users']) * 100
    adult_pct_1m = (ml1m_data['age_distribution'].get('Adult', 0) / ml1m_data['total_users']) * 100
    
    print(f"\n🎂 IDADE:")
    print(f"  • Ambos datasets têm maioria adulta")
    print(f"  • ML-100k: {adult_pct_100k:.1f}% adultos")
    print(f"  • ML-1M: {adult_pct_1m:.1f}% adultos (inclui Middle_Age)")
    print(f"  • Grupos minoritários: Senior em ambos")
    
    # Tamanho e representatividade
    print(f"\n📊 REPRESENTATIVIDADE:")
    print(f"  • ML-1M é {ml1m_data['total_users']/ml100k_data['total_users']:.1f}x maior que ML-100k")
    print(f"  • Distribuições demográficas similares")
    print(f"  • ML-1M oferece mais dados para grupos minoritários")
    
    # Implicações para fairness
    print(f"\n⚖️ IMPLICAÇÕES PARA FAIRNESS:")
    print(f"  • Desbalanceamento de gênero consistente em ambos")
    print(f"  • Grupos etários minoritários (Senior) em ambos")
    print(f"  • ML-1M permite análises mais robustas")
    print(f"  • Padrões de fairness devem ser similares")


def main():
    """Função principal"""
    
    print("🎯 COMPARAÇÃO DEMOGRÁFICA: MOVIELENS 100K vs 1M")
    print("=" * 60)
    
    # Analisa ambos datasets
    ml100k_data = analyze_ml100k_demographics()
    ml1m_data = analyze_ml1m_demographics()
    
    # Compara datasets
    compare_datasets(ml100k_data, ml1m_data)
    
    # Gera insights
    generate_summary_insights(ml100k_data, ml1m_data)
    
    print(f"\n" + "=" * 80)
    print("✅ ANÁLISE COMPARATIVA COMPLETA!")
    print("=" * 80)


if __name__ == "__main__":
    main()



