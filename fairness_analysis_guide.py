#!/usr/bin/env python3
"""
Guia completo para análise de métricas de fairness em sistemas de recomendação.
"""

import pandas as pd
import numpy as np


def explain_demographic_parity():
    """Explica Demographic Parity em detalhes"""
    
    print("=" * 80)
    print("📊 DEMOGRAPHIC PARITY (PARIDADE DEMOGRÁFICA)")
    print("=" * 80)
    
    print("\n🎯 DEFINIÇÃO:")
    print("Mede se a taxa de recomendações positivas é igual entre grupos demográficos")
    
    print("\n📐 FÓRMULA MATEMÁTICA:")
    print("P(ŷ = 1 | A = 0) = P(ŷ = 1 | A = 1)")
    print("Onde:")
    print("  • ŷ = predição (1 = recomendado, 0 = não recomendado)")
    print("  • A = atributo sensível (ex: gênero, idade)")
    
    print("\n🔢 CÁLCULO PRÁTICO:")
    print("Demographic Parity Difference = |P(ŷ=1|Male) - P(ŷ=1|Female)|")
    
    print("\n📊 EXEMPLO COM NOSSOS DADOS:")
    print("Modelo SVD - Gênero:")
    print("  • Taxa de recomendação para Homens: 52.3%")
    print("  • Taxa de recomendação para Mulheres: 50.3%")
    print("  • Diferença: |52.3% - 50.3%| = 2.0% = 0.020")
    print("  • Status: ✅ Justo (< 0.1 = 10%)")
    
    print("\n⚖️ INTERPRETAÇÃO:")
    print("  • 0.000: Perfeita paridade (ideal)")
    print("  • < 0.100: Aceitável (diferença < 10%)")
    print("  • 0.100-0.200: Moderadamente injusto")
    print("  • > 0.200: Altamente injusto")
    
    print("\n🎯 O QUE SIGNIFICA NA PRÁTICA:")
    print("  • Valor baixo: Sistema recomenda igualmente para todos os grupos")
    print("  • Valor alto: Sistema favorece um grupo sobre outro")
    print("  • Problema: Pode mascarar diferenças legítimas de preferência")


def explain_equalized_odds():
    """Explica Equalized Odds em detalhes"""
    
    print("\n\n" + "=" * 80)
    print("⚖️ EQUALIZED ODDS (ODDS EQUALIZADAS)")
    print("=" * 80)
    
    print("\n🎯 DEFINIÇÃO:")
    print("Mede se as taxas de verdadeiros e falsos positivos são iguais entre grupos")
    
    print("\n📐 FÓRMULA MATEMÁTICA:")
    print("P(ŷ = 1 | y = 1, A = 0) = P(ŷ = 1 | y = 1, A = 1)  [TPR]")
    print("P(ŷ = 1 | y = 0, A = 0) = P(ŷ = 1 | y = 0, A = 1)  [FPR]")
    print("Onde:")
    print("  • y = verdade (1 = item relevante, 0 = não relevante)")
    print("  • TPR = True Positive Rate (sensibilidade)")
    print("  • FPR = False Positive Rate")
    
    print("\n🔢 CÁLCULO PRÁTICO:")
    print("Equalized Odds Difference = max(|TPR_diff|, |FPR_diff|)")
    
    print("\n📊 EXEMPLO COM NOSSOS DADOS:")
    print("Modelo SVD - Gênero:")
    print("  • TPR Homens: 68.5% (acerta 68.5% dos itens relevantes)")
    print("  • TPR Mulheres: 66.2% (acerta 66.2% dos itens relevantes)")
    print("  • Diferença TPR: |68.5% - 66.2%| = 2.3%")
    print("  • Status: ✅ Justo (< 10%)")
    
    print("\n⚖️ INTERPRETAÇÃO:")
    print("  • 0.000: Perfeita equidade (ideal)")
    print("  • < 0.100: Aceitável")
    print("  • 0.100-0.200: Moderadamente injusto")
    print("  • > 0.200: Altamente injusto")
    
    print("\n🎯 O QUE SIGNIFICA NA PRÁTICA:")
    print("  • Valor baixo: Sistema tem mesma precisão para todos os grupos")
    print("  • Valor alto: Sistema é mais preciso para um grupo que outro")
    print("  • Mais rigorosa que Demographic Parity")


def analyze_our_results():
    """Analisa nossos resultados específicos"""
    
    print("\n\n" + "=" * 80)
    print("🔍 ANÁLISE DOS NOSSOS RESULTADOS")
    print("=" * 80)
    
    # Dados dos nossos modelos
    results = {
        'Modelo': ['Baseline', 'KNN', 'KNN Baseline', 'SVD', 'SVD++'],
        'Demo_Parity_Gender': [0.0186, 0.0222, 0.0159, 0.0202, 0.0169],
        'Demo_Parity_Age': [0.0879, 0.1231, 0.1050, 0.0806, 0.0734],
        'Equal_Odds_Gender': [0.0255, 0.0531, 0.0224, 0.0228, 0.0176],
        'Equal_Odds_Age': [0.0943, 0.1226, 0.1039, 0.0996, 0.1063]
    }
    
    df = pd.DataFrame(results)
    
    print("\n📊 ANÁLISE POR FEATURE SENSÍVEL:")
    
    print("\n🚹🚺 GÊNERO:")
    print("  Demographic Parity:")
    for _, row in df.iterrows():
        dp_gender = row['Demo_Parity_Gender']
        status = "✅" if dp_gender < 0.1 else "❌"
        percentage = dp_gender * 100
        print(f"    {row['Modelo']:<12}: {dp_gender:.4f} ({percentage:.1f}%) {status}")
    
    print("\n  Equalized Odds:")
    for _, row in df.iterrows():
        eo_gender = row['Equal_Odds_Gender']
        status = "✅" if eo_gender < 0.1 else "❌"
        percentage = eo_gender * 100
        print(f"    {row['Modelo']:<12}: {eo_gender:.4f} ({percentage:.1f}%) {status}")
    
    print("\n🎂 IDADE:")
    print("  Demographic Parity:")
    for _, row in df.iterrows():
        dp_age = row['Demo_Parity_Age']
        status = "✅" if dp_age < 0.1 else "❌"
        percentage = dp_age * 100
        print(f"    {row['Modelo']:<12}: {dp_age:.4f} ({percentage:.1f}%) {status}")
    
    print("\n  Equalized Odds:")
    for _, row in df.iterrows():
        eo_age = row['Equal_Odds_Age']
        status = "✅" if eo_age < 0.1 else "❌"
        percentage = eo_age * 100
        print(f"    {row['Modelo']:<12}: {eo_age:.4f} ({percentage:.1f}%) {status}")


def provide_interpretation_guidelines():
    """Fornece diretrizes de interpretação"""
    
    print("\n\n" + "=" * 80)
    print("📋 DIRETRIZES DE INTERPRETAÇÃO")
    print("=" * 80)
    
    print("\n🎯 THRESHOLDS RECOMENDADOS:")
    print("  • < 0.05 (5%):  Excelente fairness")
    print("  • 0.05-0.10:    Aceitável")
    print("  • 0.10-0.20:    Problemático - requer atenção")
    print("  • > 0.20 (20%): Inaceitável - requer intervenção")
    
    print("\n⚠️ CONTEXTO IMPORTANTE:")
    print("  • Nem toda diferença é discriminação")
    print("  • Preferências legítimas podem variar entre grupos")
    print("  • Considere o contexto do domínio")
    print("  • Balance fairness com utilidade")
    
    print("\n🔄 TRADE-OFFS:")
    print("  • Demographic Parity vs. Individual Fairness")
    print("  • Fairness vs. Accuracy")
    print("  • Fairness entre diferentes grupos")
    print("  • Curto prazo vs. Longo prazo")
    
    print("\n🚨 SINAIS DE ALERTA:")
    print("  • Diferenças > 10% consistentes")
    print("  • Padrões sistemáticos de bias")
    print("  • Grupos minoritários sempre prejudicados")
    print("  • Métricas piorando ao longo do tempo")


def provide_actionable_insights():
    """Fornece insights acionáveis"""
    
    print("\n\n" + "=" * 80)
    print("💡 INSIGHTS ACIONÁVEIS DOS NOSSOS RESULTADOS")
    print("=" * 80)
    
    print("\n✅ PONTOS POSITIVOS:")
    print("  • Todos os modelos são justos para GÊNERO")
    print("  • SVD e Baseline são completamente justos")
    print("  • Diferenças de gênero são mínimas (< 2.5%)")
    print("  • Nenhum modelo tem bias extremo")
    
    print("\n⚠️ ÁREAS DE PREOCUPAÇÃO:")
    print("  • IDADE é problemática em 3 de 5 modelos")
    print("  • KNN tem os piores resultados de fairness")
    print("  • Grupos etários minoritários podem estar sendo prejudicados")
    
    print("\n🎯 RECOMENDAÇÕES ESPECÍFICAS:")
    print("\n  Para PRODUÇÃO:")
    print("    1. Use SVD (melhor equilíbrio performance/fairness)")
    print("    2. Monitore grupos etários continuamente")
    print("    3. Evite KNN sem ajustes")
    
    print("\n  Para MELHORIA:")
    print("    1. Colete mais dados de grupos minoritários")
    print("    2. Implemente re-balanceamento por idade")
    print("    3. Considere post-processing fairness")
    print("    4. Teste algoritmos com constraints de fairness")
    
    print("\n  Para MONITORAMENTO:")
    print("    1. Acompanhe métricas por grupo mensalmente")
    print("    2. Defina alertas para diferenças > 8%")
    print("    3. Analise feedback de usuários por grupo")
    print("    4. Meça satisfação por segmento demográfico")


def main():
    """Função principal"""
    
    print("🎓 GUIA COMPLETO DE ANÁLISE DE FAIRNESS")
    print("Sistema de Recomendação - MovieLens 100k")
    
    explain_demographic_parity()
    explain_equalized_odds()
    analyze_our_results()
    provide_interpretation_guidelines()
    provide_actionable_insights()
    
    print("\n\n" + "=" * 80)
    print("✅ GUIA DE ANÁLISE COMPLETO!")
    print("=" * 80)
    print("\nPróximos passos:")
    print("1. Implemente monitoramento contínuo")
    print("2. Teste estratégias de mitigação")
    print("3. Valide com stakeholders do negócio")
    print("4. Documente decisões de fairness")


if __name__ == "__main__":
    main()
