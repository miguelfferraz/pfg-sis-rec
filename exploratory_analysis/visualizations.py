import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataset_loaders import create_loader

# Configuração para gráficos mais bonitos
plt.style.use('default')
sns.set_palette("husl")


def plot_rating_distribution(dataset_name: str, base_path: str = "/Users/miguelferraz/Projects/Personal/unicamp/pfg-sis-rec/datasets/extracted", 
                            save_path: str = "visualizations", figsize: tuple = (10, 6)) -> None:
    Path(save_path).mkdir(exist_ok=True)
    
    try:
        loader = create_loader(dataset_name, base_path)
        loader.load_ratings()
        
        if loader.ratings_df is None or loader.ratings_df.empty:
            print(f"Dataset {dataset_name} não possui ratings explícitos")
            return
        
        info = loader.get_dataset_info()
        
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(f'Distribuição de Ratings - {info["dataset_name"]}', fontsize=16, fontweight='bold')
        
        ratings = loader.ratings_df['rating']
        
        ax.hist(ratings, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequência')
        ax.set_title('Histograma de Ratings')
        ax.grid(True, alpha=0.3)
        
        mean_rating = ratings.mean()
        median_rating = ratings.median()
        ax.axvline(mean_rating, color='red', linestyle='--', label=f'Média: {mean_rating:.2f}')
        ax.axvline(median_rating, color='orange', linestyle='--', label=f'Mediana: {median_rating:.2f}')
        ax.legend()
        
        stats_text = f"""Estatísticas:
Total de Ratings: {len(ratings):,}
Mínimo: {ratings.min():.1f}
Máximo: {ratings.max():.1f}
Média: {mean_rating:.2f}
Mediana: {median_rating:.2f}
Desvio Padrão: {ratings.std():.2f}
Escala: {info.get('rating_scale', 'N/A')}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        filename = f"{save_path}/rating_distribution_{dataset_name.lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo: {filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"Erro ao gerar gráfico para {dataset_name}: {str(e)}")


def plot_rating_distribution_by_value(dataset_name: str, base_path: str = "/Users/miguelferraz/Projects/Personal/unicamp/pfg-sis-rec/datasets/extracted",
                                     save_path: str = "visualizations", figsize: tuple = (12, 8)) -> None:
    Path(save_path).mkdir(exist_ok=True)
    
    try:
        loader = create_loader(dataset_name, base_path)
        loader.load_ratings()
        
        if loader.ratings_df is None or loader.ratings_df.empty:
            print(f"Dataset {dataset_name} não possui ratings explícitos")
            return
        
        info = loader.get_dataset_info()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle(f'Análise Detalhada de Ratings - {info["dataset_name"]}', fontsize=16, fontweight='bold')
        
        ratings = loader.ratings_df['rating']
        
        rating_counts = ratings.value_counts().sort_index()
        bars = ax1.bar(rating_counts.index, rating_counts.values, alpha=0.7, color='lightcoral', edgecolor='black')
        ax1.set_xlabel('Valor do Rating')
        ax1.set_ylabel('Quantidade')
        ax1.set_title('Distribuição por Valor de Rating')
        ax1.grid(True, alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=9)
        
        rating_percentages = (rating_counts / rating_counts.sum()) * 100
        bars2 = ax2.bar(rating_percentages.index, rating_percentages.values, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Valor do Rating')
        ax2.set_ylabel('Percentual (%)')
        ax2.set_title('Distribuição Percentual por Valor de Rating')
        ax2.grid(True, alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        filename = f"{save_path}/rating_values_{dataset_name.lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo: {filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"Erro ao gerar gráfico detalhado para {dataset_name}: {str(e)}")


def plot_comparative_ratings(datasets: List[str] = ['amazonmusic', 'anime', 'bookcrossing'],
                           base_path: str = "/Users/miguelferraz/Projects/Personal/unicamp/pfg-sis-rec/datasets/extracted",
                           save_path: str = "visualizations", figsize: tuple = (15, 10)) -> None:
    Path(save_path).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Comparação de Distribuições de Ratings entre Datasets', fontsize=16, fontweight='bold')
    
    all_ratings = {}
    dataset_info = {}
    
    for dataset_name in datasets:
        try:
            loader = create_loader(dataset_name, base_path)
            loader.load_ratings()
            
            if loader.ratings_df is not None and not loader.ratings_df.empty:
                all_ratings[dataset_name] = loader.ratings_df['rating']
                dataset_info[dataset_name] = loader.get_dataset_info()
                
        except Exception as e:
            print(f"Erro ao carregar {dataset_name}: {str(e)}")
    
    if not all_ratings:
        print("Nenhum dataset com ratings foi carregado com sucesso")
        return
    
    ax1 = axes[0]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    for i, (name, ratings) in enumerate(all_ratings.items()):
        info = dataset_info[name]
        ax1.hist(ratings, bins=20, alpha=0.6, label=f'{info["dataset_name"]} (μ={ratings.mean():.2f})', 
                color=colors[i % len(colors)], density=True)
    ax1.set_xlabel('Rating')
    ax1.set_ylabel('Densidade')
    ax1.set_title('Distribuições Normalizadas')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    stats_data = []
    for name, ratings in all_ratings.items():
        stats_data.append({
            'Dataset': dataset_info[name]["dataset_name"],
            'Média': ratings.mean(),
            'Mediana': ratings.median(),
            'Desvio Padrão': ratings.std()
        })
    
    stats_df = pd.DataFrame(stats_data)
    x = np.arange(len(stats_df))
    width = 0.25
    
    ax2.bar(x - width, stats_df['Média'], width, label='Média', color='skyblue')
    ax2.bar(x, stats_df['Mediana'], width, label='Mediana', color='lightcoral')
    ax2.bar(x + width, stats_df['Desvio Padrão'], width, label='Desvio Padrão', color='lightgreen')
    
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Valor')
    ax2.set_title('Estatísticas Comparativas')
    ax2.set_xticks(x)
    ax2.set_xticklabels(stats_df['Dataset'], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    ax3.axis('tight')
    ax3.axis('off')
    
    table_data = []
    for name, ratings in all_ratings.items():
        info = dataset_info[name]
        table_data.append([
            info["dataset_name"],
            f'{len(ratings):,}',
            f'{ratings.min():.1f} - {ratings.max():.1f}',
            f'{ratings.mean():.2f}',
            f'{ratings.std():.2f}'
        ])
    
    table = ax3.table(cellText=table_data,
                     colLabels=['Dataset', 'Total Ratings', 'Escala', 'Média', 'Desvio Padrão'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax3.set_title('Resumo Estatístico', pad=20)
    
    plt.tight_layout()
    
    filename = f"{save_path}/comparative_ratings.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Gráfico comparativo salvo: {filename}")
    
    plt.show()


def generate_all_visualizations(base_path: str = "/Users/miguelferraz/Projects/Personal/unicamp/pfg-sis-rec/datasets/extracted") -> None:
    datasets_with_ratings = ['amazonmusic', 'anime', 'bookcrossing']
    
    print("Gerando visualizações de distribuição de ratings...")
    print("=" * 60)
    
    for dataset_name in datasets_with_ratings:
        print(f"\nGerando gráficos para {dataset_name.upper()}...")
        plot_rating_distribution(dataset_name, base_path)
        plot_rating_distribution_by_value(dataset_name, base_path)
    
    print(f"\nGerando gráfico comparativo...")
    plot_comparative_ratings(datasets_with_ratings, base_path)
    
    print("\n" + "=" * 60)
    print("Todas as visualizações foram geradas com sucesso!")
    print("Verifique a pasta 'visualizations' para ver os gráficos.")


if __name__ == "__main__":
    generate_all_visualizations()
