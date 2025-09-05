import sys
from pathlib import Path
import pandas as pd
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent))

from dataset_loaders import create_loader


def format_stats_table(stats: Dict[str, Any]) -> str:
    def format_number(value):
        if isinstance(value, float):
            if value < 0.01:
                return f"{value:.4f}"
            else:
                return f"{value:.2f}"
        elif isinstance(value, int):
            return f"{value:,}"
        else:
            return str(value)
    
    table = "\n\n" + "="*80 + "\n"
    table += f"DATASET: {stats.get('dataset_name', 'N/A')}\n"
    # table += "="*80 + "\n"
    # table += f"Tipo: {stats.get('dataset_type', 'N/A')}\n"
    # table += f"Domínio: {stats.get('domain', 'N/A')}\n"
    # table += f"Formato: {stats.get('data_format', 'N/A')}\n"
    # table += f"Escala de Rating: {stats.get('rating_scale', 'N/A')}\n"
    # table += f"Possui Metadados: {'Sim' if stats.get('has_metadata', False) else 'Não'}\n"
    table += "\n" + "-"*80 + "\n"
    table += "ESTATÍSTICAS BÁSICAS\n"
    table += "-"*80 + "\n"
    
    # Verificar se o dataset tem ratings explícitos
    has_explicit_ratings = stats.get('has_explicit_ratings', True)
    
    if has_explicit_ratings and stats.get('total_ratings', 0) > 0:
        table += f"Total de Ratings: {format_number(stats.get('total_ratings', 0))}\n"
        table += f"Usuários Únicos: {format_number(stats.get('unique_users', 0))}\n"
        table += f"Items Únicos: {format_number(stats.get('unique_items', 0))}\n"
        table += f"Rating Mínimo: {format_number(stats.get('rating_min', 0))}\n"
        table += f"Rating Máximo: {format_number(stats.get('rating_max', 0))}\n"
        table += f"Rating Médio: {format_number(stats.get('rating_mean', 0))}\n"
        table += f"Desvio Padrão: {format_number(stats.get('rating_std', 0))}\n"
        table += f"Esparsidade: {(stats.get('sparsity', 0)*100):.4f}%\n"
        
        density = 1 - stats.get('sparsity', 1)
        table += f"Densidade: {(density*100):.4f}%\n"
        
        avg_ratings_per_user = stats.get('total_ratings', 0) / max(stats.get('unique_users', 1), 1)
        avg_ratings_per_item = stats.get('total_ratings', 0) / max(stats.get('unique_items', 1), 1)
        
        table += f"Ratings/Usuário: {format_number(avg_ratings_per_user)}\n"
        table += f"Ratings/Item: {format_number(avg_ratings_per_item)}\n"
    else:
        # Dataset sem ratings explícitos (como Steam)
        table += f"Usuários Únicos: {format_number(stats.get('unique_users', 0))}\n"
        table += f"Items Únicos: {format_number(stats.get('unique_items', 0))}\n"
        table += f"Tipo de Dados: {stats.get('data_description', 'Dados comportamentais')}\n"
    
    if 'total_history_records' in stats:
        table += "\n" + "-"*80 + "\n"
        table += "ESTATÍSTICAS DE HISTÓRICO\n"
        table += "-"*80 + "\n"
        table += f"Total de Registros de Histórico: {format_number(stats.get('total_history_records', 0))}\n"
        table += f"Usuários Únicos: {format_number(stats.get('unique_users_history', 0))}\n"
        table += f"Items Únicos: {format_number(stats.get('unique_items_history', 0))}\n"
        table += f"Items/Usuário: {format_number(stats.get('avg_items_per_user_history', 0))}\n"
    
    if 'total_users_info' in stats:
        table += "\n" + "-"*80 + "\n"
        table += "INFORMAÇÕES DEMOGRÁFICAS DOS USUÁRIOS\n"
        table += "-"*80 + "\n"
        table += f"Total de Usuários com Info: {format_number(stats.get('total_users_info', 0))}\n"
        # table += f"Usuários com Idade Válida: {format_number(stats.get('users_with_age', 0))}\n"
        
        if stats.get('avg_user_age') is not None:
            table += f"Idade Média: {format_number(stats.get('avg_user_age', 0))} anos\n"
            table += f"Idade Mínima: {format_number(stats.get('min_user_age', 0))} anos\n"
            table += f"Idade Máxima: {format_number(stats.get('max_user_age', 0))} anos\n"
        else:
            table += "Dados de idade não disponíveis\n"
    
    if 'total_play_hours' in stats:
        table += "\n" + "-"*80 + "\n"
        table += "DADOS DE HORAS JOGADAS (STEAM)\n"
        table += "-"*80 + "\n"
        table += f"Total de Sessões de Jogo: {format_number(stats.get('total_play_sessions', 0))}\n"
        table += f"Total de Horas Jogadas: {format_number(stats.get('total_play_hours', 0))} horas\n"
        table += f"Usuários que Jogaram: {format_number(stats.get('unique_users_playing', 0))}\n"
        table += f"Jogos com Dados: {format_number(stats.get('unique_games_played', 0))}\n"
        table += f"Horas/Jogo: {format_number(stats.get('avg_hours_per_game', 0))} horas\n"
        table += f"Maior Tempo em um Jogo: {format_number(stats.get('max_hours_single_game', 0))} horas\n"
    
    table += "="*80 + "\n"
    
    return table


def analyze_dataset(dataset_name: str, base_path: str = "/Users/miguelferraz/Projects/Personal/unicamp/pfg-sis-rec/datasets/extracted") -> Dict[str, Any]:
    try:
        loader = create_loader(dataset_name, base_path)
        
        loader.load_ratings()
        loader.load_metadata()
        
        stats = loader.get_dataset_info()
        
        return stats
        
    except Exception as e:
        print(f"Erro ao analisar dataset {dataset_name}: {str(e)}")
        return {
            'dataset_name': dataset_name,
            'error': str(e),
            'status': 'failed'
        }


def analyze_all_datasets(base_path: str = "/Users/miguelferraz/Projects/Personal/unicamp/pfg-sis-rec/datasets/extracted") -> Dict[str, Dict[str, Any]]:
    datasets_to_analyze = ['amazonmusic', 'anime', 'bookcrossing', 'steam']
    
    results = {}
    
    for dataset_name in datasets_to_analyze:
        
        try:
            stats = analyze_dataset(dataset_name, base_path)
            results[dataset_name] = stats
            
            if 'error' not in stats:
                print(format_stats_table(stats))
            else:
                print(f"ERRO ao processar {dataset_name}: {stats['error']}")
                
        except Exception as e:
            error_msg = f"Erro inesperado ao processar {dataset_name}: {str(e)}"
            print(error_msg)
            print(f"ERRO: {error_msg}")
            results[dataset_name] = {'error': str(e), 'status': 'failed'}
    
    return results


def generate_comparison_summary(results: Dict[str, Dict[str, Any]]) -> None:
    print("\n" + "="*100)
    print("RESUMO COMPARATIVO DOS DATASETS")
    print("="*100)
    
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not successful_results:
        print("Nenhum dataset foi carregado com sucesso para comparação.")
        return
    
    comparison_data = []
    
    for dataset_name, stats in successful_results.items():
        comparison_data.append({
            'Dataset': stats.get('dataset_name', dataset_name),
            'Domínio': stats.get('domain', 'N/A'),
            'Total Ratings': stats.get('total_ratings', 0),
            'Usuários': stats.get('unique_users', 0),
            'Items': stats.get('unique_items', 0),
            'Esparsidade (%)': f"{stats.get('sparsity', 0)*100:.2f}%",
            'Rating Médio': f"{stats.get('rating_mean', 0):.2f}",
            'Escala': f"{stats.get('rating_scale', (0, 0))[0]}-{stats.get('rating_scale', (0, 0))[1]}"
        })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        print(f"\n{'-'*60}")
        print("ESTATÍSTICAS GERAIS:")
        print(f"{'-'*60}")
        
        total_ratings = sum(stats.get('total_ratings', 0) for stats in successful_results.values())
        total_users = sum(stats.get('unique_users', 0) for stats in successful_results.values())
        total_items = sum(stats.get('unique_items', 0) for stats in successful_results.values())
        
        print(f"Total de Ratings (todos datasets): {total_ratings:,}")
        print(f"Total de Usuários (todos datasets): {total_users:,}")
        print(f"Total de Items (todos datasets): {total_items:,}")
        
        sparsities = [(name, stats.get('sparsity', 1)) for name, stats in successful_results.items()]
        if sparsities:
            most_sparse = min(sparsities, key=lambda x: x[1])
            least_sparse = max(sparsities, key=lambda x: x[1])
            
            print(f"Dataset mais denso: {successful_results[most_sparse[0]].get('dataset_name')} ({(1-most_sparse[1])*100:.2f}% densidade)")
            print(f"Dataset mais esparso: {successful_results[least_sparse[0]].get('dataset_name')} ({least_sparse[1]*100:.2f}% esparsidade)")


def main():
    try:
        results = analyze_all_datasets()
        
        # generate_comparison_summary(results)
        
        failed = len([r for r in results.values() if 'error' in r])
        
        print(f"Datasets com erro: {failed}")
        
        if failed > 0:
            print("\nDatasets com erro:")
            for name, result in results.items():
                if 'error' in result:
                    print(f"  - {name}: {result['error']}")
        
    except Exception as e:
        print(f"Erro na execução principal: {str(e)}")
        print(f"Erro na execução: {str(e)}")


if __name__ == "__main__":
    main()
