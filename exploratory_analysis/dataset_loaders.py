from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
import pandas as pd
import json
from pathlib import Path
from surprise import Dataset, Reader
from surprise.dataset import DatasetAutoFolds


class BaseDatasetLoader(ABC):    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.ratings_df: Optional[pd.DataFrame] = None
        self.metadata_df: Optional[pd.DataFrame] = None
        self.surprise_dataset: Optional[DatasetAutoFolds] = None
        self._validate_path()
    
    def _validate_path(self) -> None:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path não encontrado: {self.dataset_path}")
    
    @abstractmethod
    def load_ratings(self) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def load_metadata(self) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        pass
    
    def to_surprise_dataset(self, rating_scale: Tuple[float, float] = (1, 5)) -> DatasetAutoFolds:
        if self.ratings_df is None:
            self.load_ratings()
        
        reader = Reader(rating_scale=rating_scale)
        
        self.surprise_dataset = Dataset.load_from_df(
            self.ratings_df[['user_id', 'item_id', 'rating']], 
            reader
        )
        
        return self.surprise_dataset
    
    def get_basic_stats(self) -> Dict[str, Any]:
        if self.ratings_df is None:
            self.load_ratings()
            
        stats = {
            'total_ratings': len(self.ratings_df),
            'unique_users': self.ratings_df['user_id'].nunique(),
            'unique_items': self.ratings_df['item_id'].nunique(),
            'rating_min': self.ratings_df['rating'].min(),
            'rating_max': self.ratings_df['rating'].max(),
            'rating_mean': self.ratings_df['rating'].mean(),
            'rating_std': self.ratings_df['rating'].std(),
            'sparsity': 1 - (len(self.ratings_df) / 
                           (self.ratings_df['user_id'].nunique() * 
                            self.ratings_df['item_id'].nunique()))
        }
        
        return stats


class AmazonMusicLoader(BaseDatasetLoader):
    """
    Este dataset contém reviews de produtos musicais da Amazon,
    incluindo ratings e metadados dos produtos.
    """
    
    def load_ratings(self) -> pd.DataFrame:
        json_file = self.dataset_path / "Digital_Music_5.json"
        
        if not json_file.exists():
            raise FileNotFoundError(f"Arquivo de ratings não encontrado: {json_file}")
        
        ratings_data = []
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    review = json.loads(line.strip())
                    ratings_data.append({
                        'user_id': review['reviewerID'],
                        'item_id': review['asin'],
                        'rating': float(review['overall']),
                        'timestamp': review.get('unixReviewTime', None),
                        'helpful': review.get('helpful', [0, 0])
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Erro ao processar linha: {e}")
                    continue
        
        self.ratings_df = pd.DataFrame(ratings_data)
        
        return self.ratings_df
    
    def load_metadata(self) -> pd.DataFrame:
        csv_file = self.dataset_path / "amazon_music_metadata.csv"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"Arquivo de metadados não encontrado: {csv_file}")
        
        try:
            self.metadata_df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Erro ao carregar metadados: {e}")
            self.metadata_df = pd.DataFrame()
        
        return self.metadata_df
    
    def get_dataset_info(self) -> Dict[str, Any]:
        if self.ratings_df is None:
            self.load_ratings()
        
        basic_stats = self.get_basic_stats()
        
        info = {
            **basic_stats,
            'dataset_name': 'Amazon Music',
            'dataset_type': 'E-commerce Reviews',
            'domain': 'Digital Music',
            'has_metadata': self.metadata_df is not None and not self.metadata_df.empty,
            'rating_scale': (1.0, 5.0),
            'data_format': 'JSON + CSV'
        }
        
        return info


class AnimeLoader(BaseDatasetLoader):
    """
    Este dataset contém ratings de animes do MyAnimeList.
    Inclui ratings explícitos e histórico de acessos.
    """
    
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        self.history_df: Optional[pd.DataFrame] = None
    
    def load_ratings(self) -> pd.DataFrame:
        ratings_file = self.dataset_path / "anime_ratings.dat"
        
        if not ratings_file.exists():
            raise FileNotFoundError(f"Arquivo de ratings não encontrado: {ratings_file}")
        
        self.ratings_df = pd.read_csv(
            ratings_file, 
            sep='\t',
            header=0,
            dtype={'User_ID': 'int32', 'Anime_ID': 'int32', 'Feedback': 'float32'}
        )
        
        self.ratings_df = self.ratings_df.rename(columns={
            'User_ID': 'user_id',
            'Anime_ID': 'item_id', 
            'Feedback': 'rating'
        })
        
        return self.ratings_df
    
    def load_metadata(self) -> pd.DataFrame:
        info_file = self.dataset_path / "anime_info.dat"
        
        if info_file.exists():
            try:
                self.metadata_df = pd.read_csv(info_file, sep='\t')
            except Exception as e:
                print(f"Erro ao carregar metadados: {e}")
                self.metadata_df = pd.DataFrame()
        else:
            self.metadata_df = pd.DataFrame()
        
        return self.metadata_df
    
    def load_history(self) -> pd.DataFrame:        
        history_file = self.dataset_path / "anime_history.dat"
        
        if not history_file.exists():
            raise FileNotFoundError(f"Arquivo de histórico não encontrado: {history_file}")
        
        # Ler o arquivo com separador de tab
        self.history_df = pd.read_csv(
            history_file, 
            sep='\t',
            header=0,
            dtype={'User_ID': 'int32', 'Anime_ID': 'int32', 'Feedback': 'int8'}
        )
        
        # Renomear colunas para padronizar
        self.history_df = self.history_df.rename(columns={
            'User_ID': 'user_id',
            'Anime_ID': 'item_id',
            'Feedback': 'accessed'
        })
        
        return self.history_df
    
    def get_dataset_info(self) -> Dict[str, Any]:
        if self.ratings_df is None:
            self.load_ratings()
        
        # Carregar também os dados de histórico
        if self.history_df is None:
            self.load_history()
        
        basic_stats = self.get_basic_stats()
        
        # Estatísticas do histórico
        history_stats = {}
        if self.history_df is not None and not self.history_df.empty:
            history_stats = {
                'total_history_records': len(self.history_df),
                'unique_users_history': self.history_df['user_id'].nunique(),
                'unique_items_history': self.history_df['item_id'].nunique(),
                'avg_items_per_user_history': len(self.history_df) / max(self.history_df['user_id'].nunique(), 1)
            }
        
        info = {
            **basic_stats,
            **history_stats,
            'dataset_name': 'Anime Recommendations',
            'dataset_type': 'Entertainment Ratings',
            'domain': 'Anime/Manga',
            'has_metadata': self.metadata_df is not None and not self.metadata_df.empty,
            'rating_scale': (1.0, 10.0),
            'data_format': 'DAT (Tab-separated)'
        }
        
        return info


class BookCrossingLoader(BaseDatasetLoader):
    """
    Este dataset contém ratings de livros do Book-Crossing community.
    Inclui ratings explícitos, histórico de acessos e informações demográficas dos usuários.
    """
    
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        self.history_df: Optional[pd.DataFrame] = None
        self.users_info_df: Optional[pd.DataFrame] = None
    
    def load_ratings(self) -> pd.DataFrame:
        ratings_file = self.dataset_path / "book_ratings.dat"
        
        if not ratings_file.exists():
            raise FileNotFoundError(f"Arquivo de ratings não encontrado: {ratings_file}")
        
        self.ratings_df = pd.read_csv(
            ratings_file, 
            sep='\t',
            header=0,
            dtype={'user': 'int32', 'item': 'int32', 'rating': 'float32'}
        )
        
        self.ratings_df = self.ratings_df.rename(columns={
            'user': 'user_id',
            'item': 'item_id'
        })
        
        return self.ratings_df
    
    def load_metadata(self) -> pd.DataFrame:
        info_file = self.dataset_path / "items_info.dat"
        
        if info_file.exists():
            try:
                self.metadata_df = pd.read_csv(
                    info_file, 
                    sep='\t',
                    encoding='utf-8',
                    on_bad_lines='skip'
                )
            except Exception as e:
                print(f"Erro ao carregar metadados: {e}")
                self.metadata_df = pd.DataFrame()
        else:
            self.metadata_df = pd.DataFrame()
        
        return self.metadata_df
    
    def load_history(self) -> pd.DataFrame:
        history_file = self.dataset_path / "book_history.dat"
        
        if not history_file.exists():
            raise FileNotFoundError(f"Arquivo de histórico não encontrado: {history_file}")
        
        self.history_df = pd.read_csv(
            history_file, 
            sep='\t',
            header=0,
            dtype={'user': 'int32', 'item': 'int32', 'accessed': 'int8'}
        )
        
        self.history_df = self.history_df.rename(columns={
            'user': 'user_id',
            'item': 'item_id'
        })

        return self.history_df
    
    def load_users_info(self) -> pd.DataFrame:
        users_file = self.dataset_path / "users_info.dat"
        
        if users_file.exists():
            try:
                self.users_info_df = pd.read_csv(
                    users_file, 
                    sep=r'\s+',  # Usar regex para múltiplos espaços
                    encoding='utf-8',
                    on_bad_lines='skip',
                    engine='python'  # Necessário para regex sep
                )
                
                if 'User-ID' in self.users_info_df.columns:
                    self.users_info_df['User-ID'] = pd.to_numeric(self.users_info_df['User-ID'], errors='coerce').astype('Int32')
                if 'Age' in self.users_info_df.columns:
                    self.users_info_df['Age'] = pd.to_numeric(self.users_info_df['Age'], errors='coerce')
                
                self.users_info_df = self.users_info_df.rename(columns={
                    'User-ID': 'user_id'
                })
            except Exception as e:
                print(f"Erro ao carregar informações dos usuários: {e}")
                self.users_info_df = pd.DataFrame()
        else:
            self.users_info_df = pd.DataFrame()
        
        return self.users_info_df
    
    def get_dataset_info(self) -> Dict[str, Any]:
        if self.ratings_df is None:
            self.load_ratings()
        
        if self.history_df is None:
            self.load_history()
        if self.users_info_df is None:
            self.load_users_info()
        
        basic_stats = self.get_basic_stats()
        
        history_stats = {}
        if self.history_df is not None and not self.history_df.empty:
            history_stats = {
                'total_history_records': len(self.history_df),
                'unique_users_history': self.history_df['user_id'].nunique(),
                'unique_items_history': self.history_df['item_id'].nunique(),
                'avg_items_per_user_history': len(self.history_df) / max(self.history_df['user_id'].nunique(), 1)
            }
        
        users_stats = {}
        if self.users_info_df is not None and not self.users_info_df.empty:
            valid_ages = self.users_info_df['Age'][(self.users_info_df['Age'] >= 5) & (self.users_info_df['Age'] <= 100)]
            
            users_stats = {
                'total_users_info': len(self.users_info_df),
                'users_with_age': len(valid_ages),
                'avg_user_age': valid_ages.mean() if len(valid_ages) > 0 else None,
                'min_user_age': valid_ages.min() if len(valid_ages) > 0 else None,
                'max_user_age': valid_ages.max() if len(valid_ages) > 0 else None
            }
        
        info = {
            **basic_stats,
            **history_stats,
            **users_stats,
            'dataset_name': 'Book Crossing',
            'dataset_type': 'Book Ratings',
            'domain': 'Books/Literature',
            'has_metadata': self.metadata_df is not None and not self.metadata_df.empty,
            'rating_scale': (1.0, 10.0),
            'data_format': 'DAT (Tab-separated)'
        }
        
        return info


def create_loader(dataset_name: str, base_path: str = "datasets/extracted") -> BaseDatasetLoader:
    loaders = {
        'amazonmusic': AmazonMusicLoader,
        'anime': AnimeLoader,
        'bookcrossing': BookCrossingLoader,
        'steam': SteamLoader,
        # 'retailrocket': RetailRocketLoader,
    }
    
    dataset_name = dataset_name.lower()
    if dataset_name not in loaders:
        raise ValueError(f"Dataset '{dataset_name}' não suportado. Opções: {list(loaders.keys())}")
    
    path_mapping = {
        'amazonmusic': 'AmazonMusic',
        'anime': 'anime',
        'bookcrossing': 'book_crossing',
        'steam': 'steam',
        'retailrocket': 'RetailRocket_Ecommerce'
    }
    
    dataset_path = Path(base_path) / path_mapping[dataset_name]
    
    return loaders[dataset_name](str(dataset_path))


class SteamLoader(BaseDatasetLoader):
    """
    Este dataset contém dados de jogos da Steam.
    Inclui dados de compras e horas jogadas como feedback implícito.
    """
    
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        self.purchase_df: Optional[pd.DataFrame] = None
        self.play_hours_df: Optional[pd.DataFrame] = None
        self.users_info_df: Optional[pd.DataFrame] = None
    
    def load_ratings(self) -> pd.DataFrame:
        """
        Steam não possui ratings explícitos tradicionais.
        Este método retorna um DataFrame vazio para manter compatibilidade.
        """
        # Steam não tem ratings explícitos, apenas horas jogadas
        self.ratings_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating'])
        return self.ratings_df
    
    def load_play_hours(self) -> pd.DataFrame:
        """Carrega dados de horas jogadas pelos usuários."""
        play_file = self.dataset_path / "game_play.dat"
        
        if not play_file.exists():
            raise FileNotFoundError(f"Arquivo de play hours não encontrado: {play_file}")
        
        self.play_hours_df = pd.read_csv(
            play_file, 
            sep='\t',
            header=0,
            dtype={'User_ID': 'int32', 'Game_ID': 'int32', 'Hours': 'float32'}
        )
        
        # Renomear colunas para padronizar
        self.play_hours_df = self.play_hours_df.rename(columns={
            'User_ID': 'user_id',
            'Game_ID': 'item_id'
        })
        
        return self.play_hours_df
    
    def load_metadata(self) -> pd.DataFrame:
        info_file = self.dataset_path / "item_info.dat"
        
        if info_file.exists():
            try:
                self.metadata_df = pd.read_csv(
                    info_file, 
                    sep='\t',
                    encoding='utf-8',
                    on_bad_lines='skip'
                )
            except Exception as e:
                print(f"Erro ao carregar metadados: {e}")
                self.metadata_df = pd.DataFrame()
        else:
            self.metadata_df = pd.DataFrame()
        
        return self.metadata_df
    
    def load_purchases(self) -> pd.DataFrame:
        """Carrega dados de compras dos usuários (feedback implícito)."""
        purchase_file = self.dataset_path / "game_purchase.dat"
        
        if not purchase_file.exists():
            raise FileNotFoundError(f"Arquivo de compras não encontrado: {purchase_file}")
        
        self.purchase_df = pd.read_csv(
            purchase_file, 
            sep='\t',
            header=0,
            dtype={'User_ID': 'int32', 'Game_ID': 'int32', 'Purchase': 'int8'}
        )
        
        # Renomear colunas para padronizar
        self.purchase_df = self.purchase_df.rename(columns={
            'User_ID': 'user_id',
            'Game_ID': 'item_id'
        })
        
        return self.purchase_df
    
    def load_users_info(self) -> pd.DataFrame:
        """Carrega informações dos usuários (mapeamento de IDs)."""
        users_file = self.dataset_path / "user_info.dat"
        
        if users_file.exists():
            try:
                self.users_info_df = pd.read_csv(
                    users_file, 
                    sep='\t',
                    encoding='utf-8',
                    on_bad_lines='skip'
                )
                
                # Renomear coluna para padronizar
                self.users_info_df = self.users_info_df.rename(columns={
                    'New_ID': 'user_id'
                })
                
            except Exception as e:
                print(f"Erro ao carregar informações dos usuários: {e}")
                self.users_info_df = pd.DataFrame()
        else:
            self.users_info_df = pd.DataFrame()
        
        return self.users_info_df
    
    def get_dataset_info(self) -> Dict[str, Any]:
        # Carregar os dados específicos do Steam
        if self.purchase_df is None:
            self.load_purchases()
        if self.play_hours_df is None:
            self.load_play_hours()
        if self.users_info_df is None:
            self.load_users_info()
        
        # Steam não tem ratings explícitos tradicionais
        # Carregar ratings vazio para compatibilidade
        if self.ratings_df is None:
            self.load_ratings()
        
        # Estatísticas de compras (equivale ao histórico)
        purchase_stats = {}
        if self.purchase_df is not None and not self.purchase_df.empty:
            purchase_stats = {
                'total_history_records': len(self.purchase_df),  # Usando compras como "histórico"
                'unique_users_history': self.purchase_df['user_id'].nunique(),
                'unique_items_history': self.purchase_df['item_id'].nunique(),
                'avg_items_per_user_history': len(self.purchase_df) / max(self.purchase_df['user_id'].nunique(), 1)
            }
        
        # Estatísticas de horas jogadas (dados específicos do Steam)
        play_stats = {}
        if self.play_hours_df is not None and not self.play_hours_df.empty:
            play_stats = {
                'total_play_hours': self.play_hours_df['Hours'].sum(),
                'avg_hours_per_game': self.play_hours_df['Hours'].mean(),
                'max_hours_single_game': self.play_hours_df['Hours'].max(),
                'total_play_sessions': len(self.play_hours_df),
                'unique_users_playing': self.play_hours_df['user_id'].nunique(),
                'unique_games_played': self.play_hours_df['item_id'].nunique(),
                'users_with_play_data': len(self.users_info_df) if self.users_info_df is not None else 0
            }
        
        # Estatísticas básicas (mas Steam não tem ratings reais)
        basic_stats = {
            'total_ratings': 0,  # Steam não tem ratings explícitos
            'unique_users': purchase_stats.get('unique_users_history', 0),
            'unique_items': purchase_stats.get('unique_items_history', 0),
            'rating_min': None,
            'rating_max': None,
            'rating_mean': None,
            'rating_std': None,
            'sparsity': None  # Não aplicável sem ratings
        }
        
        info = {
            **basic_stats,
            **purchase_stats,
            **play_stats,
            'dataset_name': 'Steam Games',
            'dataset_type': 'Gaming Platform Data',
            'domain': 'Video Games',
            'has_metadata': self.metadata_df is not None and not self.metadata_df.empty,
            'has_explicit_ratings': False,  # Steam não tem ratings explícitos
            'has_play_hours': True,
            'has_purchase_data': True,
            'data_format': 'DAT (Tab-separated)',
            'data_description': 'Horas jogadas e dados de compras (sem ratings explícitos)'
        }
        
        return info
