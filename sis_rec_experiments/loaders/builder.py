from pathlib import Path

from sis_rec_experiments.loaders.amazon_music_loader import AmazonMusicLoader
from sis_rec_experiments.loaders.anime_loader import AnimeLoader
from sis_rec_experiments.loaders.base_loader import BaseDatasetLoader
from sis_rec_experiments.loaders.book_crossing_loader import BookCrossingLoader
from sis_rec_experiments.loaders.movielens_loader import MovieLensLoader
from sis_rec_experiments.loaders.movielens_1m_loader import MovieLens1MLoader
from sis_rec_experiments.loaders.steam_loader import SteamLoader


def create_loader(dataset_name: str, base_path: str = "sis_rec_experiments/datasets/extracted") -> BaseDatasetLoader:
    loaders = {
        "amazonmusic": AmazonMusicLoader,
        "anime": AnimeLoader,
        "bookcrossing": BookCrossingLoader,
        "movielens": MovieLensLoader,
        "movielens1m": MovieLens1MLoader,
        "steam": SteamLoader,
    }

    dataset_name = dataset_name.lower()
    if dataset_name not in loaders:
        raise ValueError(f"Dataset '{dataset_name}' não suportado. Opções: {list(loaders.keys())}")

    path_mapping = {
        "amazonmusic": "AmazonMusic",
        "anime": "anime",
        "bookcrossing": "book_crossing",
        "movielens": "movielens",
        "movielens1m": "ml-1m",
        "steam": "steam",
        "retailrocket": "RetailRocket_Ecommerce",
    }

    dataset_path = Path(base_path) / path_mapping[dataset_name]

    return loaders[dataset_name](str(dataset_path))
