from pathlib import Path

from src.loaders.amazon_music_loader import AmazonMusicLoader
from src.loaders.anime_loader import AnimeLoader
from src.loaders.base_loader import BaseDatasetLoader
from src.loaders.book_crossing_loader import BookCrossingLoader
from src.loaders.movielens_1m_loader import MovieLens1MLoader
from src.loaders.movielens_loader import MovieLensLoader


def create_loader(dataset_name: str, base_path: str = "sis_rec_experiments/datasets/extracted") -> BaseDatasetLoader:
    loaders = {
        "amazonmusic": AmazonMusicLoader,
        "anime": AnimeLoader,
        "bookcrossing": BookCrossingLoader,
        "movielens": MovieLensLoader,
        "movielens1m": MovieLens1MLoader,
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
    }

    dataset_path = Path(base_path) / path_mapping[dataset_name]

    return loaders[dataset_name](str(dataset_path))
