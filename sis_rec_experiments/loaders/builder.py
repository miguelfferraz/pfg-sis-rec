from pathlib import Path

from sis_rec_experiments.loaders import (
    AmazonMusicLoader,
    AnimeLoader,
    BaseDatasetLoader,
    BookCrossingLoader,
    SteamLoader,
)


def create_loader(dataset_name: str, base_path: str = "datasets/extracted") -> BaseDatasetLoader:
    loaders = {
        "amazonmusic": AmazonMusicLoader,
        "anime": AnimeLoader,
        "bookcrossing": BookCrossingLoader,
        "steam": SteamLoader,
    }

    dataset_name = dataset_name.lower()
    if dataset_name not in loaders:
        raise ValueError(f"Dataset '{dataset_name}' não suportado. Opções: {list(loaders.keys())}")

    path_mapping = {
        "amazonmusic": "AmazonMusic",
        "anime": "anime",
        "bookcrossing": "book_crossing",
        "steam": "steam",
        "retailrocket": "RetailRocket_Ecommerce",
    }

    dataset_path = Path(base_path) / path_mapping[dataset_name]

    return loaders[dataset_name](str(dataset_path))
