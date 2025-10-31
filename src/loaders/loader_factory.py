from pathlib import Path
from typing import Dict, List, Optional, Type

from src.loaders.base_loader import BaseDatasetLoader


class LoaderRegistry:

    def __init__(self):
        self._loaders: Dict[str, Dict] = {}

    def register(self, name: str, loader_class: Type[BaseDatasetLoader], path_suffix: str, description: str = ""):
        self._loaders[name.lower()] = {"class": loader_class, "path_suffix": path_suffix, "description": description}

    def get_loader_info(self, name: str) -> Dict:
        name = name.lower()
        if name not in self._loaders:
            raise ValueError(f"Loader '{name}' not registered")
        return self._loaders[name]

    def list_available(self) -> List[str]:
        return list(self._loaders.keys())

    def is_registered(self, name: str) -> bool:
        return name.lower() in self._loaders


class LoaderFactory:
    def __init__(self, base_path: str = "src/datasets/extracted"):
        self.base_path = Path(base_path)
        self.registry = LoaderRegistry()
        self._register_default_loaders()

    def _register_default_loaders(self):
        from src.loaders.amazon_music_loader import AmazonMusicLoader
        from src.loaders.anime_loader import AnimeLoader
        from src.loaders.book_crossing_loader import BookCrossingLoader
        from src.loaders.movielens_1m_loader import MovieLens1MLoader
        from src.loaders.movielens_loader import MovieLensLoader

        self.registry.register(
            "amazonmusic", AmazonMusicLoader, "AmazonMusic", "Amazon Music reviews dataset with ratings and metadata"
        )

        self.registry.register(
            "anime", AnimeLoader, "anime", "Anime ratings dataset with user preferences and metadata"
        )

        self.registry.register(
            "bookcrossing",
            BookCrossingLoader,
            "book_crossing",
            "Book-Crossing dataset with ratings and demographic data",
        )

        self.registry.register(
            "movielens", MovieLensLoader, "movielens", "MovieLens 100k dataset with movie ratings and user demographics"
        )

        self.registry.register(
            "movielens1m", MovieLens1MLoader, "ml-1m", "MovieLens 1M dataset with movie ratings and user demographics"
        )

    def create_loader(self, dataset_name: str, custom_path: Optional[str] = None) -> BaseDatasetLoader:
        """
        Args:
            dataset_name: Dataset name
            custom_path: Custom path (optional)

        Returns:
            Appropriate loader instance for the dataset

        Raises:
            ValueError: If the dataset is not supported
            FileNotFoundError: If the dataset path does not exist
        """
        dataset_name = dataset_name.lower()

        if not self.registry.is_registered(dataset_name):
            available = self.registry.list_available()
            raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {available}")

        loader_info = self.registry.get_loader_info(dataset_name)

        if custom_path:
            dataset_path = Path(custom_path)
        else:
            dataset_path = self.base_path / loader_info["path_suffix"]

        return loader_info["class"](str(dataset_path))

    def register_custom_loader(
        self, name: str, loader_class: Type[BaseDatasetLoader], path_suffix: str, description: str = ""
    ):
        """
        Args:
            name: Loader name
            loader_class: Loader class
            path_suffix: Path suffix
            description: Loader description
        """
        if not issubclass(loader_class, BaseDatasetLoader):
            raise TypeError("Loader class must inherit from BaseDatasetLoader")

        self.registry.register(name, loader_class, path_suffix, description)

    def list_available_loaders(self) -> List[Dict[str, str]]:
        result = []
        for name in self.registry.list_available():
            loader_info = self.registry.get_loader_info(name)
            result.append(
                {"name": name, "class": loader_info["class"].__name__, "description": loader_info["description"]}
            )
        return result

    def get_loader_description(self, dataset_name: str) -> str:
        loader_info = self.registry.get_loader_info(dataset_name)
        return loader_info["description"]
