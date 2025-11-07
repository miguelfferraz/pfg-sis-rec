import pytest

from src.loaders.base_loader import BaseDatasetLoader
from src.loaders.loader_factory import LoaderFactory, LoaderRegistry


class TestLoaderRegistry:

    def test_register_and_get_loader_info(self):
        registry = LoaderRegistry()

        class TestLoader(BaseDatasetLoader):
            DEFAULT_RATING_SCALE = (1, 5)

            def _load_ratings(self):
                pass

            def _load_users(self):
                pass

            def _load_items(self):
                pass

        registry.register("test", TestLoader, "test_path", "Test description")

        info = registry.get_loader_info("test")
        assert info["class"] == TestLoader
        assert info["path_suffix"] == "test_path"
        assert info["description"] == "Test description"

    def test_case_insensitive_registration(self):
        registry = LoaderRegistry()

        class TestLoader(BaseDatasetLoader):
            DEFAULT_RATING_SCALE = (1, 5)

            def _load_ratings(self):
                pass

            def _load_users(self):
                pass

            def _load_items(self):
                pass

        registry.register("TEST", TestLoader, "test_path")

        assert registry.is_registered("test")
        assert registry.is_registered("TEST")
        assert registry.is_registered("Test")

    def test_list_available(self):
        registry = LoaderRegistry()

        class TestLoader1(BaseDatasetLoader):
            DEFAULT_RATING_SCALE = (1, 5)

            def _load_ratings(self):
                pass

            def _load_users(self):
                pass

            def _load_items(self):
                pass

        class TestLoader2(BaseDatasetLoader):
            DEFAULT_RATING_SCALE = (1, 5)

            def _load_ratings(self):
                pass

            def _load_users(self):
                pass

            def _load_items(self):
                pass

        registry.register("loader1", TestLoader1, "path1")
        registry.register("loader2", TestLoader2, "path2")

        available = registry.list_available()
        assert "loader1" in available
        assert "loader2" in available
        assert len(available) == 2

    def test_get_nonexistent_loader_raises_error(self):
        registry = LoaderRegistry()

        with pytest.raises(ValueError, match="Loader 'nonexistent' not registered"):
            registry.get_loader_info("nonexistent")


class TestLoaderFactory:

    def test_factory_initialization(self):
        factory = LoaderFactory()

        available = factory.registry.list_available()
        expected_loaders = ["amazonmusic", "anime", "bookcrossing", "movielens", "movielens1m"]

        for loader in expected_loaders:
            assert loader in available

    def test_create_loader_invalid_dataset(self):
        factory = LoaderFactory()

        with pytest.raises(ValueError, match="Dataset 'invalid' not supported"):
            factory.create_loader("invalid")

    def test_register_custom_loader(self):
        factory = LoaderFactory()

        class CustomLoader(BaseDatasetLoader):
            DEFAULT_RATING_SCALE = (1, 5)

            def _load_ratings(self):
                pass

            def _load_users(self):
                pass

            def _load_items(self):
                pass

        factory.register_custom_loader("custom", CustomLoader, "custom_path", "Custom loader")

        assert factory.registry.is_registered("custom")
        info = factory.registry.get_loader_info("custom")
        assert info["class"] == CustomLoader
        assert info["description"] == "Custom loader"

    def test_register_invalid_loader_class_raises_error(self):
        factory = LoaderFactory()

        class InvalidLoader:
            pass

        with pytest.raises(TypeError, match="Loader class must inherit from BaseDatasetLoader"):
            factory.register_custom_loader("invalid", InvalidLoader, "path")

    def test_list_available_loaders(self):
        factory = LoaderFactory()

        loaders = factory.list_available_loaders()

        assert isinstance(loaders, list)
        assert len(loaders) > 0

        for loader in loaders:
            assert "name" in loader
            assert "class" in loader
            assert "description" in loader

    def test_get_loader_description(self):
        factory = LoaderFactory()

        description = factory.get_loader_description("movielens")
        assert isinstance(description, str)
        assert len(description) > 0

    def test_create_loader_with_custom_path(self, temp_dir):
        factory = LoaderFactory()

        custom_path = temp_dir / "custom_movielens"
        custom_path.mkdir()
        ratings_file = custom_path / "u.data"
        ratings_file.write_text("1\t1\t5\t123456789\n")

        loader = factory.create_loader("movielens", str(custom_path))
        assert loader is not None
        assert loader.dataset_path == custom_path
