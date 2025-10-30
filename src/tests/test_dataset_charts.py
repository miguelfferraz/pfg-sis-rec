import unittest
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import pandas as pd

from sis_rec_experiments.outputs.dataset_charts import (
    ChartConfig,
    ChartData,
    ComparativeChart,
    DatasetVisualizer,
    DetailedRatingChart,
    RatingDistributionChart,
)


class TestChartData(unittest.TestCase):

    @patch("sis_rec_experiments.outputs.dataset_charts.create_loader")
    def test_chart_data_from_loader_success(self, mock_create_loader):
        mock_loader = MagicMock()
        mock_loader.ratings_df = pd.DataFrame(
            {"user_id": [1, 2, 3, 4, 5], "item_id": [1, 1, 2, 2, 3], "rating": [4.0, 5.0, 3.0, 4.5, 2.0]}
        )
        mock_loader.get_dataset_info.return_value = {"dataset_name": "Test Dataset", "rating_scale": (1.0, 5.0)}
        mock_create_loader.return_value = mock_loader

        data = ChartData.from_loader("test_dataset")

        self.assertTrue(data.has_ratings)
        self.assertEqual(data.dataset_name, "test_dataset")
        self.assertEqual(data.display_name, "Test Dataset")
        self.assertEqual(len(data.ratings), 5)
        self.assertAlmostEqual(data.stats["mean"], 3.7)
        self.assertEqual(data.stats["count"], 5)

    @patch("sis_rec_experiments.outputs.dataset_charts.create_loader")
    def test_chart_data_from_loader_no_ratings(self, mock_create_loader):
        mock_loader = MagicMock()
        mock_loader.ratings_df = None
        mock_create_loader.return_value = mock_loader

        data = ChartData.from_loader("steam_dataset")

        self.assertFalse(data.has_ratings)
        self.assertEqual(data.dataset_name, "steam_dataset")
        self.assertEqual(len(data.ratings), 0)

    @patch("sis_rec_experiments.outputs.dataset_charts.create_loader")
    def test_chart_data_from_loader_error(self, mock_create_loader):
        mock_create_loader.side_effect = Exception("Dataset not found")

        data = ChartData.from_loader("invalid_dataset")

        self.assertFalse(data.has_ratings)
        self.assertEqual(data.dataset_name, "invalid_dataset")


class TestChartConfig(unittest.TestCase):

    def test_chart_config_defaults(self):
        config = ChartConfig()

        self.assertEqual(config.style, "default")
        self.assertEqual(config.palette, "husl")
        self.assertEqual(config.dpi, 300)
        self.assertEqual(config.figsize_single, (10, 6))
        self.assertEqual(config.alpha, 0.7)

    @patch("matplotlib.pyplot.style.use")
    @patch("seaborn.set_palette")
    def test_chart_config_apply_style(self, mock_set_palette, mock_style_use):
        config = ChartConfig()
        config.apply_style()

        mock_style_use.assert_called_once_with("default")
        mock_set_palette.assert_called_once_with("husl")


class TestRatingDistributionChart(unittest.TestCase):

    def setUp(self):
        self.sample_data = ChartData(
            dataset_name="test",
            display_name="Test Dataset",
            ratings=pd.Series([4.0, 5.0, 3.0, 4.5, 2.0]),
            stats={
                "mean": 3.7,
                "median": 4.0,
                "std": 1.1,
                "min": 2.0,
                "max": 5.0,
                "count": 5,
                "rating_scale": (1.0, 5.0),
            },
            has_ratings=True,
        )

    @patch("matplotlib.pyplot.show")
    def test_create_chart_success(self, mock_show):
        chart = RatingDistributionChart()
        fig = chart.create_chart(self.sample_data)

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_chart_no_ratings(self):
        no_rating_data = ChartData(
            dataset_name="steam", display_name="Steam", ratings=pd.Series(dtype=float), stats={}, has_ratings=False
        )

        chart = RatingDistributionChart()

        with self.assertRaises(ValueError):
            chart.create_chart(no_rating_data)

    @patch("pathlib.Path.mkdir")
    @patch("matplotlib.pyplot.Figure.savefig")
    def test_save_chart(self, mock_savefig, mock_mkdir):
        chart = RatingDistributionChart()
        fig = plt.figure()

        chart.save_chart(fig, "test.png", "test_path")

        mock_mkdir.assert_called_once_with(exist_ok=True)
        mock_savefig.assert_called_once()
        plt.close(fig)


class TestDetailedRatingChart(unittest.TestCase):

    def setUp(self):
        self.sample_data = ChartData(
            dataset_name="test",
            display_name="Test Dataset",
            ratings=pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]),
            stats={
                "mean": 3.14,
                "median": 3.0,
                "std": 1.35,
                "min": 1.0,
                "max": 5.0,
                "count": 7,
                "rating_scale": (1.0, 5.0),
            },
            has_ratings=True,
        )

    @patch("matplotlib.pyplot.show")
    def test_create_detailed_chart(self, mock_show):
        chart = DetailedRatingChart()
        fig = chart.create_chart(self.sample_data)

        self.assertIsInstance(fig, plt.Figure)
        # Verificar se tem 2 subplots
        self.assertEqual(len(fig.axes), 2)
        plt.close(fig)


class TestComparativeChart(unittest.TestCase):

    def setUp(self):
        self.datasets = [
            ChartData(
                dataset_name="dataset1",
                display_name="Dataset 1",
                ratings=pd.Series([3.0, 4.0, 5.0]),
                stats={"mean": 4.0, "median": 4.0, "std": 1.0, "count": 3, "min": 3.0, "max": 5.0},
                has_ratings=True,
            ),
            ChartData(
                dataset_name="dataset2",
                display_name="Dataset 2",
                ratings=pd.Series([2.0, 3.0, 4.0]),
                stats={"mean": 3.0, "median": 3.0, "std": 1.0, "count": 3, "min": 2.0, "max": 4.0},
                has_ratings=True,
            ),
        ]

    @patch("matplotlib.pyplot.show")
    def test_create_comparative_chart(self, mock_show):
        chart = ComparativeChart()
        fig = chart.create_chart(self.datasets)

        self.assertIsInstance(fig, plt.Figure)
        # Deve ter 3 subplots (distribuição, stats, tabela)
        self.assertEqual(len(fig.axes), 3)
        plt.close(fig)

    def test_create_chart_no_datasets(self):
        chart = ComparativeChart()

        with self.assertRaises(ValueError):
            chart.create_chart([])


class TestDatasetVisualizer(unittest.TestCase):

    @patch("sis_rec_experiments.outputs.dataset_charts.ChartData.from_loader")
    def test_discover_available_datasets(self, mock_from_loader):
        def side_effect(dataset_name, path):
            if dataset_name in ["movielens", "amazonmusic"]:
                return ChartData(dataset_name, dataset_name, pd.Series([1, 2, 3]), {}, True)
            else:
                return ChartData(dataset_name, dataset_name, pd.Series(dtype=float), {}, False)

        mock_from_loader.side_effect = side_effect

        visualizer = DatasetVisualizer()
        available = visualizer.discover_available_datasets()

        self.assertIn("movielens", available)
        self.assertIn("amazonmusic", available)
        self.assertNotIn("steam", available)

    @patch("sis_rec_experiments.outputs.dataset_charts.ChartData.from_loader")
    @patch("matplotlib.pyplot.show")
    def test_create_single_distribution_success(self, mock_show, mock_from_loader):
        mock_data = ChartData(
            "test",
            "Test",
            pd.Series([1, 2, 3, 4, 5]),
            {"mean": 3, "median": 3, "std": 1.5, "count": 5, "min": 1, "max": 5, "rating_scale": (1, 5)},
            True,
        )
        mock_from_loader.return_value = mock_data

        visualizer = DatasetVisualizer()
        fig = visualizer.create_single_distribution("test", show=False)

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    @patch("sis_rec_experiments.outputs.dataset_charts.ChartData.from_loader")
    def test_create_single_distribution_error(self, mock_from_loader):
        mock_from_loader.side_effect = Exception("Error")

        visualizer = DatasetVisualizer()
        fig = visualizer.create_single_distribution("invalid", show=False)

        self.assertIsNone(fig)

    @patch("sis_rec_experiments.outputs.dataset_charts.ChartData.from_loader")
    @patch("matplotlib.pyplot.show")
    def test_create_comparative_analysis(self, mock_show, mock_from_loader):
        def side_effect(dataset_name, path):
            return ChartData(
                dataset_name,
                dataset_name.title(),
                pd.Series([1, 2, 3]),
                {"mean": 2, "median": 2, "std": 1, "count": 3, "min": 1, "max": 3},
                True,
            )

        mock_from_loader.side_effect = side_effect

        visualizer = DatasetVisualizer()
        fig = visualizer.create_comparative_analysis(["test1", "test2"], show=False)

        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
