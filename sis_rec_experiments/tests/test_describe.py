import unittest
from unittest.mock import MagicMock, patch

from sis_rec_experiments.outputs.describe import DatasetAnalyzer, DatasetReport, ReportFormatter


class TestDatasetReport(unittest.TestCase):

    def test_dataset_report_success(self):
        stats = {
            "total_ratings": 100000,
            "unique_users": 943,
            "unique_items": 1682,
            "rating_scale": (1.0, 5.0),
            "rating_mean": 3.53,
            "sparsity": 0.937,
            "domain": "Movies",
        }
        report = DatasetReport("movielens", stats)

        self.assertTrue(report.is_success)
        self.assertEqual(report.total_ratings, 100000)
        self.assertEqual(report.unique_users, 943)
        self.assertEqual(report.rating_scale, (1.0, 5.0))
        self.assertAlmostEqual(report.rating_mean, 3.53)

    def test_dataset_report_error(self):
        report = DatasetReport("invalid", {}, error="Dataset not found")

        self.assertFalse(report.is_success)
        self.assertEqual(report.error, "Dataset not found")


class TestDatasetAnalyzer(unittest.TestCase):

    @patch("sis_rec_experiments.outputs.describe.create_loader")
    def test_analyze_single_success(self, mock_create_loader):
        mock_loader = MagicMock()
        mock_loader.get_dataset_info.return_value = {"total_ratings": 1000}
        mock_create_loader.return_value = mock_loader

        analyzer = DatasetAnalyzer()
        report = analyzer.analyze_single("test_dataset")

        self.assertTrue(report.is_success)
        self.assertEqual(report.name, "test_dataset")
        mock_loader.load_ratings.assert_called_once()
        mock_loader.load_metadata.assert_called_once()

    @patch("sis_rec_experiments.outputs.describe.create_loader")
    def test_analyze_single_error(self, mock_create_loader):
        mock_create_loader.side_effect = FileNotFoundError("Dataset not found")

        analyzer = DatasetAnalyzer()
        report = analyzer.analyze_single("invalid_dataset")

        self.assertFalse(report.is_success)
        self.assertIn("Dataset not found", report.error)

    @patch("sis_rec_experiments.outputs.describe.create_loader")
    def test_discover_available_datasets(self, mock_create_loader):
        def side_effect(dataset_name, path):
            if dataset_name in ["movielens", "amazonmusic"]:
                return MagicMock()
            raise FileNotFoundError()

        mock_create_loader.side_effect = side_effect

        analyzer = DatasetAnalyzer()
        available = analyzer.discover_available_datasets()

        self.assertIn("movielens", available)
        self.assertIn("amazonmusic", available)
        self.assertNotIn("invalid", available)


class TestReportFormatter(unittest.TestCase):

    def test_format_single_success(self):
        stats = {
            "dataset_name": "MovieLens 100k",
            "total_ratings": 100000,
            "unique_users": 943,
            "unique_items": 1682,
            "rating_scale": (1.0, 5.0),
            "rating_mean": 3.53,
            "sparsity": 0.937,
            "domain": "Movies",
        }
        report = DatasetReport("movielens", stats)

        output = ReportFormatter.format_single(report)

        self.assertIn("MovieLens 100k", output)
        self.assertIn("943 users", output)
        self.assertIn("100,000 ratings", output)
        self.assertIn("1.0-5.0", output)
        self.assertIn("avg: 3.53", output)

    def test_format_single_error(self):
        report = DatasetReport("invalid", {}, error="Not found")

        output = ReportFormatter.format_single(report)

        self.assertIn("❌", output)
        self.assertIn("invalid", output)
        self.assertIn("Not found", output)

    def test_format_single_no_ratings(self):
        stats = {
            "dataset_name": "Steam Games",
            "total_ratings": 0,
            "unique_users": 1000,
            "unique_items": 500,
            "domain": "Video Games",
            "data_description": "Play hours data",
        }
        report = DatasetReport("steam", stats)

        output = ReportFormatter.format_single(report)

        self.assertIn("Steam Games", output)
        self.assertIn("1,000 users", output)
        self.assertNotIn("ratings", output)
        self.assertIn("Play hours data", output)

    def test_format_summary(self):
        reports = [
            DatasetReport(
                "movielens",
                {
                    "dataset_name": "MovieLens",
                    "total_ratings": 100000,
                    "unique_users": 943,
                    "unique_items": 1682,
                    "sparsity": 0.937,
                },
            ),
            DatasetReport(
                "anime",
                {
                    "dataset_name": "Anime",
                    "total_ratings": 50000,
                    "unique_users": 500,
                    "unique_items": 1000,
                    "sparsity": 0.9,
                },
            ),
        ]

        output = ReportFormatter.format_summary(reports)

        self.assertIn("DATASET SUMMARY", output)
        self.assertIn("150,000", output)  # Total ratings
        self.assertIn("Most dense:", output)
        self.assertIn("Most sparse:", output)


if __name__ == "__main__":
    unittest.main()
