from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from sis_rec_experiments.loaders.builder import create_loader


@dataclass
class DatasetReport:
    name: str
    stats: Dict[str, Any]
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.error is None

    @property
    def total_ratings(self) -> int:
        return self.stats.get("total_ratings", 0)

    @property
    def unique_users(self) -> int:
        return self.stats.get("unique_users", 0)

    @property
    def unique_items(self) -> int:
        return self.stats.get("unique_items", 0)

    @property
    def rating_scale(self) -> tuple:
        return self.stats.get("rating_scale", (0, 0))

    @property
    def rating_mean(self) -> float:
        return self.stats.get("rating_mean", 0.0)

    @property
    def sparsity(self) -> float:
        return self.stats.get("sparsity", 0.0)

    @property
    def domain(self) -> str:
        return self.stats.get("domain", "Unknown")


class DatasetAnalyzer:
    def __init__(self, base_path: str = "sis_rec_experiments/datasets/extracted"):
        self.base_path = base_path

    def analyze_single(self, dataset_name: str) -> DatasetReport:
        try:
            loader = create_loader(dataset_name, self.base_path)
            loader.load_ratings()
            loader.load_metadata()
            stats = loader.get_dataset_info()
            return DatasetReport(name=dataset_name, stats=stats)
        except Exception as e:
            return DatasetReport(name=dataset_name, stats={}, error=str(e))

    def analyze_multiple(self, dataset_names: List[str]) -> List[DatasetReport]:
        return [self.analyze_single(name) for name in dataset_names]

    def discover_available_datasets(self) -> List[str]:
        available = []
        candidates = ["amazonmusic", "anime", "bookcrossing", "movielens", "steam"]

        for dataset in candidates:
            try:
                create_loader(dataset, self.base_path)
                available.append(dataset)
            except (FileNotFoundError, ValueError):
                continue

        return available


class ReportFormatter:
    @staticmethod
    def format_single(report: DatasetReport) -> str:
        if not report.is_success:
            return f"❌ {report.name}: {report.error}"

        stats = report.stats
        dataset_name = stats.get("dataset_name", report.name)

        output = f"\n{dataset_name}\n"

        if report.total_ratings > 0:
            output += (
                "├─ "
                + f"{report.unique_users:,} users, {report.unique_items:,} items, {report.total_ratings:,} ratings\n"
            )
            if report.rating_mean is not None:
                output += (
                    "├─ "
                    + f"Rating scale: {report.rating_scale[0]:.1f}-{report.rating_scale[1]:.1f} (avg: {report.rating_mean:.2f})\n"
                )
            output += "├─ " + f"Sparsity: {report.sparsity*100:.1f}% | Density: {(1-report.sparsity)*100:.1f}%\n"
        else:
            output += "├─ " + f"{report.unique_users:,} users, {report.unique_items:,} items\n"
            output += "├─ " + f"Data type: {stats.get('data_description', 'Implicit feedback')}\n"

        output += "└─ " + f"Domain: {report.domain}\n"

        return output

    @staticmethod
    def format_summary(reports: List[DatasetReport]) -> str:
        successful = [r for r in reports if r.is_success]
        failed = [r for r in reports if not r.is_success]

        if not successful:
            return "No datasets loaded successfully."

        output = f"\n{'='*60}\n"
        output += f"DATASET SUMMARY ({len(successful)} datasets)\n"
        output += f"{'='*60}\n"

        total_ratings = sum(r.total_ratings for r in successful)
        total_users = sum(r.unique_users for r in successful)
        total_items = sum(r.unique_items for r in successful)

        output += f"Total ratings: {total_ratings:,}\n"
        output += f"Total users: {total_users:,}\n"
        output += f"Total items: {total_items:,}\n"

        if len(successful) > 1:
            rating_datasets = [r for r in successful if r.total_ratings > 0]
            if len(rating_datasets) > 1:
                sparsities = [(r.stats.get("dataset_name", r.name), r.sparsity) for r in rating_datasets]
                most_dense = min(sparsities, key=lambda x: x[1])
                most_sparse = max(sparsities, key=lambda x: x[1])

                output += f"\nMost dense: {most_dense[0]} ({(1-most_dense[1])*100:.1f}%)\n"
                output += f"Most sparse: {most_sparse[0]} ({most_sparse[1]*100:.1f}%)\n"

        if failed:
            output += f"\nFailed datasets: {', '.join(r.name for r in failed)}\n"

        output += f"{'='*60}\n"
        return output


def main():
    import sys

    analyzer = DatasetAnalyzer()
    formatter = ReportFormatter()

    args = sys.argv[1:]

    if not args:
        # No arguments: analyze all available datasets
        available_datasets = analyzer.discover_available_datasets()
        if not available_datasets:
            print("No datasets found. Please extract datasets first using 'make datasets-extract'")
            return

        print(f"Found {len(available_datasets)} datasets: {', '.join(available_datasets)}")
        reports = analyzer.analyze_multiple(available_datasets)

        for report in reports:
            print(formatter.format_single(report))

        if len(reports) > 1:
            print(formatter.format_summary(reports))

    elif args[0] == "--summary":
        available_datasets = analyzer.discover_available_datasets()
        reports = analyzer.analyze_multiple(available_datasets)
        print(formatter.format_summary(reports))

    else:
        dataset_names = args
        reports = analyzer.analyze_multiple(dataset_names)

        for report in reports:
            print(formatter.format_single(report))


if __name__ == "__main__":
    main()
