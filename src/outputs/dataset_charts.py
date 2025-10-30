from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sis_rec_experiments.loaders.builder import create_loader


@dataclass
class ChartData:
    dataset_name: str
    display_name: str
    ratings: pd.Series
    stats: Dict[str, float]
    has_ratings: bool = True

    @classmethod
    def from_loader(cls, dataset_name: str, base_path: str = "sis_rec_experiments/datasets/extracted"):
        try:
            loader = create_loader(dataset_name, base_path)
            loader.load_ratings()

            if loader.ratings_df is None or loader.ratings_df.empty:
                return cls(
                    dataset_name=dataset_name,
                    display_name=dataset_name.title(),
                    ratings=pd.Series(dtype=float),
                    stats={},
                    has_ratings=False,
                )

            info = loader.get_dataset_info()
            ratings = loader.ratings_df["rating"]

            stats = {
                "mean": ratings.mean(),
                "median": ratings.median(),
                "std": ratings.std(),
                "min": ratings.min(),
                "max": ratings.max(),
                "count": len(ratings),
                "rating_scale": info.get("rating_scale", (ratings.min(), ratings.max())),
            }

            return cls(
                dataset_name=dataset_name,
                display_name=info.get("dataset_name", dataset_name.title()),
                ratings=ratings,
                stats=stats,
                has_ratings=True,
            )

        except Exception:
            return cls(
                dataset_name=dataset_name,
                display_name=dataset_name.title(),
                ratings=pd.Series(dtype=float),
                stats={},
                has_ratings=False,
            )


class ChartConfig:
    def __init__(self):
        self.style = "default"
        self.palette = "husl"
        self.dpi = 300
        self.figsize_single = (10, 6)
        self.figsize_detailed = (12, 8)
        self.figsize_comparative = (15, 10)
        self.alpha = 0.7

    def apply_style(self):
        plt.style.use(self.style)
        sns.set_palette(self.palette)


class BaseChart(ABC):
    def __init__(self, config: ChartConfig = None):
        self.config = config or ChartConfig()
        self.config.apply_style()

    @abstractmethod
    def create_chart(self, data: ChartData, save_path: Optional[str] = None) -> plt.Figure:
        pass

    def save_chart(self, fig: plt.Figure, filename: str, save_path: str = "visualizations"):
        Path(save_path).mkdir(exist_ok=True)
        full_path = Path(save_path) / filename
        fig.savefig(full_path, dpi=self.config.dpi, bbox_inches="tight")
        print(f"Chart saved: {full_path}")
        return full_path


class RatingDistributionChart(BaseChart):
    def create_chart(self, data: ChartData, save_path: Optional[str] = None) -> plt.Figure:
        if not data.has_ratings:
            raise ValueError(f"Dataset {data.dataset_name} has no explicit ratings")

        fig, ax = plt.subplots(figsize=self.config.figsize_single)
        fig.suptitle(f"Rating Distribution - {data.display_name}", fontsize=16, fontweight="bold")

        ax.hist(data.ratings, bins=30, alpha=self.config.alpha, color="skyblue", edgecolor="black")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Frequency")
        ax.set_title("Rating Histogram")
        ax.grid(True, alpha=0.3)

        mean_rating = data.stats["mean"]
        median_rating = data.stats["median"]
        ax.axvline(mean_rating, color="red", linestyle="--", label=f"Mean: {mean_rating:.2f}")
        ax.axvline(median_rating, color="orange", linestyle="--", label=f"Median: {median_rating:.2f}")
        ax.legend()

        stats_text = f"""Statistics:
Total Ratings: {data.stats['count']:,}
Min: {data.stats['min']:.1f}
Max: {data.stats['max']:.1f}
Mean: {mean_rating:.2f}
Median: {median_rating:.2f}
Std Dev: {data.stats['std']:.2f}
Scale: {data.stats['rating_scale']}"""

        fig.text(
            0.02, 0.02, stats_text, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8)
        )

        plt.tight_layout()

        if save_path:
            filename = f"rating_distribution_{data.dataset_name.lower()}.png"
            self.save_chart(fig, filename, save_path)

        return fig


class DetailedRatingChart(BaseChart):
    def create_chart(self, data: ChartData, save_path: Optional[str] = None) -> plt.Figure:
        if not data.has_ratings:
            raise ValueError(f"Dataset {data.dataset_name} has no explicit ratings")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figsize_detailed)
        fig.suptitle(f"Detailed Rating Analysis - {data.display_name}", fontsize=16, fontweight="bold")

        rating_counts = data.ratings.value_counts().sort_index()

        bars = ax1.bar(
            rating_counts.index, rating_counts.values, alpha=self.config.alpha, color="lightcoral", edgecolor="black"
        )
        ax1.set_xlabel("Rating Value")
        ax1.set_ylabel("Count")
        ax1.set_title("Distribution by Rating Value")
        ax1.grid(True, alpha=0.3)

        self._add_value_labels(ax1, bars)

        rating_percentages = (rating_counts / rating_counts.sum()) * 100
        bars2 = ax2.bar(
            rating_percentages.index,
            rating_percentages.values,
            alpha=self.config.alpha,
            color="lightgreen",
            edgecolor="black",
        )
        ax2.set_xlabel("Rating Value")
        ax2.set_ylabel("Percentage (%)")
        ax2.set_title("Percentage Distribution by Rating Value")
        ax2.grid(True, alpha=0.3)

        self._add_percentage_labels(ax2, bars2)

        plt.tight_layout()

        if save_path:
            filename = f"rating_values_{data.dataset_name.lower()}.png"
            self.save_chart(fig, filename, save_path)

        return fig

    def _add_value_labels(self, ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{int(height):,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    def _add_percentage_labels(self, ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )


class ComparativeChart(BaseChart):
    def create_chart(self, datasets: List[ChartData], save_path: Optional[str] = None) -> plt.Figure:
        datasets_with_ratings = [d for d in datasets if d.has_ratings]

        if not datasets_with_ratings:
            raise ValueError("No datasets with ratings provided")

        fig, axes = plt.subplots(1, 3, figsize=self.config.figsize_comparative)
        fig.suptitle("Rating Distribution Comparison Between Datasets", fontsize=16, fontweight="bold")

        self._create_distribution_plot(axes[0], datasets_with_ratings)
        self._create_stats_plot(axes[1], datasets_with_ratings)
        self._create_summary_table(axes[2], datasets_with_ratings)

        plt.tight_layout()

        if save_path:
            filename = "comparative_ratings.png"
            self.save_chart(fig, filename, save_path)

        return fig

    def _create_distribution_plot(self, ax, datasets):
        colors = ["skyblue", "lightcoral", "lightgreen", "gold", "mediumpurple"]
        for i, data in enumerate(datasets):
            ax.hist(
                data.ratings,
                bins=20,
                alpha=0.6,
                label=f'{data.display_name} (μ={data.stats["mean"]:.2f})',
                color=colors[i % len(colors)],
                density=True,
            )

        ax.set_xlabel("Rating")
        ax.set_ylabel("Density")
        ax.set_title("Normalized Distributions")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _create_stats_plot(self, ax, datasets):
        stats_data = []
        for data in datasets:
            stats_data.append(
                {
                    "Dataset": data.display_name,
                    "Mean": data.stats["mean"],
                    "Median": data.stats["median"],
                    "Std Dev": data.stats["std"],
                }
            )

        stats_df = pd.DataFrame(stats_data)
        x = np.arange(len(stats_df))
        width = 0.25

        ax.bar(x - width, stats_df["Mean"], width, label="Mean", color="skyblue")
        ax.bar(x, stats_df["Median"], width, label="Median", color="lightcoral")
        ax.bar(x + width, stats_df["Std Dev"], width, label="Std Dev", color="lightgreen")

        ax.set_xlabel("Dataset")
        ax.set_ylabel("Value")
        ax.set_title("Comparative Statistics")
        ax.set_xticks(x)
        ax.set_xticklabels(stats_df["Dataset"], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _create_summary_table(self, ax, datasets):
        ax.axis("tight")
        ax.axis("off")

        table_data = []
        for data in datasets:
            table_data.append(
                [
                    data.display_name,
                    f"{data.stats['count']:,}",
                    f"{data.stats['min']:.1f} - {data.stats['max']:.1f}",
                    f"{data.stats['mean']:.2f}",
                    f"{data.stats['std']:.2f}",
                ]
            )

        table = ax.table(
            cellText=table_data,
            colLabels=["Dataset", "Total Ratings", "Scale", "Mean", "Std Dev"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.set_title("Statistical Summary", pad=20)


class DatasetVisualizer:
    def __init__(self, base_path: str = "sis_rec_experiments/datasets/extracted"):
        self.base_path = base_path
        self.config = ChartConfig()

    def create_single_distribution(
        self, dataset_name: str, save_path: str = "visualizations", show: bool = True
    ) -> Optional[plt.Figure]:
        try:
            data = ChartData.from_loader(dataset_name, self.base_path)
            chart = RatingDistributionChart(self.config)
            fig = chart.create_chart(data, save_path)

            if show:
                plt.show()

            return fig

        except Exception as e:
            print(f"Error generating chart for {dataset_name}: {str(e)}")
            return None

    def create_detailed_analysis(
        self, dataset_name: str, save_path: str = "visualizations", show: bool = True
    ) -> Optional[plt.Figure]:
        try:
            data = ChartData.from_loader(dataset_name, self.base_path)
            chart = DetailedRatingChart(self.config)
            fig = chart.create_chart(data, save_path)

            if show:
                plt.show()

            return fig

        except Exception as e:
            print(f"Error generating detailed chart for {dataset_name}: {str(e)}")
            return None

    def create_comparative_analysis(
        self, dataset_names: List[str], save_path: str = "visualizations", show: bool = True
    ) -> Optional[plt.Figure]:
        try:
            datasets = []
            for name in dataset_names:
                data = ChartData.from_loader(name, self.base_path)
                if data.has_ratings:
                    datasets.append(data)
                else:
                    print(f"Warning: {name} has no explicit ratings - ignored in comparison")

            if not datasets:
                print("No datasets with ratings loaded for comparison")
                return None

            chart = ComparativeChart(self.config)
            fig = chart.create_chart(datasets, save_path)

            if show:
                plt.show()

            return fig

        except Exception as e:
            print(f"Error generating comparative chart: {str(e)}")
            return None

    def discover_available_datasets(self) -> List[str]:
        available = []
        candidates = ["amazonmusic", "anime", "bookcrossing", "movielens", "steam"]

        for dataset in candidates:
            try:
                data = ChartData.from_loader(dataset, self.base_path)
                if data.has_ratings:
                    available.append(dataset)
            except:
                continue

        return available

    def generate_all_visualizations(self, save_path: str = "visualizations") -> None:
        available_datasets = self.discover_available_datasets()

        if not available_datasets:
            print("No datasets with ratings found")
            return

        print(f"Generating visualizations for {len(available_datasets)} datasets...")
        print("=" * 60)

        for dataset_name in available_datasets:
            print(f"\nGenerating charts for {dataset_name.upper()}...")
            self.create_single_distribution(dataset_name, save_path, show=False)
            self.create_detailed_analysis(dataset_name, save_path, show=False)

        if len(available_datasets) > 1:
            print(f"\nGenerating comparative chart...")
            self.create_comparative_analysis(available_datasets, save_path, show=False)

        print("\n" + "=" * 60)
        print("All visualizations generated successfully!")
        print(f"Check '{save_path}' folder for charts.")


def main():
    import sys

    visualizer = DatasetVisualizer()
    args = sys.argv[1:]

    if not args:
        available_datasets = visualizer.discover_available_datasets()
        if not available_datasets:
            print("No datasets with ratings found. Run 'make datasets-extract' first")
            return

        print(f"Generating visualizations for {len(available_datasets)} datasets: {', '.join(available_datasets)}")
        visualizer.generate_all_visualizations()

    elif args[0] == "--list":
        available = visualizer.discover_available_datasets()
        print(f"Available datasets for visualization: {', '.join(available)}")

    elif args[0] in ["--distribution", "-d"]:
        if len(args) < 2:
            print("Usage: python -m sis_rec_experiments.outputs.dataset_charts --distribution DATASET")
            return

        dataset_name = args[1]
        print(f"Generating distribution chart for {dataset_name}...")
        visualizer.create_single_distribution(dataset_name, show=True)

    elif args[0] in ["--detailed", "-dt"]:
        if len(args) < 2:
            print("Usage: python -m sis_rec_experiments.outputs.dataset_charts --detailed DATASET")
            return

        dataset_name = args[1]
        print(f"Generating detailed analysis for {dataset_name}...")
        visualizer.create_detailed_analysis(dataset_name, show=True)

    elif args[0] in ["--compare", "-c"]:
        if len(args) < 2:
            available = visualizer.discover_available_datasets()
            dataset_names = available
            print(f"Comparing all available datasets: {', '.join(dataset_names)}")
        else:
            dataset_names = args[1:]
            print(f"Comparing datasets: {', '.join(dataset_names)}")

        visualizer.create_comparative_analysis(dataset_names, show=True)

    elif args[0] in ["--all-for", "-a"]:
        if len(args) < 2:
            print("Usage: python -m sis_rec_experiments.outputs.dataset_charts --all-for DATASET")
            return

        dataset_name = args[1]
        print(f"Generating all visualizations for {dataset_name}...")
        visualizer.create_single_distribution(dataset_name, show=False)
        visualizer.create_detailed_analysis(dataset_name, show=False)
        print("Visualizations generated successfully!")

    elif args[0] == "--help":
        print(
            """Usage: python -m sis_rec_experiments.outputs.dataset_charts [OPTION] [ARGUMENTS]

Options:
  (no arguments)      Generate all visualizations for all datasets
  --list              List available datasets
  -d, --distribution  Distribution chart for specific dataset
  -dt, --detailed     Detailed analysis for specific dataset  
  -c, --compare       Comparative chart (all or specific)
  -a, --all-for       All visualizations for specific dataset
  --help              Show this help

Examples:
  python -m sis_rec_experiments.outputs.dataset_charts
  python -m sis_rec_experiments.outputs.dataset_charts --distribution movielens
  python -m sis_rec_experiments.outputs.dataset_charts --compare movielens amazonmusic
  python -m sis_rec_experiments.outputs.dataset_charts --all-for anime"""
        )

    else:
        dataset_names = args
        print(f"Generating charts for: {', '.join(dataset_names)}")
        for dataset_name in dataset_names:
            visualizer.create_single_distribution(dataset_name, show=False)
            visualizer.create_detailed_analysis(dataset_name, show=False)


if __name__ == "__main__":
    main()
