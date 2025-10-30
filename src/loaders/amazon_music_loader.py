import json
from typing import Any, Dict

import pandas as pd

from src.loaders.base_loader import BaseDatasetLoader


class AmazonMusicLoader(BaseDatasetLoader):
    """
    This dataset contains reviews of musical products from Amazon,
    including ratings and metadata of the products.
    """

    DEFAULT_RATING_SCALE = (1.0, 5.0)

    def load_ratings(self) -> pd.DataFrame:
        json_file = self.dataset_path / "Digital_Music_5.json"

        if not json_file.exists():
            raise FileNotFoundError(f"Rating file not found: {json_file}")

        ratings_data = []
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    review = json.loads(line.strip())
                    ratings_data.append(
                        {
                            "user_id": review["reviewerID"],
                            "item_id": review["asin"],
                            "rating": float(review["overall"]),
                            "timestamp": review.get("unixReviewTime", None),
                            "helpful": review.get("helpful", [0, 0]),
                        }
                    )
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error processing line: {e}")
                    continue

        self.ratings_df = pd.DataFrame(ratings_data)

        return self.ratings_df

    def load_metadata(self) -> pd.DataFrame:
        csv_file = self.dataset_path / "amazon_music_metadata.csv"

        if not csv_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {csv_file}")

        try:
            self.metadata_df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error loading metadata: {e}")
            self.metadata_df = pd.DataFrame()

        return self.metadata_df

    def get_dataset_info(self) -> Dict[str, Any]:
        if self.ratings_df is None:
            self.load_ratings()

        basic_stats = self.get_basic_stats()

        info = {
            **basic_stats,
            "dataset_name": "Amazon Music",
            "dataset_type": "E-commerce Reviews",
            "domain": "Digital Music",
            "has_metadata": self.metadata_df is not None and not self.metadata_df.empty,
            "rating_scale": (1.0, 5.0),
            "data_format": "JSON + CSV",
        }

        return info
