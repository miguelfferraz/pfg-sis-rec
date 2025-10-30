from typing import Any, Dict, Optional

import pandas as pd
from surprise import Dataset

from sis_rec_experiments.loaders.base_loader import BaseDatasetLoader


class SteamLoader(BaseDatasetLoader):
    """
    This dataset contains data of games from Steam.
    Includes purchase and play hours as implicit feedback.
    """

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        self.purchase_df: Optional[pd.DataFrame] = None
        self.play_hours_df: Optional[pd.DataFrame] = None
        self.users_info_df: Optional[pd.DataFrame] = None

    def load_ratings(self) -> pd.DataFrame:
        self.ratings_df = pd.DataFrame(columns=["user_id", "item_id", "rating"])
        return self.ratings_df

    def load_play_hours(self) -> pd.DataFrame:
        play_file = self.dataset_path / "game_play.dat"

        if not play_file.exists():
            raise FileNotFoundError(f"Play hours file not found: {play_file}")

        self.play_hours_df = pd.read_csv(
            play_file, sep="\t", header=0, dtype={"User_ID": "int32", "Game_ID": "int32", "Hours": "float32"}
        )

        self.play_hours_df = self.play_hours_df.rename(columns={"User_ID": "user_id", "Game_ID": "item_id"})

        return self.play_hours_df

    def load_metadata(self) -> pd.DataFrame:
        info_file = self.dataset_path / "item_info.dat"

        if info_file.exists():
            try:
                self.metadata_df = pd.read_csv(info_file, sep="\t", encoding="utf-8", on_bad_lines="skip")
            except Exception as e:
                print(f"Error loading metadata: {e}")
                self.metadata_df = pd.DataFrame()
        else:
            self.metadata_df = pd.DataFrame()

        return self.metadata_df

    def load_purchases(self) -> pd.DataFrame:
        purchase_file = self.dataset_path / "game_purchase.dat"

        if not purchase_file.exists():
            raise FileNotFoundError(f"Purchase file not found: {purchase_file}")

        self.purchase_df = pd.read_csv(
            purchase_file, sep="\t", header=0, dtype={"User_ID": "int32", "Game_ID": "int32", "Purchase": "int8"}
        )

        self.purchase_df = self.purchase_df.rename(columns={"User_ID": "user_id", "Game_ID": "item_id"})

        return self.purchase_df

    def load_users_info(self) -> pd.DataFrame:
        users_file = self.dataset_path / "user_info.dat"

        if users_file.exists():
            try:
                self.users_info_df = pd.read_csv(users_file, sep="\t", encoding="utf-8", on_bad_lines="skip")

                self.users_info_df = self.users_info_df.rename(columns={"New_ID": "user_id"})

            except Exception as e:
                print(f"Error loading users info: {e}")
                self.users_info_df = pd.DataFrame()
        else:
            self.users_info_df = pd.DataFrame()

        return self.users_info_df

    def get_dataset_info(self) -> Dict[str, Any]:
        if self.purchase_df is None:
            self.load_purchases()
        if self.play_hours_df is None:
            self.load_play_hours()
        if self.users_info_df is None:
            self.load_users_info()
        if self.ratings_df is None:
            self.load_ratings()

        purchase_stats = {}
        if self.purchase_df is not None and not self.purchase_df.empty:
            purchase_stats = {
                "total_history_records": len(self.purchase_df),
                "unique_users_history": self.purchase_df["user_id"].nunique(),
                "unique_items_history": self.purchase_df["item_id"].nunique(),
                "avg_items_per_user_history": len(self.purchase_df) / max(self.purchase_df["user_id"].nunique(), 1),
            }

        play_stats = {}
        if self.play_hours_df is not None and not self.play_hours_df.empty:
            play_stats = {
                "total_play_hours": self.play_hours_df["Hours"].sum(),
                "avg_hours_per_game": self.play_hours_df["Hours"].mean(),
                "max_hours_single_game": self.play_hours_df["Hours"].max(),
                "total_play_sessions": len(self.play_hours_df),
                "unique_users_playing": self.play_hours_df["user_id"].nunique(),
                "unique_games_played": self.play_hours_df["item_id"].nunique(),
                "users_with_play_data": len(self.users_info_df) if self.users_info_df is not None else 0,
            }

        basic_stats = {
            "total_ratings": 0,
            "unique_users": purchase_stats.get("unique_users_history", 0),
            "unique_items": purchase_stats.get("unique_items_history", 0),
            "rating_min": None,
            "rating_max": None,
            "rating_mean": None,
            "rating_std": None,
            "sparsity": None,
        }

        info = {
            **basic_stats,
            **purchase_stats,
            **play_stats,
            "dataset_name": "Steam Games",
            "dataset_type": "Gaming Platform Data",
            "domain": "Video Games",
            "has_metadata": self.metadata_df is not None and not self.metadata_df.empty,
            "has_explicit_ratings": False,
            "has_play_hours": True,
            "has_purchase_data": True,
            "data_format": "DAT (Tab-separated)",
            "data_description": "Play hours and purchase data (no explicit ratings)",
        }

        return info

    def to_surprise_dataset(self, rating_scale=None, reader=None) -> Dataset:
        raise NotImplementedError(
            "Dataset Steam does not have explicit ratings. "
            "To use with Surprise, it would be necessary to convert play hours to implicit ratings first."
        )
