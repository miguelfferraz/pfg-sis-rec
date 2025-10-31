import pandas as pd

from src.loaders.base_loader import BaseDatasetLoader


class AnimeLoader(BaseDatasetLoader):
    """
    This dataset contains ratings of anime from MyAnimeList.
    Includes explicit ratings and access history.
    """

    DEFAULT_RATING_SCALE = (1.0, 10.0)

    def _load_ratings(self) -> pd.DataFrame:
        ratings_file = self.dataset_path / "anime_ratings.dat"

        if not ratings_file.exists():
            raise FileNotFoundError(f"Rating file not found: {ratings_file}")

        ratings_df = pd.read_csv(
            ratings_file, sep="\t", header=0, dtype={"User_ID": "int32", "Anime_ID": "int32", "Feedback": "float32"}
        )

        return ratings_df.rename(columns={"User_ID": "user_id", "Anime_ID": "item_id", "Feedback": "rating"})

    def _load_users(self) -> pd.DataFrame:
        history_file = self.dataset_path / "anime_history.dat"

        if self._ratings_df is None:
            self._ratings_df = self._load_ratings()

        users_from_ratings = (
            self._ratings_df.groupby("user_id").agg({"rating": ["count", "mean"], "item_id": "nunique"}).reset_index()
        )

        users_data = []
        for _, row in users_from_ratings.iterrows():
            user_data = {
                "user_id": row["user_id"],
                "total_ratings": row[("rating", "count")],
                "avg_rating": row[("rating", "mean")],
                "unique_items_rated": row[("item_id", "nunique")],
            }
            users_data.append(user_data)

        users_df = pd.DataFrame(users_data)

        if history_file.exists():
            try:
                history_df = pd.read_csv(
                    history_file,
                    sep="\t",
                    header=0,
                    dtype={"User_ID": "int32", "Anime_ID": "int32", "Feedback": "int8"},
                )
                history_df = history_df.rename(
                    columns={"User_ID": "user_id", "Anime_ID": "item_id", "Feedback": "accessed"}
                )

                history_stats = (
                    history_df.groupby("user_id").agg({"accessed": "sum", "item_id": "nunique"}).reset_index()
                )
                history_stats.columns = ["user_id", "total_accessed", "unique_items_accessed"]

                users_df = users_df.merge(history_stats, on="user_id", how="left")
            except Exception:
                pass

        return users_df

    def _load_items(self) -> pd.DataFrame:
        info_file = self.dataset_path / "anime_info.dat"

        if self._ratings_df is None:
            self._ratings_df = self._load_ratings()

        items_from_ratings = (
            self._ratings_df.groupby("item_id").agg({"rating": ["count", "mean"], "user_id": "nunique"}).reset_index()
        )

        items_data = []
        for _, row in items_from_ratings.iterrows():
            item_data = {
                "item_id": row["item_id"],
                "total_ratings": row[("rating", "count")],
                "avg_rating": row[("rating", "mean")],
                "unique_users": row[("user_id", "nunique")],
            }
            items_data.append(item_data)

        items_df = pd.DataFrame(items_data)

        if info_file.exists():
            try:
                metadata_df = pd.read_csv(info_file, sep="\t")
                items_df = items_df.merge(metadata_df, on="item_id", how="left")
            except Exception:
                pass

        return items_df
