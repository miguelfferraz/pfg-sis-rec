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

        users_stats = (
            self._ratings_df.groupby("user_id")
            .agg({"rating": ["count", "mean", "std"], "item_id": "nunique"})
            .reset_index()
        )

        users_stats.columns = ["user_id", "total_ratings", "avg_rating", "rating_std", "unique_items_rated"]

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

                users_stats = users_stats.merge(history_stats, on="user_id", how="left")
                users_stats["total_accessed"] = users_stats["total_accessed"].fillna(0)
                users_stats["unique_items_accessed"] = users_stats["unique_items_accessed"].fillna(0)
            except Exception:
                users_stats["total_accessed"] = 0
                users_stats["unique_items_accessed"] = 0

        activity_bins = [0, 5, 20, 50, float("inf")]
        activity_labels = ["low", "medium", "high", "very_high"]
        users_stats["activity_level"] = pd.cut(
            users_stats["total_ratings"], bins=activity_bins, labels=activity_labels, include_lowest=True
        )

        rating_bins = [0, 6.0, 7.5, 8.5, 10.0]
        rating_labels = ["critical", "moderate", "positive", "very_positive"]
        users_stats["rating_behavior"] = pd.cut(
            users_stats["avg_rating"], bins=rating_bins, labels=rating_labels, include_lowest=True
        )

        users_stats["engagement_ratio"] = users_stats["total_accessed"] / (users_stats["total_ratings"] + 1)
        engagement_bins = [0, 2.0, 5.0, float("inf")]
        engagement_labels = ["focused", "moderate", "explorer"]
        users_stats["engagement_level"] = pd.cut(
            users_stats["engagement_ratio"], bins=engagement_bins, labels=engagement_labels, include_lowest=True
        )

        activity_encoded = self._create_categorical_mapping(
            users_stats["activity_level"].astype(str), "activity_level", self.user_mappings
        )
        rating_behavior_encoded = self._create_categorical_mapping(
            users_stats["rating_behavior"].astype(str), "rating_behavior", self.user_mappings
        )
        engagement_encoded = self._create_categorical_mapping(
            users_stats["engagement_level"].astype(str), "engagement_level", self.user_mappings
        )

        users_df = pd.DataFrame(
            {
                "user_id": users_stats["user_id"],
                "activity_level": activity_encoded,
                "rating_behavior": rating_behavior_encoded,
                "engagement_level": engagement_encoded,
            }
        )

        return users_df

    def _load_items(self) -> pd.DataFrame:
        info_file = self.dataset_path / "anime_info.dat"

        if not info_file.exists():
            return pd.DataFrame(columns=["item_id"])

        try:
            metadata_df = pd.read_csv(info_file, sep="\t")

            if "anime_ids" in metadata_df.columns:
                metadata_df = metadata_df.rename(columns={"anime_ids": "item_id"})

            primary_genres = []
            for _, row in metadata_df.iterrows():
                if pd.isna(row.get("genre", "")) or row.get("genre", "") == "":
                    primary_genres.append("unknown")
                else:
                    genre_list = str(row["genre"]).split(", ")
                    primary_genre = genre_list[0] if genre_list else "unknown"
                    primary_genres.append(primary_genre)

            anime_types = []
            for _, row in metadata_df.iterrows():
                anime_type = row.get("type", "unknown")
                if pd.isna(anime_type) or anime_type == "":
                    anime_types.append("unknown")
                else:
                    anime_types.append(str(anime_type))

            episode_categories = []
            for _, row in metadata_df.iterrows():
                episodes = row.get("episodes", 0)
                if pd.isna(episodes):
                    episodes = 0
                else:
                    try:
                        episodes = int(episodes)
                    except:
                        episodes = 0

                if episodes == 1:
                    episode_categories.append("single")
                elif episodes <= 12:
                    episode_categories.append("short")
                elif episodes <= 26:
                    episode_categories.append("standard")
                else:
                    episode_categories.append("long")

            primary_genre_encoded = self._create_categorical_mapping(
                pd.Series(primary_genres), "primary_genre", self.item_mappings
            )
            anime_type_encoded = self._create_categorical_mapping(
                pd.Series(anime_types), "anime_type", self.item_mappings
            )
            episode_category_encoded = self._create_categorical_mapping(
                pd.Series(episode_categories), "episode_category", self.item_mappings
            )

            items_df = pd.DataFrame(
                {
                    "item_id": metadata_df["item_id"],
                    "primary_genre": primary_genre_encoded,
                    "anime_type": anime_type_encoded,
                    "episode_category": episode_category_encoded,
                }
            )

            return items_df

        except Exception:
            return pd.DataFrame(columns=["item_id"])
