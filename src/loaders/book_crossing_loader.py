import pandas as pd

from src.loaders.base_loader import BaseDatasetLoader


class BookCrossingLoader(BaseDatasetLoader):
    """
    This dataset contains ratings of books from the Book-Crossing community.
    Includes explicit ratings, access history and demographic information of users.
    """

    DEFAULT_RATING_SCALE = (1.0, 10.0)

    def _load_ratings(self) -> pd.DataFrame:
        ratings_file = self.dataset_path / "book_ratings.dat"

        if not ratings_file.exists():
            raise FileNotFoundError(f"Rating file not found: {ratings_file}")

        ratings_df = pd.read_csv(
            ratings_file, sep="\t", header=0, dtype={"user": "int32", "item": "int32", "rating": "float32"}
        )

        return ratings_df.rename(columns={"user": "user_id", "item": "item_id"})

    def _load_users(self) -> pd.DataFrame:
        users_file = self.dataset_path / "users_info.dat"
        history_file = self.dataset_path / "book_history.dat"

        if self._ratings_df is None:
            self._ratings_df = self._load_ratings()

        users_stats = (
            self._ratings_df.groupby("user_id").agg({"rating": ["count", "mean"], "item_id": "nunique"}).reset_index()
        )

        users_stats.columns = ["user_id", "total_ratings", "avg_rating", "unique_items_rated"]

        countries = []

        if users_file.exists():
            try:
                rows = []
                with open(users_file, "r", encoding="utf-8") as f:
                    _ = f.readline().strip().split("\t")

                    for line in f:
                        parts = line.strip().split("\t")
                        while len(parts) > 3 and parts[-1] == "":
                            parts.pop()

                        if len(parts) >= 3:
                            rows.append(parts[:3])

                users_info_df = pd.DataFrame(rows, columns=["User-ID", "Location", "Age"])

                if "User-ID" in users_info_df.columns:
                    users_info_df["User-ID"] = pd.to_numeric(users_info_df["User-ID"], errors="coerce").astype("Int32")
                    users_info_df = users_info_df.rename(columns={"User-ID": "user_id"})

                if "Age" in users_info_df.columns:
                    users_info_df["Age"] = pd.to_numeric(users_info_df["Age"], errors="coerce")

                users_stats = users_stats.merge(users_info_df, on="user_id", how="left")

                for _, row in users_stats.iterrows():
                    location = row.get("Location", "")
                    if pd.isna(location) or location == "":
                        countries.append("unknown")
                    else:
                        location_parts = str(location).split(",")
                        location_parts = [part.strip() for part in location_parts]

                        if len(location_parts) >= 3:
                            country = location_parts[-1]
                            countries.append(country)
                        elif len(location_parts) == 2:
                            country = location_parts[1]
                            countries.append(country)
                        elif len(location_parts) == 1:
                            country = location_parts[0]
                            countries.append(country)
                        else:
                            countries.append("unknown")

            except Exception:
                countries = ["unknown"] * len(users_stats)
        else:
            countries = ["unknown"] * len(users_stats)

        if history_file.exists():
            try:
                history_df = pd.read_csv(
                    history_file, sep="\t", header=0, dtype={"user": "int32", "item": "int32", "accessed": "int8"}
                )
                history_df = history_df.rename(columns={"user": "user_id", "item": "item_id"})

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

        activity_bins = [0, 5, 15, 50, float("inf")]
        activity_labels = ["low", "medium", "high", "very_high"]
        users_stats["activity_level"] = pd.cut(
            users_stats["total_ratings"], bins=activity_bins, labels=activity_labels, include_lowest=True
        )

        rating_bins = [0, 6.0, 7.5, 8.5, 10.0]
        rating_labels = ["critical", "moderate", "positive", "very_positive"]
        users_stats["rating_behavior"] = pd.cut(
            users_stats["avg_rating"], bins=rating_bins, labels=rating_labels, include_lowest=True
        )

        country_encoded = self._create_categorical_mapping(pd.Series(countries), "country", self.user_mappings)
        activity_encoded = self._create_categorical_mapping(
            users_stats["activity_level"].astype(str), "activity_level", self.user_mappings
        )
        rating_behavior_encoded = self._create_categorical_mapping(
            users_stats["rating_behavior"].astype(str), "rating_behavior", self.user_mappings
        )

        users_df = pd.DataFrame(
            {
                "user_id": users_stats["user_id"],
                "country": country_encoded,
                "age": users_stats.get("Age", pd.Series([None] * len(users_stats))).fillna(0),
                "activity_level": activity_encoded,
                "rating_behavior": rating_behavior_encoded,
            }
        )

        return users_df

    def _load_items(self) -> pd.DataFrame:
        info_file = self.dataset_path / "items_info.dat"

        if not info_file.exists():
            return pd.DataFrame(columns=["item_id"])

        try:
            rows = []
            with open(info_file, "r", encoding="utf-8", errors="ignore") as f:
                header = f.readline().strip().split("\t")

                for line in f:
                    parts = line.strip().split("\t")

                    if len(parts) > len(header):
                        fixed_parts = parts[:5]

                        publisher_parts = parts[5 : len(parts) - 3]
                        publisher = " ".join(publisher_parts) if publisher_parts else ""
                        fixed_parts.append(publisher)

                        fixed_parts.extend(parts[-3:])
                        parts = fixed_parts

                    while len(parts) < len(header):
                        parts.append("")

                    rows.append(parts[: len(header)])

            metadata_df = pd.DataFrame(rows, columns=header)

            if "Book_ID" in metadata_df.columns:
                metadata_df = metadata_df.rename(columns={"Book_ID": "item_id"})

            publication_years = []
            publication_counts = []

            publisher_counts = {}
            if "Publisher" in metadata_df.columns:
                publisher_counts = metadata_df["Publisher"].value_counts().to_dict()

            for _, row in metadata_df.iterrows():
                year = row.get("Year-Of-Publication", None)
                if pd.isna(year) or year is None:
                    publication_years.append(0)
                else:
                    try:
                        year_int = int(year)
                        publication_years.append(year_int)
                    except ValueError:
                        publication_years.append(0)

                publisher = row.get("Publisher", "")
                if pd.isna(publisher) or publisher == "":
                    publication_counts.append(0)
                else:
                    publisher_count = publisher_counts.get(publisher, 0)
                    publication_counts.append(publisher_count)

            items_df = pd.DataFrame(
                {
                    "item_id": metadata_df["item_id"],
                    "publication_year": publication_years,
                    "publisher_book_count": publication_counts,
                }
            )

            return items_df

        except Exception:
            return pd.DataFrame(columns=["item_id"])
