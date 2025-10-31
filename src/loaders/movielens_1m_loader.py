import pandas as pd

from src.loaders.base_loader import BaseDatasetLoader


class MovieLens1MLoader(BaseDatasetLoader):
    """
    This dataset contains movie ratings from MovieLens 1M.
    Dataset with 1,000,209 ratings from 6,040 users on 3,883 movies.
    """

    DEFAULT_RATING_SCALE = (1.0, 5.0)

    def _load_ratings(self) -> pd.DataFrame:
        ratings_file = self.dataset_path / "ratings.dat"

        if not ratings_file.exists():
            raise FileNotFoundError(f"Rating file not found: {ratings_file}")

        return pd.read_csv(
            ratings_file,
            sep="::",
            header=None,
            names=["user_id", "item_id", "rating", "timestamp"],
            dtype={"user_id": "int32", "item_id": "int32", "rating": "float32", "timestamp": "int64"},
            engine="python",
        )

    def _load_users(self) -> pd.DataFrame:
        users_file = self.dataset_path / "users.dat"

        if self._ratings_df is None:
            self._ratings_df = self._load_ratings()

        users_from_ratings = (
            self._ratings_df.groupby("user_id")
            .agg({"rating": ["count", "mean"], "item_id": "nunique", "timestamp": ["min", "max"]})
            .reset_index()
        )

        users_data = []
        for _, row in users_from_ratings.iterrows():
            user_data = {
                "user_id": row["user_id"],
                "total_ratings": row[("rating", "count")],
                "avg_rating": row[("rating", "mean")],
                "unique_items_rated": row[("item_id", "nunique")],
                "first_rating": row[("timestamp", "min")],
                "last_rating": row[("timestamp", "max")],
            }
            users_data.append(user_data)

        users_df = pd.DataFrame(users_data)

        if users_file.exists():
            try:
                demographics_df = pd.read_csv(
                    users_file,
                    sep="::",
                    header=None,
                    names=["user_id", "gender", "age", "occupation", "zip_code"],
                    dtype={
                        "user_id": "int32",
                        "gender": "str",
                        "age": "int32",
                        "occupation": "int32",
                        "zip_code": "str",
                    },
                    engine="python",
                )

                occupation_map = {
                    0: "other",
                    1: "academic/educator",
                    2: "artist",
                    3: "clerical/admin",
                    4: "college/grad student",
                    5: "customer service",
                    6: "doctor/health care",
                    7: "executive/managerial",
                    8: "farmer",
                    9: "homemaker",
                    10: "K-12 student",
                    11: "lawyer",
                    12: "programmer",
                    13: "retired",
                    14: "sales/marketing",
                    15: "scientist",
                    16: "self-employed",
                    17: "technician/engineer",
                    18: "tradesman/craftsman",
                    19: "unemployed",
                    20: "writer",
                }

                demographics_df["occupation_name"] = demographics_df["occupation"].map(occupation_map)
                users_df = users_df.merge(demographics_df, on="user_id", how="left")
            except Exception:
                pass

        return users_df

    def _load_items(self) -> pd.DataFrame:
        movies_file = self.dataset_path / "movies.dat"

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

        if movies_file.exists():
            try:
                metadata_df = pd.read_csv(
                    movies_file,
                    sep="::",
                    header=None,
                    encoding="latin-1",
                    on_bad_lines="skip",
                    engine="python",
                    names=["item_id", "title", "genres"],
                )

                if "genres" in metadata_df.columns:
                    metadata_df["genres_list"] = metadata_df["genres"].str.split("|")

                items_df = items_df.merge(metadata_df, on="item_id", how="left")
            except Exception:
                pass

        return items_df
