from typing import Any, Dict

import pandas as pd

from sis_rec_experiments.loaders.base_loader import BaseDatasetLoader


class MovieLens1MLoader(BaseDatasetLoader):
    """
    This dataset contains movie ratings from MovieLens 1M.
    Dataset with 1,000,209 ratings from 6,040 users on 3,883 movies.
    """

    DEFAULT_RATING_SCALE = (1.0, 5.0)

    def load_ratings(self) -> pd.DataFrame:
        ratings_file = self.dataset_path / "ratings.dat"

        if not ratings_file.exists():
            raise FileNotFoundError(f"Rating file not found: {ratings_file}")

        self.ratings_df = pd.read_csv(
            ratings_file,
            sep="::",
            header=None,
            names=["user_id", "item_id", "rating", "timestamp"],
            dtype={"user_id": "int32", "item_id": "int32", "rating": "float32", "timestamp": "int64"},
            engine="python",
        )

        return self.ratings_df

    def load_metadata(self) -> pd.DataFrame:
        movies_file = self.dataset_path / "movies.dat"

        if movies_file.exists():
            try:
                self.metadata_df = pd.read_csv(
                    movies_file,
                    sep="::",
                    header=None,
                    encoding="latin-1",
                    on_bad_lines="skip",
                    engine="python",
                    names=["item_id", "title", "genres"],
                )

                if "genres" in self.metadata_df.columns:
                    self.metadata_df["genres_list"] = self.metadata_df["genres"].str.split("|")

            except Exception as e:
                print(f"Error loading metadata: {e}")
                self.metadata_df = pd.DataFrame()
        else:
            self.metadata_df = pd.DataFrame()

        return self.metadata_df

    def load_users(self) -> pd.DataFrame:
        users_file = self.dataset_path / "users.dat"

        if not users_file.exists():
            print(f"Users file not found: {users_file}")
            return pd.DataFrame()

        try:
            users_df = pd.read_csv(
                users_file,
                sep="::",
                header=None,
                names=["user_id", "gender", "age", "occupation", "zip_code"],
                dtype={"user_id": "int32", "gender": "str", "age": "int32", "occupation": "int32", "zip_code": "str"},
                engine="python",
            )

            age_map = {
                1: 17,
                18: 21,
                25: 29,
                35: 39,
                45: 47,
                50: 52,
                56: 60,
            }

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

            users_df["occupation_name"] = users_df["occupation"].map(occupation_map)

            return users_df

        except Exception as e:
            print(f"Error loading users: {e}")
            return pd.DataFrame()

    def get_dataset_info(self) -> Dict[str, Any]:
        if self.ratings_df is None:
            self.load_ratings()

        basic_stats = self.get_basic_stats()

        if self.metadata_df is None:
            self.load_metadata()

        info = {
            **basic_stats,
            "dataset_name": "MovieLens 1M",
            "dataset_type": "Movie Ratings",
            "domain": "Movies/Entertainment",
            "has_metadata": self.metadata_df is not None and not self.metadata_df.empty,
            "has_user_demographics": True,
            "rating_scale": (1.0, 5.0),
            "data_format": ":: (Double colon separated)",
            "description": "MovieLens 1M dataset with 1,000,209 ratings from 6,040 users on 3,883 movies",
            "collection_period": "2000",
            "source": "GroupLens Research Project, University of Minnesota",
        }

        return info

    def get_genre_statistics(self) -> Dict[str, Any]:
        if self.metadata_df is None or self.metadata_df.empty:
            self.load_metadata()

        if self.metadata_df.empty or "genres_list" not in self.metadata_df.columns:
            return {}

        all_genres = []
        for genres_list in self.metadata_df["genres_list"]:
            if isinstance(genres_list, list):
                all_genres.extend(genres_list)

        genre_counts = pd.Series(all_genres).value_counts()

        return {
            "total_unique_genres": len(genre_counts),
            "most_common_genres": genre_counts.head(10).to_dict(),
            "genre_distribution": genre_counts.to_dict(),
        }

    def get_user_demographics_stats(self) -> Dict[str, Any]:
        users_df = self.load_users()

        if users_df.empty:
            return {}

        return {
            "gender_distribution": users_df["gender"].value_counts().to_dict(),
            "age_statistics": {
                "mean": float(users_df["age"].mean()),
                "median": float(users_df["age"].median()),
                "min": int(users_df["age"].min()),
                "max": int(users_df["age"].max()),
                "std": float(users_df["age"].std()),
            },
            "occupation_distribution": users_df["occupation_name"].value_counts().head(10).to_dict(),
            "total_users": len(users_df),
        }
