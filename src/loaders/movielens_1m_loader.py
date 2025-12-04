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

        if not users_file.exists():
            return pd.DataFrame(columns=["user_id"])

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

            demographics_df["occupation_name"] = demographics_df["occupation"].map(occupation_map)

            gender_encoded = self._create_categorical_mapping(demographics_df["gender"], "gender", self.user_mappings)
            occupation_encoded = self._create_categorical_mapping(
                demographics_df["occupation_name"], "occupation", self.user_mappings
            )

            users_df = pd.DataFrame(
                {
                    "user_id": demographics_df["user_id"],
                    "gender": gender_encoded,
                    "age": demographics_df["age"],
                    "occupation": occupation_encoded,
                }
            )

            return users_df

        except Exception:
            return pd.DataFrame(columns=["user_id"])

    def _load_items(self) -> pd.DataFrame:
        movies_file = self.dataset_path / "movies.dat"

        if not movies_file.exists():
            return pd.DataFrame(columns=["item_id"])

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

            if "genres" not in metadata_df.columns:
                return pd.DataFrame(columns=["item_id"])

            primary_genres = []
            all_genres = set()

            for genres_str in metadata_df["genres"]:
                if pd.isna(genres_str) or genres_str == "":
                    primary_genres.append("unknown")
                else:
                    genre_list = genres_str.split("|")
                    primary_genre = genre_list[0] if genre_list else "unknown"
                    primary_genres.append(primary_genre)
                    all_genres.update(genre_list)

            unique_genres = sorted(list(all_genres))
            if "unknown" not in unique_genres:
                unique_genres.insert(0, "unknown")

            self.item_mappings["primary_genre"] = {genre: idx for idx, genre in enumerate(unique_genres)}

            items_df = pd.DataFrame(
                {
                    "item_id": metadata_df["item_id"],
                    "primary_genre": [self.item_mappings["primary_genre"][genre] for genre in primary_genres],
                }
            )

            return items_df

        except Exception:
            return pd.DataFrame(columns=["item_id"])
