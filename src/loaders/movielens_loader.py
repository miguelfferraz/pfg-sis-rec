import pandas as pd

from src.loaders.base_loader import BaseDatasetLoader


class MovieLensLoader(BaseDatasetLoader):
    """
    This dataset contains movie ratings from MovieLens 100k.
    Classic dataset with 100,000 ratings from 943 users on 1682 movies.
    """

    DEFAULT_RATING_SCALE = (1.0, 5.0)

    def _load_ratings(self) -> pd.DataFrame:
        ratings_file = self.dataset_path / "u.data"

        if not ratings_file.exists():
            raise FileNotFoundError(f"Rating file not found: {ratings_file}")

        return pd.read_csv(
            ratings_file,
            sep="\t",
            header=None,
            names=["user_id", "item_id", "rating", "timestamp"],
            dtype={"user_id": "int32", "item_id": "int32", "rating": "float32", "timestamp": "int64"},
        )

    def _load_users(self) -> pd.DataFrame:
        users_file = self.dataset_path / "u.user"

        if not users_file.exists():
            return pd.DataFrame(columns=["user_id"])

        try:
            demographics_df = pd.read_csv(
                users_file,
                sep="|",
                header=None,
                names=["user_id", "age", "gender", "occupation", "zip_code"],
                dtype={"user_id": "int32", "age": "int32", "gender": "str", "occupation": "str", "zip_code": "str"},
            )

            gender_encoded = self._create_categorical_mapping(demographics_df["gender"], "gender", self.user_mappings)
            occupation_encoded = self._create_categorical_mapping(
                demographics_df["occupation"], "occupation", self.user_mappings
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
        movies_file = self.dataset_path / "u.item"
        genre_file = self.dataset_path / "u.genre"

        if not movies_file.exists():
            return pd.DataFrame(columns=["item_id"])

        try:
            genre_names = []
            if genre_file.exists():
                with open(genre_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and "|" in line:
                            genre_name = line.split("|")[0]
                            if genre_name != "unknown":
                                genre_names.append(genre_name)

            movies_df = pd.read_csv(movies_file, sep="|", header=None, encoding="latin-1", on_bad_lines="skip")

            if len(movies_df.columns) < 6:
                return pd.DataFrame(columns=["item_id"])

            genre_columns = movies_df.iloc[:, 5 : 5 + len(genre_names)]

            primary_genres = []
            for _, row in genre_columns.iterrows():
                genre_indices = row[row == 1].index
                if len(genre_indices) > 0:
                    genre_idx = genre_indices[0] - 5
                    primary_genres.append(genre_idx)
                else:
                    primary_genres.append(0)

            self.item_mappings["primary_genre"] = {i: i for i in range(len(genre_names) + 1)}

            items_df = pd.DataFrame({"item_id": movies_df.iloc[:, 0], "primary_genre": primary_genres})

            return items_df

        except Exception:
            return pd.DataFrame(columns=["item_id"])
