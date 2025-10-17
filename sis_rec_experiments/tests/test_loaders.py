import unittest

from sis_rec_experiments.loaders.builder import create_loader


class TestLoaders(unittest.TestCase):

    def test_amazon_music_loader(self):
        loader = create_loader("amazonmusic")

        ratings_df = loader.load_ratings()
        self.assertIsNotNone(ratings_df)
        self.assertGreater(len(ratings_df), 0)
        self.assertIn("user_id", ratings_df.columns)
        self.assertIn("item_id", ratings_df.columns)
        self.assertIn("rating", ratings_df.columns)

        metadata_df = loader.load_metadata()
        self.assertIsNotNone(metadata_df)

    def test_anime_loader(self):
        loader = create_loader("anime")

        ratings_df = loader.load_ratings()
        self.assertIsNotNone(ratings_df)
        self.assertGreater(len(ratings_df), 0)
        self.assertIn("user_id", ratings_df.columns)
        self.assertIn("item_id", ratings_df.columns)
        self.assertIn("rating", ratings_df.columns)

        metadata_df = loader.load_metadata()
        self.assertIsNotNone(metadata_df)

    def test_book_crossing_loader(self):
        loader = create_loader("bookcrossing")

        ratings_df = loader.load_ratings()
        self.assertIsNotNone(ratings_df)
        self.assertGreater(len(ratings_df), 0)
        self.assertIn("user_id", ratings_df.columns)
        self.assertIn("item_id", ratings_df.columns)
        self.assertIn("rating", ratings_df.columns)

        metadata_df = loader.load_metadata()
        self.assertIsNotNone(metadata_df)

    def test_movielens_loader(self):
        loader = create_loader("movielens")

        ratings_df = loader.load_ratings()
        self.assertIsNotNone(ratings_df)
        self.assertGreater(len(ratings_df), 0)
        self.assertIn("user_id", ratings_df.columns)
        self.assertIn("item_id", ratings_df.columns)
        self.assertIn("rating", ratings_df.columns)

        metadata_df = loader.load_metadata()
        self.assertIsNotNone(metadata_df)

    def test_movielens_1m_loader(self):
        loader = create_loader("movielens1m")

        ratings_df = loader.load_ratings()
        self.assertIsNotNone(ratings_df)
        self.assertGreater(len(ratings_df), 0)
        self.assertIn("user_id", ratings_df.columns)
        self.assertIn("item_id", ratings_df.columns)
        self.assertIn("rating", ratings_df.columns)
        self.assertIn("timestamp", ratings_df.columns)

        self.assertGreaterEqual(ratings_df["rating"].min(), 1.0)
        self.assertLessEqual(ratings_df["rating"].max(), 5.0)

        self.assertGreater(len(ratings_df), 900000)

        metadata_df = loader.load_metadata()
        self.assertIsNotNone(metadata_df)
        self.assertGreater(len(metadata_df), 0)
        self.assertIn("item_id", metadata_df.columns)
        self.assertIn("title", metadata_df.columns)
        self.assertIn("genres", metadata_df.columns)
        self.assertIn("genres_list", metadata_df.columns)

        users_df = loader.load_users()
        self.assertIsNotNone(users_df)
        self.assertGreater(len(users_df), 0)
        self.assertIn("user_id", users_df.columns)
        self.assertIn("gender", users_df.columns)
        self.assertIn("age", users_df.columns)
        self.assertIn("occupation", users_df.columns)
        self.assertIn("occupation_name", users_df.columns)

        self.assertEqual(len(users_df), 6040)


if __name__ == "__main__":
    unittest.main()
