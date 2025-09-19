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


if __name__ == "__main__":
    unittest.main()
