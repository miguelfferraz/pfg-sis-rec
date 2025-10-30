import unittest

from sis_rec_experiments.loaders.builder import create_loader


class TestSurpriseConversion(unittest.TestCase):

    def test_amazon_music_surprise(self):
        loader = create_loader("amazonmusic")
        loader.load_ratings()

        surprise_dataset = loader.to_surprise_dataset()
        self.assertIsNotNone(surprise_dataset)

        trainset = surprise_dataset.build_full_trainset()
        self.assertGreater(trainset.n_users, 0)
        self.assertGreater(trainset.n_items, 0)
        self.assertGreater(trainset.n_ratings, 0)
        self.assertEqual(trainset.rating_scale, (1.0, 5.0))

    def test_anime_surprise(self):
        loader = create_loader("anime")
        loader.load_ratings()

        surprise_dataset = loader.to_surprise_dataset()
        self.assertIsNotNone(surprise_dataset)

        trainset = surprise_dataset.build_full_trainset()
        self.assertGreater(trainset.n_users, 0)
        self.assertGreater(trainset.n_items, 0)
        self.assertGreater(trainset.n_ratings, 0)
        self.assertEqual(trainset.rating_scale, (1.0, 10.0))

    def test_book_crossing_surprise(self):
        loader = create_loader("bookcrossing")
        loader.load_ratings()

        surprise_dataset = loader.to_surprise_dataset()
        self.assertIsNotNone(surprise_dataset)

        trainset = surprise_dataset.build_full_trainset()
        self.assertGreater(trainset.n_users, 0)
        self.assertGreater(trainset.n_items, 0)
        self.assertGreater(trainset.n_ratings, 0)
        self.assertEqual(trainset.rating_scale, (1.0, 10.0))

    def test_rating_scale_override(self):
        loader = create_loader("amazonmusic")
        loader.load_ratings()

        custom_scale = (0.0, 1.0)
        surprise_dataset = loader.to_surprise_dataset(rating_scale=custom_scale)
        trainset = surprise_dataset.build_full_trainset()
        self.assertEqual(trainset.rating_scale, custom_scale)


if __name__ == "__main__":
    unittest.main()
