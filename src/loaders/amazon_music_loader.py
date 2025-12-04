import json

import pandas as pd

from src.loaders.base_loader import BaseDatasetLoader


class AmazonMusicLoader(BaseDatasetLoader):
    """
    This dataset contains reviews of musical products from Amazon,
    including ratings and metadata of the products.
    """

    DEFAULT_RATING_SCALE = (1.0, 5.0)

    def _load_ratings(self) -> pd.DataFrame:
        json_file = self.dataset_path / "Digital_Music_5.json"

        if not json_file.exists():
            raise FileNotFoundError(f"Rating file not found: {json_file}")

        ratings_data = []
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    review = json.loads(line.strip())
                    ratings_data.append(
                        {
                            "user_id": review["reviewerID"],
                            "item_id": review["asin"],
                            "rating": float(review["overall"]),
                            "timestamp": review.get("unixReviewTime", None),
                            "helpful": review.get("helpful", [0, 0]),
                        }
                    )
                except (json.JSONDecodeError, KeyError):
                    continue

        return pd.DataFrame(ratings_data)

    def _load_users(self) -> pd.DataFrame:
        if self._ratings_df is None:
            self._ratings_df = self._load_ratings()

        users_stats = (
            self._ratings_df.groupby("user_id")
            .agg(
                {
                    "rating": ["count", "mean"],
                    "timestamp": ["min", "max"],
                    "helpful": lambda x: x.apply(lambda h: h[0] if len(h) >= 2 else 0).sum(),
                }
            )
            .reset_index()
        )

        users_stats.columns = ["user_id", "total_ratings", "avg_rating", "first_review", "last_review", "total_helpful"]

        activity_bins = [0, 1, 5, 15, float("inf")]
        activity_labels = ["low", "medium", "high", "very_high"]
        users_stats["activity_level"] = pd.cut(
            users_stats["total_ratings"], bins=activity_bins, labels=activity_labels, include_lowest=True
        )

        rating_bins = [0, 3.0, 4.0, 4.5, 5.0]
        rating_labels = ["critical", "moderate", "positive", "very_positive"]
        users_stats["rating_behavior"] = pd.cut(
            users_stats["avg_rating"], bins=rating_bins, labels=rating_labels, include_lowest=True
        )

        helpful_bins = [0, 1, 5, float("inf")]
        helpful_labels = ["not_helpful", "somewhat_helpful", "very_helpful"]
        users_stats["helpfulness_level"] = pd.cut(
            users_stats["total_helpful"], bins=helpful_bins, labels=helpful_labels, include_lowest=True
        )

        activity_encoded = self._create_categorical_mapping(
            users_stats["activity_level"].astype(str), "activity_level", self.user_mappings
        )
        rating_behavior_encoded = self._create_categorical_mapping(
            users_stats["rating_behavior"].astype(str), "rating_behavior", self.user_mappings
        )
        helpfulness_encoded = self._create_categorical_mapping(
            users_stats["helpfulness_level"].astype(str), "helpfulness_level", self.user_mappings
        )

        users_df = pd.DataFrame(
            {
                "user_id": users_stats["user_id"],
                "activity_level": activity_encoded,
                "rating_behavior": rating_behavior_encoded,
                "helpfulness_level": helpfulness_encoded,
            }
        )

        return users_df

    def _load_items(self) -> pd.DataFrame:
        csv_file = self.dataset_path / "amazon_music_metadata.csv"

        if not csv_file.exists():
            return pd.DataFrame(columns=["item_id"])

        try:
            metadata_df = pd.read_csv(csv_file)
            if "asin" in metadata_df.columns:
                metadata_df = metadata_df.rename(columns={"asin": "item_id"})

            exclude_columns = [
                "item_id",
                "title",
                "Accessories",
                "Air Tool Accessories",
                "Arts & Crafts Supplies",
                "Arts, Crafts & Sewing",
                "Baby Products",
                "Bath & Body",
                "Bathroom Fixtures",
                "Beauty",
                "Book Lights",
                "Brass Accessories",
                "Cards & Card Stock",
                "CD Players",
                "CDs & Vinyl",
                "Classroom Decorations",
                "Cleansers",
                "Clothing, Shoes & Jewelry",
                "Collated Fasteners",
                "Collated Nails",
                "Cushions",
                "DJ Equipment",
                "DJ, Electronic Music & Karaoke",
                "Desk Accessories & Workspace Organizers",
                "Electrical",
                "Exercise",
                "Face",
                "Fan Shop",
                "Fasteners",
                "Fastening Tool Accessories",
                "Files & Rasps",
                "Fishing",
                "Flashlights",
                "Flies",
                "Fragrance",
                "Game Calls",
                "Gardening & Lawn Care",
                "Gear",
                "General Accessories",
                "Gun Holsters",
                "Gun Holsters, Cases & Bags",
                "Hand Tools",
                "Handheld Flashlights",
                "Health & Personal Care",
                "Health Care",
                "Hoes",
                "Holiday & Wedding",
                "Hose Reels",
                "Hunting",
                "Hunting & Fishing",
                "Hunting Optics",
                "Ice Hockey",
                "Industrial & Scientific",
                "Instrument Accessories",
                "Jewelry Accessories",
                "Karaoke Equipment",
                "Keyboards",
                "Kitchen & Bath Fixtures",
                "Kwanzaa",
                "Lacrosse",
                "Light Switches",
                "Lighting & Ceiling Fans",
                "Live Sound & Stage",
                "Lures, Baits & Attractants",
                "Mail & Suggestion Boxes",
                "Makeup",
                "Musical Instruments",
                "Mutes",
                "Novelty Lighting",
                "Novelty, Costumes & More",
                "Office & School Supplies",
                "Office Products",
                "PA Systems",
                "Pain Relievers",
                "Paper",
                "Patio Furniture & Accessories",
                "Patio Seating",
                "Patio, Lawn & Garden",
                "Pinner Nails",
                "Postcards",
                "Power & Hand Tools",
                "Power Tool Accessories",
                "Professional Development Resources",
                "Sheet Music Folders",
                "Showers",
                "Sports & Outdoors",
                "Steam Showers",
                "Studio Recording Equipment",
                "Switches",
                "Teaching Materials",
                "Team Sports",
                "Tools & Accessories",
                "Tools & Home Improvement",
                "Trim & Embellishments",
                "Vitamins & Dietary Supplements",
                "Voices",
                "Walkers",
                "Wall Stickers",
                "Wall Switches",
                "Washers",
                "Wave Washers & Wave Springs",
            ]

            genre_columns = [col for col in metadata_df.columns if col not in exclude_columns]

            primary_genres = []
            for _, row in metadata_df.iterrows():
                item_genres = []
                for genre in genre_columns:
                    if genre in row and row[genre] == 1.0:
                        item_genres.append(genre)

                if item_genres:
                    primary_genres.append(item_genres[0])
                else:
                    primary_genres.append("unknown")

            primary_genre_series = pd.Series(primary_genres)
            primary_genre_encoded = self._create_categorical_mapping(
                primary_genre_series, "primary_genre", self.item_mappings
            )

            items_df = pd.DataFrame(
                {
                    "item_id": metadata_df["item_id"],
                    "primary_genre": primary_genre_encoded,
                }
            )

            return items_df

        except Exception:
            return pd.DataFrame(columns=["item_id"])
