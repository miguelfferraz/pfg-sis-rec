from sis_rec_experiments.loaders.builder import create_loader

if __name__ == "__main__":
    loader = create_loader("amazonmusic")
    ratings_df = loader.load_ratings()
    metadata_df = loader.load_metadata()

    print("Ratings:")
    print(ratings_df.head())

    print("\nMetadata:")
    print(metadata_df.head())
