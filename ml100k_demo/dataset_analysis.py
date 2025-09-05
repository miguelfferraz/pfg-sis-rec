import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from surprise import Dataset, get_dataset_dir


def load_original_dataset():
    """Load the original MovieLens 100k dataset."""
    print("Loading MovieLens 100k dataset...")
    
    # Load using surprise to ensure dataset is available
    dataset = Dataset.load_builtin("ml-100k")
    
    # Get the directory where the dataset is stored
    data_dir = get_dataset_dir() + "/ml-100k/ml-100k"
    
    # Load ratings data
    ratings_file = os.path.join(data_dir, "u.data")
    ratings = pd.read_csv(
        ratings_file, 
        sep='\t', 
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    
    # Load movie data
    movies_file = os.path.join(data_dir, "u.item")
    movies = pd.read_csv(
        movies_file, 
        sep='|', 
        encoding='latin-1',
        names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + 
              [f'genre_{i}' for i in range(19)]
    )
    
    # Load user data
    users_file = os.path.join(data_dir, "u.user")
    users = pd.read_csv(
        users_file,
        sep='|',
        names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
    )
    
    print("Dataset loaded successfully!")
    return ratings, movies, users


def basic_dataset_info(ratings, movies, users):
    """Display basic information about the dataset."""
    print("\n" + "="*50)
    print("MOVIELENS 100K DATASET - BASIC INFO")
    print("="*50)
    
    print(f"Total ratings: {len(ratings):,}")
    print(f"Total movies: {len(movies):,}")
    print(f"Total users: {len(users):,}")
    print(f"Unique movies with ratings: {ratings['movie_id'].nunique():,}")
    print(f"Unique users with ratings: {ratings['user_id'].nunique():,}")
    
    print(f"\nRating scale: {ratings['rating'].min()} - {ratings['rating'].max()}")
    print(f"Average rating: {ratings['rating'].mean():.2f}")
    print(f"Rating distribution:")
    print(ratings['rating'].value_counts().sort_index())
    
    print(f"\nDataset density: {len(ratings) / (ratings['user_id'].nunique() * ratings['movie_id'].nunique()) * 100:.2f}%")


def ratings_analysis(ratings):
    """Analyze rating patterns."""
    print("\n" + "="*50)
    print("RATINGS ANALYSIS")
    print("="*50)
    
    # Ratings per user
    ratings_per_user = ratings.groupby('user_id').size()
    print(f"Ratings per user - Mean: {ratings_per_user.mean():.1f}, Std: {ratings_per_user.std():.1f}")
    print(f"Min ratings per user: {ratings_per_user.min()}")
    print(f"Max ratings per user: {ratings_per_user.max()}")
    
    # Ratings per movie
    ratings_per_movie = ratings.groupby('movie_id').size()
    print(f"\nRatings per movie - Mean: {ratings_per_movie.mean():.1f}, Std: {ratings_per_movie.std():.1f}")
    print(f"Min ratings per movie: {ratings_per_movie.min()}")
    print(f"Max ratings per movie: {ratings_per_movie.max()}")
    
    # Most rated movies
    most_rated = ratings_per_movie.nlargest(10)
    print(f"\nTop 10 most rated movies (by movie_id):")
    print(most_rated)


def movies_analysis(movies, ratings):
    """Analyze movie data."""
    print("\n" + "="*50)
    print("MOVIES ANALYSIS")
    print("="*50)
    
    # Extract year from title (most titles have year in parentheses)
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
    
    # Movies by decade
    decade_counts = movies['year'].dropna().apply(lambda x: int(x//10)*10).value_counts().sort_index()
    print("Movies by decade:")
    print(decade_counts)
    
    # Genre analysis
    genre_cols = [col for col in movies.columns if col.startswith('genre_')]
    genre_sums = movies[genre_cols].sum()
    
    # Map genre columns to actual genre names
    genre_names = [
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    
    genre_data = pd.Series(genre_sums.values, index=genre_names).sort_values(ascending=False)
    print(f"\nTop 10 genres by movie count:")
    print(genre_data.head(10))
    
    # Movies with most ratings
    movie_ratings = ratings.groupby('movie_id').agg({
        'rating': ['count', 'mean']
    }).round(2)
    movie_ratings.columns = ['num_ratings', 'avg_rating']
    movie_ratings = movie_ratings.sort_values('num_ratings', ascending=False)
    
    # Merge with movie titles
    top_rated_movies = movie_ratings.head(10).merge(
        movies[['movie_id', 'title']], on='movie_id'
    )
    
    print(f"\nTop 10 movies by number of ratings:")
    for _, row in top_rated_movies.iterrows():
        print(f"{row['title'][:50]:50} | {row['num_ratings']:4d} ratings | {row['avg_rating']:.2f} avg")


def users_analysis(users, ratings):
    """Analyze user demographics."""
    print("\n" + "="*50)
    print("USERS ANALYSIS")
    print("="*50)
    
    print("Age distribution:")
    print(f"Mean age: {users['age'].mean():.1f}")
    print(f"Age range: {users['age'].min()} - {users['age'].max()}")
    
    print(f"\nGender distribution:")
    print(users['gender'].value_counts())
    
    print(f"\nTop 10 occupations:")
    print(users['occupation'].value_counts().head(10))
    
    # User activity analysis
    user_activity = ratings.groupby('user_id').size()
    active_users = users.merge(
        user_activity.to_frame('num_ratings'), 
        left_on='user_id', 
        right_index=True
    )
    
    print(f"\nUser activity by gender:")
    gender_activity = active_users.groupby('gender')['num_ratings'].agg(['mean', 'std']).round(1)
    print(gender_activity)


def create_visualizations(ratings, movies, users):
    """Create simple visualizations."""
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    # Create visualizations directory
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Rating distribution
    axes[0, 0].hist(ratings['rating'], bins=5, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Rating Distribution')
    axes[0, 0].set_xlabel('Rating')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Ratings per user distribution
    ratings_per_user = ratings.groupby('user_id').size()
    axes[0, 1].hist(ratings_per_user, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Ratings per User Distribution')
    axes[0, 1].set_xlabel('Number of Ratings')
    axes[0, 1].set_ylabel('Number of Users')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Age distribution
    axes[1, 0].hist(users['age'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 0].set_title('User Age Distribution')
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Number of Users')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Top genres
    genre_cols = [col for col in movies.columns if col.startswith('genre_')]
    genre_names = [
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    genre_sums = movies[genre_cols].sum()
    genre_data = pd.Series(genre_sums.values, index=genre_names).sort_values(ascending=False)
    
    top_genres = genre_data.head(8)
    axes[1, 1].bar(range(len(top_genres)), top_genres.values, alpha=0.7, color='gold', edgecolor='black')
    axes[1, 1].set_title('Top 8 Movie Genres')
    axes[1, 1].set_xlabel('Genre')
    axes[1, 1].set_ylabel('Number of Movies')
    axes[1, 1].set_xticks(range(len(top_genres)))
    axes[1, 1].set_xticklabels(top_genres.index, rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'dataset_overview.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to {viz_dir}/dataset_overview.png")


def main():
    """Run complete dataset analysis."""
    print("MOVIELENS 100K DATASET ANALYSIS")
    print("="*60)
    
    try:
        # Load data
        ratings, movies, users = load_original_dataset()
        
        # Run analyses
        basic_dataset_info(ratings, movies, users)
        ratings_analysis(ratings)
        movies_analysis(movies, ratings)
        users_analysis(users, ratings)
        create_visualizations(ratings, movies, users)
        
        print("\n" + "="*60)
        print("DATASET ANALYSIS COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the MovieLens dataset is downloaded first by running the main recommendation system.")


if __name__ == "__main__":
    main()
