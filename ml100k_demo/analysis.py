import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import Counter

# Create visualizations directory
VIZ_DIR = "visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)


def load_recommendations(filename="recommendations.csv"):
    """Load recommendations CSV file."""
    try:
        df = pd.read_csv(filename)
        print(f"Successfully loaded {len(df)} recommendations from {filename}")
        return df
    except FileNotFoundError:
        print(f"File {filename} not found. Please run the main program first to generate recommendations.")
        return None


def basic_info(df):
    """Display basic information about the dataset."""
    print("\n" + "="*50)
    print("BASIC DATASET INFORMATION")
    print("="*50)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of unique users: {df['user_id'].nunique()}")
    print(f"Number of unique movies: {df['movie_id'].nunique()}")
    print(f"Total recommendations: {len(df)}")
    
    print("\nColumn types:")
    print(df.dtypes)
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nBasic statistics:")
    print(df.describe())


def rating_analysis(df):
    """Analyze predicted ratings distribution."""
    print("\n" + "="*50)
    print("PREDICTED RATINGS ANALYSIS")
    print("="*50)
    
    print(f"Rating range: {df['predicted_rating'].min():.3f} - {df['predicted_rating'].max():.3f}")
    print(f"Mean rating: {df['predicted_rating'].mean():.3f}")
    print(f"Median rating: {df['predicted_rating'].median():.3f}")
    print(f"Standard deviation: {df['predicted_rating'].std():.3f}")
    
    # Rating distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['predicted_rating'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Predicted Ratings')
    plt.xlabel('Predicted Rating')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot(df['predicted_rating'])
    plt.title('Predicted Ratings Boxplot')
    plt.ylabel('Predicted Rating')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'rating_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()


def user_analysis(df):
    """Analyze user recommendation patterns."""
    print("\n" + "="*50)
    print("USER ANALYSIS")
    print("="*50)
    
    # Users with most recommendations
    user_counts = df['user_id'].value_counts()
    print(f"Recommendations per user - Mean: {user_counts.mean():.1f}, Std: {user_counts.std():.1f}")
    
    print("\nTop 10 users by number of recommendations:")
    print(user_counts.head(10))
    
    # User rating preferences
    user_avg_ratings = df.groupby('user_id')['predicted_rating'].agg(['mean', 'std', 'count'])
    
    print(f"\nUser average rating preferences:")
    print(f"Mean of user averages: {user_avg_ratings['mean'].mean():.3f}")
    print(f"Users with highest average predicted ratings:")
    print(user_avg_ratings.nlargest(5, 'mean'))


def movie_analysis(df):
    """Analyze movie recommendation patterns."""
    print("\n" + "="*50)
    print("MOVIE ANALYSIS")
    print("="*50)
    
    # Most recommended movies
    movie_counts = df.groupby(['movie_id', 'movie_name']).size().reset_index(name='recommendation_count')
    movie_counts = movie_counts.sort_values('recommendation_count', ascending=False)
    
    print("Top 15 most recommended movies:")
    print(movie_counts.head(15).to_string(index=False))
    
    # Movie rating analysis
    movie_ratings = df.groupby(['movie_id', 'movie_name'])['predicted_rating'].agg(['mean', 'std', 'count']).reset_index()
    movie_ratings = movie_ratings.sort_values('mean', ascending=False)
    
    print(f"\nTop 10 movies by average predicted rating:")
    print(movie_ratings.head(10).to_string(index=False))
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    top_movies = movie_counts.head(10)
    plt.barh(range(len(top_movies)), top_movies['recommendation_count'])
    plt.yticks(range(len(top_movies)), [name[:30] + '...' if len(name) > 30 else name for name in top_movies['movie_name']])
    plt.xlabel('Number of Recommendations')
    plt.title('Top 10 Most Recommended Movies')
    plt.gca().invert_yaxis()
    
    plt.subplot(1, 2, 2)
    plt.hist(movie_counts['recommendation_count'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title('Distribution of Movie Recommendation Counts')
    plt.xlabel('Number of Recommendations')
    plt.ylabel('Number of Movies')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'movie_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()


def recommendation_patterns(df):
    """Analyze recommendation patterns."""
    print("\n" + "="*50)
    print("RECOMMENDATION PATTERNS ANALYSIS")
    print("="*50)
    
    # User-movie matrix insights
    user_movie_counts = df.groupby('user_id')['movie_id'].count()
    print(f"Recommendations per user statistics:")
    print(f"Min: {user_movie_counts.min()}, Max: {user_movie_counts.max()}")
    print(f"Mean: {user_movie_counts.mean():.2f}, Std: {user_movie_counts.std():.2f}")
    
    # Most popular movies across users
    movie_popularity = df['movie_name'].value_counts()
    print(f"\nMovie popularity distribution:")
    print(f"Most recommended movie appears {movie_popularity.iloc[0]} times")
    print(f"Least recommended movie appears {movie_popularity.iloc[-1]} times")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(user_movie_counts, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Recommendations per User')
    plt.xlabel('Number of Recommendations')
    plt.ylabel('Number of Users')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    top_movies = movie_popularity.head(8)
    plt.pie(top_movies.values, labels=[name[:20] + '...' if len(name) > 20 else name for name in top_movies.index], 
            autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Top 8 Movies')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'recommendation_patterns.png'), dpi=300, bbox_inches='tight')
    plt.show()


def correlation_analysis(df):
    """Analyze correlations between variables."""
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)
    
    # Create numerical features for correlation
    numeric_df = df[['user_id', 'movie_id', 'predicted_rating']].copy()
    
    correlation_matrix = numeric_df.corr()
    print("Correlation matrix:")
    print(correlation_matrix)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Run complete exploratory data analysis."""
    print("RECOMMENDATIONS DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_recommendations()
    if df is None:
        return
    
    # Run all analyses
    basic_info(df)
    rating_analysis(df)
    user_analysis(df)
    movie_analysis(df)
    recommendation_patterns(df)
    correlation_analysis(df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Generated visualizations in '{VIZ_DIR}/' folder:")
    print("- rating_distribution.png")
    print("- movie_analysis.png")
    print("- recommendation_patterns.png")
    print("- correlation_matrix.png")


if __name__ == "__main__":
    main()
