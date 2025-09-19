import pandas as pd
import numpy as np
import os
from surprise import Dataset, get_dataset_dir
from collections import Counter


def load_recommendations():
    """Load generated recommendations."""
    try:
        return pd.read_csv("recommendations.csv")
    except FileNotFoundError:
        print("recommendations.csv not found. Run main.py first.")
        return None


def load_original_data():
    """Load original MovieLens data."""
    # Ensure dataset is available
    Dataset.load_builtin("ml-100k")
    
    data_dir = get_dataset_dir() + "/ml-100k/ml-100k"
    
    # Load ratings
    ratings = pd.read_csv(
        os.path.join(data_dir, "u.data"),
        sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    
    # Load movies
    movies = pd.read_csv(
        os.path.join(data_dir, "u.item"),
        sep='|',
        encoding='latin-1',
        names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + 
              [f'genre_{i}' for i in range(19)]
    )
    
    return ratings, movies


def analyze_recommendation_quality(recommendations, original_ratings, movies):
    """Analyze quality and issues with recommendations."""
    print("="*60)
    print("MODEL EVALUATION AND RECOMMENDATION ANALYSIS")
    print("="*60)
    
    # Basic comparison
    print("\nBASIC COMPARISON:")
    print(f"Original dataset movies: {original_ratings['movie_id'].nunique():,}")
    print(f"Recommended movies: {recommendations['movie_id'].nunique()}")
    print(f"Percentage of catalog recommended: {recommendations['movie_id'].nunique() / original_ratings['movie_id'].nunique() * 100:.1f}%")
    
    # Check if recommended movies exist in original
    original_movie_ids = set(original_ratings['movie_id'].unique())
    recommended_movie_ids = set(recommendations['movie_id'].unique())
    
    missing_movies = recommended_movie_ids - original_movie_ids
    if missing_movies:
        print(f"ERROR: {len(missing_movies)} recommended movies not in original dataset!")
        print(f"Missing movie IDs: {list(missing_movies)}")
    
    print("\nRECOMMENDED MOVIES ANALYSIS:")
    rec_movie_stats = recommendations.groupby(['movie_id', 'movie_name']).size().reset_index(name='recommendation_count')
    rec_movie_stats = rec_movie_stats.sort_values('recommendation_count', ascending=False)
    
    for _, row in rec_movie_stats.iterrows():
        movie_id = row['movie_id']
        original_stats = original_ratings[original_ratings['movie_id'] == movie_id]
        
        if len(original_stats) > 0:
            avg_rating = original_stats['rating'].mean()
            num_ratings = len(original_stats)
            print(f"{row['movie_name'][:40]:40} | {row['recommendation_count']:4d} recs | {num_ratings:3d} orig ratings | {avg_rating:.2f} avg")
        else:
            print(f"{row['movie_name'][:40]:40} | {row['recommendation_count']:4d} recs | NOT FOUND in original")


def analyze_rating_predictions(recommendations, original_ratings):
    """Analyze predicted ratings vs original patterns."""
    print("\n" + "="*60)
    print("RATING PREDICTION ANALYSIS")
    print("="*60)
    
    print(f"Predicted ratings range: {recommendations['predicted_rating'].min()} - {recommendations['predicted_rating'].max()}")
    print(f"Original ratings range: {original_ratings['rating'].min()} - {original_ratings['rating'].max()}")
    
    print(f"\nPredicted ratings distribution:")
    print(recommendations['predicted_rating'].value_counts().sort_index())
    
    print(f"\nOriginal ratings distribution:")
    print(original_ratings['rating'].value_counts().sort_index())
    
    # Check for issues
    print(f"\n🚨 ISSUES IDENTIFIED:")
    if recommendations['predicted_rating'].nunique() == 1:
        print("❌ ALL predictions are the same value - model is not learning patterns")
    
    if recommendations['predicted_rating'].min() == recommendations['predicted_rating'].max() == 5:
        print("❌ All predictions are maximum rating (5) - likely overfitting")
    
    predicted_mean = recommendations['predicted_rating'].mean()
    original_mean = original_ratings['rating'].mean()
    print(f"❌ Predicted mean ({predicted_mean:.2f}) vs Original mean ({original_mean:.2f}) - {abs(predicted_mean - original_mean):.2f} difference")


def analyze_movie_popularity_bias(recommendations, original_ratings, movies):
    """Check if model is biased toward popular movies."""
    print("\n" + "="*60)
    print("POPULARITY BIAS ANALYSIS")
    print("="*60)
    
    # Get movie popularity from original data
    movie_popularity = original_ratings.groupby('movie_id').agg({
        'rating': ['count', 'mean']
    }).round(2)
    movie_popularity.columns = ['num_ratings', 'avg_rating']
    movie_popularity = movie_popularity.reset_index()
    
    # Merge with movie names
    movie_popularity = movie_popularity.merge(
        movies[['movie_id', 'title']], on='movie_id'
    )
    
    # Get recommended movies stats
    recommended_movies = recommendations['movie_id'].unique()
    
    print("RECOMMENDED MOVIES POPULARITY IN ORIGINAL DATASET:")
    for movie_id in recommended_movies:
        movie_data = movie_popularity[movie_popularity['movie_id'] == movie_id]
        if len(movie_data) > 0:
            row = movie_data.iloc[0]
            popularity_rank = (movie_popularity['num_ratings'] >= row['num_ratings']).sum()
            print(f"{row['title'][:50]:50} | {row['num_ratings']:3d} ratings | Rank #{popularity_rank:4d} | {row['avg_rating']:.2f} avg")
    
    # Check if only recommending popular movies
    top_100_popular = movie_popularity.nlargest(100, 'num_ratings')['movie_id'].values
    recommended_in_top100 = len(set(recommended_movies) & set(top_100_popular))
    
    print(f"\n📊 POPULARITY BIAS METRICS:")
    print(f"Recommended movies in top 100 most popular: {recommended_in_top100}/{len(recommended_movies)} ({recommended_in_top100/len(recommended_movies)*100:.1f}%)")
    
    if recommended_in_top100 / len(recommended_movies) > 0.8:
        print("❌ HIGH POPULARITY BIAS - Model mostly recommends popular movies")
    elif recommended_in_top100 / len(recommended_movies) < 0.2:
        print("❌ LOW POPULARITY - Model recommends very unpopular movies")
    else:
        print("✅ BALANCED - Good mix of popular and less popular movies")


def identify_model_issues(recommendations, original_ratings):
    """Identify specific issues with the recommendation model."""
    print("\n" + "="*60)
    print("MODEL ISSUES IDENTIFIED")
    print("="*60)
    
    issues = []
    
    # Issue 1: No rating diversity
    if recommendations['predicted_rating'].nunique() == 1:
        issues.append("🔴 CRITICAL: No rating diversity - all predictions identical")
    
    # Issue 2: Only maximum ratings
    if recommendations['predicted_rating'].min() == 5:
        issues.append("🔴 CRITICAL: Only predicting maximum ratings (5.0)")
    
    # Issue 3: Limited movie diversity
    total_movies = original_ratings['movie_id'].nunique()
    recommended_movies = recommendations['movie_id'].nunique()
    diversity_ratio = recommended_movies / total_movies
    
    if diversity_ratio < 0.01:
        issues.append(f"🔴 CRITICAL: Very low movie diversity - only {recommended_movies}/{total_movies} movies ({diversity_ratio*100:.1f}%)")
    elif diversity_ratio < 0.05:
        issues.append(f"🟡 WARNING: Low movie diversity - only {recommended_movies}/{total_movies} movies ({diversity_ratio*100:.1f}%)")
    
    # Issue 4: Check if using anti-testset correctly
    users_in_recs = set(recommendations['user_id'].unique())
    users_in_original = set(original_ratings['user_id'].unique())
    
    if users_in_recs != users_in_original:
        missing_users = users_in_original - users_in_recs
        issues.append(f"🟡 WARNING: {len(missing_users)} users missing recommendations")
    
    # Issue 5: All users get same number of recommendations
    recs_per_user = recommendations.groupby('user_id').size()
    if recs_per_user.nunique() == 1:
        issues.append(f"🟡 INFO: All users get exactly {recs_per_user.iloc[0]} recommendations")
    
    if not issues:
        print("✅ No critical issues found!")
    else:
        for issue in issues:
            print(issue)


def suggest_improvements():
    """Suggest specific improvements for the recommendation model."""
    print("\n" + "="*60)
    print("SUGGESTED MODEL IMPROVEMENTS")
    print("="*60)
    
    improvements = [
        "🎯 ALGORITHM IMPROVEMENTS:",
        "1. Try different similarity metrics (Pearson, cosine with mean-centering)",
        "2. Implement matrix factorization (SVD, NMF) instead of KNN",
        "3. Use user-based collaborative filtering instead of item-based",
        "4. Add baseline predictors to handle rating bias",
        "",
        "🔧 PARAMETER TUNING:",
        "5. Tune KNN parameters (k neighbors, minimum support)",
        "6. Apply rating normalization/centering",
        "7. Use cross-validation for parameter selection",
        "",
        "📊 DATA PREPROCESSING:",
        "8. Filter out movies with very few ratings (< 10)",
        "9. Filter out users with very few ratings (< 20)",
        "10. Handle cold start problem for new items/users",
        "",
        "🎲 RECOMMENDATION DIVERSITY:",
        "11. Implement re-ranking for diversity",
        "12. Add genre diversity constraints",
        "13. Use ensemble methods combining multiple algorithms",
        "",
        "📈 EVALUATION:",
        "14. Implement proper train/test split instead of anti-testset",
        "15. Use evaluation metrics (RMSE, MAE, Precision@K, Recall@K)",
        "16. Add A/B testing framework",
        "",
        "🚀 ADVANCED TECHNIQUES:",
        "17. Deep learning approaches (Neural Collaborative Filtering)",
        "18. Content-based features (genres, actors, directors)",
        "19. Hybrid recommender combining collaborative + content-based",
        "20. Implement temporal dynamics (time-aware recommendations)"
    ]
    
    for improvement in improvements:
        print(improvement)


def main():
    """Run complete model evaluation."""
    print("RECOMMENDATION MODEL EVALUATION")
    print("="*60)
    
    # Load data
    recommendations = load_recommendations()
    if recommendations is None:
        return
    
    original_ratings, movies = load_original_data()
    
    # Run analyses
    analyze_recommendation_quality(recommendations, original_ratings, movies)
    analyze_rating_predictions(recommendations, original_ratings)
    analyze_movie_popularity_bias(recommendations, original_ratings, movies)
    identify_model_issues(recommendations, original_ratings)
    suggest_improvements()
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
