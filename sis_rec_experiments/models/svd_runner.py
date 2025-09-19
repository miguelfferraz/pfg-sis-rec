from .pipeline import ModelPipeline


def main():
    print("Running SVD experiments with preprocessing...")
    print("="*70)
    
    print("\n1. SVD EXPERIMENT (Raw Data)")
    print("-"*50)
    pipeline_raw = ModelPipeline(apply_preprocessing=False)
    results_svd_raw = pipeline_raw.run_experiment(
        dataset_name="movielens",
        model_name="svd"
    )
    
    print("\n2. SVD EXPERIMENT (Preprocessed Data)")
    print("-"*50)
    pipeline_filtered = ModelPipeline(
        apply_preprocessing=True,
        preprocessing_params={
            "min_user_ratings": 20,
            "min_item_ratings": 10,
            "max_user_ratings": 500
        }
    )
    results_svd_filtered = pipeline_filtered.run_experiment(
        dataset_name="movielens", 
        model_name="svd"
    )
    
    print("\n3. SVD COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<20} {'Raw Data':<15} {'Preprocessed':<15} {'Improvement':<15}")
    print("-"*70)
    
    raw_rmse = results_svd_raw['summary']['rmse']
    filtered_rmse = results_svd_filtered['summary']['rmse']
    rmse_improvement = (raw_rmse - filtered_rmse) / raw_rmse * 100
    
    raw_mae = results_svd_raw['summary']['mae'] 
    filtered_mae = results_svd_filtered['summary']['mae']
    mae_improvement = (raw_mae - filtered_mae) / raw_mae * 100
    
    raw_time = results_svd_raw['summary']['total_time']
    filtered_time = results_svd_filtered['summary']['total_time']
    time_change = (filtered_time - raw_time) / raw_time * 100
    
    print(f"{'RMSE':<20} {raw_rmse:<15.4f} {filtered_rmse:<15.4f} {rmse_improvement:<+14.2f}%")
    print(f"{'MAE':<20} {raw_mae:<15.4f} {filtered_mae:<15.4f} {mae_improvement:<+14.2f}%")
    print(f"{'Time (s)':<20} {raw_time:<15.2f} {filtered_time:<15.2f} {time_change:<+14.2f}%")
    
    if "preprocessing_report" in results_svd_filtered:
        preproc = results_svd_filtered["preprocessing_report"]["reduction_summary"]
        print(f"\nData Reduction:")
        print(f"  Ratings: {preproc['ratings_reduction']*100:.1f}% removed")
        print(f"  Users: {preproc['users_reduction']*100:.1f}% removed")  
        print(f"  Items: {preproc['items_reduction']*100:.1f}% removed")
        print(f"  Sparsity: {preproc['sparsity_change']*100:.2f}% points improvement")


if __name__ == "__main__":
    main()
