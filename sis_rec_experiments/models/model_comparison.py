import matplotlib.pyplot as plt
import numpy as np

from .pipeline import ModelPipeline


def run_model_comparison():
    print("Running comprehensive model comparison...")
    print("="*80)
    
    pipeline = ModelPipeline(
        apply_preprocessing=True,
        preprocessing_params={
            "min_user_ratings": 5,
            "min_item_ratings": 10,
            "max_user_ratings": 500
        }
    )
    
    models = ["baseline", "svd"]
    results = {}
    
    for model in models:
        print(f"\nRunning {model.upper()} experiment...")
        print("-"*50)
        results[model] = pipeline.run_experiment(
            dataset_name="movielens",
            model_name=model
        )
    
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'Time (s)':<10} {'Factors':<15}")
    print("-"*80)
    
    for model_name, result in results.items():
        rmse = result['summary']['rmse']
        mae = result['summary']['mae']
        time = result['summary']['total_time']
        
        if model_name == "svd":
            factors = result['model_params']['n_factors']
            factors_str = str(factors)
        else:
            factors_str = "N/A"
        
        print(f"{model_name.upper():<15} {rmse:<10.4f} {mae:<10.4f} {time:<10.2f} {factors_str:<15}")
    
    baseline_rmse = results['baseline']['summary']['rmse']
    svd_rmse = results['svd']['summary']['rmse']
    rmse_improvement = (baseline_rmse - svd_rmse) / baseline_rmse * 100
    
    baseline_mae = results['baseline']['summary']['mae']
    svd_mae = results['svd']['summary']['mae']
    mae_improvement = (baseline_mae - svd_mae) / baseline_mae * 100
    
    print(f"\nSVD Improvements over Baseline:")
    print(f"  RMSE: {rmse_improvement:+.2f}%")
    print(f"  MAE: {mae_improvement:+.2f}%")
    
    generate_comparison_plot(results)
    
    return results


def generate_comparison_plot(results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    models = list(results.keys())
    model_names = [m.upper() for m in models]
    
    # RMSE comparison
    rmse_values = [results[m]['summary']['rmse'] for m in models]
    rmse_stds = [results[m]['aggregated_metrics']['std_rmse'] for m in models]
    
    bars1 = ax1.bar(model_names, rmse_values, yerr=rmse_stds, 
                    capsize=5, alpha=0.7, color=['lightblue', 'lightcoral'])
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE Comparison')
    ax1.grid(True, alpha=0.3)
    
    for i, (bar, value) in enumerate(zip(bars1, rmse_values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + rmse_stds[i] + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # MAE comparison
    mae_values = [results[m]['summary']['mae'] for m in models]
    mae_stds = [results[m]['aggregated_metrics']['std_mae'] for m in models]
    
    bars2 = ax2.bar(model_names, mae_values, yerr=mae_stds,
                    capsize=5, alpha=0.7, color=['lightblue', 'lightcoral'])
    ax2.set_ylabel('MAE')
    ax2.set_title('MAE Comparison')
    ax2.grid(True, alpha=0.3)
    
    for i, (bar, value) in enumerate(zip(bars2, mae_values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + mae_stds[i] + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Training time comparison
    time_values = [results[m]['summary']['total_time'] for m in models]
    
    bars3 = ax3.bar(model_names, time_values, alpha=0.7, color=['lightblue', 'lightcoral'])
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Training Time Comparison')
    ax3.grid(True, alpha=0.3)
    
    for bar, value in zip(bars3, time_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    # Fold-wise RMSE comparison
    baseline_folds = [f['rmse'] for f in results['baseline']['fold_results']]
    svd_folds = [f['rmse'] for f in results['svd']['fold_results']]
    
    x = np.arange(1, 6)
    width = 0.35
    
    ax4.bar(x - width/2, baseline_folds, width, label='Baseline', alpha=0.7, color='lightblue')
    ax4.bar(x + width/2, svd_folds, width, label='SVD', alpha=0.7, color='lightcoral')
    
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('RMSE')
    ax4.set_title('RMSE by Fold')
    ax4.set_xticks(x)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = "sis_rec_experiments/models/results/model_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved: {plot_path}")
    plt.show()


if __name__ == "__main__":
    run_model_comparison()
