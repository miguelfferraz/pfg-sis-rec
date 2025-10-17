import matplotlib.pyplot as plt
import numpy as np

from .pipeline import ModelPipeline


def run_model_comparison():
    print("Running comprehensive model comparison...")
    print("=" * 80)

    pipeline = ModelPipeline(
        apply_preprocessing=True,
        preprocessing_params={"min_user_ratings": 5, "min_item_ratings": 10, "max_user_ratings": 500},
    )

    models = ["baseline", "svd"]
    results = {}

    for model in models:
        print(f"\nRunning {model.upper()} experiment...")
        print("-" * 50)
        results[model] = pipeline.run_experiment(dataset_name="movielens", model_name=model)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)

    print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'MSE':<10} {'FCP':<10} {'Time (s)':<10} {'Factors':<15}")
    print("-" * 105)

    for model_name, result in results.items():
        rmse = result["summary"]["rmse"]
        mae = result["summary"]["mae"]
        mse = result["summary"]["mse"]
        fcp = result["summary"]["fcp"]
        time = result["summary"]["total_time"]

        if model_name == "svd":
            factors = result["model_params"]["n_factors"]
            factors_str = str(factors)
        else:
            factors_str = "N/A"

        print(
            f"{model_name.upper():<15} {rmse:<10.4f} {mae:<10.4f} {mse:<10.4f} {fcp:<10.4f} {time:<10.2f} {factors_str:<15}"
        )

    baseline_rmse = results["baseline"]["summary"]["rmse"]
    svd_rmse = results["svd"]["summary"]["rmse"]
    rmse_improvement = (baseline_rmse - svd_rmse) / baseline_rmse * 100

    baseline_mae = results["baseline"]["summary"]["mae"]
    svd_mae = results["svd"]["summary"]["mae"]
    mae_improvement = (baseline_mae - svd_mae) / baseline_mae * 100

    baseline_mse = results["baseline"]["summary"]["mse"]
    svd_mse = results["svd"]["summary"]["mse"]
    mse_improvement = (baseline_mse - svd_mse) / baseline_mse * 100

    baseline_fcp = results["baseline"]["summary"]["fcp"]
    svd_fcp = results["svd"]["summary"]["fcp"]
    fcp_improvement = (svd_fcp - baseline_fcp) / baseline_fcp * 100

    print(f"\nSVD Improvements over Baseline:")
    print(f"  RMSE: {rmse_improvement:+.2f}%")
    print(f"  MAE: {mae_improvement:+.2f}%")
    print(f"  MSE: {mse_improvement:+.2f}%")
    print(f"  FCP: {fcp_improvement:+.2f}%")

    generate_comparison_plot(results)

    return results


def generate_comparison_plot(results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    models = list(results.keys())
    model_names = [m.upper() for m in models]

    # RMSE comparison
    rmse_values = [results[m]["summary"]["rmse"] for m in models]
    rmse_stds = [results[m]["aggregated_metrics"]["std_rmse"] for m in models]

    bars1 = ax1.bar(model_names, rmse_values, yerr=rmse_stds, capsize=5, alpha=0.7, color=["lightblue", "lightcoral"])
    ax1.set_ylabel("RMSE")
    ax1.set_title("RMSE Comparison")
    ax1.grid(True, alpha=0.3)

    for i, (bar, value) in enumerate(zip(bars1, rmse_values)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + rmse_stds[i] + 0.001,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # MAE comparison
    mae_values = [results[m]["summary"]["mae"] for m in models]
    mae_stds = [results[m]["aggregated_metrics"]["std_mae"] for m in models]

    bars2 = ax2.bar(model_names, mae_values, yerr=mae_stds, capsize=5, alpha=0.7, color=["lightblue", "lightcoral"])
    ax2.set_ylabel("MAE")
    ax2.set_title("MAE Comparison")
    ax2.grid(True, alpha=0.3)

    for i, (bar, value) in enumerate(zip(bars2, mae_values)):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + mae_stds[i] + 0.001,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # MSE comparison
    mse_values = [results[m]["summary"]["mse"] for m in models]
    mse_stds = [results[m]["aggregated_metrics"]["std_mse"] for m in models]

    bars3 = ax3.bar(model_names, mse_values, yerr=mse_stds, capsize=5, alpha=0.7, color=["lightblue", "lightcoral"])
    ax3.set_ylabel("MSE")
    ax3.set_title("MSE Comparison")
    ax3.grid(True, alpha=0.3)

    for i, (bar, value) in enumerate(zip(bars3, mse_values)):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + mse_stds[i] + 0.001,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # FCP comparison
    fcp_values = [results[m]["summary"]["fcp"] for m in models]
    fcp_stds = [results[m]["aggregated_metrics"]["std_fcp"] for m in models]

    bars4 = ax4.bar(model_names, fcp_values, yerr=fcp_stds, capsize=5, alpha=0.7, color=["lightblue", "lightcoral"])
    ax4.set_ylabel("FCP")
    ax4.set_title("FCP Comparison (Higher is Better)")
    ax4.grid(True, alpha=0.3)

    for i, (bar, value) in enumerate(zip(bars4, fcp_values)):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + fcp_stds[i] + 0.001,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    plot_path = "sis_rec_experiments/models/results/model_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nComparison plot saved: {plot_path}")
    plt.show()


if __name__ == "__main__":
    run_model_comparison()
