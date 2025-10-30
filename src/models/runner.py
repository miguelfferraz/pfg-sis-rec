from .pipeline import ModelPipeline


def main():
    pipeline = ModelPipeline()

    print("Running BaselineOnly experiment on MovieLens dataset...")
    results = pipeline.run_experiment(dataset_name="movielens", model_name="baseline")

    print("\nExperiment Summary:")
    print(f"Model: {results['model_name']}")
    print(f"Dataset: {results['dataset_name']}")
    print(f"RMSE: {results['summary']['rmse']:.4f}")
    print(f"MAE: {results['summary']['mae']:.4f}")
    print(f"Total Time: {results['summary']['total_time']:.2f}s")


if __name__ == "__main__":
    main()
