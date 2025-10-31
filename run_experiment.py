#!/usr/bin/env python3
"""
Script to run a single experiment.

Usage:
    python run_experiment.py <experiment_name>

Example:
    python run_experiment.py movielens_100k_fairness
"""
import sys
from pathlib import Path

from src.pipeline_builder import PipelineBuilder


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <experiment_name>")
        print("\nAvailable experiments:")
        experiments_dir = Path("experiments")
        for exp_file in sorted(experiments_dir.glob("*_fairness.json")):
            print(f"  - {exp_file.stem}")
        sys.exit(1)

    experiment_name = sys.argv[1]
    if not experiment_name.endswith(".json"):
        experiment_name = f"{experiment_name}.json"

    experiment_path = Path("experiments") / experiment_name

    if not experiment_path.exists():
        print(f"Error: Experiment file not found: {experiment_path}")
        sys.exit(1)

    print(f"\nRunning experiment: {experiment_path.stem}")
    print("=" * 80)

    pipeline = PipelineBuilder.from_json_file(str(experiment_path), verbose=True)
    result = pipeline.run()

    print("\n" + "=" * 80)
    print("Experiment completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

