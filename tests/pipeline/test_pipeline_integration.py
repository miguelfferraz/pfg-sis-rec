import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline_builder import PipelineBuilder


class TestPipelineIntegration:
    def test_pipeline_from_dict(self, sample_config):
        pipeline = PipelineBuilder.from_dict(sample_config)
        assert pipeline is not None
        assert len(pipeline.stages) > 0

    def test_pipeline_run_with_preprocessing_and_training(self, sample_config):
        pipeline = PipelineBuilder.from_dict(sample_config)
        result = pipeline.run()

        assert "predictions_df" in result
        assert "training_results" in result
        assert isinstance(result["predictions_df"], pd.DataFrame)
        assert len(result["predictions_df"]) > 0

        predictions_df = result["predictions_df"]
        assert "user_id" in predictions_df.columns
        assert "item_id" in predictions_df.columns
        assert "rating_true" in predictions_df.columns
        assert "rating_pred" in predictions_df.columns
        assert "model_name" in predictions_df.columns

        self._cleanup(sample_config["output"]["save_path"])

    def test_pipeline_cross_validate(self, sample_config_cross_validate):
        pipeline = PipelineBuilder.from_dict(sample_config_cross_validate)
        result = pipeline.run()

        assert "predictions_df" in result
        predictions_df = result["predictions_df"]
        assert "fold" in predictions_df.columns
        assert predictions_df["fold"].nunique() == 3

        training_results = result["training_results"]
        assert "baseline" in training_results
        assert training_results["baseline"]["validation_type"] == "cross_validate"

        self._cleanup(sample_config_cross_validate["output"]["save_path"])

    def test_pipeline_saves_results(self, sample_config):
        pipeline = PipelineBuilder.from_dict(sample_config)
        pipeline.run()

        output_path = Path(sample_config["output"]["save_path"])
        assert output_path.exists()
        assert (output_path / "training_results.json").exists()
        assert (output_path / "predictions.csv").exists()

        self._cleanup(sample_config["output"]["save_path"])

    def test_pipeline_without_preprocessing(self, minimal_config):
        pipeline = PipelineBuilder.from_dict(minimal_config)
        result = pipeline.run()

        assert "predictions_df" in result
        assert len(result["predictions_df"]) > 0

    def test_pipeline_multiple_models(self):
        config = {
            "dataset": {"name": "movielens"},
            "training": {
                "models": [{"name": "svd", "params": {"n_factors": 5}}, {"name": "baseline", "params": {}}],
                "validation": {"type": "train_test_split", "test_size": 0.2, "metrics": ["rmse"]},
            },
        }

        pipeline = PipelineBuilder.from_dict(config)
        result = pipeline.run()

        predictions_df = result["predictions_df"]
        assert "svd" in predictions_df["model_name"].values
        assert "baseline" in predictions_df["model_name"].values

        training_results = result["training_results"]
        assert "svd" in training_results
        assert "baseline" in training_results

    def test_pipeline_builder_validation(self):
        invalid_config = {"training": {"models": []}}

        with pytest.raises(ValueError, match="must contain 'dataset'"):
            PipelineBuilder.from_dict(invalid_config)

    def test_pipeline_builder_from_json_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            PipelineBuilder.from_json_file("nonexistent.json")

    def _cleanup(self, path: str):
        output_path = Path(path)
        if output_path.exists():
            shutil.rmtree(output_path)
