from src.pipeline_builder import PipelineBuilder


class TestFairnessIntegration:
    def test_fairness_stage_with_full_pipeline(self):
        config = {
            "dataset": {"name": "movielens"},
            "training": {
                "models": [{"name": "baseline", "params": {}}],
                "validation": {"type": "train_test_split", "test_size": 0.2, "metrics": ["rmse", "mae"]},
            },
            "fairness": {
                "user_attributes": [{"name": "gender", "type": "categorical"}],
                "item_attributes": [{"name": "genre", "type": "categorical"}],
                "positive_threshold": 4.0,
            },
        }

        pipeline = PipelineBuilder.from_dict(config)
        result = pipeline.run()

        assert "fairness_results" in result
        fairness_results = result["fairness_results"]

        assert "baseline" in fairness_results
        baseline_results = fairness_results["baseline"]

        assert "overall_metrics" in baseline_results
        assert "by_fold" in baseline_results
        assert "aggregated" in baseline_results

        aggregated = baseline_results["aggregated"]
        assert "user_fairness" in aggregated
        assert "item_fairness" in aggregated

        if "gender" in aggregated["user_fairness"]:
            gender_fairness = aggregated["user_fairness"]["gender"]
            assert "groups" in gender_fairness
            assert "fairness_metrics" in gender_fairness
            assert "disparities" in gender_fairness

    def test_fairness_with_numeric_bins(self):
        config = {
            "dataset": {"name": "movielens"},
            "training": {
                "models": [{"name": "baseline", "params": {}}],
                "validation": {"type": "train_test_split", "test_size": 0.2, "metrics": ["rmse"]},
            },
            "fairness": {
                "user_attributes": [
                    {
                        "name": "age",
                        "type": "numeric",
                        "bins": [
                            {"label": "young", "min": 0, "max": 25},
                            {"label": "adult", "min": 26, "max": 50},
                            {"label": "senior", "min": 51, "max": 100},
                        ],
                    }
                ],
                "positive_threshold": 4.0,
            },
        }

        pipeline = PipelineBuilder.from_dict(config)
        result = pipeline.run()

        fairness_results = result["fairness_results"]
        baseline_results = fairness_results["baseline"]

        age_fairness = baseline_results["aggregated"]["user_fairness"]["age"]
        assert "groups" in age_fairness

        groups = age_fairness["groups"]
        assert any(label in groups for label in ["young", "adult", "senior"])

    def test_fairness_with_cross_validate(self):
        config = {
            "dataset": {"name": "movielens"},
            "training": {
                "models": [{"name": "baseline", "params": {}}],
                "validation": {"type": "cross_validate", "cv": 3, "metrics": ["rmse"]},
            },
            "fairness": {
                "user_attributes": [{"name": "gender", "type": "categorical"}],
                "positive_threshold": 4.0,
            },
        }

        pipeline = PipelineBuilder.from_dict(config)
        result = pipeline.run()

        fairness_results = result["fairness_results"]
        baseline_results = fairness_results["baseline"]

        assert len(baseline_results["by_fold"]) == 3

        for fold_id in ["0", "1", "2"]:
            assert fold_id in baseline_results["by_fold"]
            fold_result = baseline_results["by_fold"][fold_id]
            assert "user_fairness" in fold_result

    def test_fairness_metrics_structure(self):
        config = {
            "dataset": {"name": "movielens"},
            "training": {
                "models": [{"name": "baseline", "params": {}}],
                "validation": {"type": "train_test_split", "test_size": 0.2, "metrics": ["rmse", "mae"]},
            },
            "fairness": {
                "user_attributes": [{"name": "gender", "type": "categorical"}],
                "positive_threshold": 4.0,
            },
        }

        pipeline = PipelineBuilder.from_dict(config)
        result = pipeline.run()

        gender_fairness = result["fairness_results"]["baseline"]["aggregated"]["user_fairness"]["gender"]

        fairness_metrics = gender_fairness["fairness_metrics"]
        assert "equal_opportunity" in fairness_metrics
        assert "equalized_odds" in fairness_metrics
        assert "demographic_parity" in fairness_metrics

        assert "ratio" in fairness_metrics["equal_opportunity"]
        assert "difference" in fairness_metrics["equal_opportunity"]

        disparities = gender_fairness["disparities"]
        assert "rmse" in disparities or "mae" in disparities
