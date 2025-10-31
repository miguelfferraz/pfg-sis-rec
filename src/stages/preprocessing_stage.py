from typing import Any, Dict

from src.preprocessors.min_ratings_filter import MinRatingsFilter
from src.preprocessors.rating_normalizer import RatingNormalizer
from src.stages.base_stage import BaseStage


class PreprocessingStage(BaseStage):
    PREPROCESSOR_MAP = {"min_ratings_filter": MinRatingsFilter, "rating_normalizer": RatingNormalizer}

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ratings_df = context["ratings_df"]
        users_df = context["users_df"]
        items_df = context["items_df"]

        preprocessors_config = self.config.get("preprocessing", [])

        for preprocessor_config in preprocessors_config:
            preprocessor_type = preprocessor_config["type"]
            preprocessor_params = preprocessor_config.get("params", {})

            if preprocessor_type not in self.PREPROCESSOR_MAP:
                raise ValueError(f"Unknown preprocessor type: {preprocessor_type}")

            preprocessor_class = self.PREPROCESSOR_MAP[preprocessor_type]
            preprocessor = preprocessor_class(preprocessor_params)

            ratings_df, users_df, items_df = preprocessor.process(ratings_df, users_df, items_df)

        context["ratings_df"] = ratings_df
        context["users_df"] = users_df
        context["items_df"] = items_df

        return context
