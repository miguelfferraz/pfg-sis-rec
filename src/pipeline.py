from typing import Any, Dict, List

import pandas as pd

from src.loaders.loader_factory import LoaderFactory
from src.logger import Logger
from src.stages.base_stage import BaseStage
from src.stages.fairness_stage import FairnessStage
from src.stages.output_stage import OutputStage
from src.stages.preprocessing_stage import PreprocessingStage
from src.stages.training_stage import TrainingStage


class Pipeline:
    def __init__(self, config: Dict[str, Any], verbose: bool = True):
        self.config = config
        self.context: Dict[str, Any] = {}
        self.stages: List[BaseStage] = []
        self.logger = Logger(enabled=verbose)
        self._build_stages()

    def _build_stages(self):
        if "preprocessing" in self.config and self.config["preprocessing"]:
            self.stages.append(PreprocessingStage(self.config))

        if "training" in self.config:
            self.stages.append(TrainingStage(self.config))

        if "fairness" in self.config:
            self.stages.append(FairnessStage(self.config))

        if "output" in self.config:
            self.stages.append(OutputStage(self.config))

    def run(self) -> Dict[str, Any]:
        self.logger.info("Starting pipeline execution")
        
        self.context["logger"] = self.logger
        
        self.logger.stage_start("Dataset Loading")
        self._load_dataset()
        self.logger.stage_end("Dataset Loading")

        for stage in self.stages:
            stage_name = stage.__class__.__name__
            self.logger.stage_start(stage_name)
            self.context = stage.execute(self.context)
            self.logger.stage_end(stage_name)

        self.logger.info("Pipeline execution completed")
        return self.context

    def _load_dataset(self):
        dataset_config = self.config.get("dataset", {})
        dataset_name = dataset_config.get("name")

        self.logger.info(f"Loading dataset: {dataset_name}")
        
        factory = LoaderFactory()
        loader = factory.create_loader(dataset_name)

        surprise_dataset = loader.get_ratings()
        users_df = loader.get_users()
        items_df = loader.get_items()

        self.logger.info(f"Dataset loaded: {len(users_df)} users, {len(items_df)} items")

        trainset = surprise_dataset.build_full_trainset()
        ratings_data = []
        for uid, iid, rating in trainset.all_ratings():
            ratings_data.append(
                {"user_id": trainset.to_raw_uid(uid), "item_id": trainset.to_raw_iid(iid), "rating": rating}
            )
        ratings_df = pd.DataFrame(ratings_data)

        self.logger.info(f"Total ratings: {len(ratings_df)}")

        self.context["ratings_df"] = ratings_df
        self.context["users_df"] = users_df
        self.context["items_df"] = items_df
        self.context["loader"] = loader
        self.context["rating_scale"] = loader.DEFAULT_RATING_SCALE

    def get_predictions(self) -> pd.DataFrame:
        return self.context.get("predictions_df", pd.DataFrame())

    def get_training_results(self) -> Dict[str, Any]:
        return self.context.get("training_results", {})
