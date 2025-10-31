import json
from pathlib import Path
from typing import Any, Dict, Union

from src.pipeline import Pipeline


class PipelineBuilder:
    @staticmethod
    def from_json_file(json_path: Union[str, Path], verbose: bool = True) -> Pipeline:
        json_path = Path(json_path)

        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")

        with open(json_path, "r") as f:
            config = json.load(f)

        return PipelineBuilder.from_dict(config, verbose=verbose)

    @staticmethod
    def from_dict(config: Dict[str, Any], verbose: bool = True) -> Pipeline:
        PipelineBuilder._validate_config(config)
        return Pipeline(config, verbose=verbose)

    @staticmethod
    def _validate_config(config: Dict[str, Any]):
        if "dataset" not in config:
            raise ValueError("Configuration must contain 'dataset' section")
