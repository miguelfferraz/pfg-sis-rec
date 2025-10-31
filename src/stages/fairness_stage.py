from typing import Any, Dict

from src.stages.base_stage import BaseStage


class FairnessStage(BaseStage):
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return context
