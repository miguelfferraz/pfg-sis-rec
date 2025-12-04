from typing import Any, Dict

from src.stages.base_stage import BaseStage


class OutputStage(BaseStage):
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return context
