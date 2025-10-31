import time
from typing import Optional


class Logger:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.stage_start_time: Optional[float] = None

    def info(self, message: str):
        if self.enabled:
            print(f"[INFO] {message}")

    def stage_start(self, stage_name: str):
        if self.enabled:
            print(f"\n{'='*60}")
            print(f"[STAGE] Starting: {stage_name}")
            print(f"{'='*60}")
            self.stage_start_time = time.time()

    def stage_end(self, stage_name: str):
        if self.enabled and self.stage_start_time:
            elapsed = time.time() - self.stage_start_time
            print(f"[STAGE] Completed: {stage_name} ({elapsed:.2f}s)")
            print(f"{'='*60}\n")
            self.stage_start_time = None

    def warning(self, message: str):
        if self.enabled:
            print(f"[WARNING] {message}")

    def error(self, message: str):
        if self.enabled:
            print(f"[ERROR] {message}")

