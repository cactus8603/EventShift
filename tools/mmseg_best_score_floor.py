from mmengine.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class BestScoreFloorHook(Hook):
    """Initialize mmengine best-score bookkeeping to a required floor."""

    priority = "LOW"

    def __init__(self, floors, primary_metric=None):
        self.floors = {str(metric): float(score) for metric, score in floors.items()}
        self.primary_metric = primary_metric

    def _apply(self, runner):
        for metric, floor in self.floors.items():
            keys = [f"best_score_{metric}"]
            if self.primary_metric is None or metric == self.primary_metric:
                keys.append("best_score")
            for key in keys:
                current = runner.message_hub.runtime_info.get(key)
                if current is None or float(current) < floor:
                    runner.message_hub.update_info(key, floor)

    def before_train(self, runner):
        self._apply(runner)

    def after_val_epoch(self, runner, metrics=None):
        self._apply(runner)
