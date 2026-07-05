from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np
from mmseg.evaluation.metrics import IoUMetric
from mmseg.registry import METRICS


@METRICS.register_module()
class DomainSplitIoUMetric(IoUMetric):
    """Compute mIoU for CoSEC day/night and ACDC validation samples in one pass."""

    def _split_names(self, img_path):
        path = Path(img_path)
        parts = path.parts
        text = str(path)
        if "/rgb_anon/" in text or "rgb_anon" in parts:
            if "night" in parts:
                return ("acdc", "acdc_night")
            return ("acdc",)
        for part in reversed(parts):
            if part.startswith("Day_"):
                return ("day",)
            if part.startswith("Night_"):
                return ("night",)
        return ("other",)

    def process(self, data_batch, data_samples):
        num_classes = len(self.dataset_meta["classes"])
        for data_sample in data_samples:
            pred_label = data_sample["pred_sem_seg"]["data"].squeeze()
            label = data_sample["gt_sem_seg"]["data"].squeeze().to(pred_label)
            result = self.intersect_and_union(pred_label, label, num_classes, self.ignore_index)
            self.results.append((self._split_names(data_sample["img_path"]), result))

    def _metrics_from_results(self, results, prefix=""):
        results = tuple(zip(*results))
        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            self.nan_to_num,
            self.beta,
        )
        summary = OrderedDict()
        for key, values in ret_metrics.items():
            metric_name = key if key == "aAcc" else "m" + key
            summary[f"{prefix}{metric_name}" if prefix else metric_name] = float(
                np.round(np.nanmean(values) * 100, 4)
            )
            if key == "IoU":
                for class_name, value in zip(self.dataset_meta["classes"], values):
                    metric_key = f"{prefix}IoU-{class_name}" if prefix else f"IoU-{class_name}"
                    summary[metric_key] = float(np.round(value * 100, 4))
        return summary

    def compute_metrics(self, results):
        grouped = defaultdict(list)
        all_results = []
        for splits, result in results:
            all_results.append(result)
            for split in splits:
                grouped[split].append(result)

        metrics = OrderedDict()
        if all_results:
            metrics.update(self._metrics_from_results(all_results))
        for split in ("day", "night", "acdc", "acdc_night"):
            if grouped[split]:
                metrics.update(self._metrics_from_results(grouped[split], prefix=f"{split}_"))
        return metrics
