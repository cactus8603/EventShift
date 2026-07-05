"""Evaluation helpers."""

from __future__ import annotations

import torch

from eventshift.utils.metrics import confusion_matrix, mean_iou


@torch.no_grad()
def evaluate_model(model, dataloader, num_classes: int = 19, device="cuda", ignore_index: int = 255) -> dict[str, float]:
    model.eval()
    hist = None
    for batch in dataloader:
        image = batch["image"].to(device)
        event = batch.get("event")
        event_stats = batch.get("event_stats")
        if event is not None:
            event = event.to(device)
        if event_stats is not None:
            event_stats = event_stats.to(device)
        pred = model(image, event=event, event_stats=event_stats).argmax(dim=1).cpu().numpy()
        target = batch["sem_seg"].cpu().numpy()
        for pred_i, target_i in zip(pred, target):
            cm = confusion_matrix(pred_i, target_i, num_classes=num_classes, ignore_index=ignore_index)
            hist = cm if hist is None else hist + cm
    return mean_iou(hist)

