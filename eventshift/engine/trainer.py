"""Small native trainer for lightweight EventShift experiments."""

from __future__ import annotations

import torch

from eventshift.losses import segmentation_loss


def train_one_epoch(model, dataloader, optimizer, device="cuda", ignore_index: int = 255) -> dict[str, float]:
    model.train()
    total = 0.0
    steps = 0
    for batch in dataloader:
        image = batch["image"].to(device)
        target = batch["sem_seg"].to(device)
        event = batch.get("event")
        event_stats = batch.get("event_stats")
        if event is not None:
            event = event.to(device)
        if event_stats is not None:
            event_stats = event_stats.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image, event=event, event_stats=event_stats)
        loss = segmentation_loss(logits, target, ignore_index=ignore_index)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
        steps += 1
    return {"loss": total / max(steps, 1)}

