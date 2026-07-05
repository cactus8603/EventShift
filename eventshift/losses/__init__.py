"""Loss functions."""

from .edge_loss import binary_edge_loss
from .segmentation_loss import segmentation_loss

__all__ = ["binary_edge_loss", "segmentation_loss"]

