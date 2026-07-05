# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

__all__ = ["MaskFormerSemanticDatasetMapper"]


class MaskFormerSemanticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        event_num_bins,
        event_support_percentile,
        event_temporal_threshold,
        event_support_dilation,
        event_dropout_prob,
        event_bin_dropout_prob,
        event_edge_dropout_prob,
        event_force_zero,
        event_edge_window_radii_ms,
        event_boundary_radii,
        event_class_boundary_radius,
        event_class_boundary_num_classes,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.event_num_bins = event_num_bins
        self.event_support_percentile = event_support_percentile
        self.event_temporal_threshold = event_temporal_threshold
        self.event_support_dilation = event_support_dilation
        self.event_dropout_prob = event_dropout_prob
        self.event_bin_dropout_prob = event_bin_dropout_prob
        self.event_edge_dropout_prob = event_edge_dropout_prob
        self.event_force_zero = event_force_zero
        self.event_edge_window_radii_ms = tuple(int(value) for value in event_edge_window_radii_ms)
        self.event_boundary_radii = tuple(int(value) for value in event_boundary_radii)
        self.event_class_boundary_radius = int(event_class_boundary_radius)
        self.event_class_boundary_num_classes = int(event_class_boundary_num_classes)

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        if is_train:
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                )
            ]
            if cfg.INPUT.CROP.ENABLED:
                augs.append(
                    T.RandomCrop_CategoryAreaConstraint(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                        cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                        cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    )
                )
            if cfg.INPUT.COLOR_AUG_SSD:
                augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            augs.append(T.RandomFlip())
        else:
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TEST,
                    cfg.INPUT.MAX_SIZE_TEST,
                    "choice",
                )
            ]

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "event_num_bins": cfg.INPUT.EVENT.NUM_BINS,
            "event_support_percentile": cfg.INPUT.EVENT.SUPPORT_PERCENTILE,
            "event_temporal_threshold": cfg.INPUT.EVENT.TEMPORAL_THRESHOLD,
            "event_support_dilation": cfg.INPUT.EVENT.SUPPORT_DILATION,
            "event_dropout_prob": cfg.INPUT.EVENT.DROPOUT_PROB,
            "event_bin_dropout_prob": cfg.INPUT.EVENT.BIN_DROPOUT_PROB,
            "event_edge_dropout_prob": cfg.INPUT.EVENT.EDGE_DROPOUT_PROB,
            "event_force_zero": cfg.INPUT.EVENT.ZERO_EVENT,
            "event_edge_window_radii_ms": cfg.INPUT.EVENT.EDGE_WINDOW_RADII_MS,
            "event_boundary_radii": cfg.INPUT.EVENT.BOUNDARY_RADII,
            "event_class_boundary_radius": (
                cfg.MODEL.EVENT_EDGE.CLASS_BOUNDARY_RADIUS if cfg.MODEL.EVENT_EDGE.CLASS_AWARE else 0
            ),
            "event_class_boundary_num_classes": cfg.MODEL.EVENT_EDGE.NUM_CLASSES,
        }
        return ret

    def _load_event(self, dataset_dict, image_shape):
        event_channels = self.event_num_bins * 2
        height, width = image_shape[:2]
        if "event_h5" not in dataset_dict:
            return (
                np.zeros((height, width, event_channels), dtype=np.float32),
                np.zeros((height, width, 4), dtype=np.float32),
            )

        from cosec_event_dataset import load_event_representation  # noqa: WPS433

        event, aux = load_event_representation(
            dataset_dict,
            image_shape,
            num_bins=self.event_num_bins,
        )
        return event.transpose(1, 2, 0), aux.transpose(1, 2, 0)

    def _load_event_edge(self, dataset_dict, image_shape):
        height, width = image_shape[:2]
        channels = len(self.event_edge_window_radii_ms) * 3
        if "event_h5" not in dataset_dict or channels == 0:
            return np.zeros((height, width, channels), dtype=np.float32)

        from cosec_event_dataset import load_event_edge_representation  # noqa: WPS433

        event_edge = load_event_edge_representation(
            dataset_dict,
            image_shape,
            self.event_edge_window_radii_ms,
        )
        return event_edge.transpose(1, 2, 0)

    def _event_stats_from_aux(self, aux):
        old_density = aux[:, :, 0]
        new_density = aux[:, :, 1]
        pos_count = aux[:, :, 2]
        neg_count = aux[:, :, 3]
        density_raw = old_density + new_density
        density_total = np.log1p(density_raw)
        temporal_balance = 1.0 - np.abs(old_density - new_density) / (density_raw + 1e-6)
        polarity_balance = 1.0 - np.abs(pos_count - neg_count) / (pos_count + neg_count + 1e-6)

        nonzero_density = density_total[density_total > 0]
        if nonzero_density.size > 0:
            threshold = np.percentile(nonzero_density, self.event_support_percentile)
            support = (density_total > threshold) & (temporal_balance > self.event_temporal_threshold)
        else:
            support = np.zeros_like(density_total, dtype=bool)
        if self.event_support_dilation > 0 and support.any():
            kernel_size = 2 * self.event_support_dilation + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            support = cv2.dilate(support.astype(np.uint8), kernel, iterations=1).astype(bool)

        return np.stack(
            [
                density_total,
                temporal_balance,
                polarity_balance,
                support.astype(np.float32),
            ],
            axis=2,
        ).astype(np.float32)

    def _augment_event(self, event, event_aux, is_train):
        if self.event_force_zero:
            return np.zeros_like(event), np.zeros_like(event_aux)
        if not is_train:
            return event, event_aux
        if self.event_dropout_prob > 0 and np.random.random() < self.event_dropout_prob:
            return np.zeros_like(event), np.zeros_like(event_aux)
        if self.event_bin_dropout_prob > 0:
            for channel_idx in range(event.shape[2]):
                if np.random.random() < self.event_bin_dropout_prob:
                    event[:, :, channel_idx] = 0
        return event, event_aux

    def _augment_event_edge(self, event_edge, is_train):
        if self.event_force_zero:
            return np.zeros_like(event_edge)
        if not is_train:
            return event_edge
        if self.event_edge_dropout_prob > 0 and np.random.random() < self.event_edge_dropout_prob:
            return np.zeros_like(event_edge)
        return event_edge

    def _apply_geometric_transforms(self, transforms, array):
        if array.ndim == 3 and array.shape[2] == 0:
            return array
        for transform in getattr(transforms, "transforms", [transforms]):
            if "Color" in transform.__class__.__name__:
                continue
            array = transform.apply_image(array)
        return array

    def _semantic_boundary(self, sem_seg_gt, radius):
        valid = sem_seg_gt != self.ignore_label
        if radius <= 0 or not np.any(valid):
            return np.zeros(sem_seg_gt.shape, dtype=np.float32)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * int(radius) + 1, 2 * int(radius) + 1),
        )
        low = sem_seg_gt.astype(np.float32, copy=True)
        high = sem_seg_gt.astype(np.float32, copy=True)
        low[~valid] = -1000.0
        high[~valid] = 1000.0
        local_max = cv2.dilate(low, kernel)
        local_min = cv2.erode(high, kernel)
        return (valid & (local_max != local_min)).astype(np.float32)

    def _class_semantic_boundary(self, sem_seg_gt, radius, num_classes):
        if radius <= 0 or num_classes <= 0:
            return None
        valid = sem_seg_gt != self.ignore_label
        if not np.any(valid):
            return np.zeros((num_classes, sem_seg_gt.shape[0], sem_seg_gt.shape[1]), dtype=np.float32)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * int(radius) + 1, 2 * int(radius) + 1),
        )
        targets = np.zeros((num_classes, sem_seg_gt.shape[0], sem_seg_gt.shape[1]), dtype=np.float32)
        for class_id in range(num_classes):
            mask = ((sem_seg_gt == class_id) & valid).astype(np.uint8)
            if not np.any(mask):
                continue
            dilated = cv2.dilate(mask, kernel, iterations=1)
            eroded = cv2.erode(mask, kernel, iterations=1)
            targets[class_id] = ((dilated != eroded) & valid).astype(np.float32)
        return targets

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        if image.shape[:2] != sem_seg_gt.shape[:2]:
            image = cv2.resize(
                image,
                (sem_seg_gt.shape[1], sem_seg_gt.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            dataset_dict["height"], dataset_dict["width"] = sem_seg_gt.shape[:2]

        event, event_aux = self._load_event(dataset_dict, image.shape)
        event_edge = self._load_event_edge(dataset_dict, image.shape)
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg
        event = self._apply_geometric_transforms(transforms, event)
        event_aux = self._apply_geometric_transforms(transforms, event_aux)
        event_edge = self._apply_geometric_transforms(transforms, event_edge)
        event, event_aux = self._augment_event(event, event_aux, self.is_train)
        event_edge = self._augment_event_edge(event_edge, self.is_train)
        event_stats = self._event_stats_from_aux(event_aux)
        boundary_targets = {
            int(radius): self._semantic_boundary(sem_seg_gt, int(radius))
            for radius in self.event_boundary_radii
        }
        class_boundary_target = self._class_semantic_boundary(
            sem_seg_gt,
            self.event_class_boundary_radius,
            self.event_class_boundary_num_classes,
        )

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        event = torch.as_tensor(np.ascontiguousarray(event.transpose(2, 0, 1))).float()
        event_edge = torch.as_tensor(np.ascontiguousarray(event_edge.transpose(2, 0, 1))).float()
        event_stats = torch.as_tensor(np.ascontiguousarray(event_stats.transpose(2, 0, 1))).float()
        boundary_targets = {
            radius: torch.as_tensor(np.ascontiguousarray(target)).float()
            for radius, target in boundary_targets.items()
        }
        if class_boundary_target is not None:
            class_boundary_target = torch.as_tensor(np.ascontiguousarray(class_boundary_target)).float()
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            event = F.pad(event, padding_size, value=0).contiguous()
            event_edge = F.pad(event_edge, padding_size, value=0).contiguous()
            event_stats = F.pad(event_stats, padding_size, value=0).contiguous()
            boundary_targets = {
                radius: F.pad(target, padding_size, value=0).contiguous()
                for radius, target in boundary_targets.items()
            }
            if class_boundary_target is not None:
                class_boundary_target = F.pad(class_boundary_target, padding_size, value=0).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image
        dataset_dict["event"] = event
        dataset_dict["event_edge"] = event_edge
        dataset_dict["event_stats"] = event_stats
        for radius, target in boundary_targets.items():
            dataset_dict[f"boundary_r{radius}"] = target
        if class_boundary_target is not None:
            dataset_dict[f"class_boundary_r{self.event_class_boundary_radius}"] = class_boundary_target

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict
