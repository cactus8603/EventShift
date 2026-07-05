"""Pseudo-label dataset helpers for CoSEC test / REAL self-training.

The labels created here are intentionally conservative: low-confidence or
teacher-disagreement pixels are set to 255 so the standard semantic loss ignores
them. This lets us try FixMatch/ST++-style pseudo supervision without changing
the Mask2Former loss stack.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError

from cosec_finetune_splits import CLASSES


def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()
IGNORE_LABEL = 255
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}


BALANCED_CLASS_CAPS = {
    "road": 0.36,
    "vegetation": 0.36,
    "sidewalk": 0.09,
    "building": 0.09,
    "sky": 0.05,
    "wall": 0.04,
    "fence": 0.04,
}

BALANCED_CLASS_THRESHOLDS = {
    "road": 224,
    "vegetation": 224,
    "sidewalk": 208,
    "building": 208,
    "sky": 208,
}

RARE_BOUNDARY_FULL_CLASSES = {
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
}

RARE_BOUNDARY_ONLY_CLASSES = {
    "road",
    "sidewalk",
    "vegetation",
    "terrain",
    "sky",
}

GAP_FOCUS_FULL_CLASSES = {
    "building",
    "wall",
    "fence",
    "pole",
    "traffic sign",
    "person",
    "rider",
    "motorcycle",
    "bicycle",
}

GAP_FOCUS_BOUNDARY_CLASSES = {
    "road",
    "sidewalk",
    "vegetation",
    "sky",
}


REAL_EVENT_VARIANT_OPTIONS = {
    # Strict version: only the strongest event activity is trusted. It keeps
    # high-confidence teacher pixels in event-supported regions and adds
    # medium-confidence semantic-boundary pixels when events support them.
    "swinl_eventedge": {
        "event_percentile": 80,
        "event_delta_ms": 50,
        "event_dilate_radius": 4,
        "semantic_boundary_radius": 3,
        "boundary_conf_threshold": 192,
        "base_requires_event": True,
        "boundary_requires_event": True,
        "boundary_only": False,
    },
    # Looser version: useful when the strict edge filter makes the pseudo-label
    # support too small. Still ignores pixels with no event activity.
    "swinl_eventactive": {
        "event_percentile": 0,
        "event_delta_ms": 50,
        "event_dilate_radius": 2,
        "semantic_boundary_radius": 0,
        "boundary_conf_threshold": None,
        "base_requires_event": True,
        "boundary_requires_event": False,
        "boundary_only": False,
    },
    # A longer-window edge variant for REAL, where motion is often slow and
    # short windows can miss usable contours.
    "swinl_eventedge100": {
        "event_percentile": 80,
        "event_delta_ms": 100,
        "event_dilate_radius": 4,
        "semantic_boundary_radius": 3,
        "boundary_conf_threshold": 192,
        "base_requires_event": True,
        "boundary_requires_event": True,
        "boundary_only": False,
    },
    # REAL "segment_co" is a SegFormer-style pseudo label, not human GT. Use it
    # as a second teacher: keep only Swin-L pseudo pixels that agree with
    # segment_co and have event support.
    "swinl_segmentco": {
        "event_percentile": None,
        "event_delta_ms": None,
        "event_dilate_radius": 0,
        "semantic_boundary_radius": 0,
        "boundary_conf_threshold": None,
        "base_requires_event": False,
        "boundary_requires_event": False,
        "boundary_only": False,
        "agree_dir": "segment_co",
    },
    "swinl_segmentco_eventactive": {
        "event_percentile": 0,
        "event_delta_ms": 50,
        "event_dilate_radius": 2,
        "semantic_boundary_radius": 0,
        "boundary_conf_threshold": None,
        "base_requires_event": True,
        "boundary_requires_event": False,
        "boundary_only": False,
        "agree_dir": "segment_co",
    },
    "swinl_segmentco_eventedge": {
        "event_percentile": 80,
        "event_delta_ms": 50,
        "event_dilate_radius": 4,
        "semantic_boundary_radius": 3,
        "boundary_conf_threshold": 192,
        "base_requires_event": True,
        "boundary_requires_event": True,
        "boundary_only": False,
        "agree_dir": "segment_co",
    },
}


def _read_gray(path):
    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = array[:, :, 0]
    return array


def _masked_label_path(seq_name, image_stem, variant, threshold, source):
    return (
        ROOT
        / "work_dirs"
        / "cache"
        / "pseudo_labels"
        / f"{source}_{variant}_conf{threshold}"
        / seq_name
        / f"{image_stem}.png"
    )


def _safe_cache_name(text):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))


def _masked_prediction_label_path(seq_name, image_stem, prediction_name, variant):
    return (
        ROOT
        / "work_dirs"
        / "cache"
        / "pseudo_labels"
        / f"test_prediction_{_safe_cache_name(prediction_name)}_{_safe_cache_name(variant)}"
        / seq_name
        / f"{image_stem}.png"
    )


def _class_option_indices(options):
    return {
        CLASS_TO_INDEX[name]: value
        for name, value in options.items()
        if name in CLASS_TO_INDEX
    }


def _apply_class_thresholds(label, conf, keep, class_thresholds):
    if not class_thresholds:
        return keep
    adjusted = keep.copy()
    for class_index, class_threshold in _class_option_indices(class_thresholds).items():
        adjusted[label == class_index] &= conf[label == class_index] >= int(class_threshold)
    return adjusted


def _apply_class_caps(label, conf, keep, class_caps):
    if not class_caps:
        return keep
    adjusted = keep.copy()
    total_pixels = int(label.size)
    for class_index, max_fraction in _class_option_indices(class_caps).items():
        class_keep = np.flatnonzero(adjusted.reshape(-1) & (label.reshape(-1) == class_index))
        max_keep = int(round(float(max_fraction) * total_pixels))
        if class_keep.size <= max_keep:
            continue
        if max_keep <= 0:
            adjusted.reshape(-1)[class_keep] = False
            continue
        class_conf = conf.reshape(-1)[class_keep]
        selected_local = np.argpartition(class_conf, -max_keep)[-max_keep:]
        selected = class_keep[selected_local]
        adjusted_flat = adjusted.reshape(-1)
        adjusted_flat[class_keep] = False
        adjusted_flat[selected] = True
    return adjusted


def _semantic_boundary_mask(label, radius=3):
    if int(radius) <= 0:
        return np.zeros(label.shape, dtype=bool)
    valid = label != IGNORE_LABEL
    boundary = np.zeros(label.shape, dtype=bool)
    padded_label = np.pad(label, int(radius), mode="edge")
    padded_valid = np.pad(valid, int(radius), mode="edge")
    center = padded_label[radius : radius + label.shape[0], radius : radius + label.shape[1]]
    center_valid = padded_valid[radius : radius + label.shape[0], radius : radius + label.shape[1]]
    for dy in range(-int(radius), int(radius) + 1):
        for dx in range(-int(radius), int(radius) + 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = padded_label[
                radius + dy : radius + dy + label.shape[0],
                radius + dx : radius + dx + label.shape[1],
            ]
            neighbor_valid = padded_valid[
                radius + dy : radius + dy + label.shape[0],
                radius + dx : radius + dx + label.shape[1],
            ]
            boundary |= center_valid & neighbor_valid & (center != neighbor)
    return boundary


def _apply_rare_boundary_filter(label, keep, radius=3):
    full_indices = set(_class_option_indices({name: 1 for name in RARE_BOUNDARY_FULL_CLASSES}).keys())
    boundary_indices = set(_class_option_indices({name: 1 for name in RARE_BOUNDARY_ONLY_CLASSES}).keys())
    full_mask = np.isin(label, list(full_indices))
    boundary_mask = _semantic_boundary_mask(label, radius=radius) & np.isin(label, list(boundary_indices))
    return keep & (full_mask | boundary_mask)


def _apply_focus_class_filter(label, keep, full_classes=None, boundary_classes=None, radius=3):
    full_classes = full_classes or set()
    boundary_classes = boundary_classes or set()
    full_indices = set(_class_option_indices({name: 1 for name in full_classes}).keys())
    boundary_indices = set(_class_option_indices({name: 1 for name in boundary_classes}).keys())
    focus_mask = np.zeros(label.shape, dtype=bool)
    if full_indices:
        focus_mask |= np.isin(label, list(full_indices))
    if boundary_indices:
        boundary_mask = _semantic_boundary_mask(label, radius=radius)
        focus_mask |= boundary_mask & np.isin(label, list(boundary_indices))
    return keep & focus_mask


def _binary_dilate(mask, radius=1):
    radius = int(radius)
    if radius <= 0:
        return mask.astype(bool, copy=False)
    padded = np.pad(mask.astype(bool, copy=False), radius, mode="constant", constant_values=False)
    dilated = np.zeros(mask.shape, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            dilated |= padded[
                radius + dy : radius + dy + mask.shape[0],
                radius + dx : radius + dx + mask.shape[1],
            ]
    return dilated


def _real_event_path(seq_dir):
    for name in ("events.h5", "events_co_left.h5"):
        path = seq_dir / name
        if path.exists():
            return path
    return None


@lru_cache(maxsize=128)
def _real_timestamps(seq_dir_str):
    return np.loadtxt(Path(seq_dir_str) / "timestamps.txt", dtype=np.int64, encoding="utf-8-sig")


def _real_event_activity_mask(seq_dir, frame_id, shape, delta_ms=50, percentile=80, dilate_radius=3):
    """Return a binary event activity map aligned to REAL RGB/pseudo masks."""
    event_path = _real_event_path(seq_dir)
    timestamps_path = seq_dir / "timestamps.txt"
    if event_path is None or not timestamps_path.exists():
        return np.zeros(shape, dtype=bool)

    import h5py

    timestamps = _real_timestamps(str(seq_dir))
    frame_id = int(frame_id)
    if frame_id < 0 or frame_id >= len(timestamps):
        return np.zeros(shape, dtype=bool)

    ts_end = int(timestamps[frame_id])
    ts_start = max(0, ts_end - int(delta_ms) * 1000)
    counts = np.zeros(shape, dtype=np.uint16)

    with h5py.File(event_path, "r") as h5f:
        ms_to_idx = h5f["ms_to_idx"]
        start_ms = max(0, int(np.floor(ts_start / 1000.0)))
        end_ms = min(len(ms_to_idx) - 1, int(np.ceil(ts_end / 1000.0)))
        start_idx = int(ms_to_idx[start_ms])
        end_idx = int(ms_to_idx[end_ms])
        if end_idx <= start_idx:
            return np.zeros(shape, dtype=bool)
        t = np.asarray(h5f["t"][start_idx:end_idx])
        keep = (t >= ts_start) & (t <= ts_end)
        if not np.any(keep):
            return np.zeros(shape, dtype=bool)
        x = np.asarray(h5f["x"][start_idx:end_idx])[keep].astype(np.int64, copy=False)
        y = np.asarray(h5f["y"][start_idx:end_idx])[keep].astype(np.int64, copy=False)

    valid = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
    if not np.any(valid):
        return np.zeros(shape, dtype=bool)
    np.add.at(counts, (y[valid], x[valid]), 1)
    if float(percentile) <= 0:
        active = counts > 0
    else:
        nonzero = counts[counts > 0]
        if nonzero.size == 0:
            active = counts > 0
        else:
            threshold = max(1.0, float(np.percentile(nonzero, float(percentile))))
            active = counts >= threshold
    return _binary_dilate(active, radius=dilate_radius)


def _real_event_variant_options(variant):
    return REAL_EVENT_VARIANT_OPTIONS.get(variant)


def _write_real_event_pseudo_label(label_path, conf_path, output_path, threshold, seq_dir, image_stem, variant):
    label = _read_gray(label_path).astype(np.uint8, copy=False)
    conf = _read_gray(conf_path)
    options = _real_event_variant_options(variant)
    if options is None:
        valid_fraction = _write_masked_label(label_path, conf_path, output_path, threshold)
        return valid_fraction

    agree_dir = options.get("agree_dir")
    agree_label = None
    if agree_dir:
        agree_path = seq_dir / str(agree_dir) / f"{image_stem}.png"
        if not agree_path.exists():
            return 0.0
        agree_label = _read_gray(agree_path).astype(np.uint8, copy=False)
        if agree_label.shape != label.shape:
            agree_label = np.asarray(
                Image.fromarray(agree_label, mode="L").resize(
                    (label.shape[1], label.shape[0]), resample=Image.Resampling.NEAREST
                )
            )

    needs_event = bool(options["base_requires_event"]) or (
        options.get("boundary_conf_threshold") is not None and bool(options["boundary_requires_event"])
    )
    if needs_event:
        event_mask = _real_event_activity_mask(
            seq_dir,
            int(image_stem),
            label.shape,
            delta_ms=options["event_delta_ms"],
            percentile=options["event_percentile"],
            dilate_radius=options["event_dilate_radius"],
        )
    else:
        event_mask = np.ones(label.shape, dtype=bool)
    keep = conf >= int(threshold)
    if agree_label is not None:
        keep &= label == agree_label
    if options["base_requires_event"]:
        keep &= event_mask

    boundary_threshold = options.get("boundary_conf_threshold")
    if boundary_threshold is not None:
        boundary = _semantic_boundary_mask(label, radius=options["semantic_boundary_radius"])
        boundary_keep = (conf >= int(boundary_threshold)) & boundary
        if agree_label is not None:
            boundary_keep &= label == agree_label
        if options["boundary_requires_event"]:
            boundary_keep &= event_mask
        keep |= boundary_keep

    if options.get("boundary_only"):
        keep &= _semantic_boundary_mask(label, radius=options["semantic_boundary_radius"])

    masked = np.full(label.shape, IGNORE_LABEL, dtype=np.uint8)
    masked[keep] = label[keep]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(masked, mode="L").save(output_path)
    return float(keep.mean())


def _write_masked_label(
    label_path,
    conf_path,
    output_path,
    threshold,
    agree_path=None,
    agree_conf_path=None,
    class_caps=None,
    class_thresholds=None,
    rare_boundary=False,
    rare_boundary_radius=3,
    focus_full_classes=None,
    focus_boundary_classes=None,
    focus_boundary_radius=3,
):
    label = _read_gray(label_path).astype(np.uint8, copy=False)
    conf = _read_gray(conf_path)
    keep = conf >= int(threshold)

    if agree_path is not None:
        agree_label = _read_gray(agree_path).astype(np.uint8, copy=False)
        keep &= label == agree_label
        if agree_conf_path is not None and agree_conf_path.exists():
            keep &= _read_gray(agree_conf_path) >= int(threshold)

    keep = _apply_class_thresholds(label, conf, keep, class_thresholds)
    keep = _apply_class_caps(label, conf, keep, class_caps)
    if rare_boundary:
        keep = _apply_rare_boundary_filter(label, keep, radius=rare_boundary_radius)
    if focus_full_classes or focus_boundary_classes:
        keep = _apply_focus_class_filter(
            label,
            keep,
            full_classes=focus_full_classes,
            boundary_classes=focus_boundary_classes,
            radius=focus_boundary_radius,
        )

    masked = np.full(label.shape, IGNORE_LABEL, dtype=np.uint8)
    masked[keep] = label[keep]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(masked, mode="L").save(output_path)
    return float(keep.mean())


def _test_agreement_paths(seq_dir, image_stem, variant):
    if variant == "consensus":
        # Historical name kept for compatibility. This pair later turned out to
        # be effectively identical, so prefer segformer_consensus for new runs.
        return (
            seq_dir / "prior_mask2former_large_ft_cc_submission" / f"{image_stem}.png",
            seq_dir / "prior_mask2former_large_ft_cc_submission_conf" / f"{image_stem}.png",
        )
    if variant in {"segformer_consensus", "segformer_balcap", "segformer_rare_boundary"}:
        return (
            seq_dir / "prior_segformer_ft_cc_submission" / f"{image_stem}.png",
            seq_dir / "prior_segformer_ft_cc_submission_conf" / f"{image_stem}.png",
        )
    return None, None


def _test_variant_options(variant):
    if variant == "segformer_balcap":
        return {
            "class_caps": BALANCED_CLASS_CAPS,
            "class_thresholds": BALANCED_CLASS_THRESHOLDS,
            "rare_boundary": False,
            "rare_boundary_radius": 3,
        }
    if variant == "segformer_rare_boundary":
        return {
            "class_caps": None,
            "class_thresholds": {
                "road": 224,
                "vegetation": 224,
                "sky": 208,
                "sidewalk": 208,
            },
            "rare_boundary": True,
            "rare_boundary_radius": 3,
        }
    return {
        "class_caps": None,
        "class_thresholds": None,
        "rare_boundary": False,
        "rare_boundary_radius": 3,
    }


def _prediction_variant_options(variant):
    if variant == "all":
        return {
            "agree_dir": None,
            "agree_conf_dir": None,
            "confidence_threshold": None,
            "class_caps": None,
            "class_thresholds": None,
            "rare_boundary": False,
            "rare_boundary_radius": 3,
            "focus_full_classes": None,
            "focus_boundary_classes": None,
            "focus_boundary_radius": 3,
        }
    if variant == "segformer_agree_conf192":
        return {
            "agree_dir": "prior_segformer_ft_cc_submission",
            "agree_conf_dir": "prior_segformer_ft_cc_submission_conf",
            "confidence_threshold": 192,
            "class_caps": None,
            "class_thresholds": None,
            "rare_boundary": False,
            "rare_boundary_radius": 3,
            "focus_full_classes": None,
            "focus_boundary_classes": None,
            "focus_boundary_radius": 3,
        }
    if variant == "segformer_agree_rare_boundary_conf192":
        return {
            "agree_dir": "prior_segformer_ft_cc_submission",
            "agree_conf_dir": "prior_segformer_ft_cc_submission_conf",
            "confidence_threshold": 192,
            "class_caps": None,
            "class_thresholds": {
                "road": 224,
                "vegetation": 224,
                "sky": 208,
                "sidewalk": 208,
            },
            "rare_boundary": True,
            "rare_boundary_radius": 3,
            "focus_full_classes": None,
            "focus_boundary_classes": None,
            "focus_boundary_radius": 3,
        }
    if variant == "segformer_agree_gap_focus_conf192":
        return {
            "agree_dir": "prior_segformer_ft_cc_submission",
            "agree_conf_dir": "prior_segformer_ft_cc_submission_conf",
            "confidence_threshold": 192,
            "class_caps": {
                "building": 0.30,
                "wall": 0.12,
                "fence": 0.18,
                "road": 0.04,
                "sidewalk": 0.04,
                "vegetation": 0.04,
                "sky": 0.04,
            },
            "class_thresholds": {
                "building": 208,
                "wall": 208,
                "fence": 208,
                "road": 224,
                "sidewalk": 224,
                "vegetation": 224,
                "sky": 224,
            },
            "rare_boundary": False,
            "rare_boundary_radius": 3,
            "focus_full_classes": GAP_FOCUS_FULL_CLASSES,
            "focus_boundary_classes": GAP_FOCUS_BOUNDARY_CLASSES,
            "focus_boundary_radius": 3,
        }
    raise ValueError(f"Unknown prediction pseudo variant: {variant}")


def _ensure_prediction_pseudo_label(seq_dir, image_stem, prediction_name, variant):
    prediction_root = ROOT / "work_dirs" / "submissions" / "prediction_dirs" / prediction_name
    label_path = prediction_root / seq_dir.name / "segment_co" / f"{image_stem}.png"
    if not label_path.exists():
        return None, 0.0

    options = _prediction_variant_options(variant)
    if variant == "all":
        label = _read_gray(label_path)
        return label_path, float((label != IGNORE_LABEL).mean())

    output_path = _masked_prediction_label_path(seq_dir.name, image_stem, prediction_name, variant)
    if output_path.exists():
        try:
            cached = _read_gray(output_path)
            return output_path, float((cached != IGNORE_LABEL).mean())
        except (OSError, UnidentifiedImageError):
            output_path.unlink(missing_ok=True)

    agree_path = None
    agree_conf_path = None
    if options["agree_dir"]:
        agree_path = seq_dir / options["agree_dir"] / f"{image_stem}.png"
        if not agree_path.exists():
            return None, 0.0
    if options["agree_conf_dir"]:
        agree_conf_path = seq_dir / options["agree_conf_dir"] / f"{image_stem}.png"
        if not agree_conf_path.exists():
            return None, 0.0

    # The prediction-dir teacher has no stored per-pixel confidence. For
    # high-confidence variants, use agreement and the secondary teacher's
    # confidence as a conservative mask.
    if options["confidence_threshold"] is None:
        conf = np.full(_read_gray(label_path).shape, 255, dtype=np.uint8)
        conf_path = output_path.with_name(f"{output_path.stem}_conf255.png")
        conf_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(conf, mode="L").save(conf_path)
        threshold = 0
    else:
        conf_path = agree_conf_path
        threshold = int(options["confidence_threshold"])

    valid_fraction = _write_masked_label(
        label_path,
        conf_path,
        output_path,
        threshold,
        agree_path=agree_path,
        agree_conf_path=agree_conf_path,
        class_caps=options["class_caps"],
        class_thresholds=options["class_thresholds"],
        rare_boundary=options["rare_boundary"],
        rare_boundary_radius=options["rare_boundary_radius"],
        focus_full_classes=options.get("focus_full_classes"),
        focus_boundary_classes=options.get("focus_boundary_classes"),
        focus_boundary_radius=options.get("focus_boundary_radius", 3),
    )
    return output_path, valid_fraction


def _ensure_test_pseudo_label(seq_dir, image_stem, variant, threshold):
    label_path = seq_dir / "prior_swinL_ft" / f"{image_stem}.png"
    conf_path = seq_dir / "prior_swinL_ft_conf" / f"{image_stem}.png"
    if not label_path.exists() or not conf_path.exists():
        return None, 0.0

    output_path = _masked_label_path(seq_dir.name, image_stem, variant, threshold, "test")
    if output_path.exists():
        try:
            cached = _read_gray(output_path)
            return output_path, float((cached != IGNORE_LABEL).mean())
        except (OSError, UnidentifiedImageError):
            output_path.unlink(missing_ok=True)

    agree_path, agree_conf_path = _test_agreement_paths(seq_dir, image_stem, variant)
    if agree_path is not None:
        if not agree_path.exists():
            return None, 0.0
    options = _test_variant_options(variant)

    valid_fraction = _write_masked_label(
        label_path,
        conf_path,
        output_path,
        threshold,
        agree_path=agree_path,
        agree_conf_path=agree_conf_path,
        class_caps=options["class_caps"],
        class_thresholds=options["class_thresholds"],
        rare_boundary=options["rare_boundary"],
        rare_boundary_radius=options["rare_boundary_radius"],
    )
    return output_path, valid_fraction


def _ensure_real_pool_pseudo_label(seq_dir, image_stem, variant, threshold):
    label_path = seq_dir / "prior_swinL_ft" / f"{image_stem}.png"
    conf_path = seq_dir / "prior_swinL_ft_conf" / f"{image_stem}.png"
    if not label_path.exists() or not conf_path.exists():
        return None, 0.0

    output_path = _masked_label_path(seq_dir.name, image_stem, variant, threshold, "realpool")
    if output_path.exists():
        try:
            cached = _read_gray(output_path)
            return output_path, float((cached != IGNORE_LABEL).mean())
        except (OSError, UnidentifiedImageError):
            output_path.unlink(missing_ok=True)

    valid_fraction = _write_real_event_pseudo_label(
        label_path,
        conf_path,
        output_path,
        threshold,
        seq_dir,
        image_stem,
        variant,
    )
    return output_path, valid_fraction


def _test_sequence_prefixes(domain):
    if domain == "day":
        return ("Day_",)
    if domain == "night":
        return ("Night_",)
    if domain == "real":
        return ("REAL_",)
    if domain == "daynight":
        return ("Day_", "Night_")
    if domain == "all":
        return ("Day_", "Night_", "REAL_")
    raise ValueError(f"Unknown test pseudo domain: {domain}")


def _evenly_spaced_subset(records, limit):
    if limit is None or int(limit) <= 0 or int(limit) >= len(records):
        return list(records)
    selected = []
    keep_count = int(limit)
    for rank in range(keep_count):
        index = int((rank + 0.5) * len(records) / keep_count)
        selected.append(records[min(index, len(records) - 1)])
    return selected


@lru_cache(maxsize=None)
def load_cosec_test_pseudo_dicts(
    domain,
    variant="consensus",
    threshold=192,
    repeat=1,
    min_valid_fraction=0.05,
    limit=None,
):
    records = []
    test_root = ROOT / "data" / "test"
    prefixes = _test_sequence_prefixes(domain)
    for seq_dir in sorted(path for path in test_root.iterdir() if path.is_dir() and path.name.startswith(prefixes)):
        image_dir = seq_dir / "img_co_left"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.glob("*.png")):
            pseudo_path, valid_fraction = _ensure_test_pseudo_label(seq_dir, image_path.stem, variant, threshold)
            if pseudo_path is None or valid_fraction < float(min_valid_fraction):
                continue
            records.append(
                {
                    "file_name": str(image_path),
                    "sem_seg_file_name": str(pseudo_path),
                    "image_id": f"pseudo_{domain}_{seq_dir.name}_{image_path.stem}",
                    "source": f"test_{domain}_{variant}_conf{threshold}",
                    "pseudo_valid_fraction": valid_fraction,
                }
            )
    records = _evenly_spaced_subset(records, limit)
    return tuple(records) * int(repeat)


@lru_cache(maxsize=None)
def load_cosec_test_prediction_pseudo_dicts(
    domain,
    prediction_name,
    variant="all",
    repeat=1,
    min_valid_fraction=0.05,
    limit=None,
):
    records = []
    test_root = ROOT / "data" / "test"
    prefixes = _test_sequence_prefixes(domain)
    for seq_dir in sorted(path for path in test_root.iterdir() if path.is_dir() and path.name.startswith(prefixes)):
        image_dir = seq_dir / "img_co_left"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.glob("*.png")):
            pseudo_path, valid_fraction = _ensure_prediction_pseudo_label(
                seq_dir,
                image_path.stem,
                prediction_name,
                variant,
            )
            if pseudo_path is None or valid_fraction < float(min_valid_fraction):
                continue
            records.append(
                {
                    "file_name": str(image_path),
                    "sem_seg_file_name": str(pseudo_path),
                    "image_id": f"pseudo_prediction_{domain}_{variant}_{seq_dir.name}_{image_path.stem}",
                    "source": f"test_prediction_{prediction_name}_{domain}_{variant}",
                    "pseudo_valid_fraction": valid_fraction,
                }
            )
    records = _evenly_spaced_subset(records, limit)
    return tuple(records) * int(repeat)


@lru_cache(maxsize=None)
def load_real_pool_pseudo_dicts(variant="swinl", threshold=224, repeat=1, limit=600, min_valid_fraction=0.05):
    records = []
    real_root = ROOT / "data" / "REAL_dataset"
    candidates = []
    for seq_dir in sorted(path for path in real_root.iterdir() if path.is_dir()):
        image_dir = seq_dir / "gt"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.glob("*.png")):
            candidates.append((seq_dir, image_path))

    candidates = _evenly_spaced_subset(candidates, limit)
    for seq_dir, image_path in candidates:
        pseudo_path, valid_fraction = _ensure_real_pool_pseudo_label(seq_dir, image_path.stem, variant, threshold)
        if pseudo_path is None or valid_fraction < float(min_valid_fraction):
            continue
        records.append(
            {
                "file_name": str(image_path),
                "sem_seg_file_name": str(pseudo_path),
                "image_id": f"pseudo_realpool_{seq_dir.name}_{image_path.stem}",
                "source": f"realpool_{variant}_conf{threshold}",
                "real_sequence": seq_dir.name,
                "real_frame": image_path.stem,
                "pseudo_variant": variant,
                "pseudo_valid_fraction": valid_fraction,
            }
        )
    return tuple(records) * int(repeat)
