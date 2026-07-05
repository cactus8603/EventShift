from functools import lru_cache
from pathlib import Path


SWIN_L_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = WORKSPACE_ROOT / "BRENet" / "projects" / "brenet_cosec" / "splits"
TRAIN_SPLIT_FILE = SPLIT_DIR / "train_seq_holdout_305.txt"
VAL_SPLIT_FILE = SPLIT_DIR / "val_seq_holdout_305.txt"
DEFAULT_COSEC_TRAIN_ROOT = SWIN_L_ROOT / "data" / "train"
DEFAULT_KFOLD_COUNT = 3


CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

PALETTE = (
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
)

def mmseg_split_name(seq_name, frame_id):
    return f"{seq_name}/{frame_id:06d}"


def cosec_domain(seq_name):
    if seq_name.startswith("Day_"):
        return "day"
    if seq_name.startswith("Night_"):
        return "night"
    raise ValueError(f"Unknown CoSEC sequence domain: {seq_name}")


def _read_split_file(path):
    if not path.exists():
        raise FileNotFoundError(f"CoSEC split file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return frozenset(line.strip() for line in f if line.strip() and not line.lstrip().startswith("#"))


def _file_split_spec(split):
    suffix_specs = (
        ("day_train", "train", "Day_"),
        ("night_train", "train", "Night_"),
        ("day_val", "val", "Day_"),
        ("night_val", "val", "Night_"),
        ("train", "train", None),
        ("val", "val", None),
    )
    for suffix, subset, seq_prefix in suffix_specs:
        token = f"_{suffix}"
        if not split.endswith(token):
            continue
        prefix = split[: -len(token)]
        if not prefix:
            continue
        path = SPLIT_DIR / f"{subset}_{prefix}.txt"
        if path.exists():
            return path, seq_prefix
    return None


@lru_cache(maxsize=None)
def _file_split_ids(split):
    spec = _file_split_spec(split)
    if spec is None:
        return None
    path, seq_prefix = spec
    sample_ids = _read_split_file(path)
    if seq_prefix is not None:
        sample_ids = frozenset(sample_id for sample_id in sample_ids if sample_id.startswith(seq_prefix))
    return sample_ids


@lru_cache(maxsize=None)
def _train_ids():
    return _read_split_file(TRAIN_SPLIT_FILE)


@lru_cache(maxsize=None)
def _val_ids():
    return _read_split_file(VAL_SPLIT_FILE)


def holdout_sample_group(seq_name, frame_id):
    sample_name = mmseg_split_name(seq_name, frame_id)
    try:
        domain = cosec_domain(seq_name)
    except ValueError:
        return None
    if sample_name in _val_ids():
        return f"{domain}_val"
    if sample_name in _train_ids():
        return f"{domain}_train"
    return None


def sample_split(seq_name, frame_id):
    sample_group = holdout_sample_group(seq_name, frame_id)
    if sample_group in {"day_train", "night_train"}:
        return "train"
    return sample_group


def parse_kfold_split(split):
    parts = split.split("_")
    if not parts or not parts[0].startswith("kfold"):
        return None
    if len(parts) < 3 or not parts[1].startswith("fold"):
        raise ValueError(f"Malformed CoSEC k-fold split: {split}")
    try:
        folds = int(parts[0][len("kfold") :])
        fold_index = int(parts[1][len("fold") :])
    except ValueError as error:
        raise ValueError(f"Malformed CoSEC k-fold split: {split}") from error
    if folds < 2:
        raise ValueError(f"CoSEC k-fold split requires at least 2 folds: {split}")
    if fold_index < 0 or fold_index >= folds:
        raise ValueError(f"CoSEC k-fold split fold index out of range: {split}")

    tail = parts[2:]
    if len(tail) == 1 and tail[0] in {"train", "val"}:
        domain = None
        subset = tail[0]
    elif len(tail) == 2 and tail[0] in {"day", "night"} and tail[1] in {"train", "val"}:
        domain = tail[0]
        subset = tail[1]
    else:
        raise ValueError(f"Malformed CoSEC k-fold split: {split}")
    return {
        "folds": folds,
        "fold_index": fold_index,
        "domain": domain,
        "subset": subset,
        "requested_split": f"{domain + '_' if domain else ''}{subset}",
    }


def _group_matches_requested_split(sample_group, requested_split):
    if sample_group is None:
        return False
    if requested_split == "train":
        return sample_group in {"day_train", "night_train"}
    if requested_split == "val":
        return sample_group in {"day_val", "night_val"}
    if requested_split in {"day_train", "night_train", "day_val", "night_val"}:
        return sample_group == requested_split
    raise ValueError(f"Unknown split: {requested_split}")


def iter_cosec_sequence_infos(root=DEFAULT_COSEC_TRAIN_ROOT):
    root = Path(root)
    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not seq_dir.name.startswith(("Day_", "Night_")):
            continue
        img_dir = seq_dir / "img_co_left"
        label_dir = seq_dir / "segment_co"
        if not img_dir.is_dir() or not label_dir.is_dir():
            continue
        frame_ids = []
        for img_path in sorted(img_dir.glob("*.png")):
            label_path = label_dir / img_path.name
            if label_path.exists():
                frame_ids.append(int(img_path.stem))
        if frame_ids:
            yield {
                "seq_name": seq_dir.name,
                "domain": cosec_domain(seq_dir.name),
                "frame_count": len(frame_ids),
                "frame_ids": tuple(frame_ids),
            }


@lru_cache(maxsize=None)
def _cached_kfold_sequence_sets(root, folds):
    infos = tuple(iter_cosec_sequence_infos(root))
    if len(infos) < folds:
        raise ValueError(f"Cannot build {folds}-fold CoSEC split from only {len(infos)} sequences")

    val_sequence_sets = [set() for _ in range(folds)]
    for domain in ("day", "night"):
        domain_infos = [info for info in infos if info["domain"] == domain]
        if len(domain_infos) < folds:
            raise ValueError(
                f"Cannot build domain-aware sequence-level {folds}-fold CoSEC split: "
                f"{domain} has only {len(domain_infos)} sequences. "
                f"Use folds <= {len(domain_infos)} for leakage-free domain-aware splits."
            )
        fold_frame_counts = [0 for _ in range(folds)]
        fold_seq_counts = [0 for _ in range(folds)]
        for info in sorted(domain_infos, key=lambda item: (-item["frame_count"], item["seq_name"])):
            fold_index = min(range(folds), key=lambda idx: (fold_frame_counts[idx], fold_seq_counts[idx], idx))
            val_sequence_sets[fold_index].add(info["seq_name"])
            fold_frame_counts[fold_index] += info["frame_count"]
            fold_seq_counts[fold_index] += 1

    all_sequences = frozenset(info["seq_name"] for info in infos)
    return tuple(
        {
            "train": frozenset(all_sequences - frozenset(val_sequences)),
            "val": frozenset(val_sequences),
        }
        for val_sequences in val_sequence_sets
    )


def build_cosec_kfold_sequence_sets(root=DEFAULT_COSEC_TRAIN_ROOT, folds=DEFAULT_KFOLD_COUNT):
    root = str(Path(root).resolve())
    return _cached_kfold_sequence_sets(root, int(folds))


def kfold_sample_group(seq_name, fold_index, folds=DEFAULT_KFOLD_COUNT, root=DEFAULT_COSEC_TRAIN_ROOT):
    domain = cosec_domain(seq_name)
    fold_sequences = build_cosec_kfold_sequence_sets(root=root, folds=folds)[fold_index]
    subset = "val" if seq_name in fold_sequences["val"] else "train"
    return f"{domain}_{subset}"


def sample_group_for_split(seq_name, frame_id, split, root=DEFAULT_COSEC_TRAIN_ROOT):
    kfold_spec = parse_kfold_split(split)
    if kfold_spec:
        return kfold_sample_group(
            seq_name,
            kfold_spec["fold_index"],
            folds=kfold_spec["folds"],
            root=root,
        )
    return holdout_sample_group(seq_name, frame_id)


def split_contains_sample(seq_name, frame_id, split, root=DEFAULT_COSEC_TRAIN_ROOT):
    file_ids = _file_split_ids(split)
    if file_ids is not None:
        return mmseg_split_name(seq_name, frame_id) in file_ids
    kfold_spec = parse_kfold_split(split)
    requested_split = kfold_spec["requested_split"] if kfold_spec else split
    sample_group = sample_group_for_split(seq_name, frame_id, split, root=root)
    return _group_matches_requested_split(sample_group, requested_split)


def iter_cosec_samples(root="data/train", split="train"):
    root = Path(root)
    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not seq_dir.name.startswith(("Day_", "Night_")):
            continue
        img_dir = seq_dir / "img_co_left"
        label_dir = seq_dir / "segment_co"
        if not img_dir.is_dir() or not label_dir.is_dir():
            continue
        for img_path in sorted(img_dir.glob("*.png")):
            frame_id = int(img_path.stem)
            label_path = label_dir / img_path.name
            if not label_path.exists():
                continue
            if split_contains_sample(seq_dir.name, frame_id, split, root=root):
                yield seq_dir.name, frame_id, img_path, label_path
