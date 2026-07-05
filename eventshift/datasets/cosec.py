"""CoSEC event manifest loader used by EventShift."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable


def eventshift_root() -> Path:
    return Path(os.environ.get("EVENTSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()


def default_brenet_root() -> Path:
    return Path(os.environ.get("BRENET_ROOT", eventshift_root() / "data" / "BRENet")).resolve()


def infer_manifest_root(manifest_path: str | Path) -> Path:
    manifest = Path(manifest_path).expanduser().resolve()
    parent = manifest.parent
    if (
        parent.name == "manifests"
        and parent.parent.name == "brenet_cosec"
        and parent.parent.parent.name == "projects"
    ):
        return parent.parent.parent.parent
    return parent


def default_manifest_path() -> Path:
    return Path(
        os.environ.get(
            "EVENTSHIFT_COSEC_MANIFEST",
            default_brenet_root() / "projects" / "brenet_cosec" / "manifests" / "cosec_train_bidir_50ms.json",
        )
    ).resolve()


@dataclass(frozen=True)
class CoSECEventSample:
    sequence: str
    frame_id: int
    image: Path
    label: Path
    event_h5: Path
    event_old: tuple[int, int]
    event_new: tuple[int, int]

    @property
    def image_id(self) -> str:
        return f"{self.sequence}_{self.frame_id:06d}"


def _resolve_manifest_path(path: str | Path, roots: list[Path]) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = [root / path for root in roots]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _legacy_split_contains_sample(sequence: str, frame_id: int, split: str) -> bool:
    legacy_tools = eventshift_root() / "legacy" / "traincode_04111" / "tools"
    if legacy_tools.is_dir() and str(legacy_tools) not in sys.path:
        sys.path.insert(0, str(legacy_tools))
    try:
        from cosec_finetune_splits import split_contains_sample  # type: ignore
    except Exception:
        return True
    return bool(split_contains_sample(sequence, frame_id, split))


@lru_cache(maxsize=8)
def _load_payload(manifest_path: str) -> dict:
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest_samples(
    manifest_path: str | Path | None = None,
    brenet_root: str | Path | None = None,
    require_exists: bool = True,
) -> list[CoSECEventSample]:
    manifest = Path(manifest_path or default_manifest_path()).resolve()
    roots = []
    if brenet_root is not None:
        roots.append(Path(brenet_root).expanduser().resolve())
    roots.extend([infer_manifest_root(manifest), manifest.parent, default_brenet_root()])
    deduped_roots = []
    seen_roots = set()
    for root in roots:
        key = str(root)
        if key in seen_roots:
            continue
        seen_roots.add(key)
        deduped_roots.append(root)
    payload = _load_payload(str(manifest))
    samples: list[CoSECEventSample] = []
    for row in payload.get("samples", []):
        if not row.get("valid", True):
            continue
        image = _resolve_manifest_path(row["image"], deduped_roots)
        label = _resolve_manifest_path(row["label"], deduped_roots)
        event_h5 = _resolve_manifest_path(row["event_h5"], deduped_roots)
        if require_exists and not (image.exists() and label.exists() and event_h5.exists()):
            continue
        samples.append(
            CoSECEventSample(
                sequence=str(row["sequence"]),
                frame_id=int(row["frame_id"]),
                image=image,
                label=label,
                event_h5=event_h5,
                event_old=tuple(int(value) for value in row["event_old"]),
                event_new=tuple(int(value) for value in row["event_new"]),
            )
        )
    return samples


def load_cosec_event_dicts(
    split: str,
    manifest_path: str | Path | None = None,
    brenet_root: str | Path | None = None,
    require_exists: bool = True,
) -> list[dict]:
    records = []
    for sample in load_manifest_samples(manifest_path, brenet_root, require_exists=require_exists):
        if not _legacy_split_contains_sample(sample.sequence, sample.frame_id, split):
            continue
        records.append(
            {
                "file_name": str(sample.image),
                "sem_seg_file_name": str(sample.label),
                "image_id": sample.image_id,
                "event_h5": str(sample.event_h5),
                "event_old": list(sample.event_old),
                "event_new": list(sample.event_new),
            }
        )
    return records


def iter_image_ids(records: Iterable[dict]) -> Iterable[str]:
    for record in records:
        yield str(record.get("image_id") or Path(record["file_name"]).stem)

