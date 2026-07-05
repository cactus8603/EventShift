#!/usr/bin/env python
"""Export Day test masks with train-learned event pair correction.

This exporter is intentionally narrow. It uses an existing RGB Day prediction
directory as the strong anchor, runs the event model only to get an event
candidate, and applies the train-learned class-pair route saved by
diagnose_train_learned_tta_event_pair_router.py.

The test manifest is used only for image/event paths and event time windows.
No test semantic label is read.
"""

import argparse
import copy
import json
import os
import re
import shutil
import sys
import importlib.util
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "third_party" / "Mask2Former"))
if importlib.util.find_spec("detectron2") is None:
    sys.path.insert(0, str(ROOT / "third_party" / "detectron2"))

from cosec_finetune_splits import CLASSES  # noqa: E402
from detectron2.data import detection_utils as utils  # noqa: E402
from detectron2.data import transforms as T  # noqa: E402
from mask2former import MaskFormerSemanticDatasetMapper  # noqa: E402
from train_mask2former_cosec import register_cosec  # noqa: E402

from diagnose_train_learned_tta_event_pair_router import (  # noqa: E402
    condition_mask,
    make_regions,
    merge_with_pairs,
)
from diagnose_tta_event_class_routing import (  # noqa: E402
    build_model,
    infer_prob,
    setup_cfg,
    top_conf_margin,
)


ROUTE_RE = re.compile(
    r"^(?P<region>.+)_b(?P<radius>\d+)_baseconf(?P<base_conf>[0-9.]+)"
    r"_eventconf(?P<event_conf>[0-9.]+)_(?P<margin_mode>.+)$"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-config", required=True)
    parser.add_argument("--event-weights", required=True)
    parser.add_argument("--diagnostic-json", required=True)
    parser.add_argument("--base-day-dir", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--path-root", default=str(ROOT))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zip", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--route-method", default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def resolve_path(path, path_root):
    path = Path(path)
    if path.is_absolute():
        return path
    rooted = Path(path_root) / path
    if rooted.exists():
        return rooted
    return ROOT / path


def load_records(manifest_path, path_root, sequences=None, limit=None):
    keep = set(sequences) if sequences else None
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        samples = json.load(f)
    records = []
    for sample in samples:
        seq_name = sample["sequence"]
        if not seq_name.startswith("Day_"):
            continue
        if keep is not None and seq_name not in keep:
            continue
        image_path = resolve_path(sample["image"], path_root)
        event_h5 = resolve_path(sample["event_h5"], path_root)
        records.append(
            {
                "sequence": seq_name,
                "file_name": str(image_path),
                "event_h5": str(event_h5),
                "event_old": [int(value) for value in sample["event_old"]],
                "event_new": [int(value) for value in sample["event_new"]],
                "frame_name": image_path.name,
            }
        )
    records.sort(key=lambda item: (item["sequence"], item["frame_name"]))
    if limit is not None:
        records = records[:limit]
    return records


def selected_route_from_diagnostic(path, route_method=None):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    routes = data.get("top_eval_routes", [])
    if not routes:
        raise ValueError(f"No top_eval_routes found in {path}")
    if route_method is None:
        route = routes[0]
    else:
        matches = [row for row in routes if row.get("method") == route_method]
        if not matches:
            raise ValueError(f"Route method not found in diagnostic: {route_method}")
        route = matches[0]
    match = ROUTE_RE.match(route["method"])
    if not match:
        raise ValueError(f"Could not parse route method: {route['method']}")
    spec = match.groupdict()
    spec["radius"] = int(spec["radius"])
    spec["base_conf"] = float(spec["base_conf"])
    spec["event_conf"] = float(spec["event_conf"])
    if spec["base_conf"] != 0.0 or spec["margin_mode"] != "none":
        raise ValueError(
            "This exporter uses anchor masks only, so it supports routes with "
            "baseconf0 and margin mode 'none' only. Requested: "
            f"{route['method']}"
        )
    selected_pairs = []
    for row in route.get("selected_pairs", []):
        src_name, dst_name = row["pair"].split("->", 1)
        selected_pairs.append((CLASSES.index(src_name), CLASSES.index(dst_name)))
    if not selected_pairs:
        raise ValueError(f"Route has no selected pairs: {route['method']}")
    return route, spec, set(selected_pairs), data


def make_event_mapper(cfg):
    mapper = MaskFormerSemanticDatasetMapper(cfg, False)

    def map_record(record):
        dataset_dict = copy.deepcopy(record)
        image = utils.read_image(dataset_dict["file_name"], format=mapper.img_format)
        original_shape = image.shape[:2]
        event, event_aux = mapper._load_event(dataset_dict, image.shape)
        event_edge = mapper._load_event_edge(dataset_dict, image.shape)

        aug_input = T.AugInput(image)
        aug_input, transforms = T.apply_transform_gens(mapper.tfm_gens, aug_input)
        image = aug_input.image
        event = mapper._apply_geometric_transforms(transforms, event)
        event_aux = mapper._apply_geometric_transforms(transforms, event_aux)
        event_edge = mapper._apply_geometric_transforms(transforms, event_edge)
        event_stats = mapper._event_stats_from_aux(event_aux)

        image_tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        event_tensor = torch.as_tensor(np.ascontiguousarray(event.transpose(2, 0, 1))).float()
        event_edge_tensor = torch.as_tensor(np.ascontiguousarray(event_edge.transpose(2, 0, 1))).float()
        event_stats_tensor = torch.as_tensor(np.ascontiguousarray(event_stats.transpose(2, 0, 1))).float()

        if mapper.size_divisibility > 0:
            image_size = (image_tensor.shape[-2], image_tensor.shape[-1])
            padding_size = [
                0,
                mapper.size_divisibility - image_size[1],
                0,
                mapper.size_divisibility - image_size[0],
            ]
            image_tensor = F.pad(image_tensor, padding_size, value=128).contiguous()
            event_tensor = F.pad(event_tensor, padding_size, value=0).contiguous()
            event_edge_tensor = F.pad(event_edge_tensor, padding_size, value=0).contiguous()
            event_stats_tensor = F.pad(event_stats_tensor, padding_size, value=0).contiguous()

        dataset_dict["image"] = image_tensor
        dataset_dict["event"] = event_tensor
        dataset_dict["event_edge"] = event_edge_tensor
        dataset_dict["event_stats"] = event_stats_tensor
        dataset_dict["height"] = original_shape[0]
        dataset_dict["width"] = original_shape[1]
        return dataset_dict

    return map_record


def load_base_mask(base_day_dir, record):
    path = Path(base_day_dir) / record["sequence"] / "segment_co" / record["frame_name"]
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read base mask: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.int64)


def write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8)):
        raise RuntimeError(f"Could not write mask: {path}")


def zip_directory(src_dir, zip_path):
    src_dir = Path(src_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    max_path_len = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if not path.is_file() or path.name == "manifest.json":
                continue
            rel_path = path.relative_to(src_dir)
            max_path_len = max(max_path_len, len(str(rel_path)))
            zf.write(path, rel_path)
            count += 1
    return {"entries": count, "max_path_len": max_path_len}


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    route, spec, selected_pairs, diagnostic = selected_route_from_diagnostic(
        args.diagnostic_json,
        args.route_method,
    )
    records = load_records(args.test_manifest, args.path_root, args.sequences, args.limit)
    if not records:
        raise ValueError("No Day records matched the requested manifest/sequences.")

    event_cfg = setup_cfg(args.event_config, args.event_weights, args.device)
    event_model = build_model(event_cfg)
    event_mapper = make_event_mapper(event_cfg)

    counts = {
        "total": 0,
        "changed_pixels": 0,
        "by_sequence": {},
    }
    for record in tqdm(records, desc="Export Day event pair route"):
        dst_path = out_dir / record["sequence"] / "segment_co" / record["frame_name"]
        if args.skip_existing and dst_path.exists():
            continue

        base_pred = load_base_mask(args.base_day_dir, record)
        mapped = event_mapper(record)
        event_prob = infer_prob(event_model, mapped, base_pred.shape, use_flip=False)
        event_pred = event_prob.argmax(dim=0).numpy()
        event_conf, event_margin = top_conf_margin(event_prob)
        valid = np.ones(base_pred.shape, dtype=bool)
        regions = make_regions(mapped["event_stats"].float(), base_pred.shape, valid)
        ctx = {
            "valid": valid,
            "regions": regions,
            "base_pred": base_pred,
            "event_pred": event_pred,
            "base_conf": np.ones(base_pred.shape, dtype=np.float32),
            "base_margin": np.ones(base_pred.shape, dtype=np.float32),
            "event_conf": event_conf,
            "event_margin": event_margin,
        }
        cond = condition_mask(
            ctx,
            spec["region"],
            spec["radius"],
            spec["base_conf"],
            spec["event_conf"],
            spec["margin_mode"],
        )
        merged = merge_with_pairs(ctx, cond, selected_pairs)
        changed = int((merged != base_pred).sum())
        write_mask(dst_path, merged)

        counts["total"] += 1
        counts["changed_pixels"] += changed
        seq_stats = counts["by_sequence"].setdefault(
            record["sequence"],
            {"count": 0, "changed_pixels": 0},
        )
        seq_stats["count"] += 1
        seq_stats["changed_pixels"] += changed

    zip_info = zip_directory(out_dir, args.zip) if args.zip else None
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "event_config": str(Path(args.event_config).resolve()),
        "event_weights": str(Path(args.event_weights).resolve()),
        "diagnostic_json": str(Path(args.diagnostic_json).resolve()),
        "base_day_dir": str(Path(args.base_day_dir).resolve()),
        "test_manifest": str(Path(args.test_manifest).resolve()),
        "path_root": str(Path(args.path_root).resolve()),
        "out_dir": str(out_dir.resolve()),
        "zip": str(Path(args.zip).resolve()) if args.zip else None,
        "device": args.device,
        "route": route,
        "selected_route_method": route["method"],
        "diagnostic_rgb_tta": diagnostic.get("results", {}).get("rgb_tta"),
        "counts": counts,
        "zip_info": zip_info,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote Day event-pair masks to: {out_dir}")
    if args.zip:
        print(f"Wrote zip: {args.zip}")
    print(json.dumps({"counts": counts, "zip_info": zip_info}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
