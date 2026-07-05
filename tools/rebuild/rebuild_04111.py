#!/usr/bin/env python3
"""Recipe-driven rebuild for the 0.4111 EventShift submission."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from eventshift.backends import backend_pythonpath, get_export_backend
from eventshift.utils.config import compose_configs, eventshift_config_path, load_config


DEFAULT_RECIPE = "configs/eventshift/recipes/rebuild_04111_b75.yaml"


def repo_root() -> Path:
    return Path(os.environ.get("EVENTSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the 0.4111 b75 submission from the recipe, bundled checkpoints, and artifacts."
    )
    parser.add_argument("--recipe", default=DEFAULT_RECIPE, help="Recipe YAML under configs/eventshift/recipes.")
    parser.add_argument("--test-root", default=os.environ.get("TEST_ROOT"), help="CoSEC test root containing sequence/img_co_left folders.")
    parser.add_argument("--out-root", default=os.environ.get("OUT_ROOT"), help="Output root.")
    parser.add_argument("--conda", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--m2f-env", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mmseg-env", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--device", default=None, help="Torch device.")
    parser.add_argument("--smoke-limit", default=os.environ.get("SMOKE_LIMIT"), help="Export only first N frames per model and stop before composition.")
    parser.add_argument("--skip-inference", action="store_true", help="Reuse existing raw masks under --out-root.")
    parser.add_argument("--run-inference", action="store_true", help="Run model inference. This is the default.")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic torch settings. This is the default.")
    parser.add_argument("--non-deterministic", action="store_true", help="Disable deterministic torch settings.")
    return parser.parse_args()


def bool_text(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def as_path(value: str | Path, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def check_path(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Missing required path: {path}")


def count_pngs(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.rglob("*.png") if item.is_file())


def expected_count(test_root: Path, sequences: list[str]) -> int:
    total = 0
    for seq in sequences:
        img_dir = test_root / seq / "img_co_left"
        if not img_dir.is_dir():
            raise SystemExit(f"Missing image directory for expected count: {img_dir}")
        total += sum(1 for item in img_dir.glob("*.png") if item.is_file())
    return total


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_env(root: Path, backend_name: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    parts = [str(root)]
    if backend_name:
        backend = get_export_backend(backend_name)
        backend_path = backend_pythonpath(root, backend)
        if backend_path:
            parts.append(backend_path)
    else:
        parts.extend([
            str(root / "tools"),
            str(root / "third_party" / "Mask2Former"),
            str(root / "third_party" / "detectron2"),
        ])
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(part for part in parts if part)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def python_command(conda: Path | None, env_name: str | None, script: Path, args: list[str]) -> list[str]:
    if conda and env_name:
        return [str(conda), "run", "--no-capture-output", "-n", env_name, "python", str(script), *args]
    return [sys.executable, str(script), *args]


def run_command(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, check=True, env=env)


def resolve_runtime(args: argparse.Namespace, rebuild_cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    runtime = rebuild_cfg.get("runtime", {})
    deterministic = bool_text(runtime.get("deterministic"), default=True)
    if os.environ.get("EVENTSHIFT_DETERMINISTIC") is not None:
        deterministic = bool_text(os.environ.get("EVENTSHIFT_DETERMINISTIC"), default=deterministic)
    if args.deterministic:
        deterministic = True
    if args.non_deterministic:
        deterministic = False

    run_inference = bool_text(os.environ.get("RUN_INFERENCE"), default=True)
    if args.skip_inference:
        run_inference = False
    if args.run_inference:
        run_inference = True

    conda_value = args.conda or runtime.get("conda") or None
    return {
        "conda": as_path(conda_value, root) if conda_value else None,
        "m2f_env": args.m2f_env or runtime.get("m2f_env") or os.environ.get("CONDA_DEFAULT_ENV") or "ebmv_seg",
        "mmseg_env": args.mmseg_env or runtime.get("mmseg_env") or os.environ.get("CONDA_DEFAULT_ENV") or "ebmv_seg",
        "device": args.device or runtime.get("device", "cuda:0"),
        "deterministic": deterministic,
        "run_inference": run_inference,
    }


def load_export_config(model: str, variant: str) -> dict[str, Any]:
    return compose_configs(eventshift_config_path(model=model, variant=variant))


def exporter_command(
    root: Path,
    conda: Path,
    env_name: str,
    test_root: Path,
    raw_dir: Path,
    runtime: dict[str, Any],
    export_cfg: dict[str, Any],
    export_item: dict[str, Any],
    sequences: list[str],
    smoke_limit: str | None,
) -> tuple[list[str], dict[str, str]]:
    model_cfg = export_cfg.get("model", {})
    inference_cfg = export_cfg.get("inference", {})
    checkpoints = export_cfg.get("checkpoints", {})
    backend_name = model_cfg.get("exporter") or model_cfg.get("backend")
    backend = get_export_backend(str(backend_name))

    out_dir = raw_dir / str(export_item["name"])
    command_args = [
        backend.config_flag,
        str(as_path(model_cfg["backend_config"], root)),
        backend.weights_flag,
        str(as_path(checkpoints["init_weights"], root)),
        "--test-root",
        str(test_root),
        "--out-dir",
        str(out_dir),
        "--device",
        str(runtime["device"]),
        "--progress-desc",
        str(inference_cfg.get("progress_desc") or export_item["name"]),
        "--sequences",
        *sequences,
        "--skip-existing",
    ]
    if smoke_limit not in {None, ""}:
        command_args.extend(["--limit", str(smoke_limit)])

    extra_args = as_list(inference_cfg.get("extra_args"))
    if backend.name in {"mmseg", "segformer"} and os.environ.get("MMSEG_TTA_FLIP", "1") != "1":
        extra_args = [arg for arg in extra_args if arg != "--flip"]
    command_args.extend(extra_args)

    command = python_command(conda, env_name, root / backend.script, command_args)
    return command, make_env(root, backend.name)


def run_exports(
    root: Path,
    recipe: dict[str, Any],
    runtime: dict[str, Any],
    test_root: Path,
    raw_dir: Path,
    smoke_limit: str | None,
) -> None:
    sequence_groups = recipe["sequence_groups"]
    expected = {
        "day": expected_count(test_root, as_list(sequence_groups["day"])),
        "night": expected_count(test_root, as_list(sequence_groups["night"])),
    }
    print(f"expected counts: day={expected['day']} night={expected['night']}", flush=True)

    for export_item in recipe["exports"]:
        name = str(export_item["name"])
        group = str(export_item["group"])
        sequences = as_list(sequence_groups[group])
        out_dir = raw_dir / name
        current = count_pngs(out_dir)
        if current == expected[group]:
            print(f"[skip] {name}: {current}/{expected[group]}", flush=True)
            continue
        print(f"[export] {name}: {current}/{expected[group]} -> {out_dir}", flush=True)
        export_cfg = load_export_config(str(export_item["model"]), str(export_item["variant"]))
        env_name = runtime["m2f_env"] if export_item.get("env") == "m2f" else runtime["mmseg_env"]
        command, env = exporter_command(
            root=root,
            conda=runtime["conda"],
            env_name=env_name,
            test_root=test_root,
            raw_dir=raw_dir,
            runtime=runtime,
            export_cfg=export_cfg,
            export_item=export_item,
            sequences=sequences,
            smoke_limit=smoke_limit,
        )
        run_command(command, env)


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    if out_dir.is_dir() and count_pngs(out_dir) > 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print(f"extracted {zip_path} -> {out_dir}", flush=True)


def run_bundle_python(root: Path, runtime: dict[str, Any], script: Path, args: list[str]) -> None:
    command = python_command(runtime["conda"], runtime["m2f_env"], script, args)
    run_command(command, make_env(root, backend_name=None))


def zip_test(path: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
    if bad:
        raise SystemExit(f"Corrupt zip entry in {path}: {bad}")


def zip_content_equal(left: Path, right: Path) -> tuple[bool, str]:
    with zipfile.ZipFile(left) as left_zip, zipfile.ZipFile(right) as right_zip:
        left_names = sorted(name for name in left_zip.namelist() if not name.endswith("/"))
        right_names = sorted(name for name in right_zip.namelist() if not name.endswith("/"))
        if left_names != right_names:
            return False, f"entry lists differ: {len(left_names)} != {len(right_names)}"
        for name in left_names:
            if left_zip.read(name) != right_zip.read(name):
                return False, f"content differs at {name}"
    return True, f"entries={len(left_names)}"


def run_postprocess(
    root: Path,
    recipe: dict[str, Any],
    runtime: dict[str, Any],
    test_root: Path,
    raw_dir: Path,
    composed_dir: Path,
    zip_dir: Path,
    report_dir: Path,
    extract_dir: Path,
) -> Path:
    postprocess_dir = root / "tools" / "postprocess"
    filter_script = postprocess_dir / "filter_submission_delta_by_transition.py"
    compose_script = postprocess_dir / "compose_domain_submission.py"
    check_path(filter_script)
    check_path(compose_script)

    artifacts = recipe["artifacts"]
    anchor_zip = as_path(artifacts["anchor_04075_zip"], root)
    night_valpair_zip = as_path(artifacts["night_valpair_zip"], root)
    realgate_zip = as_path(artifacts["realgate_zip"], root)
    expected_final_zip = as_path(recipe["expected_final_zip"], root)
    for path in (anchor_zip, night_valpair_zip, realgate_zip, expected_final_zip):
        check_path(path)

    extract_zip(anchor_zip, extract_dir / anchor_zip.stem)
    extract_zip(night_valpair_zip, extract_dir / night_valpair_zip.stem)
    extract_zip(realgate_zip, extract_dir / realgate_zip.stem)

    pp = recipe["postprocess"]
    allow_pairs = str(pp["allow_pairs"])
    eventseg = pp["eventseg_p70"]
    main_keepreal = pp["main_keepreal"]
    eventseg_keepreal = pp["eventseg_keepreal"]
    pipeline_b75 = pp["pipeline_b75"]
    final = pp["final"]

    eventseg_out = composed_dir / eventseg["out_name"]
    run_bundle_python(root, runtime, filter_script, [
        "--base", str(raw_dir / "mask2former_night_full_desc"),
        "--candidate", str(raw_dir / "segformer_night_event"),
        "--out-dir", str(eventseg_out),
        "--zip", str(zip_dir / f"{eventseg['out_name']}.zip"),
        "--summary", str(report_dir / eventseg["summary_name"]),
        "--domains", "Night",
        "--allow-pairs", allow_pairs,
        "--component-min-boundary5-rate", str(eventseg["component_min_boundary5_rate"]),
        "--component-max-area", str(eventseg["component_max_area"]),
        "--overwrite",
    ])

    main_keepreal_out = composed_dir / main_keepreal["out_name"]
    run_bundle_python(root, runtime, compose_script, [
        "--day-dir", str(raw_dir / "mask2former_day_event"),
        "--night-dir", str(extract_dir / night_valpair_zip.stem),
        "--real-dir", str(extract_dir / anchor_zip.stem),
        "--test-root", str(test_root),
        "--out-dir", str(main_keepreal_out),
        "--zip", str(zip_dir / f"{main_keepreal['out_name']}.zip"),
        "--overwrite",
    ])

    eventseg_keepreal_out = composed_dir / eventseg_keepreal["out_name"]
    run_bundle_python(root, runtime, compose_script, [
        "--day-dir", str(raw_dir / "mask2former_day_event"),
        "--night-dir", str(eventseg_out),
        "--real-dir", str(extract_dir / anchor_zip.stem),
        "--test-root", str(test_root),
        "--out-dir", str(eventseg_keepreal_out),
        "--zip", str(zip_dir / f"{eventseg_keepreal['out_name']}.zip"),
        "--overwrite",
    ])

    pipeline_b75_out = composed_dir / pipeline_b75["out_name"]
    run_bundle_python(root, runtime, filter_script, [
        "--base", str(zip_dir / f"{main_keepreal['out_name']}.zip"),
        "--candidate", str(zip_dir / f"{eventseg_keepreal['out_name']}.zip"),
        "--out-dir", str(pipeline_b75_out),
        "--zip", str(zip_dir / f"{pipeline_b75['out_name']}.zip"),
        "--summary", str(report_dir / pipeline_b75["summary_name"]),
        "--domains", "Night",
        "--allow-pairs", allow_pairs,
        "--component-min-boundary5-rate", str(pipeline_b75["component_min_boundary5_rate"]),
        "--component-max-area", str(pipeline_b75["component_max_area"]),
        "--overwrite",
    ])

    final_zip = zip_dir / f"{final['out_name']}.zip"
    run_bundle_python(root, runtime, compose_script, [
        "--day-dir", str(pipeline_b75_out),
        "--night-dir", str(pipeline_b75_out),
        "--real-dir", str(extract_dir / realgate_zip.stem),
        "--test-root", str(test_root),
        "--out-dir", str(composed_dir / final["out_name"]),
        "--zip", str(final_zip),
        "--overwrite",
    ])
    return final_zip


def main() -> None:
    args = parse_args()
    root = repo_root()
    recipe_path = as_path(args.recipe, root)
    cfg = load_config(recipe_path)
    recipe = cfg["rebuild"]
    runtime = resolve_runtime(args, recipe, root)

    if not args.test_root:
        raise SystemExit("--test-root is required. Example: bash scripts/rebuild_04111.sh --test-root /path/to/test")
    test_root = as_path(args.test_root, root)
    check_path(test_root)
    if runtime["conda"]:
        check_path(runtime["conda"])

    if runtime["deterministic"]:
        os.environ["EVENTSHIFT_DETERMINISTIC"] = "1"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        os.environ["EVENTSHIFT_DETERMINISTIC"] = "0"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_prefix = recipe.get("output_prefix", "rebuild_04111_b75_from_checkpoints")
    out_root = as_path(args.out_root, root) if args.out_root else root / "outputs" / f"{out_prefix}_{stamp}"
    raw_dir = out_root / "prediction_dirs" / "from_checkpoints_raw"
    composed_dir = out_root / "composed"
    zip_dir = out_root / "submission_zips"
    report_dir = out_root / "reports"
    extract_dir = out_root / "extracted_artifacts"
    for directory in (raw_dir, composed_dir, zip_dir, report_dir, extract_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"BUNDLE_DIR={root}", flush=True)
    print(f"RECIPE={recipe_path}", flush=True)
    print(f"TEST_ROOT={test_root}", flush=True)
    print(f"OUT_ROOT={out_root}", flush=True)
    print(f"DEVICE={runtime['device']}", flush=True)
    if runtime["conda"]:
        print(f"CONDA={runtime['conda']}", flush=True)
        print(f"M2F_ENV={runtime['m2f_env']} MMSEG_ENV={runtime['mmseg_env']}", flush=True)
    else:
        print(f"PYTHON={sys.executable}", flush=True)

    # Validate model configs and weights before any long-running export starts.
    for export_item in recipe["exports"]:
        export_cfg = load_export_config(str(export_item["model"]), str(export_item["variant"]))
        model_cfg = export_cfg["model"]
        checkpoints = export_cfg["checkpoints"]
        backend = get_export_backend(str(model_cfg.get("exporter") or model_cfg.get("backend")))
        check_path(root / backend.script)
        check_path(as_path(model_cfg["backend_config"], root))
        check_path(as_path(checkpoints["init_weights"], root))

    if runtime["run_inference"]:
        run_exports(root, recipe, runtime, test_root, raw_dir, args.smoke_limit)
    else:
        sequence_groups = recipe["sequence_groups"]
        day_count = expected_count(test_root, as_list(sequence_groups["day"]))
        night_count = expected_count(test_root, as_list(sequence_groups["night"]))
        print(f"expected counts: day={day_count} night={night_count}", flush=True)
        print(f"[skip] RUN_INFERENCE=0; using existing masks under {raw_dir}", flush=True)

    if args.smoke_limit not in {None, ""}:
        print(f"SMOKE_LIMIT={args.smoke_limit}; stopping after export smoke test.", flush=True)
        return

    final_zip = run_postprocess(root, recipe, runtime, test_root, raw_dir, composed_dir, zip_dir, report_dir, extract_dir)
    zip_test(final_zip)

    actual_sha = sha256(final_zip)
    expected_sha = str(recipe["expected_final_sha"])
    print(f"[sha] final_04111: {actual_sha}", flush=True)
    if actual_sha != expected_sha:
        print(f"WARNING: final_04111 SHA mismatch. Expected {expected_sha}", flush=True)

    reference_matches = []
    expected_final_zip = as_path(recipe["expected_final_zip"], root)
    if final_zip.read_bytes() == expected_final_zip.read_bytes():
        reference_matches.append("bundle_authoritative_byte")
        print("byte_compare=identical_to_bundle_authoritative_final_zip", flush=True)
    else:
        same_content, detail = zip_content_equal(final_zip, expected_final_zip)
        if same_content:
            reference_matches.append("bundle_authoritative_content")
            print(f"content_compare=identical_to_bundle_authoritative_final_zip ({detail})", flush=True)
        else:
            print(f"WARNING: final zip content differs from bundle authoritative final zip: {detail}", flush=True)

    submit_zip = root / "submit" / "sub_pipeline_b75.zip"
    if submit_zip.exists():
        same_submit_content, submit_detail = zip_content_equal(final_zip, submit_zip)
        if same_submit_content:
            reference_matches.append("submit_content")
            print(f"content_compare=identical_to_submit_sub_pipeline_b75_zip ({submit_detail})", flush=True)
        else:
            print(f"WARNING: final zip content differs from submit/sub_pipeline_b75.zip: {submit_detail}", flush=True)

    if not reference_matches:
        raise SystemExit("WARNING: final zip content does not match any local 0.4111 reference zip.")

    print("OK: rebuilt 0.4111 b75 anchor from bundle checkpoints and bundle artifacts.", flush=True)
    print(f"final_zip={final_zip}", flush=True)


if __name__ == "__main__":
    main()
