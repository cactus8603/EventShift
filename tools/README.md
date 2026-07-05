# Tools

The recommended public entry points are thin shell wrappers under `scripts/`:

```text
scripts/infer.sh          Args-first inference wrapper
scripts/train.sh          Args-first training wrapper
scripts/rebuild_04111.sh  Recipe-driven 0.4111 rebuild wrapper
scripts/eval.sh           Evaluation wrapper
scripts/prepare_data.sh   Dataset preparation notes/checks
```

Implementation tools are grouped by purpose:

```text
training/     Mask2Former training and calibration training
export/       Mask2Former / MMSeg prediction export and TTA
rebuild/      Recipe-driven 0411 rebuild runner plus smoke tests
postprocess/  Submission composition, filtering, voting, repair routing
diagnostics/  Event alignment, event support, routing and gap diagnostics
data/         Dataset split builders, event-edge cache builders, converters
analysis/     Summaries, audits, run analysis
launchers/    Historical experiment launcher scripts
cache/        Prediction and feature cache helpers
misc/         Less frequently used utilities
```

Historical one-off shell scripts are preserved under `scripts/archive/` and `tools/rebuild/archive/`. Several shared modules remain directly under `tools/` for compatibility with the original imports used by Mask2Former dataset mappers and older scripts.
