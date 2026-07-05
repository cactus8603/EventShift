# Tools

The 0411 scripts are grouped by purpose:

```text
training/     Mask2Former training and calibration training
export/       Mask2Former / MMSeg prediction export and TTA
rebuild/      0411 rebuild scripts
postprocess/  Submission composition, filtering, voting, repair routing
diagnostics/  Event alignment, event support, routing and gap diagnostics
data/         Dataset split builders, event-edge cache builders, converters
analysis/     Summaries, audits, run analysis
launchers/    Historical experiment launcher scripts
cache/        Prediction and feature cache helpers
misc/         Less frequently used utilities
```

Several shared modules remain directly under `tools/` for compatibility with the
original imports used by Mask2Former dataset mappers and older scripts.
