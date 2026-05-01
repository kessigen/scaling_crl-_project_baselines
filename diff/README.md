# Diff — Earliest Code Snapshot

This folder contains the earliest version of the project's code, preserved for reference.

The codebase has since been **restructured and refactored**, which is why the file names and directory layout here differ from the current top-level project. Functionality has been split into more focused modules and renamed for clarity.

## Where the code lives now

- The code that used to drive **pose-based SAC** training/eval (originally under `baselines_old/`) now lives in [`baselines/pose_sac_old/`](../baselines/pose_sac_old/).
- The code that used to drive **image-based SAC** training/eval (originally under `baselines_old/image_based/`) now lives in [`baselines/image_sac/`](../baselines/image_sac/).

Refer to those directories for the current, maintained implementations. The contents of `diff/` are kept only as a historical reference and are not used by the active training or evaluation pipelines.
