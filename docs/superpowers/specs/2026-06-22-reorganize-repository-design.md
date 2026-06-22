# Reorganize Repository Design

## Goal

Make `/cpfs/user/guowenqi/lingbot-map-copy` easier to scan by grouping scripts, outputs, checkpoints, reports, and temporary files while preserving the main package and demo entrypoint.

## Scope

This is a medium reorganization. The public quick-start surface stays stable: `README.md`, `LICENSE.txt`, `pyproject.toml`, `demo.py`, `lingbot_map/`, `assets/`, and `example/` remain at the repository root.

Training, evaluation, visualization, diagnostic, and utility scripts move under `scripts/`. Experiment outputs, logs, caches, checkpoints, reports, and vendor packages move into purpose-specific top-level directories.

## Target Layout

```text
scripts/
  train/
  eval/
  visualize/
  tools/
checkpoints/
outputs/
  eval/
  logs/
  cache/
reports/
vendor/
```

## Compatibility

Shell scripts that previously assumed Python files and checkpoints lived next to them should compute `REPO_DIR` from their new location and call files by absolute paths inside the repository. README references should point to the new script and checkpoint locations. `demo.py` remains at the root to keep the common demo workflow simple.

## Verification

After moving files, verify that imports still compile, root-level clutter is reduced, key shell scripts pass `bash -n`, and README no longer points users to moved root-level training/evaluation scripts.
