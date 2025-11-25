# Architecture Overview

## Context
The toolkit exposes a CLI pipeline (`code_01_qeeg.py`) for running configurable qEEG feature calculations backed by MNE and AntroPy utilities housed in `utils/`.

## High-Level Components
- **CLI Orchestrator**: Parses arguments (config path, directories, feature filters, logging level, dry-run), loads the JSON configuration, discovers `.fif`/`.edf` EEG files, and coordinates feature execution.
- **Feature Registry / Dependency Injection**: `build_feature_registry()` produces a mapping from feature names to callables (`absolute_power`, `relative_power`, `permutation_entropy`). The registry can be overridden when embedding the pipeline, satisfying the dependency-injection requirement and easing future extensions.
- **Feature Pipeline**: `FeaturePipeline` loads each EEG file via an MNE reader and invokes handlers sequentially while emitting structured logs.
- **Utilities**:
  - `utils/basefun.py` implements absolute and relative power by computing Welch PSDs (via `psd_array_welch`) over configurable bands.
  - `utils/entropy.py` implements permutation entropy per channel using AntroPy, with helper logic to pick channels from the loaded Raw object.

## Data Flow
1. Config (`configs/cal_qEEG_all.json`) defines `features` – name plus feature-specific parameters (band definitions, entropy embedding parameters, etc.).
2. CLI filters features via `--feature` (optional) and iterates over EEG files discovered in `data/EEG_DATA/`.
3. Each handler returns structured results (e.g., `{channels: [...], bands: {...}}`), which are summarized via logging. Persisting QC reports/CSVs will be added in future milestones.

## Notes
- Data inputs remain within `data/EEG_DATA/` and are ignored by version control.
- `result/` folders (timestamped by the operator) are created automatically to host downstream QC reports, long-format CSVs, and logs.
