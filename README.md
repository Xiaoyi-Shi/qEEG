# Quantitative EEG Toolkit

Reference implementation of a configurable qEEG processing flow built on top of MNE and AntroPy. `code_01_qeeg.py` is the primary CLI that orchestrates discovery, feature extraction, tidy exports, and QC HTML generation. The `sample/` folder keeps the earlier prototype for comparison/regression.

## Repository layout

- `code_01_qeeg.py`: end-to-end CLI pipeline that reads configs, runs features, and produces QC artifacts.
- `sample/`: frozen prototype (`code_01_qeeg_pipeline.py`) that mirrors the production logic.
- `configs/`: JSON configuration files describing bands, Welch parameters, entropy settings, and reporting metadata.
- `data/EEG_DATA/`: drop `.fif` or `.edf` recordings here (ignored by git).
- `result/`: timestamped run directories containing CSV exports, QC HTML, and logs.
- `docs/`: planning notes, architecture docs, and change history.
- `utils/`: reusable feature calculators consumed by the CLI.

## Configuration schema

```json
{
  "paths": {
    "data_dir": "data/EEG_DATA",
    "output_dir": "result"
  },
  "bands": {
    "delta": [1, 4],
    "theta": [4, 8],
    "alpha": [8, 13],
    "beta": [13, 30]
  },
  "welch": {
    "n_fft": 4096,
    "n_overlap": 2048,
    "n_per_seg": null
  },
  "entropy": {
    "bands": {
      "delta": [1, 4],
      "alpha": [8, 13]
    },
    "order": 3,
    "delay": 1,
    "normalize": true
  },
  "report": {
    "title": "qEEG QC Report",
    "author": "Data Ops"
  }
}
```

- `paths`: relative or absolute locations for the input EEG directory and the result root.
- `bands`: named `[fmin, fmax]` pairs that drive the absolute/relative power calculations.
- `welch`: optional PSD overrides that are passed to MNE's Welch-based PSD computation.
- `entropy`: optional AntroPy permutation entropy settings; omit or leave `bands` empty to disable the feature.
- `report`: metadata for the HTML QC title page.

## Running the pipeline

Activate the `mne_1.9.0` conda environment and execute:

```bash
python code_01_qeeg.py --config configs/cal_qEEG_all.json
```

Optional flags:

- `--data-dir` / `--result-dir` override the config paths.
- `--feature absolute_power` (repeatable) restricts computation to selected metrics (`absolute_power`, `relative_power`, `permutation_entropy`).
- `--dry-run` exercises discovery/logging without loading recordings or writing outputs.
- `--log-level DEBUG` surfaces verbose diagnostics.

Each run creates `result/<timestamp>/` containing:

1. `qEEG_result.csv`: tidy dataset (`subject_id`, `channel`, `band`, `metric`, `power`).
2. `QC.html`: Plotly-backed QC report with metadata summaries and per-feature distributions (channel + subject means, z-scores, descriptive statistics).
3. `log/pipeline.log`: structured execution log mirrored to stdout.

## Working with the reference prototype

The `sample/` folder keeps the earliest pipeline plus helper modules for regression or experimentation. It is no longer required for day-to-day execution but remains useful when comparing algorithmic tweaks.

## Contributing

- Update `docs/Architecture.md` whenever pipeline responsibilities shift.
- Capture noteworthy changes in `docs/CHANGELOG.md` (new configs, feature handlers, reporting tweaks, etc.).
- Keep PHI out of version control: `data/` stays ignored, and generated reports should only be shared after de-identification.
