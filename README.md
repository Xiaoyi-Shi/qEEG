# Quantitative EEG Toolkit

Reference implementation of a configurable qEEG processing flow built on top of MNE and AntroPy. `code_01_qeeg.py` is the primary CLI that orchestrates discovery, feature extraction, tidy exports, and QC HTML generation. 

## Repository layout

- `code_01_qeeg.py`: end-to-end CLI pipeline that reads configs, runs features, and produces QC artifacts.
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
    "bids_dir": null,
    "output_dir": "result"
  },
  "preprocessing": {
    "resample_hz": 250,
    "bandpass": {"l_freq": 1, "h_freq": 40},
    "notch": {"freqs": [50]},
    "montage": {"name": "standard_1020"},
    "reference": {"kind": "average"}
  },
  "power": {
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
    }
  },
  "entropy": {
    "permutation": {
      "bands": {
        "delta": [1, 4],
        "alpha": [8, 13]
      },
      "order": 3,
      "delay": 1,
      "normalize": true
    },
    "spectral": {
      "band_label": "full",
      "method": "fft",
      "nperseg": null,
      "normalize": true
    }
  },
  "report": {
    "title": "qEEG QC Report",
    "author": "Data Ops"
  }
}
```

- `paths`: relative or absolute locations for the input EEG directory (flat `data_dir` or `bids_dir`) and the result root.
- `preprocessing`: unified knobs for resampling (`resample_hz`), bandpass filtering (`bandpass` block mirrors `Raw.filter` kwargs), notch filtering (`notch.freqs`), montage selection (built-in `name` or custom `path`), and reference strategy (`reference.kind` = `average`, `channels`, or `none` with optional `channels` list).
- `power.bands`: named `[fmin, fmax]` pairs that drive the absolute/relative power calculations.
- `power.welch`: optional PSD overrides that are passed to MNE's Welch-based PSD computation.
- `entropy.permutation`: permutation entropy settings (requires `bands` to be populated to enable the feature).
- `entropy.spectral`: parameters for AntroPy's spectral entropy (band label for reporting plus method/nperseg/normalize).
- `report`: metadata for the HTML QC title page.

## Running the pipeline

Activate the `mne_1.9.0` conda environment and execute:

```bash
python code_01_qeeg.py --config configs/cal_qEEG_all.json
```

Optional flags:

- `--data-dir` / `--result-dir` override the config paths.
- `--bids-dir` points the run at a BIDS dataset root (expects EEG files under `sub-*/[ses-*/]eeg/`).
- `--feature absolute_power` (repeatable) restricts computation to selected metrics (`absolute_power`, `relative_power`, `permutation_entropy`, `spectral_entropy`).
- `--dry-run` exercises discovery/logging without loading recordings or writing outputs.
- `--log-level DEBUG` surfaces verbose diagnostics.

To process BIDS-formatted studies that live under `data/BIDS/`:

```bash
python code_01_qeeg.py --config configs/cal_qEEG_all.json --bids-dir data/BIDS
```

Each run creates `result/<timestamp>/` containing:

1. `qEEG_result.csv`: tidy dataset (`subject_id`, `channel`, `band`, `metric`, `power`).
2. `QC.html`: Plotly-backed QC report with metadata summaries and per-feature distributions (channel + subject means, z-scores, descriptive statistics).
3. `log/pipeline.log`: structured execution log mirrored to stdout.

## Contributing

- Update `docs/Architecture.md` whenever pipeline responsibilities shift.
- Capture noteworthy changes in `docs/CHANGELOG.md` (new configs, feature handlers, reporting tweaks, etc.).
- Keep PHI out of version control: `data/` stays ignored, and generated reports should only be shared after de-identification.
