# Architecture Overview

## Context
`code_01_qeeg.py` is the primary CLI that loads JSON configs, discovers EEG recordings (either from a flat directory or a BIDS root), invokes feature calculators from `utils/`, and persists both a tidy CSV (`subject_id`, `channel`, `band`, `metric`, `power`) and a Plotly-backed QC report. The helper modules `utils/basefun.py` (power features), `utils/entropy.py` (entropy family), and `utils/QC.py` (QC rendering) are the main extension points.

## High-Level Components
- **CLI Orchestrator**: Parses CLI flags (config, directory overrides including `--bids-dir`, feature filters, logging, dry-run), loads the JSON spec, resolves relative paths, and manages timestamped run folders (`result/<timestamp>/` containing CSV/QC/logs).
- **Preprocessing Layer**: `utils.basefun.preprocess_raw` loads montage metadata, executes notch/bandpass filters, resamples, and enforces the desired reference strategy before any feature calculators touch the data. Defaults to an average reference to preserve historical behavior.
- **Feature Execution Loop**: Loads each `.fif`/`.edf` via MNE, captures recording metadata, and conditionally runs absolute power, relative power, permutation entropy, and spectral entropy calculators based on config + CLI switches.
- **Reporting/Persistence**: `tidy_power_table` merges feature DataFrames; the CLI writes CSV + QC HTML with metadata coverage, per-feature histograms, and z-score-based status flags.
- **Utilities**:
  - `utils/basefun.py` exposes Welch PSD helpers, absolute and relative band power calculators, metadata summarization, and tidy-frame helpers.
  - `utils/entropy.py` hosts AntroPy-backed entropy features. Initially only permutation entropy was supported; v1.02 adds spectral entropy with parameter containers for consistent config parsing.
  - `utils/QC.py` isolates the HTML/QC rendering helpers (`generate_qc_report`, histogram/table builders) so the CLI stays focused on orchestration.

## Data Flow
1. Config (`configs/cal_qEEG_all.json`) defines `paths` (flat `data_dir` or `bids_dir`), optional `preprocessing` (resample/filter/notch/montage/reference), a `power` block (band definitions + Welch overrides), an `entropy` block (permutation + spectral parameters), and QC metadata.
2. CLI resolves directories, discovers EEG files via the appropriate strategy, logs coverage (or exits if none and not dry-run).
3. For each file: load raw, run `preprocess_raw` (montage assignment, notch/bandpass filters, resampling, reference), collect metadata, compute enabled features -> individual pandas DataFrames.
4. `tidy_power_table` concatenates DataFrames, aligning columns. The CLI writes `qEEG_result.csv`.
5. `utils/QC.generate_qc_report` ingests metadata rows + tidy frame to produce the interactive QC HTML.

## Spectral Entropy Extension (v1.02)
The updated PRS introduces **Entity 4**: spectral entropy per channel. Implementation touches both utilities and the CLI.

### Scope of Impact
- `utils/entropy.py`: add `SpectralEntropyParams` dataclass and `compute_spectral_entropy` helper returning tidy frames.
- `code_01_qeeg.py`: extend supported feature flags, parse the `entropy.spectral` config, call the new helper, and merge results into the tidy dataset.
- `configs/cal_qEEG_all.json`: include an `entropy.spectral` section (method/nperseg/normalize/band label).
- `README.md`, `docs/CHANGELOG.md`, `docs/Proj_Planning.md`: document the new capability.

### Implementation Plan
1. **Utility Extension**
   - Introduce `@dataclass SpectralEntropyParams` mirroring AntroPy arguments (`method`, `nperseg`, `normalize`).
   - Implement `compute_spectral_entropy(raw, subject_id, *, params, picks=None, band_label="full") -> pd.DataFrame`.
   - Function will pick EEG channels, iterate over channel traces, apply `ant.spectral_entropy(trace, raw.info["sfreq"], ...)`, and build tidy rows.

   ```python
   def compute_spectral_entropy(raw, subject_id, *, params, picks=None, band_label="full"):
       params = SpectralEntropyParams.from_mapping(params)
       picks = picks or mne.pick_types(raw.info, eeg=True)
       data = raw.get_data(picks=picks)
       for idx, channel in enumerate(np.array(raw.ch_names)[picks]):
           value = ant.spectral_entropy(
               data[idx],
               sfreq=raw.info["sfreq"],
               method=params.method,
               nperseg=params.nperseg,
               normalize=params.normalize,
           )
           rows.append({
               "subject_id": subject_id,
               "channel": channel,
               "band": band_label,
               "metric": "spectral_entropy",
               "power": float(value),
           })
       return pd.DataFrame(rows)
   ```

2. **CLI Integration**
   - Update `SUPPORTED_FEATURES` and `_select_feature_flags` to include `spectral_entropy`.
    - Parse config via `config.get("entropy", {}).get("spectral")`; build params object and decide enablement.
   - Within processing loop, call `compute_spectral_entropy` when enabled and append to tidy frames.

3. **Config + Docs**
   - Expand `configs/cal_qEEG_all.json` with the new block.
   - Update README + Architecture/Changelog with references to the new feature and configuration knobs.

### File Updates
- `utils/entropy.py`
- `code_01_qeeg.py`
- `configs/cal_qEEG_all.json`
- `README.md`
- `docs/CHANGELOG.md`

## Preprocessing Enhancements (v1.04)
To satisfy the updated PRS optional parameters, the pipeline now exposes a unified preprocessing stage controllable from JSON configs.

### Scope of Impact
- `utils/basefun.py`: introduces `preprocess_raw` plus helpers for montage loading, notch/bandpass filters, resampling, and reference strategies. The absolute power calculator now assumes the input has already been referenced.
- `code_01_qeeg.py`: invokes `preprocess_raw` immediately after loading each recording so metadata and feature computations reflect the processed signal.
- `configs/cal_qEEG_all.json`, `README.md`, `docs/Project-Requirements-Specification.md`: document the new `preprocessing` block (resample/bandpass/notch/montage/reference) and default behaviors.

### Configuration Notes
```json
"preprocessing": {
  "resample_hz": 250,
  "bandpass": {"l_freq": 1, "h_freq": 40},
  "notch": {"freqs": [50, 100]},
  "montage": {"name": "standard_1020"},
  "reference": {"kind": "channels", "channels": ["M1", "M2"]}
}
```
- Omit keys to skip individual steps; referencing defaults to `average` unless `reference.kind` is set to `none`.
- `montage.path` resolves relative to the config file, enabling project-specific electrode layouts.
