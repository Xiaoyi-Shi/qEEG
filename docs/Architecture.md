# Architecture Overview

## Context
`code_01_qeeg.py` is the primary CLI that loads JSON configs, discovers EEG recordings, invokes feature calculators from `utils/`, and persists both a tidy CSV (`subject_id`, `channel`, `band`, `metric`, `power`) and a Plotly-backed QC report. The helper modules `utils/basefun.py` (power features) and `utils/entropy.py` (entropy family) are the main extension points.

## High-Level Components
- **CLI Orchestrator**: Parses CLI flags (config, directory overrides, feature filters, logging, dry-run), loads the JSON spec, resolves relative paths, and manages timestamped run folders (`result/<timestamp>/` containing CSV/QC/logs).
- **Feature Execution Loop**: Loads each `.fif`/`.edf` via MNE, captures recording metadata, and conditionally runs absolute power, relative power, permutation entropy, and spectral entropy calculators based on config + CLI switches.
- **Reporting/Persistence**: `tidy_power_table` merges feature DataFrames; the CLI writes CSV + QC HTML with metadata coverage, per-feature histograms, and z-score-based status flags.
- **Utilities**:
  - `utils/basefun.py` exposes Welch PSD helpers, absolute and relative band power calculators, metadata summarization, and tidy-frame helpers.
  - `utils/entropy.py` hosts AntroPy-backed entropy features. Initially only permutation entropy was supported; v1.02 adds spectral entropy with parameter containers for consistent config parsing.

## Data Flow
1. Config (`configs/cal_qEEG_all.json`) defines `paths`, band definitions, Welch overrides, permutation entropy parameters (`entropy` section), spectral entropy parameters (`spectral_entropy`), and QC metadata.
2. CLI resolves directories, discovers EEG files, logs coverage (or exits if none and not dry-run).
3. For each file: load raw, collect metadata, compute enabled features -> individual pandas DataFrames.
4. `tidy_power_table` concatenates DataFrames, aligning columns. The CLI writes `qEEG_result.csv`.
5. `generate_qc_report` ingests metadata rows + tidy frame to produce the interactive QC HTML.

## Spectral Entropy Extension (v1.02)
The updated PRS introduces **Entity 4**: spectral entropy per channel. Implementation touches both utilities and the CLI.

### Scope of Impact
- `utils/entropy.py`: add `SpectralEntropyParams` dataclass and `compute_spectral_entropy` helper returning tidy frames.
- `code_01_qeeg.py`: extend supported feature flags, parse `spectral_entropy` config, call the new helper, and merge results into the tidy dataset.
- `configs/cal_qEEG_all.json`: include a `spectral_entropy` section (method/nperseg/normalize/band label).
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
   - Parse config via `config.get("spectral_entropy")`; build params object and decide enablement.
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
