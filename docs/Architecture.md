# Architecture Overview

## Context
`code_01_qeeg.py` is the primary CLI that loads JSON configs, discovers EEG recordings (either from a flat directory or a BIDS root), invokes feature calculators from `utils/`, and persists both a tidy CSV (`subject_id`, `channel`, `band`, `metric`, `power`) and a Plotly-backed QC report. The helper modules `utils/preprocessing.py` (loading + preprocessing), `utils/power.py` (Welch power features), `utils/entropy.py` (entropy family), `utils/microstate.py` (pycrostates-backed microstate analysis), and `utils/QC.py` (QC rendering) are the main extension points.

## High-Level Components
- **CLI Orchestrator**: Parses CLI flags (config, directory overrides including `--bids-dir`, feature filters, logging, dry-run), loads the JSON spec, resolves relative paths, and manages timestamped run folders (`result/<timestamp>/` containing CSV/QC/logs).
- **Preprocessing Layer**: `utils.preprocessing.preprocess_raw` loads montage metadata, executes notch/bandpass filters, resamples, and enforces the desired reference strategy before any feature calculators touch the data. Defaults to an average reference to preserve historical behavior.
- **Feature Execution Loop**: Loads each `.fif`/`.edf` via MNE, captures recording metadata, and conditionally runs absolute power, relative power, band-power ratios, permutation entropy, spectral entropy, and (optionally) microstate calculators based on config + CLI switches.
- **Reporting/Persistence**: `tidy_power_table` merges feature DataFrames; the CLI writes CSV + QC HTML with metadata coverage, per-feature histograms, and z-score-based status flags. Segmented runs also emit `qEEG_segment_result.csv`, a wide table keyed by (`subject_id`, `entity`, `channel`) with one column per chronological segment. Microstate runs write `microstate_result.csv` plus a dedicated `microstate_QC.html`.
- **Utilities**:
  - `utils/preprocessing.py` exposes raw loaders, montage/filtering/resampling helpers, and metadata summarization.
  - `utils/power.py` houses Welch PSD helpers plus absolute/relative/ratio band power calculators and tidy-frame utilities.
  - `utils/entropy.py` hosts AntroPy-backed entropy features. Initially only permutation entropy was supported; v1.02 adds spectral entropy with parameter containers for consistent config parsing.
  - `utils/microstate.py` loads ModKMeans templates via pycrostates, applies them to preprocessed recordings, tidies parameter/entropy/transition outputs, and renders a standalone microstate QC HTML.
  - `utils/QC.py` isolates the HTML/QC rendering helpers (`generate_qc_report`, histogram/table builders) so the CLI stays focused on orchestration.
  - `utils/config.py` standardizes JSON config loading and relative-path resolution.
  - `utils/discovery.py` encapsulates flat-directory and BIDS EEG discovery plus `RecordingDescriptor` normalization.
  - `utils/runtime.py` owns the timestamped output tree creation and logging configuration helpers used by the CLI.

## Data Flow
1. Config (`configs/cal_qEEG_all.json`) defines `paths` (flat `data_dir` or `bids_dir`), optional `preprocessing` (resample/filter/notch/montage/reference), a `power` block (enable flag, band definitions + Welch overrides), an `entropy` block (enable flag, permutation + spectral parameters), an optional `microstate` block (enable flag, template path, predict/metric knobs), an optional `Segment` block (length and bad tolerance), and QC metadata.
2. CLI resolves directories, discovers EEG files via the appropriate strategy, logs coverage (or exits if none and not dry-run).
3. For each file: load raw, run `preprocess_raw` (montage assignment, notch/bandpass filters, resampling, reference), collect metadata, compute enabled features -> individual pandas DataFrames. Microstate computation runs once per file on the full preprocessed recording (not segmented).
4. `tidy_power_table` concatenates power/entropy DataFrames, aligning columns. The CLI writes `qEEG_result.csv`.
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
- `utils/preprocessing.py`: introduces `preprocess_raw` plus helpers for montage loading, notch/bandpass filters, resampling, and reference strategies. The absolute power calculator now assumes the input has already been referenced.
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

## Segmented Feature Export (v1.05)
Segmented processing introduces per-window metrics when the `Segment` block is configured.

### Scope of Impact
- `code_01_qeeg.py` parses the `Segment` block into a `SegmentConfig`, logs enablement, and orchestrates per-window computations.
- `_build_segment_windows` in `utils/segment.py` slices each recording into contiguous windows equal to `Segment_length`, marking windows as invalid when the overlap with annotations containing `"bad"` exceeds `bad_segment_tolerance`.
- `_compute_segment_rows` (likewise in `utils/segment.py`) reuses existing feature helpers (absolute/relative/ratio power, permutation entropy, spectral entropy) on each valid window, storing results under an entity key that combines metric + band label.
- `_segment_rows_to_dataframe` pads row vectors to the global maximum number of windows and writes `qEEG_segment_result.csv` whenever segmentation is enabled.
- `configs/cal_qEEG_all.json`, `README.md`, and the PRS describe the new configuration knobs and resulting artifact.

## Band Ratios & QC Heatmaps (v1.06)
The PRS adds **Entity 3** (power ratios) plus a QC requirement to visualize segmented runs per subject.

### Scope of Impact
- `utils/power.py`: introduces `compute_power_ratios`, which pivots the absolute power table per subject/channel and evaluates numerator/denominator pairs defined by `power.ratio_bands`.
- `code_01_qeeg.py`: extends `SUPPORTED_FEATURES` with `power_ratio`, ensures the CLI auto-enables absolute power when ratios are requested, wires the ratio helper into both whole-record and segmented calculations, and plumbs segment DataFrames into the QC renderer.
- `utils/QC.py`: accepts the optional segment table, maps metric/band combinations back to their segment entities, and adds a subject dropdown + Plotly heatmap for each feature whenever segmentation is enabled.
- `configs/cal_qEEG_all.json`, `README.md`, and the PRS capture the new `ratio_bands` block and describe the QC dropdown/heatmap behavior.

## Microstate Analysis & Enable Gates (v1.07)
Adds pycrostates-based microstate support alongside explicit enable switches for the power and entropy stacks.

### Scope of Impact
- `utils/microstate.py`: parse microstate config, load ModKMeans templates, predict segmentations, tidy `compute_parameters()`, `entropy()`, and `compute_transition_matrix()` outputs, and render a dedicated QC HTML.
- `code_01_qeeg.py`: wire microstate config parsing, call microstate computation per recording (no segmentation), add microstate CSV/QC outputs, and respect `power.enable`/`entropy.enable` gates.
- `utils/runtime.py`: register microstate outputs in the run directory map.
- `configs/cal_qEEG_all.json`, `README.md`, `docs/Project-Requirements-Specification.md`: document the new config knobs, outputs, and version bump.
