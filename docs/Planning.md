# Prompt History

- 2025-11-25: Repository scaffolded per PRS instructions.
- 2025-11-30: BIDS ingestion plan approved 忌抗 1) extend CLI/config plumbing with `paths.bids_dir`/`--bids-dir`; 2) add BIDS-aware discovery helper that maps `sub-*/ses-*` EEG files to subject IDs; 3) document usage in README/PRS and sample configs.
- 2025-12-01: Preprocessing/montage/reference plan 忌抗 1) add a `preprocessing` block to configs covering resample/bandpass/notch knobs; 2) introduce a reusable helper that applies these ops plus montage + reference selection before feature extraction; 3) update CLI/docs/PRS to highlight the new optional parameters and version the spec.
- 2025-12-05: Config reorg plan 忌抗 1) review `configs/cal_qEEG_all.json` to capture existing parameter groupings; 2) restructure so top-level keys follow `paths`, `preprocessing`, `power`, `entropy` with legacy bands/Welch nested under power and entropy metrics consolidated; 3) propagate any schema or documentation updates needed after the reordering.
- 2025-12-08: Segmentation feature plan 忌抗 1) parse the new `Segment` config block (length + bad tolerance) and expose flags to the CLI; 2) chunk EEG recordings, evaluate bad segments, compute entity metrics per segment, and emit `qEEG_segment_result.csv`; 3) document the new behavior across Architecture, PRS version log, and README.
