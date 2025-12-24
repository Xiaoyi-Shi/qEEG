# Repository Guidelines

## Project Structure & Module Organization
The entry point `code_01_qeeg.py` orchestrates discovery, feature extraction, and QC exports. Configuration JSON lives in `configs/`, reusable signal utilities sit in `utils/` (`preprocessing.py` for loading/montage/filtering, `power.py` for Welch-based band metrics, `entropy.py` for perm/spectral metrics, `QC.py` for HTML generation), and contributor docs are stored in `docs/Architecture.md` plus related planning notes. Each config now carries a `preprocessing` block that controls resampling, bandpass/notch filters, montage, and reference defaults—keep this in sync with new signal-processing requirements. Drop raw recordings under `data/EEG_DATA/` (git-ignored) and inspect artifacts in `result/<timestamp>/` where CSVs, QC.html, and logs are written for each run.

## Build, Test, and Development Commands
```bash
conda activate mne_1.11.0   # activate env (In this project, the conda env has been activated)
python code_01_qeeg.py --config configs/cal_qEEG_all.json               # full pipeline
python code_01_qeeg.py --config configs/cal_qEEG_all.json --dry-run     # discovery/log smoke test
python code_01_qeeg.py --feature absolute_power --log-level DEBUG       # scoped debugging run
```
Use `--data-dir` / `--result-dir` to point at alternate staging areas and supply `--feature` repeatedly to mix metrics.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation, ≤99-character lines, and type-hinted functions (see existing signatures). Stick to snake_case for functions/variables, UpperCamelCase for parameter containers, and CONSTANT_CASE for module-level tuples. Ingest paths via `pathlib.Path`, feed structured logs through the module-level logger, and keep functions pure—return DataFrames or dicts instead of mutating globals.

## Testing Guidelines
There is no pytest suite yet; rely on deterministic pipeline runs. Validate changes by executing a `--dry-run` first, then a focused feature pass using a small `.fif` sample to ensure `result/<timestamp>/qEEG_result.csv` populates expected bands/metrics. Review the generated QC.html and `log/pipeline.log` for warnings, and attach notable log excerpts or screenshots when reporting regressions. When adding utilities, include lightweight data-shape assertions and unit-level helpers inline until a dedicated `tests/` folder lands.

## Commit & Pull Request Guidelines
Recent history favors short, imperative summaries (e.g., “Add QC module”, “rebuild arch”). Keep commits scoped to one concern, include rationale in the body if behavior changes, and reference configs or docs you touched. Pull requests should describe the dataset or configs used, list validation commands executed, link GitHub issues or tickets, and attach QC screenshots if the HTML changed. Update `docs/Architecture.md` and `docs/Project-Requirements-Specification.md` whenever responsibilities shift or new outputs are introduced.

## Data Security & Configuration Tips
Never commit PHI: `data/` and `result/` stay local. Store secrets (hospital IDs, UNC shares) in environment variables or local config overrides that remain untracked. Keep template configs generic, and capture sensitive deployment notes in `docs/Planning.md` instead of code comments or sample JSON.
