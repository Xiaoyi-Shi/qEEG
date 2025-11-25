# Repository Guidelines

## Project Structure & Module Organization
- `code_01_qeeg.py` exposes the CLI pipeline that parses configs, loads EEG files, dispatches feature handlers, and writes QC reports.
- `configs/*.json` store feature definitions and PSD parameters; keep shared defaults such as `cal_qEEG_all.json` here.
- `data/EEG_DATA/` contains raw `.fif`/`.edf` inputs and is ignored by git—never commit patient data.
- `utils/basefun.py` and `utils/entropy.py` house reusable feature calculators (band power and permutation entropy). Extend them before wiring new handlers into the CLI registry.
- `result/` and its timestamped run folders hold generated CSVs and HTML reports; `docs/` captures planning, architecture, and change logs.

## Build, Test, and Development Commands
- The current conda virtual environment(F:\ProgramData\anaconda3\envs\mne_1.9.0\python.exe) has all the required libraries installed, use this virtual environment to run python script.
- Run the pipeline on actual data: `python code_01_qeeg.py --config configs/cal_qEEG_all.json --data-dir data/EEG_DATA --result-dir result`.
- Exercise logic without loading EEG data: `python code_01_qeeg.py --dry-run --log-level DEBUG`.
- To focus on one feature, append `--feature absolute_power` (repeat per feature).

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, snake_case functions, and descriptive module-level constants.
- Maintain type hints and concise docstrings like those in `code_01_qeeg.py` for every public function.
- Use structured logging via `logging` (INFO for orchestrations, DEBUG inside feature loops).
- Keep configuration keys lower_snake_case and mirror JSON schema inside the utils modules.

## Testing Guidelines
- No automated suite exists yet; rely on `--dry-run` for fast validation and full runs on a curated EEG sample to confirm feature math.
- Compare CSV summaries against QC HTML sections to spot anomalies.
- When adding handlers, create synthetic Raw objects in notebooks or scripts under `docs/` to regress expected band outputs.

## Commit & Pull Request Guidelines
- The history shows short, imperative summaries (`Init`), so prefer messages such as “Add permutation entropy handler” with focused scopes.
- Include PR descriptions covering motivation, configs touched, and sample command invocations (paste console snippet or QC path).
- Link the relevant doc updates (e.g., `docs/Architecture.md`) and include screenshots of new QC sections when UI changes occur.

## Security & Configuration Tips
- Treat everything under `data/` as PHI-adjacent; scrub identifiers before sharing artifacts and avoid uploading raw files.
- Store per-site secrets outside the repo; configs should reference environment variables or placeholders, not absolute private paths.
