# ReGAT-BERT score-fusion ablations

The implementation plans are `FUSION_ABLATION_PLAN.md` and its extension `MULTIHOP_EXTENSION_PLAN.md`; follow them literally. Deviations and VERIFY outcomes
are logged in `NOTES.md` (log new ones there before continuing). How to run on SLURM: `RUNNING.md`.

- Code: `src/` (run modules as `python -m src.<module>` from the repo root), tests: `tests/`, configs: `configs/`.
- Experiment keys: `A`/`B` (MS MARCO), `<dataset>`, `<dataset>@<variant>` (see `src/common.py`); masks handle variable pool sizes.
- All hyperparameters live in `configs/*.yaml`; `REGAT_PROFILE=smoke` overlays `configs/smoke.yaml`.
- Never train the cross-encoder / bi-encoder; never retrain the GAT after Stage 3; never use `test`/TREC for decisions.
- Storage: only the frozen GATs go to the HF Hub (`src/hub.py`, `configs/hub.yaml`); data and caches are never committed or uploaded (rebuilt / re-downloaded). See `README.md`.
- Env: `~/.conda/envs/regat` (Python 3.11 + OpenJDK 21), created by `slurm/setup_env.sh`.
