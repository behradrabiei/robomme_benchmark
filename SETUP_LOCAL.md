# RoboMME local setup (RTX 5090 host, 2026-11-02)

Local notes for this clone (RoboMME/robomme_benchmark @ 5753639). Nothing in the LongNav conda
environments was touched. Large files live under `/mnt/vault/robomme`.

## Layout

| Path | What |
|---|---|
| `.venv/` | Simulator env, exactly `uv.lock` (`uv sync --group server --extra dev --python 3.11`). Use `uv run` here. |
| `.venv-mme-eval/` | Eval-client env: the same locked simulator stack + `openpi-client` + the policy repo's `examples/robomme/requirements.txt`. Used by `run_mme_vla_eval.sh`. Not lock-managed: never `uv sync` into it. |
| `scripts/plus_eval/robomme_vla/robomme_policy_learning/` | MME-VLA policy repo, submodule pinned at ac846a9. Its `.venv/` is `uv sync --frozen --no-dev --python 3.11` (jax 0.5.3 cuda12, torch 2.7.1) plus `pytest==8.4.2`. |
| `/mnt/vault/robomme/ckpts/mme_vla_suite/<model>/{79999/,history_config.txt}` | Checkpoints (unzipped with `scripts/unzip_ckpt.py`, zip removed). Only `perceptual-framesamp-modul` (12 GB) is downloaded. |
| `/mnt/vault/robomme/runs/` | Eval outputs. `runs` in this repo and `runs/ckpts` in the policy repo are symlinks into the vault. |
| `/mnt/vault/robomme/openpi_home` | `OPENPI_DATA_HOME` (openpi asset cache). |
| `run_mme_vla_eval.sh` | Serve one checkpoint + drive the RoboMME test-split evaluation (or `EVAL=plus` for RoboMME-Plus). |

`uv` 0.13.6 is installed at `~/.local/bin/uv` (add to PATH).

## Why two simulator envs

`uv run` re-syncs `.venv` to `uv.lock` on every call. The policy repo's eval client needs
transformers 4.57.3 (huggingface-hub < 1.0) while the lock pins huggingface-hub 1.4.1, and
`ms_swift` ships a top-level `tests` package that shadows this repo's `tests/`. Installing the
client extras into `.venv` therefore breaks the moment `uv run` is used, and breaks pytest. The
extras live in `.venv-mme-eval` instead, built from `uv export --frozen --no-dev --group server`
then the extras on top. Deviations from upstream's `requirements.txt` in that env:

- `git+...ManiSkill.git@dev` line dropped: the lock's pinned fork rev 07be6fb is used instead.
- `pyarrow<21` (20.0.0): newer pyarrow refuses numpy 1.26.4, which the lock pins.
- huggingface-hub resolves to 0.36.2 (lock has 1.4.1) because of transformers.

## Hardware notes

- SAPIEN renders on the GPU through Vulkan (NVIDIA ICD present, `DISPLAY=:1`); no `SAPIEN_RENDER_DEVICE=cpu` needed.
- JAX 0.5.3 + locked cuDNN 9.5.1 / nvcc 12.9 run on Blackwell (sm_120): bf16 matmul and the cuDNN conv path
  both work; the first conv compile autotunes slowly (~15 s), then is cached.
- Server + simulator share GPU 0 (`XLA_PYTHON_CLIENT_PREALLOCATE=false`): ~18.6 GB of the 32 GB in use during eval.
- No system `ffmpeg`; `imageio[ffmpeg]` provides the binary for mp4 writing.

## Checks run

- `uv run scripts/run_example.py` (PickXtimes, joint_angle) and `--action-space-type waypoint --task-id MoveCube`: videos written, ~6 s each.
- `uv run python -m pytest tests/lightweight/ -q`: 733 passed, 47 skipped, 7 failed. The 7 are upstream drift, not
  environment: 3 need the uninitialised `3-tier-astra` submodule (`api_client`), 2 `test_TaskGoal` compare against
  old goal wording ("back and forth" vs "back-and-forth"), 2 `test_step_error_handling` AST checks no longer match
  `DemonstrationWrapper.step()`.
- End to end: `SAVE_DIR=/mnt/vault/robomme/runs/smoke_eval ./run_mme_vla_eval.sh perceptual-framesamp-modul 7 --args.only_tasks=PickXtimes`
  → PickXtimes 50/50 test episodes: SR 0.88 (seed 7; the model card reports a 44.51 overall average across the 16 tasks, per-task numbers are in the paper),
  6.8 min wall-clock for the 49 episode gaps, i.e. ~8.3 s/episode including model calls and video writing.
- RoboMME-Plus adapter smoke: `EVAL=plus ./run_mme_vla_eval.sh perceptual-framesamp-modul 7 --tasks BinFill --episodes 0 --output-dir ...`
  → runs cleanly end to end: BinFill ep0 `fail` after 891 control steps (budget 1877), 0 errors, 56 model calls at 0.32 s mean latency, `summary.json`/`result.json`/`episode.mp4` written. (Plus is much harder than RoboMME; the leaderboard entry for this model is low.)

## Reproducing the paper's main table

```bash
export PATH=$HOME/.local/bin:$PATH
cd robomme_benchmark
for SEED in 7 42 0; do ./run_mme_vla_eval.sh perceptual-framesamp-modul $SEED; done
```

Each seed is 16 tasks × 50 test episodes with `max_steps=1300`, resumable via `progress.json`;
`log.json` holds the per-task and overall success rates when the seed finishes. PickXtimes is a
fast task; longer tasks (1300-step timeouts, video demos) will take more per episode, so budget
roughly 2–13 h per seed (8 s/ep observed on PickXtimes; a 1300-step timeout episode is ~1 min). Other checkpoints: download
`Yinpei/mme_vla_suite --include '<model>/*'`, unzip with `scripts/unzip_ckpt.py`, pass the model
name as the first argument. Symbolic variants additionally need `Yinpei/vlm_subgoal_predictor`
and the `--args.use_qwenvl/--args.subgoal_type` flags (see `scripts/eval.sh` in the policy repo).

Training data (56 GB, `Yinpei/robomme_data_h5`) was intentionally not downloaded; it is only
needed for `scripts/dataset_replay.py` and for training.
