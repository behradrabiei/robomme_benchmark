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

# SimpleMemVLA baseline (2026-11-16)

Second policy family on this clone: SimpleMemVLA (Qwen3.5-4B + ~0.9B DiT flow-matching action
head, PyTorch, in-process; memory = a 60 s / 2 fps front-camera video window of 120 frames,
4550 prompt tokens on RoboMME). Set up and smoke-tested only; no full run, no ablation code.

## Layout

| Path | What |
|---|---|
| `scripts/plus_eval/simplememvla/SimpleMemVLA/` | Policy repo in RoboMME's registered submodule slot, pinned `c564c17`. The local submodule URL is overridden in `.git/config` to `https://github.com/OpenBMB/SimpleMemVLA.git` (`.gitmodules` still names the `YinpeiDai` fork; same commit, and OpenBMB HEAD only adds README changes). RoboMME is vendored inside it at `robomme_sim/robomme/`. |
| `.venv-simplememvla/` | uv venv, Python 3.10 (8 GB). Not lock-managed; never `uv sync` into it. |
| `/mnt/vault/robomme/ckpts/simplememvla/simplememvla_robomme/` | `yinchenghust/simplememvla_robomme` (12.7 GB, one model for all 16 tasks; no base Qwen weights needed for eval). |
| `/mnt/vault/robomme/runs/simplememvla/<RUN_TAG>/` | `results.json`, `episodes.jsonl`, `eval.log`, `videos/`. |
| `/mnt/vault/robomme/cache/simplememvla/` | triton / torchinductor compile caches. |
| `run_simplememvla_eval.sh` | Standard RoboMME test split through upstream's `scripts/eval_robomme.sh`; `EVAL=plus` for the RoboMME-Plus adapter `scripts/plus_eval/simplememvla/eval_plus.py`. |
| `patches/simplememvla_local.diff` | Copy of the policy repo's local changes (`c564c17..local`, below). |

## Environment (deviations from upstream's README)

```bash
uv venv --seed --python 3.10 .venv-simplememvla
PY=$PWD/.venv-simplememvla/bin/python; SM=scripts/plus_eval/simplememvla/SimpleMemVLA
uv pip install -p $PY torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -p $PY -r <(grep -vE '^(torch|torchvision|deepspeed)' $SM/requirements.txt) transformers==5.13.1 \
  torch==2.9.1+cu128 torchvision==0.24.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
PYBIN=$PY bash $SM/scripts/install/install_robomme_sim.sh      # ManiSkill fork @ 07be6fbc, editable, in third_party/
uv pip install -p $PY sapien==3.0.3 numpy==1.26.4 opencv-python==4.11.0.86
uv pip install -p $PY --no-deps flash-linear-attention==0.2.2 && uv pip install -p $PY einops
```

- **torch 2.9.1+cu128 instead of the pinned 2.4.1**: 2.4.1 has no Blackwell (sm_120) kernels. Same
  torch / ManiSkill / numpy as the locked simulator env. transformers 5.13.1 is the version the
  checkpoint was saved with.
- **No flash-attn** (upstream's prebuilt wheel is for torch 2.4): attention runs with `sdpa`, as in
  the RoboMME team's own Plus run. **No causal-conv1d** (needs `nvcc`, not installed): torch fallback,
  so transformers still prints "The fast path is not available". `flash-linear-attention` is
  installed and its gated-delta kernels are used.
- conda's `libvulkan-loader` is not needed; the system Vulkan + NVIDIA ICD render SAPIEN.
- After the fresh submodule checkout, 16 `rmbench_sim/rmbench/description/*` files show as modified:
  upstream committed them as plain text under an LFS attribute. Not ours, not used by RoboMME.

## Local changes in the policy repo (branch `local`, copy in `patches/simplememvla_local.diff`)

With the default options none of them touches the model or its sampling; the model-side files are
upstream's. Items 5 and 6 are ablation options that are off unless set.

1. **Linear patch embedding** (`simplememvla/compat.py: apply_patch_embed_linear_compat`, called from
   `robomme_sim/__init__.py`). On torch 2.9.1 a bf16 `nn.Conv3d` with a bias takes ~700 ms for the
   256 patches of one 256×256 frame (6 ms without the bias, 0.02 ms as `F.linear`), and the eval embeds
   the wrist image plus 1–6 new video patches per decision. The patch embedding has kernel == stride,
   so it is the same linear map; vision features match the convolution to bf16 rounding (cosine
   ≥ 0.9997, relative L2 1 %). Decision time 4–6 s → 0.5 s; `SIMPLEMEMVLA_CONV3D_PATCH_EMBED=1`
   restores the convolution.
2. **Per-episode progress** (`robomme_sim/eval_success.py`): upstream prints nothing until a task
   finishes. Added one log line and one `episodes.jsonl` record per finished episode (task, episode,
   success, status, steps, n_calls, seed, group_size, wall_s).
3. **`--seed`** (`SEED=` in the run script; default 0 = upstream's hard-coded value, seeding torch and
   numpy once at start exactly as upstream does).
4. **Resume** (`--resume`, on by default): episodes already in `episodes.jsonl` are skipped and
   `results.json` covers recorded + new episodes; rerun the same command with the same `RUN_TAG`.
   `run_config.json` stores the settings that affect outcomes (checkpoint, split, seed, horizon, max
   steps, denoising steps, temperature, sub-task tokens, pipelined, dtype, attention) and a run dir
   refuses different ones; `--no-resume` starts the record afresh. Limitation, inherited from
   upstream's single global random stream: a resumed process starts that stream again, so the remaining
   episodes do not get the action noise an uninterrupted run would have given them.
5. **Context-window ablation** (`--context_frames K`, `CONTEXT_FRAMES=K` in the run script;
   `robomme_sim/policy.py`): the model's own history rule applied to a history that starts
   (K−1)·10 steps ago. Unset = upstream path. Recorded per episode and in the resume guard.
6. **Layer-depth ablation** (`--blind_layer L`, `BLIND_LAYER=L`; `robomme_sim/layer_blind.py`, two
   guarded lines in `robomme_sim/batched_policy.py`): from decoder layer L on no token reads a visual
   token. Unset = no hook installed. Not available with `--pipelined`.

## Checks run (seed 0 = upstream default, sdpa, non-pipelined, execute_horizon 16)

- `RUN_TAG=... TASKS=PickXtimes EPISODES_PER_TASK=10 GROUP_SIZE=4 ./run_simplememvla_eval.sh`
  → 10/10 with the original Conv3d (3495 s wall) and 10/10 with the linear patch embedding (303 s
  wall incl. ~80 s model load; peak 17.1 GB VRAM at group 4, 15.3 GB at group 2).
- Behavioural check of the linear patch embedding on a task below ceiling: StopCube, the same 20 test
  episodes, group 4, `SIMPLEMEMVLA_CONV3D_PATCH_EMBED=1` vs default → 15/20 on both paths, the same outcome
  on all 20 episodes, identical step counts on 16 and ±1–2 steps on the other 4 (mean 232.9 vs 232.7);
  2175 s vs 299 s wall (`runs/simplememvla/stopcube20_{conv3d,linear}/episodes.jsonl`).
- Resume: StopCube 12 episodes, group 4, killed after 6 recorded episodes, rerun → "resuming, 6/12
  already recorded", 6 run, 9/12; outcome and step count equal to an uninterrupted run on all 12
  episodes. A rerun with `SEED=1` into the same dir is refused. The evaluator from before the
  seed/resume change, rerun in the same session, also matches on all 12.
- Run-to-run variation exists at identical settings: across sessions episode 0 came out at 283 or 284
  steps and episode 11 as fail/215 or success/214 (StopCube is timing-critical, so one step can flip it).
- `TASKS="BinFill PickXtimes SwingXtimes StopCube" EPISODES_PER_TASK=2 GROUP_SIZE=2` → BinFill 2/2,
  PickXtimes 2/2, SwingXtimes 2/2, StopCube 1/2; 242 s wall for the 8 episodes.
- Per decision (1 env): 4550-token prefix 0.25 s, sub-task decode (7–11 tokens) 0.10–0.15 s, DiT
  0.06 s, vision 0.02 s → 0.45–0.5 s; simulator 0.09 s per 16 steps. The first decision of a fresh
  triton cache takes ~35 s (kernel compilation).
- RoboMME-Plus adapter: `EVAL=plus ./run_simplememvla_eval.sh --tasks BinFill --episodes 0 --output-dir ...`
  → runs end to end, `timeout` at 1877 control steps, 0 errors, 118 model calls at 0.79 s mean (the
  leaderboard entry for this model on Plus is 2 %). Its recorder wrote ~1 GB for one episode.

Not done: the full 16 × 50 standard run; runs at seeds other than 0.
For the ablations: the history window is rebuilt from the checkpoint `config.json` only
(`history_video_sec=60`, `history_video_fps=2`, `history_image_keys`; no `variable_history` key, so a
fixed clamp-padded window); eval has no CLI flag for it and the repo has no token-pruning code.
