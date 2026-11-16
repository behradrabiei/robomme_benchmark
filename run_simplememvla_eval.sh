#!/usr/bin/env bash
# Local recipe: run SimpleMemVLA (OpenBMB/SimpleMemVLA @ c564c17, PyTorch, in-process) on the
# standard RoboMME test split with upstream's own evaluator (scripts/eval_robomme.sh ->
# robomme_sim/eval_success.py; RoboMME is vendored inside the policy repo) on the single RTX 5090.
#
#   ./run_simplememvla_eval.sh [extra eval_success.py args...]
#   e.g. TASKS=PickXtimes EPISODES_PER_TASK=2 ./run_simplememvla_eval.sh
#        TASKS="BinFill PickXtimes SwingXtimes StopCube" ./run_simplememvla_eval.sh
#
# Results: /mnt/vault/robomme/runs/simplememvla/<RUN_TAG>/{results.json,episodes.jsonl,run_config.json,eval.log,videos/}
# Local additions on the policy repo's `local` branch: one "[eval] <task> ep<i>: <status> steps=..
# calls=.." line and one episodes.jsonl record per finished episode; SEED (default 0, upstream's
# value); resume.
# Resumable: rerun the same command with the same RUN_TAG; episodes already in episodes.jsonl are
# skipped and results.json covers recorded + new episodes. A run dir only accepts the settings in
# its run_config.json (checkpoint, seed, horizon, ...); pass --no-resume to start it afresh.
#
# EVAL=plus switches to the RoboMME-Plus adapter (scripts/plus_eval/simplememvla/eval_plus.py);
# extra args are then its options, e.g.  EVAL=plus ./run_simplememvla_eval.sh \
#   --tasks BinFill --episodes 0 --output-dir /mnt/vault/robomme/runs/simplememvla/plus-smoke
#
# Env overrides: CHECKPOINT GPU SEED TASKS EPISODES_PER_TASK GROUP_SIZE RUN_TAG RUN_DIR SAVE_DIR
# ATTN_IMPLEMENTATION, plus every variable upstream's scripts/eval_robomme.sh reads
# (MAX_STEPS, EXECUTE_HORIZON, PIPELINED, VIDEO_MAX_PER_TASK, ...).
# The env runs torch 2.9.1+cu128 (upstream pins 2.4.1, which has no RTX 5090 kernels), so
# flash-attn is not installed and attention defaults to sdpa; the vision patch embedding runs as
# a linear map (SIMPLEMEMVLA_CONV3D_PATCH_EMBED=1 restores the ~100x slower Conv3d). See SETUP_LOCAL.md.
set -euo pipefail

ROBOMME_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_REPO="$ROBOMME_ROOT/scripts/plus_eval/simplememvla/SimpleMemVLA"
PYBIN="$ROBOMME_ROOT/.venv-simplememvla/bin/python"
CHECKPOINT="${CHECKPOINT:-/mnt/vault/robomme/ckpts/simplememvla/simplememvla_robomme}"
SAVE_DIR="${SAVE_DIR:-/mnt/vault/robomme/runs/simplememvla}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

[[ -x "$PYBIN" ]] || { echo "missing env: $PYBIN" >&2; exit 1; }
[[ -f "$POLICY_REPO/robomme_sim/eval_success.py" ]] || { echo "submodule not initialised: $POLICY_REPO" >&2; exit 1; }
[[ -f "$CHECKPOINT/model.safetensors" ]] || { echo "missing checkpoint: $CHECKPOINT/model.safetensors" >&2; exit 1; }

# keep compile caches off the (nearly full) home disk
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/vault/robomme/cache/simplememvla/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/mnt/vault/robomme/cache/simplememvla/torchinductor}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

if [[ "${EVAL:-robomme}" == "plus" ]]; then
  echo "[eval:plus] checkpoint=$CHECKPOINT attn=$ATTN_IMPLEMENTATION extra: $*"
  cd "$ROBOMME_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYBIN" scripts/plus_eval/simplememvla/eval_plus.py \
    --checkpoint "$CHECKPOINT" --attn-implementation "$ATTN_IMPLEMENTATION" "$@"
else
  RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
  export RUN_DIR="${RUN_DIR:-$SAVE_DIR/$RUN_TAG}"
  export PYBIN CHECKPOINT GPU ATTN_IMPLEMENTATION
  export NUM_GPUS=1 GROUP_SIZE="${GROUP_SIZE:-2}"
  echo "[eval] checkpoint=$CHECKPOINT seed=$SEED tasks=${TASKS:-<all 16>} episodes/task=${EPISODES_PER_TASK:-50} group=$GROUP_SIZE attn=$ATTN_IMPLEMENTATION"
  exec bash "$POLICY_REPO/scripts/eval_robomme.sh" --seed "$SEED" "$@"
fi
