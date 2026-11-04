#!/usr/bin/env bash
# Local recipe: serve one MME-VLA checkpoint (JAX, policy-learning venv) and drive the
# RoboMME test-split evaluation (simulator venv) against it on the single RTX 5090.
#
#   ./run_mme_vla_eval.sh [MODEL=perceptual-framesamp-modul] [SEED=7] [extra eval.py args...]
#   e.g. ./run_mme_vla_eval.sh perceptual-framesamp-modul 7 --args.only_tasks=PickXtimes
#
# Results: /mnt/vault/robomme/runs/evaluation/<MODEL>/ckpt79999/seed<SEED>/{progress.json,log.json,videos/}
# Resumable: rerun the same command; eval.py skips episodes recorded in progress.json.
#
# EVAL=plus switches the client to the RoboMME-Plus adapter (scripts/plus_eval/robomme_vla/eval_plus.py);
# extra args are then its options, e.g.  EVAL=plus ./run_mme_vla_eval.sh perceptual-framesamp-modul 7 \
#   --tasks BinFill --episodes 0 --output-dir /mnt/vault/robomme/runs/framesamp-modul-plus
# Env overrides: PORT GPU CKPT_ROOT CKPT_ID SAVE_DIR OPENPI_DATA_HOME.
# Ablations (paper/robomme_ablation): ABLATION='<json>' is passed to the server as --ablation and
# ARM=<name> replaces the model name in the results path (SAVE_DIR/ARM/ckpt<ID>/seed<SEED>);
# the server then logs its per-call memory selection to policy_calls.jsonl in that directory.
set -euo pipefail

MODEL="${1:-perceptual-framesamp-modul}"; shift || true
SEED="${1:-7}"; shift || true

ROBOMME_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_REPO="$ROBOMME_ROOT/scripts/plus_eval/robomme_vla/robomme_policy_learning"
SIM_PYTHON="$ROBOMME_ROOT/.venv-mme-eval/bin/python"   # simulator stack + openpi-client + policy-repo eval extras
VLA_PYTHON="$POLICY_REPO/.venv/bin/python"
CKPT_ROOT="${CKPT_ROOT:-/mnt/vault/robomme/ckpts/mme_vla_suite}"
CKPT_ID="${CKPT_ID:-79999}"
SAVE_DIR="${SAVE_DIR:-/mnt/vault/robomme/runs/evaluation}"
PORT="${PORT:-18763}"
GPU="${GPU:-0}"
ARM="${ARM:-$MODEL}"
ABLATION="${ABLATION:-}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/vault/robomme/openpi_home}"

CKPT_DIR="$CKPT_ROOT/$MODEL/$CKPT_ID"
[[ -d "$CKPT_DIR/params" ]] || { echo "missing checkpoint: $CKPT_DIR/params" >&2; exit 1; }

mkdir -p "$SAVE_DIR"
SERVER_LOG="$SAVE_DIR/${ARM}_seed${SEED}_server.log"
SERVER_ARGS=(--seed="$SEED" --port="$PORT")
ABL_LOG=""
if [[ -n "$ABLATION" ]]; then
  SERVER_ARGS+=(--ablation="$ABLATION")
  ABL_LOG="$SAVE_DIR/$ARM/ckpt$CKPT_ID/seed$SEED/policy_calls.jsonl"
  mkdir -p "$(dirname "$ABL_LOG")"
  echo "[server] ablation: $ABLATION"
fi

echo "[server] $MODEL arm=$ARM seed=$SEED port=$PORT gpu=$GPU -> $SERVER_LOG"
(cd "$POLICY_REPO" && exec env CUDA_VISIBLE_DEVICES="$GPU" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MME_ABLATION_LOG="$ABL_LOG" \
  PYTHONPATH="$POLICY_REPO/src:$POLICY_REPO/packages/openpi-client/src" \
  "$VLA_PYTHON" scripts/serve_policy.py "${SERVER_ARGS[@]}" \
  policy:checkpoint --policy.config=mme_vla_suite --policy.dir="$CKPT_DIR") \
  > "$SERVER_LOG" 2>&1 &
VLA_PID=$!
trap 'echo "[server] stopping pid $VLA_PID"; kill "$VLA_PID" 2>/dev/null || true' EXIT

# wait for the websocket port (checkpoint load takes a while)
for _ in $(seq 1 600); do
  if ! kill -0 "$VLA_PID" 2>/dev/null; then echo "server died; see $SERVER_LOG" >&2; exit 1; fi
  if "$SIM_PYTHON" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(s.connect_ex(('127.0.0.1',$PORT)))" 2>/dev/null; then break; fi
  sleep 2
done

if [[ "${EVAL:-robomme}" == "plus" ]]; then
  echo "[eval:plus] model_seed=$SEED extra: $*"
  cd "$ROBOMME_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" VLA_CHECKPOINT="$CKPT_DIR" "$SIM_PYTHON" \
    scripts/plus_eval/robomme_vla/eval_plus.py --host 127.0.0.1 --port "$PORT" --model-seed "$SEED" "$@"
else
  echo "[eval] model_seed=$SEED only_tasks/extra: $*"
  cd "$POLICY_REPO/examples/robomme"
  CUDA_VISIBLE_DEVICES="$GPU" "$SIM_PYTHON" eval.py \
    --args.host=127.0.0.1 --args.port="$PORT" \
    --args.policy_name="$ARM" --args.model_ckpt_id="$CKPT_ID" --args.model_seed="$SEED" \
    --args.save_dir="$SAVE_DIR" "$@"
fi
