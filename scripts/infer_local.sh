#!/usr/bin/env bash
set -euo pipefail

# Sharded data-parallel inference with a merged Transformers model.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if (( $# < 1 )); then
    echo "Usage: MODEL_PATH=... GPU_IDS=0,1 $0 <input.jsonl> [extra inference args...]" >&2
    exit 2
fi
INPUT_JSONL="$1"
shift
EXTRA_ARGS=("$@")
MODEL_PATH="${MODEL_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0}"

[[ -f "$INPUT_JSONL" ]] || { echo "[ERROR] Input does not exist: $INPUT_JSONL" >&2; exit 1; }
[[ -n "$MODEL_PATH" ]] || { echo "[ERROR] MODEL_PATH is required" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "[ERROR] Python not found: $PYTHON_BIN" >&2; exit 1; }

RUN_NAME="${RUN_NAME:-$(basename "${INPUT_JSONL%.jsonl}")_$(date -u +%Y%m%d%H%M)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/inference/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/inference/${RUN_NAME}}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS//[[:space:]]/}"
NUM_SHARDS="${#GPU_ARRAY[@]}"
(( NUM_SHARDS > 0 )) || { echo "[ERROR] GPU_IDS cannot be empty" >&2; exit 1; }

PIDS=()
for ((SHARD_ID=0; SHARD_ID<NUM_SHARDS; SHARD_ID++)); do
    GPU="${GPU_ARRAY[$SHARD_ID]}"
    OUTPUT_JSONL="${OUTPUT_DIR}/shard_${SHARD_ID}.jsonl"
    LOG_FILE="${LOG_DIR}/shard_${SHARD_ID}.log"
    echo "[INFO] shard=$SHARD_ID/$NUM_SHARDS GPU=$GPU output=$OUTPUT_JSONL"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m portraitaes.inference.local \
        --input-jsonl "$INPUT_JSONL" \
        --output-jsonl "$OUTPUT_JSONL" \
        --model "$MODEL_PATH" \
        --num-shards "$NUM_SHARDS" \
        --shard-id "$SHARD_ID" \
        --resume \
        "${EXTRA_ARGS[@]}" \
        >"$LOG_FILE" 2>&1 &
    PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAILED=1
done
(( FAILED == 0 )) || { echo "[ERROR] One or more inference shards failed; inspect $LOG_DIR" >&2; exit 1; }
echo "[INFO] all shards complete: $OUTPUT_DIR"
