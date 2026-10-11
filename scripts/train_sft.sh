#!/usr/bin/env bash
set -euo pipefail

# Multi-node LoRA SFT for PortraitAes. Run the same command on every node.
# Required environment: MODEL_PATH and DATASET_PATH.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

if (( $# < 1 )); then
    echo "Usage: MODEL_PATH=... DATASET_PATH=... $0 <run-name> [extra ms-swift args...]" >&2
    exit 2
fi
RUN_NAME="$1"
shift
EXTRA_ARGS=("$@")

MODEL_PATH="${MODEL_PATH:-}"
DATASET_PATH="${DATASET_PATH:-}"
require_value MODEL_PATH "$MODEL_PATH"
require_file DATASET_PATH "$DATASET_PATH"
configure_distributed

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/sft/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/sft/${RUN_NAME}}"
AUTO_RESUME="${AUTO_RESUME:-true}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
LOG_FILE="${LOG_DIR}/node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"

RESUME_ARGS=()
# Resume only when every DeepSpeed rank has written its checkpoint state.
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    is_complete_checkpoint "$RESUME_CHECKPOINT" || die "Incomplete checkpoint: $RESUME_CHECKPOINT"
    RESUME_ARGS=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
elif [[ "$AUTO_RESUME" == "true" ]]; then
    LATEST_CHECKPOINT="$(latest_complete_checkpoint "$OUTPUT_DIR")"
    if [[ -n "$LATEST_CHECKPOINT" ]]; then
        echo "[INFO] Resuming from $LATEST_CHECKPOINT"
        RESUME_ARGS=(--resume_from_checkpoint "$LATEST_CHECKPOINT")
    fi
fi

CMD=(
    torchrun
    --nnodes="$NNODES"
    --nproc_per_node="$NPROC_PER_NODE"
    --node_rank="$NODE_RANK"
    --master_addr="$MASTER_ADDR"
    --master_port="$MASTER_PORT"
    --module portraitaes.sft.launcher
    --model "$MODEL_PATH"
    --dataset "$DATASET_PATH"
    --split_dataset_ratio "${SFT_SPLIT_DATASET_RATIO:-0.05}"
    --add_non_thinking_prefix true
    --loss_scale ignore_empty_think
    --load_from_cache_file false
    --train_type lora
    --torch_dtype bfloat16
    --num_train_epochs "${SFT_NUM_TRAIN_EPOCHS:-3}"
    --per_device_train_batch_size "${SFT_BATCH_SIZE:-1}"
    --per_device_eval_batch_size "${SFT_EVAL_BATCH_SIZE:-1}"
    --attn_impl "${SFT_ATTN_IMPL:-flash_attn}"
    --learning_rate "${SFT_LEARNING_RATE:-4e-5}"
    --lora_rank "${SFT_LORA_RANK:-64}"
    --lora_alpha "${SFT_LORA_ALPHA:-128}"
    --target_modules all-linear
    --freeze_vit true
    --freeze_aligner true
    --gradient_checkpointing true
    --vit_gradient_checkpointing false
    --gradient_accumulation_steps "${SFT_GRAD_ACCUM_STEPS:-2}"
    --output_dir "$OUTPUT_DIR"
    --eval_steps "${SFT_EVAL_STEPS:-1500}"
    --save_steps "${SFT_SAVE_STEPS:-250}"
    --save_total_limit "${SFT_SAVE_TOTAL_LIMIT:-1000}"
    --logging_steps "${SFT_LOGGING_STEPS:-10}"
    --max_length "${MAX_LENGTH:-16384}"
    --warmup_ratio "${SFT_WARMUP_RATIO:-0.05}"
    --deepspeed "${SFT_DEEPSPEED:-zero3_offload}"
    --dataset_num_proc "${DATASET_NUM_PROC:-64}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}"
    --group_by_length false
    --model_name "$RUN_NAME"
    "${RESUME_ARGS[@]}"
    "${EXTRA_ARGS[@]}"
)

echo "[INFO] run=$RUN_NAME world_size=$WORLD_SIZE node_rank=$NODE_RANK"
echo "[INFO] output=$OUTPUT_DIR log=$LOG_FILE"
printf '[INFO] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
MAX_PIXELS="${MAX_PIXELS:-1254400}" \
USE_JSONL_INMEMORY_LOADER="${USE_JSONL_INMEMORY_LOADER:-0}" \
DISABLE_HF_DATASETS_CACHE="${DISABLE_HF_DATASETS_CACHE:-1}" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
