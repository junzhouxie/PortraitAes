#!/usr/bin/env bash
set -euo pipefail

# Multi-node GRPO fine-tuning for PortraitAes.
# Required environment: MODEL_PATH, SFT_ADAPTER, and DATASET_PATH.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

if (( $# < 1 )); then
    echo "Usage: MODEL_PATH=... SFT_ADAPTER=... DATASET_PATH=... $0 <run-name> [extra ms-swift args...]" >&2
    exit 2
fi
RUN_NAME="$1"
shift
EXTRA_ARGS=("$@")

MODEL_PATH="${MODEL_PATH:-}"
SFT_ADAPTER="${SFT_ADAPTER:-}"
DATASET_PATH="${DATASET_PATH:-}"
REWARD_PLUGIN="${REWARD_PLUGIN:-${PROJECT_ROOT}/src/portraitaes/grpo/reward_plugin.py}"
require_value MODEL_PATH "$MODEL_PATH"
require_dir SFT_ADAPTER "$SFT_ADAPTER"
require_file DATASET_PATH "$DATASET_PATH"
require_file REWARD_PLUGIN "$REWARD_PLUGIN"
configure_distributed

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/grpo/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/grpo/${RUN_NAME}}"
AUTO_RESUME="${AUTO_RESUME:-true}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
LOG_FILE="${LOG_DIR}/node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"

# Stable defaults from the released experiment. Every value can be overridden.
GRPO_LEARNING_RATE="${GRPO_LEARNING_RATE:-2e-6}"
GRPO_LR_SCHEDULER="${GRPO_LR_SCHEDULER:-constant_with_warmup}"
GRPO_WARMUP_RATIO="${GRPO_WARMUP_RATIO:-0.05}"
GRPO_LOSS_TYPE="${GRPO_LOSS_TYPE:-dr_grpo}"
GRPO_SCALE_REWARDS="${GRPO_SCALE_REWARDS:-batch}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-8}"
GRPO_BETA="${GRPO_BETA:-0.02}"
GRPO_EPSILON="${GRPO_EPSILON:-0.12}"
GRPO_MAX_GRAD_NORM="${GRPO_MAX_GRAD_NORM:-0.5}"
GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-4}"
GRPO_GRAD_ACCUM_STEPS="${GRPO_GRAD_ACCUM_STEPS:-1}"
GRPO_STEPS_PER_GENERATION="${GRPO_STEPS_PER_GENERATION:-2}"
GRPO_NUM_TRAIN_EPOCHS="${GRPO_NUM_TRAIN_EPOCHS:-1}"
GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH:-512}"
GRPO_TEMPERATURE="${GRPO_TEMPERATURE:-1.0}"
GRPO_TOP_P="${GRPO_TOP_P:-0.95}"
GRPO_REPETITION_PENALTY="${GRPO_REPETITION_PENALTY:-1.05}"
GRPO_SAVE_STEPS="${GRPO_SAVE_STEPS:-50}"
GRPO_ALLOW_UNSAFE="${GRPO_ALLOW_UNSAFE:-false}"
QWEN35_FORCE_TORCH_CAUSAL_CONV1D="${QWEN35_FORCE_TORCH_CAUSAL_CONV1D:-1}"

# R1 absolute-score, R2 cross-image ranking, R3 subject-code, R4 format.
GRPO_W_R1="${GRPO_W_R1:-1.4}"
GRPO_W_R2="${GRPO_W_R2:-0.8}"
GRPO_W_R3="${GRPO_W_R3:-0.05}"
GRPO_W_R4="${GRPO_W_R4:-0.02}"
GRPO_RANK_MIN_GAP="${GRPO_RANK_MIN_GAP:-1.0}"
GRPO_RANK_MARGIN="${GRPO_RANK_MARGIN:-0.0}"
GRPO_RANK_TEMP="${GRPO_RANK_TEMP:-0.5}"
GRPO_BPR_DIRECT="${GRPO_BPR_DIRECT:-true}"
GRPO_BPR_CENTERED="${GRPO_BPR_CENTERED:-true}"
GRPO_BPR_NEUTRAL="${GRPO_BPR_NEUTRAL:-0.0}"
GRPO_W_FIT="${GRPO_W_FIT:-0.4}"
GRPO_W_VC="${GRPO_W_VC:-0.4}"
GRPO_W_ORIG="${GRPO_W_ORIG:-0.2}"
GRPO_ALPHA_FIT="${GRPO_ALPHA_FIT:-1.0}"
GRPO_ALPHA_VC="${GRPO_ALPHA_VC:-1.0}"
GRPO_ALPHA_ORIG="${GRPO_ALPHA_ORIG:-1.0}"
GRPO_BETA_FIT="${GRPO_BETA_FIT:-1.0}"
GRPO_BETA_VC="${GRPO_BETA_VC:-1.0}"
GRPO_BETA_ORIG="${GRPO_BETA_ORIG:-1.0}"
GRPO_SIGMA="${GRPO_SIGMA:-0.5}"
GRPO_R1_W_GAUSS="${GRPO_R1_W_GAUSS:-1.0}"
GRPO_R3_HIT_FULL="${GRPO_R3_HIT_FULL:-1.0}"
GRPO_R3_HIT_MAJOR="${GRPO_R3_HIT_MAJOR:-0.4}"
GRPO_R3_MISS="${GRPO_R3_MISS:--0.3}"
GRPO_PARSE_FAIL_PENALTY="${GRPO_PARSE_FAIL_PENALTY:--1.0}"
GRPO_FORMAT_SOFT_CHARS="${GRPO_FORMAT_SOFT_CHARS:-700}"
GRPO_FORMAT_MAX_CHARS="${GRPO_FORMAT_MAX_CHARS:-900}"

if [[ "$GRPO_ALLOW_UNSAFE" != "true" ]]; then
    awk "BEGIN {exit !(${GRPO_LEARNING_RATE} <= 2e-6)}" || die "GRPO_LEARNING_RATE must be <= 2e-6"
    awk "BEGIN {exit !(${GRPO_BETA} >= 0.02)}" || die "GRPO_BETA must be >= 0.02"
    (( GRPO_MAX_COMPLETION_LENGTH >= 512 )) || die "GRPO_MAX_COMPLETION_LENGTH must be >= 512"
    awk "BEGIN {exit !(${GRPO_NUM_TRAIN_EPOCHS} <= 1)}" || die "GRPO_NUM_TRAIN_EPOCHS must be <= 1"
    [[ "$GRPO_BPR_CENTERED" != "true" || "$GRPO_BPR_NEUTRAL" == "0.0" || "$GRPO_BPR_NEUTRAL" == "0" ]] || \
        die "Centered BPR requires GRPO_BPR_NEUTRAL=0"
fi

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

DIAGNOSTIC_FUNCS=()
DIAGNOSTIC_WEIGHTS=()
# Zero-weight rewards expose per-agent metrics without changing optimization.
for dimension in type_agent theme_style composition structure image_quality tone emotion; do
    for component in r1 r2 r3 r4 total; do
        DIAGNOSTIC_FUNCS+=("dim_${dimension}_${component}")
        DIAGNOSTIC_WEIGHTS+=(0)
    done
done

MAX_STEPS_ARGS=()
if (( ${GRPO_DEBUG_STEPS:-0} > 0 )); then
    MAX_STEPS_ARGS=(--max_steps "$GRPO_DEBUG_STEPS")
fi

CMD=(
    torchrun
    --nnodes="$NNODES"
    --nproc_per_node="$NPROC_PER_NODE"
    --node_rank="$NODE_RANK"
    --master_addr="$MASTER_ADDR"
    --master_port="$MASTER_PORT"
    --module portraitaes.grpo.launcher
    --rlhf_type grpo
    --model "$MODEL_PATH"
    --adapters "$SFT_ADAPTER"
    --ref_adapters "$SFT_ADAPTER"
    --lora_rank "${GRPO_LORA_RANK:-64}"
    --lora_alpha "${GRPO_LORA_ALPHA:-128}"
    --target_modules all-linear
    --freeze_vit true
    --freeze_aligner true
    --gradient_checkpointing true
    --vit_gradient_checkpointing false
    --add_non_thinking_prefix true
    --loss_scale ignore_empty_think
    --external_plugins "$REWARD_PLUGIN"
    --reward_funcs peer_relative_dir bpr_rank_cross_dir subject_code_hit format_len "${DIAGNOSTIC_FUNCS[@]}"
    --reward_weights "$GRPO_W_R1" "$GRPO_W_R2" "$GRPO_W_R3" "$GRPO_W_R4" "${DIAGNOSTIC_WEIGHTS[@]}"
    --dataset "$DATASET_PATH"
    --eval_strategy no
    --split_dataset_ratio 0
    --remove_unused_columns false
    --load_from_cache_file false
    --dataset_num_proc "${DATASET_NUM_PROC:-64}"
    --num_generations "$GRPO_NUM_GENERATIONS"
    --temperature "$GRPO_TEMPERATURE"
    --top_p "$GRPO_TOP_P"
    --repetition_penalty "$GRPO_REPETITION_PENALTY"
    --max_completion_length "$GRPO_MAX_COMPLETION_LENGTH"
    --max_length "${MAX_LENGTH:-16384}"
    --beta "$GRPO_BETA"
    --epsilon "$GRPO_EPSILON"
    --learning_rate "$GRPO_LEARNING_RATE"
    --lr_scheduler_type "$GRPO_LR_SCHEDULER"
    --max_grad_norm "$GRPO_MAX_GRAD_NORM"
    --warmup_ratio "$GRPO_WARMUP_RATIO"
    --advantage_estimator grpo
    --loss_type "$GRPO_LOSS_TYPE"
    --scale_rewards "$GRPO_SCALE_REWARDS"
    --overlong_filter true
    --log_completions true
    --num_train_epochs "$GRPO_NUM_TRAIN_EPOCHS"
    --per_device_train_batch_size "$GRPO_BATCH_SIZE"
    --gradient_accumulation_steps "$GRPO_GRAD_ACCUM_STEPS"
    --steps_per_generation "$GRPO_STEPS_PER_GENERATION"
    --logging_steps "${GRPO_LOGGING_STEPS:-5}"
    --save_steps "$GRPO_SAVE_STEPS"
    --safe_serialization false
    --save_total_limit "${GRPO_SAVE_TOTAL_LIMIT:-1000}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}"
    --deepspeed "${GRPO_DEEPSPEED:-zero3}"
    --torch_dtype bfloat16
    --attn_impl "${GRPO_ATTN_IMPL:-sdpa}"
    --report_to "${REPORT_TO:-tensorboard}"
    --output_dir "$OUTPUT_DIR"
    --model_name "$RUN_NAME"
    "${MAX_STEPS_ARGS[@]}"
    "${RESUME_ARGS[@]}"
    "${EXTRA_ARGS[@]}"
)

echo "[INFO] run=$RUN_NAME world_size=$WORLD_SIZE node_rank=$NODE_RANK"
echo "[INFO] output=$OUTPUT_DIR log=$LOG_FILE"
echo "[INFO] rewards=$GRPO_W_R1/$GRPO_W_R2/$GRPO_W_R3/$GRPO_W_R4 generations=$GRPO_NUM_GENERATIONS"
printf '[INFO] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

HF_HOME_VALUE="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
HF_DATASETS_CACHE_VALUE="${HF_DATASETS_CACHE:-${PROJECT_ROOT}/.cache/huggingface/datasets}"
mkdir -p "$HF_HOME_VALUE" "$HF_DATASETS_CACHE_VALUE"

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
MAX_PIXELS="${MAX_PIXELS:-1254400}" \
USE_JSONL_INMEMORY_LOADER="${USE_JSONL_INMEMORY_LOADER:-0}" \
DISABLE_HF_DATASETS_CACHE="${DISABLE_HF_DATASETS_CACHE:-1}" \
QWEN35_FORCE_TORCH_CAUSAL_CONV1D="$QWEN35_FORCE_TORCH_CAUSAL_CONV1D" \
HF_HOME="$HF_HOME_VALUE" \
HF_DATASETS_CACHE="$HF_DATASETS_CACHE_VALUE" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
GRPO_RANK_MIN_GAP="$GRPO_RANK_MIN_GAP" GRPO_RANK_MARGIN="$GRPO_RANK_MARGIN" GRPO_RANK_TEMP="$GRPO_RANK_TEMP" \
GRPO_BPR_DIRECT="$GRPO_BPR_DIRECT" GRPO_BPR_CENTERED="$GRPO_BPR_CENTERED" GRPO_BPR_NEUTRAL="$GRPO_BPR_NEUTRAL" \
GRPO_W_R1="$GRPO_W_R1" GRPO_W_R2="$GRPO_W_R2" GRPO_W_R3="$GRPO_W_R3" GRPO_W_R4="$GRPO_W_R4" \
GRPO_W_FIT="$GRPO_W_FIT" GRPO_W_VC="$GRPO_W_VC" GRPO_W_ORIG="$GRPO_W_ORIG" \
GRPO_ALPHA_FIT="$GRPO_ALPHA_FIT" GRPO_ALPHA_VC="$GRPO_ALPHA_VC" GRPO_ALPHA_ORIG="$GRPO_ALPHA_ORIG" \
GRPO_BETA_FIT="$GRPO_BETA_FIT" GRPO_BETA_VC="$GRPO_BETA_VC" GRPO_BETA_ORIG="$GRPO_BETA_ORIG" \
GRPO_SIGMA="$GRPO_SIGMA" GRPO_R1_W_GAUSS="$GRPO_R1_W_GAUSS" \
GRPO_R3_HIT_FULL="$GRPO_R3_HIT_FULL" GRPO_R3_HIT_MAJOR="$GRPO_R3_HIT_MAJOR" GRPO_R3_MISS="$GRPO_R3_MISS" \
GRPO_PARSE_FAIL_PENALTY="$GRPO_PARSE_FAIL_PENALTY" \
GRPO_FORMAT_SOFT_CHARS="$GRPO_FORMAT_SOFT_CHARS" GRPO_FORMAT_MAX_CHARS="$GRPO_FORMAT_MAX_CHARS" \
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
