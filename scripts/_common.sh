#!/usr/bin/env bash

# Shared helpers for training launchers. This file is meant to be sourced.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_value() {
    local name="$1"
    local value="$2"
    [[ -n "$value" ]] || die "$name is required"
}

require_file() {
    local name="$1"
    local value="$2"
    require_value "$name" "$value"
    [[ -f "$value" ]] || die "$name does not exist: $value"
}

require_dir() {
    local name="$1"
    local value="$2"
    require_value "$name" "$value"
    [[ -d "$value" ]] || die "$name does not exist: $value"
}

configure_distributed() {
    NNODES="${NNODES:-1}"
    NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_ACCELERATORS:-1}}"
    NODE_RANK="${NODE_RANK:-${RANK:-0}}"
    MASTER_ADDR="${MASTER_ADDR:-${CHIEF_IP:-127.0.0.1}}"
    MASTER_PORT="${MASTER_PORT:-29500}"
    CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-0}}"
    unset RANK

    [[ "$NNODES" =~ ^[1-9][0-9]*$ ]] || die "NNODES must be a positive integer"
    [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || die "NPROC_PER_NODE must be a positive integer"
    [[ "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] || die "NODE_RANK must be a non-negative integer"
    (( NODE_RANK < NNODES )) || die "NODE_RANK must be smaller than NNODES"
    if (( NNODES > 1 )) && [[ "$MASTER_ADDR" =~ ^(localhost|127\.0\.0\.1)$ ]]; then
        die "MASTER_ADDR must be reachable from every node for multi-node training"
    fi

    CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE//[[:space:]]/}"
    IFS=',' read -r -a VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES_VALUE"
    (( ${#VISIBLE_GPUS[@]} == NPROC_PER_NODE )) || \
        die "NPROC_PER_NODE=${NPROC_PER_NODE} does not match CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}"
    WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
}

is_complete_checkpoint() {
    local checkpoint="$1"
    local name step zero_dir model_count optimizer_count
    [[ -d "$checkpoint" ]] || return 1
    name="${checkpoint##*/}"
    step="${name#checkpoint-}"
    [[ "$step" =~ ^[0-9]+$ ]] || return 1
    [[ -f "$checkpoint/trainer_state.json" ]] || return 1
    [[ -f "$checkpoint/scheduler.pt" ]] || return 1
    [[ -f "$checkpoint/training_args.bin" ]] || return 1
    [[ -f "$checkpoint/adapter_model.safetensors" || -f "$checkpoint/adapter_model.bin" ]] || return 1
    zero_dir="$checkpoint/global_step${step}"
    [[ -d "$zero_dir" ]] || return 1
    model_count="$(find "$zero_dir" -maxdepth 1 -type f -name 'zero_pp_rank_*_model_states.pt' | wc -l | tr -d ' ')"
    optimizer_count="$(find "$zero_dir" -maxdepth 1 -type f -name '*_optim_states.pt' | wc -l | tr -d ' ')"
    (( model_count >= WORLD_SIZE && optimizer_count >= WORLD_SIZE ))
}

latest_complete_checkpoint() {
    local output_dir="$1"
    [[ -d "$output_dir" ]] || return 0
    while IFS= read -r checkpoint; do
        local name="${checkpoint##*/}"
        local step="${name#checkpoint-}"
        if [[ "$step" =~ ^[0-9]+$ ]] && is_complete_checkpoint "$checkpoint"; then
            printf '%s\t%s\n' "$step" "$checkpoint"
        fi
    done < <(find "$output_dir" -mindepth 1 -maxdepth 2 -type d -name 'checkpoint-*') | \
        sort -n | tail -1 | cut -f2-
}
