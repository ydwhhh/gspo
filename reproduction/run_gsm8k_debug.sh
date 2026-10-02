#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_DIR}"

LOSS_MODE=${1:-vanilla}
if [[ $# -gt 0 ]]; then
    shift
fi
case "${LOSS_MODE}" in
    vanilla)
        LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
        CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
        CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.2}
        ;;
    gspo|gspo_repro)
        LOSS_AGG_MODE=${LOSS_AGG_MODE:-seq-mean-token-mean}
        CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-3e-4}
        CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-4e-4}
        ;;
    *)
        echo "Unsupported loss mode: ${LOSS_MODE}. Use vanilla, gspo, or gspo_repro." >&2
        exit 2
        ;;
esac

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
DATA_DIR=${DATA_DIR:-${REPO_DIR}/data/gsm8k}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/test.parquet}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-100}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-100}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
N_GPUS=${N_GPUS:-1}
ROLLOUT_N=${ROLLOUT_N:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
PYTHON_BIN=${PYTHON_BIN:-python}

if ! PYTHON_EXECUTABLE=$(command -v "${PYTHON_BIN}"); then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
    echo "GSM8K parquet files not found under ${DATA_DIR}." >&2
    echo "Run the preprocessing command documented in reproduction/README.md first." >&2
    exit 1
fi

"${PYTHON_EXECUTABLE}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
    data.val_max_samples="${VAL_MAX_SAMPLES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.policy_loss.loss_mode="${LOSS_MODE}" \
    actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    critic.enable=False \
    reward.reward_model.enable=False \
    trainer.logger='["console"]' \
    trainer.project_name=gspo-reproduction \
    trainer.experiment_name="gsm8k-qwen2.5-1.5b-${LOSS_MODE}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    ray_kwargs.ray_init.runtime_env.py_executable="${PYTHON_EXECUTABLE}" \
    "$@"
