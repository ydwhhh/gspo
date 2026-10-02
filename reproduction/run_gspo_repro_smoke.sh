#!/usr/bin/env bash
set -euo pipefail

# Minimal end-to-end GSPO reproduction smoke test.
# Run from any directory; all defaults are resolved relative to the veRL checkout.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_DIR}"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
DATA_DIR=${DATA_DIR:-${REPO_DIR}/data/gsm8k}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/test.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs}
LOG_DIR=${LOG_DIR:-./logs}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/gspo_repro_smoke_${RUN_TIMESTAMP}.log}
PYTHON_BIN=${PYTHON_BIN:-python}

# Keep 64 examples visible to the dataloader, but consume only 8 prompts per
# step. Each prompt produces four responses, so an actor update sees 32 samples.
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-64}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-32}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}

N_GPUS=${N_GPUS:-1}
ROLLOUT_TP=${ROLLOUT_TP:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.4}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}
ACTOR_LR=${ACTOR_LR:-1e-6}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-3e-4}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-4e-4}
AUTO_PREPARE_DATA=${AUTO_PREPARE_DATA:-1}

if ! PYTHON_EXECUTABLE=$(command -v "${PYTHON_BIN}"); then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ "${MODEL_PATH}" = /* || "${MODEL_PATH}" = ./* || "${MODEL_PATH}" = ../* ]]; then
    if [[ ! -d "${MODEL_PATH}" ]]; then
        echo "Local MODEL_PATH does not exist or is not a directory: ${MODEL_PATH}" >&2
        exit 1
    fi
fi

if ! [[ "${N_GPUS}" =~ ^[1-9][0-9]*$ && "${ROLLOUT_TP}" =~ ^[1-9][0-9]*$ ]]; then
    echo "N_GPUS and ROLLOUT_TP must be positive integers." >&2
    exit 1
fi

if (( ROLLOUT_TP > N_GPUS )); then
    echo "ROLLOUT_TP=${ROLLOUT_TP} cannot exceed N_GPUS=${N_GPUS} for this single-node smoke setup." >&2
    exit 1
fi

# Fail early if Python resolves veRL from another checkout or site-packages.
"${PYTHON_EXECUTABLE}" - "${REPO_DIR}" <<'PY'
import sys
from pathlib import Path

repo_dir = Path(sys.argv[1]).resolve()
try:
    import verl
except Exception as exc:
    raise SystemExit(f"Cannot import veRL with the active Python: {exc}") from exc

verl_file = Path(verl.__file__).resolve()
try:
    verl_file.relative_to(repo_dir / "verl")
except ValueError:
    raise SystemExit(
        f"Wrong veRL import: {verl_file}\nExpected veRL from: {repo_dir / 'verl'}\n"
        "Activate the correct server environment or install this checkout in editable mode."
    )
print(f"Using veRL from: {verl_file}")
PY

if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
    if [[ "${AUTO_PREPARE_DATA}" != "1" ]]; then
        echo "GSM8K parquet files are missing:" >&2
        echo "  ${TRAIN_FILE}" >&2
        echo "  ${VAL_FILE}" >&2
        echo "Set AUTO_PREPARE_DATA=1 or prepare them with examples/data_preprocess/gsm8k.py." >&2
        exit 1
    fi

    echo "Preparing GSM8K parquet files in ${DATA_DIR} ..."
    "${PYTHON_EXECUTABLE}" examples/data_preprocess/gsm8k.py \
        --local_save_dir "${DATA_DIR}"
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "GSPO reproduction smoke configuration:"
echo "  model=${MODEL_PATH}"
echo "  train_file=${TRAIN_FILE} (max_samples=${TRAIN_MAX_SAMPLES})"
echo "  prompts_per_step=${TRAIN_BATCH_SIZE}, rollout.n=${ROLLOUT_N}"
echo "  max_response_length=${MAX_RESPONSE_LENGTH}, steps=${TOTAL_TRAINING_STEPS}"
echo "  loss_mode=gspo_repro, loss_agg_mode=seq-mean-token-mean"
echo "  python=${PYTHON_EXECUTABLE}"
echo "  GPUs=${N_GPUS}, nodes=1, rollout_tp=${ROLLOUT_TP}"
echo "  output_dir=${OUTPUT_DIR}, log=${LOG_FILE}"

RAY_RUNTIME=(
    ray_kwargs.ray_init.runtime_env.py_executable="${PYTHON_EXECUTABLE}"
)

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
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo_repro \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    critic.enable=False \
    reward.reward_model.enable=False \
    reward.reward_manager.name=naive \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=gspo-reproduction \
    trainer.experiment_name=gsm8k-qwen2.5-7b-gspo-repro-smoke \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    "${RAY_RUNTIME[@]}" \
    "$@" 2>&1 | tee "${LOG_FILE}"

# Validate the exact console metrics emitted by this veRL commit. Requiring
# actor/grad_norm also proves that backward and optimizer.step completed.
"${PYTHON_EXECUTABLE}" - "${LOG_FILE}" <<'PY'
import math
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
text = log_path.read_text(encoding="utf-8", errors="replace")
required = (
    "critic/score/mean",
    "actor/pg_loss",
    "actor/ppo_kl",
    "actor/pg_clipfrac",
    "response_length/mean",
    "actor/grad_norm",
)
number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)"

missing = []
non_finite = []
last_values = {}
for key in required:
    matches = re.findall(rf"{re.escape(key)}\s*:\s*({number})", text, flags=re.IGNORECASE)
    if not matches:
        missing.append(key)
        continue
    value = float(matches[-1])
    last_values[key] = value
    if not math.isfinite(value):
        non_finite.append(f"{key}={matches[-1]}")

if missing or non_finite:
    if missing:
        print("Missing required smoke metrics: " + ", ".join(missing), file=sys.stderr)
    if non_finite:
        print("Non-finite smoke metrics: " + ", ".join(non_finite), file=sys.stderr)
    raise SystemExit(1)

print("Smoke metric check PASS")
for key, value in last_values.items():
    print(f"  {key}={value:.8g}")
PY

echo "End-to-end gspo_repro smoke test PASS"
