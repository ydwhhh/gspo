# Minimal GSPO reproduction on veRL

This directory contains the thin experiment layer for the reproduction. The
veRL source tree remains the framework; no trainer, rollout engine, or PPO/GRPO
loop is reimplemented here.

The upstream snapshot used as the base is:

```text
repository: https://github.com/volcengine/verl.git
branch:     main
commit:     fbb4b3a8bf636f290c9c59fc346f756849e9c241
date:       2026-09-30
```

The source analysis that must precede `gspo_repro` is in
[`ANALYSIS.md`](ANALYSIS.md).

## Scope

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Dataset: GSM8K, with `data.train_max_samples=100`
- Reward: veRL's built-in strict GSM8K rule reward (`#### <number>` exact match)
- Advantage estimator: GRPO
- Policy-loss modes: `vanilla` (GRPO baseline), upstream `gspo`, and later
  `gspo_repro`
- One node only; no reward model, judge, PRM, tools, multi-turn training, MoE,
  or agent task logic

## Environment

Use a Linux machine with an NVIDIA GPU and a CUDA stack compatible with the
versions locked by the checked-out veRL commit. Python 3.12 is the upstream
development default. The macOS checkout can be used for editing and CPU unit
tests, but not for the vLLM/FSDP training run.

veRL uses `uv` and already commits `uv.lock`; do not create a second dependency
universe for this reproduction:

```bash
cd /path/to/gspo-reproduction
uv venv --python 3.12
uv sync --frozen --extra fsdp --extra vllm
```

This creates `.venv` and installs the exact training/rollout dependencies from
the upstream lock. If `uv` is not installed, install it using the method in the
root `AGENTS.md`, then rerun the two commands above.

## Data

The existing veRL preprocessor produces the expected prompt schema and stores
the numeric ground truth under `reward_model.ground_truth`:

```bash
uv run --frozen --extra fsdp --extra vllm \
  python examples/data_preprocess/gsm8k.py \
  --local_save_dir "$PWD/data/gsm8k"
```

The debug launcher selects approximately 100 training examples at dataset-load
time via `data.train_max_samples=100`; there is no need to create a forked
dataset file.

## Debug runs

Run from the repository root:

```bash
# GRPO advantage estimator + ordinary token-level PPO/GRPO policy loss
bash reproduction/run_gsm8k_debug.sh vanilla

# GRPO advantage estimator + upstream official GSPO policy loss
bash reproduction/run_gsm8k_debug.sh gspo
```

Useful overrides:

```bash
N_GPUS=2 TRAIN_MAX_SAMPLES=100 TOTAL_TRAINING_STEPS=5 \
  bash reproduction/run_gsm8k_debug.sh gspo
```

After `gspo_repro` is implemented and registered, the same launcher will accept:

```bash
bash reproduction/run_gsm8k_debug.sh gspo_repro
```

The launcher intentionally uses console logging, disables the critic, reference
KL loss, reward model, and multi-turn rollout, and saves no checkpoints during
the two-step smoke run.

## Next phase

Do not implement `gspo_repro` until the analysis checkpoint in `ANALYSIS.md` has
been reviewed. The next implementation step is limited to a new registered loss
function and a CPU numerical/gradient parity test against upstream `gspo`.

