# veRL GSPO implementation analysis

Analysis target: veRL commit `fbb4b3a8bf636f290c9c59fc346f756849e9c241`.

## Result

This veRL snapshot already supports GSPO natively. The official implementation
is `compute_policy_loss_gspo` in `verl/trainer/ppo/core_algos.py`, registered by
`@register_policy_loss("gspo")`. The ordinary PPO/GRPO policy loss is
`compute_policy_loss_vanilla`, registered as `vanilla`. GRPO itself is selected
separately as the advantage estimator (`algorithm.adv_estimator=grpo`), so the
baseline pairing is GRPO advantages plus the `vanilla` policy loss.

## Rollout-to-loss call chain

For the current default PPO trainer, the relevant path is:

1. `RayPPOTrainer.fit` repeats each prompt `rollout.n` times and calls
   `async_rollout_manager.generate_sequences`.
2. The rollout result contributes padded `responses`, `input_ids`,
   `attention_mask`, `position_ids`, and (for the current async path) an explicit
   `response_mask`; rollout log probabilities may also be returned, but normal
   decoupled PPO/GRPO does not use them as `old_log_probs`.
3. `RayPPOTrainer._compute_old_log_prob` calls
   `actor_rollout_wg.compute_log_prob` once before actor updates and stores the
   result as `old_log_probs`.
4. Rule rewards become `token_level_scores`, then `token_level_rewards` because
   `algorithm.use_kl_in_reward=False`.
5. `compute_advantage(..., adv_estimator=GRPO)` calls
   `compute_grpo_outcome_advantage` and adds `advantages` and `returns`.
6. `RayPPOTrainer._update_actor` calls `actor_rollout_wg.update_actor`, then
   `EngineWorker.update_actor -> ActorWorker.train_mini_batch -> train_batch`.
7. The model forward pass produces fresh `model_output["log_probs"]` for every
   update micro-batch. `verl.workers.utils.losses.ppo_loss` converts those values
   back to padded response layout.
8. `ppo_loss` reads `config.policy_loss.loss_mode`, resolves it through
   `get_policy_loss_fn`, and calls the selected registered policy loss.

In compact form:

```text
generate_sequences
  -> reward + response_mask
  -> _compute_old_log_prob
  -> compute_advantage(GRPO)
  -> _update_actor
  -> current actor forward
  -> ppo_loss
  -> get_policy_loss_fn(loss_mode)
  -> vanilla | gspo | later gspo_repro
```

## Tensor provenance and shapes

Let `B0` be the prompt batch size, `G=rollout.n`, `B=B0*G`, `P` the padded
prompt length, and `R` the configured padded response length.

| Tensor | Source | Shape at policy loss |
|---|---|---|
| `old_log_prob` | Actor forward in `_compute_old_log_prob`, before optimizer updates; stored as batch key `old_log_probs` | `[B, R]` |
| `log_prob` | Current actor forward inside each update micro-batch; `model_output["log_probs"]`, then `no_padding_2_padding` | `[b, R]` for the current micro-batch |
| `advantages` | `compute_grpo_outcome_advantage`: group-normalized scalar outcome reward, broadcast over valid response tokens | `[B, R]`, sliced to `[b, R]` during update |
| `response_mask` | Async rollout explicit LLM-token/loss mask; fallback is the last `R` columns of `attention_mask` | `[B, R]`, sliced to `[b, R]`; boolean in `ppo_loss` |

For the requested single-turn, no-tool run, `response_mask` is simply 1 for
generated response tokens through EOS and 0 for right padding. The more general
async path can also mask tool-response tokens, but those features are disabled
in this reproduction.

`old_log_probs` are not normally the log probabilities reported by vLLM during
sampling. veRL recomputes them with the training actor to create a stable
proximal-policy anchor and avoid rollout/training-engine numerical differences.
Only the optional rollout-correction bypass mode copies `rollout_log_probs` to
`old_log_probs`; that mode is outside this reproduction.

For GRPO, the rule reward is summed over the response to one scalar per sampled
sequence. Samples are grouped by the repeated prompt `uid`; the scalar is
centered and, by default, divided by the within-group standard deviation, then
broadcast across the response dimension and multiplied by `response_mask`.

## Official GSPO ratio and clipping

The upstream function first computes token log-ratios:

```text
delta_it = current_log_prob_it - old_log_prob_it
```

It averages valid token log-ratios within each sequence and exponentiates:

```text
log_s_i = sum_t(mask_it * delta_it) / max(sum_t(mask_it), 1)
s_i = exp(log_s_i)
```

The implementation uses a surrogate with the *value* of the sequence ratio but
token-local gradients:

```text
log_s_it = current_log_prob_it - stopgrad(current_log_prob_it)
           + stopgrad(log_s_i)
s_it = exp(clamp(log_s_it, max=10))
```

Thus every valid token in a sequence has the same forward ratio `s_i`, while
the gradient with respect to each current token log-probability remains local.
The upper log clamp is a numerical guard, not the PPO clip window.

Clipping is the standard sign-aware PPO max in loss form:

```text
loss_unclipped = -advantage * s_it
loss_clipped   = -advantage * clamp(s_it, 1-eps_low, 1+eps_high)
loss_token     = max(loss_unclipped, loss_clipped)
```

The official example uses `eps_low=3e-4`, `eps_high=4e-4`, and
`loss_agg_mode=seq-mean-token-mean`. The latter takes a token mean inside each
sequence and then an equal mean over non-empty sequences. The function accepts
other aggregation modes, but that setting matches the GSPO recipe.

## Registry and minimal change set for `gspo_repro`

The registry is the module-level `POLICY_LOSS_REGISTRY` in
`verl/trainer/ppo/core_algos.py`. `register_policy_loss(name)` inserts a
callable, and `get_policy_loss_fn(name)` is the only dispatch needed by
`verl/workers/utils/losses.py`.

The minimal implementation change is therefore:

1. `verl/trainer/ppo/core_algos.py`: add a new function decorated with
   `@register_policy_loss("gspo_repro")`. Do not edit the body of official
   `compute_policy_loss_gspo`.
2. `tests/trainer/ppo/test_gspo_repro_on_cpu.py`: compare official and
   reproduction scalar loss, metrics, and gradients on deterministic tensors,
   including variable response lengths and both positive/negative advantages.

The existing `reproduction/run_gsm8k_debug.sh` already passes the mode through,
so no trainer, rollout, worker, or registry-dispatch file needs modification.
Updating the `PolicyLossConfig` docstring or YAML comment to list the new name is
optional documentation, not a runtime requirement.

