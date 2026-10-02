# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dependency-light CPU parity tests for the GSPO policy-loss reproduction."""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch


_METRICS = ("actor/pg_clipfrac", "actor/ppo_kl")


class ActorConfig:
    """Minimal config surface used by the two production policy-loss functions."""

    def __init__(self, clip_ratio: float):
        self.clip_ratio = clip_ratio
        self.clip_ratio_low = clip_ratio
        self.clip_ratio_high = clip_ratio
        self.global_batch_info = {}


class _TorchFunctional:
    """Minimal masked reductions required by production ``agg_loss`` and GSPO."""

    @staticmethod
    def masked_sum(values, mask, axis=None):
        return (values * mask.to(dtype=values.dtype)).sum(dim=axis)

    @staticmethod
    def masked_mean(values, mask, axis=None):
        mask = mask.to(dtype=values.dtype)
        return (values * mask).sum(dim=axis) / mask.sum(dim=axis).clamp(min=1)


def _load_production_policy_losses():
    """Load only the selected function bodies without importing the veRL package."""
    core_algos_path = Path(__file__).resolve().parents[3] / "verl/trainer/ppo/core_algos.py"
    tree = ast.parse(core_algos_path.read_text())
    selected_names = {
        "agg_loss",
        "compute_policy_loss_gspo",
        "compute_policy_loss_gspo_repro",
    }
    selected_nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in selected_names:
            node.decorator_list = []
            node.returns = None
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                arg.annotation = None
            if node.args.vararg is not None:
                node.args.vararg.annotation = None
            if node.args.kwarg is not None:
                node.args.kwarg.annotation = None
            selected_nodes.append(node)

    found_names = {node.name for node in selected_nodes}
    assert found_names == selected_names, f"Missing production functions: {selected_names - found_names}"

    module = ast.fix_missing_locations(ast.Module(body=selected_nodes, type_ignores=[]))
    namespace = {
        "ActorConfig": ActorConfig,
        "torch": torch,
        "verl_F": _TorchFunctional,
    }
    exec(compile(module, core_algos_path, "exec"), namespace)
    return namespace["compute_policy_loss_gspo"], namespace["compute_policy_loss_gspo_repro"]


compute_policy_loss_gspo, compute_policy_loss_gspo_repro = _load_production_policy_losses()


def _actor_config(clip_ratio: float = 0.2) -> ActorConfig:
    return ActorConfig(clip_ratio=clip_ratio)


def _call_both(old_log_prob, log_prob, advantages, response_mask, config):
    kwargs = {
        "old_log_prob": old_log_prob,
        "advantages": advantages,
        "response_mask": response_mask,
        "loss_agg_mode": "seq-mean-token-mean",
        "config": config,
    }
    official = compute_policy_loss_gspo(log_prob=log_prob, **kwargs)
    reproduction = compute_policy_loss_gspo_repro(log_prob=log_prob, **kwargs)
    return official, reproduction


def _assert_outputs_close(case, official, reproduction):
    official_loss, official_metrics = official
    reproduction_loss, reproduction_metrics = reproduction
    loss_diff = (official_loss.detach() - reproduction_loss.detach()).abs().item()
    metric_diffs = {
        name: float(np.abs(official_metrics[name] - reproduction_metrics[name])) for name in _METRICS
    }

    try:
        torch.testing.assert_close(official_loss, reproduction_loss)
        for name in _METRICS:
            torch.testing.assert_close(
                torch.tensor(official_metrics[name]),
                torch.tensor(reproduction_metrics[name]),
            )
    except AssertionError:
        print(f"{case}: FAIL | loss_diff={loss_diff:.3e} | metric_diffs={metric_diffs}")
        raise

    print(f"{case}: PASS | loss_diff={loss_diff:.3e} | metric_diffs={metric_diffs}")


def test_gspo_repro_matches_official_on_random_tensors():
    torch.manual_seed(2026)
    batch_size, response_length = 4, 8
    old_log_prob = torch.randn(batch_size, response_length)
    log_prob = torch.randn(batch_size, response_length, requires_grad=True)
    advantages = torch.randn(batch_size, response_length)
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.float32,
    )

    official, reproduction = _call_both(
        old_log_prob, log_prob, advantages, response_mask, _actor_config()
    )
    _assert_outputs_close("random parity", official, reproduction)


def test_padding_tokens_do_not_affect_sequence_ratio():
    torch.manual_seed(7)
    old_log_prob = torch.randn(3, 8)
    log_prob = torch.randn(3, 8)
    advantages = torch.randn(3, 8)
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    changed_padding_log_prob = torch.where(response_mask.bool(), log_prob, log_prob + 100.0)
    config = _actor_config()

    official_base, reproduction_base = _call_both(
        old_log_prob, log_prob, advantages, response_mask, config
    )
    official_changed, reproduction_changed = _call_both(
        old_log_prob, changed_padding_log_prob, advantages, response_mask, config
    )

    _assert_outputs_close("variable-length parity", official_base, reproduction_base)
    torch.testing.assert_close(official_base[0], official_changed[0])
    torch.testing.assert_close(reproduction_base[0], reproduction_changed[0])
    for name in _METRICS:
        torch.testing.assert_close(
            torch.tensor(official_base[1][name]), torch.tensor(official_changed[1][name])
        )
        torch.testing.assert_close(
            torch.tensor(reproduction_base[1][name]), torch.tensor(reproduction_changed[1][name])
        )

    loss_diff = (official_base[0] - official_changed[0]).abs().item()
    metric_diffs = {
        name: abs(official_base[1][name] - official_changed[1][name]) for name in _METRICS
    }
    print(f"padding invariance: PASS | loss_diff={loss_diff:.3e} | metric_diffs={metric_diffs}")


def test_gspo_repro_matches_official_across_clipping_regions():
    clip_ratio = 0.2
    ratios = torch.tensor([0.7, 1.0, 1.3], dtype=torch.float32)
    old_log_prob = torch.zeros(3, 8)
    log_prob = ratios.log().unsqueeze(-1).expand(-1, 8).clone().requires_grad_(True)
    advantages = torch.tensor([-1.0, 1.0, 1.0]).unsqueeze(-1).expand(-1, 8)
    response_mask = torch.ones(3, 8)

    official, reproduction = _call_both(
        old_log_prob, log_prob, advantages, response_mask, _actor_config(clip_ratio)
    )
    _assert_outputs_close("clipping parity", official, reproduction)

    expected_loss = torch.tensor((0.8 - 1.0 - 1.2) / 3.0)
    expected_clipfrac = torch.tensor(2.0 / 3.0)
    torch.testing.assert_close(official[0], expected_loss)
    torch.testing.assert_close(torch.tensor(official[1]["actor/pg_clipfrac"]), expected_clipfrac)

    assert ratios[0] < 1 - clip_ratio
    assert ratios[1].item() == pytest.approx(1.0)
    assert ratios[2] > 1 + clip_ratio


def test_gspo_repro_matches_official_gradients():
    torch.manual_seed(11)
    old_log_prob = torch.randn(4, 8)
    base_log_prob = old_log_prob + 0.01 * torch.randn(4, 8)
    advantages = torch.randn(4, 8)
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.float32,
    )
    official_log_prob = base_log_prob.clone().requires_grad_(True)
    reproduction_log_prob = base_log_prob.clone().requires_grad_(True)
    config = _actor_config()

    official = compute_policy_loss_gspo(
        old_log_prob,
        official_log_prob,
        advantages,
        response_mask,
        "seq-mean-token-mean",
        config,
    )
    reproduction = compute_policy_loss_gspo_repro(
        old_log_prob,
        reproduction_log_prob,
        advantages,
        response_mask,
        "seq-mean-token-mean",
        config,
    )
    _assert_outputs_close("gradient forward parity", official, reproduction)

    official[0].backward()
    reproduction[0].backward()

    assert official_log_prob.requires_grad
    assert reproduction_log_prob.requires_grad
    assert official_log_prob.grad is not None
    assert reproduction_log_prob.grad is not None
    assert not old_log_prob.requires_grad
    assert old_log_prob.grad is None
    assert not advantages.requires_grad
    assert advantages.grad is None
    torch.testing.assert_close(official_log_prob.grad, reproduction_log_prob.grad)
    torch.testing.assert_close(
        reproduction_log_prob.grad[~response_mask.bool()],
        torch.zeros_like(reproduction_log_prob.grad[~response_mask.bool()]),
    )

    grad_diff = (official_log_prob.grad - reproduction_log_prob.grad).abs().max().item()
    print(f"gradient parity: PASS | max_grad_diff={grad_diff:.3e}")
