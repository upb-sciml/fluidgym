"""Backpropagation through differentiable environments.

Uses the small CylinderJet2D environment and a single step each, so the tests stay
cheap. They cover the gradient paths that gradient-based methods (BPTT, SHAC)
rely on: reward w.r.t. action, and the observation (resampled from the blocks to
the sensors) w.r.t. action and flow state, for single and batched environments.
"""

import torch

import fluidgym
from fluidgym.envs.util.diff_tools import mark_state_differentiable

ENV_ID = "CylinderJet2D-easy-v0"


def _assert_useful_grad(grad: torch.Tensor | None, name: str) -> None:
    assert grad is not None, f"No gradient w.r.t. {name}"
    assert torch.isfinite(grad).all(), f"Non-finite gradient w.r.t. {name}"
    assert grad.abs().sum() > 0, f"Zero gradient w.r.t. {name}"


def test_reward_gradient_wrt_action():
    """The reward is differentiable w.r.t. the action."""
    env = fluidgym.make(ENV_ID, differentiable=True)
    env.reset(seed=42)

    action = env.sample_action().requires_grad_(True)
    _, reward, _, _, _ = env.step(action)
    reward.sum().backward()

    _assert_useful_grad(action.grad, "the action")
    env.detach()


def test_observation_gradient_wrt_state():
    """The observation is differentiable w.r.t. the flow state."""
    env = fluidgym.make(ENV_ID, differentiable=True)
    env.reset(seed=42)

    state = mark_state_differentiable(env)
    obs, _, _, _, _ = env.step(env.sample_action())
    obs["velocity"].sum().backward()

    for i, tensor in enumerate(state):
        if tensor.grad is not None:
            assert torch.isfinite(tensor.grad).all(), f"Non-finite gradient (state {i})"
    assert any(t.grad is not None and t.grad.abs().sum() > 0 for t in state), (
        "Zero gradient of the observation w.r.t. the flow state"
    )
    env.detach()


def test_batched_gradients_stay_per_env():
    """In a batch, each env's observation depends only on its own action."""
    env = fluidgym.make(ENV_ID, differentiable=True, n_envs=2)
    env.reset(seed=42)

    action = env.sample_action().requires_grad_(True)
    obs, _, _, _, _ = env.step(action)

    # The observation of env 0 depends on its own action, not on that of env 1
    (grad,) = torch.autograd.grad(obs["velocity"][0].sum(), action)
    _assert_useful_grad(grad[0], "the action of the same env")
    # Relative bound instead of exact zero, for round-off in the batched solvers
    assert grad[1].abs().max() <= 1e-6 * grad[0].abs().max(), (
        "Gradient leaks into another env of the batch"
    )
    env.detach()
