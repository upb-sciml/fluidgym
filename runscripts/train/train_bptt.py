"""Train a policy with truncated backpropagation through time.

The gradient of the discounted return is taken through the simulator itself: one
window of `horizon` environment steps, one backward pass, one optimizer step.
Unlike actor-critic methods such as SHAC/SAPO there is no critic and no
bootstrap, so the return beyond the window is simply dropped.

Usage:
    python runscripts/train/train_bptt.py env_id=RBCMHD3D-easy-v0 rl_mode=marl seed=0
"""

import copy
import logging
import os
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime as dt
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import fluidgym
from fluidgym.types import FluidEnvLike
from fluidgym.wrappers import (
    FlattenObservation,
    PowerPenalty,
    VideoRecorder,
)
from normalize_obs import NormalizeObservation

logger = logging.getLogger("fluidgym.bptt")

# The fluidgym namespace is pinned to WARNING on import, which would swallow every
# logger.info below as well as the solver logs
fluidgym.set_verbosity(logging.INFO)

OmegaConf.register_new_resolver("eval", lambda x: eval(x), replace=True)

local_data_path = os.environ.get("FLUIDGYM_LOCAL_DATA_PATH", "./local_data")
fluidgym.config.update("local_data_path", to_absolute_path(local_data_path))


class BPTTPolicy(nn.Module):
    """A continuous control policy network for BPTT.

    The trunk matches the actor of the earlier SHAC/SAPO baselines, so a BPTT
    baseline differs from them in the update rather than in capacity. The
    output of the final tanh is mapped onto the environment's own action bounds,
    which are taken from the action space, so no action range is assumed.

    Parameters
    ----------
    obs_dim: int
        Size of the (flattened) observation vector.
    action_dim: int
        Size of the action vector.
    action_low, action_high: torch.Tensor
        Bounds of the environment's action space.
    units: tuple[int, ...]
        Widths of the hidden layers.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        units: tuple[int, ...] = (128, 64, 32),
    ):
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for width in units:
            layers += [nn.Linear(in_dim, width), nn.LayerNorm(width), nn.ELU()]
            in_dim = width
        # Bounds the output to [-1, 1]
        layers += [nn.Linear(in_dim, action_dim), nn.Tanh()]

        self.net = nn.Sequential(*layers)

        # These scale the [-1, 1] tanh output to the environment's action bounds
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the action for `obs`, scaled to the environment's bounds."""
        x = self.net(obs.to(dtype=self.action_scale.dtype))
        return x * self.action_scale + self.action_bias


class ConstantPolicy(nn.Module):
    """An open-loop policy: one learnable action, whatever the observation.

    BPTT on it finds the best constant actuation, the baseline a feedback policy
    has to beat. The action is ``tanh(theta)`` mapped onto the action bounds, so it
    starts at the centre of the action space and never leaves it.

    Parameters
    ----------
    action_dim: int
        Size of the action vector.
    action_low, action_high: torch.Tensor
        Bounds of the environment's action space.
    """

    def __init__(
        self,
        action_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(action_dim))
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the constant action, broadcast over the batch of ``obs``."""
        action = torch.tanh(self.theta) * self.action_scale + self.action_bias
        return action.expand(*obs.shape[:-1], -1)


def mlp(in_dim: int, out_dim: int, units: tuple[int, ...]) -> nn.Sequential:
    """A Linear-LayerNorm-ELU trunk with a linear head, as in :class:`BPTTPolicy`."""
    layers: list[nn.Module] = []
    for width in units:
        layers += [nn.Linear(in_dim, width), nn.LayerNorm(width), nn.ELU()]
        in_dim = width
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class QCritic(nn.Module):
    """An off-policy twin Q-network with Polyak-averaged targets, SAC style.

    It learns ``Q(s, a)`` of the current policy from a replay buffer, one or more
    gradient steps per environment step, with the clipped double-Q target
    ``r + gamma (1 - d) min_i Q'_i(s', pi(s'))``. The policy is deterministic, so
    there is no entropy term. Its use in BPTT is the terminal value
    ``V(s_H) = min_i Q_i(s_H, pi(s_H))`` of a window, through which the gradient
    reaches the policy both through the action and, via the observation, through
    the simulator.

    Q is in units of the raw reward. The actions are mapped onto ``[-1, 1]`` with
    the environment's own bounds before they enter the networks.

    Parameters
    ----------
    obs_dim: int
        Size of the (flattened) observation vector.
    action_dim: int
        Size of the flat action vector.
    action_low, action_high: torch.Tensor
        Bounds of the environment's action space.
    policy: nn.Module
        The policy whose value is learned, taking a flat observation batch to an
        action batch of any shape with ``action_dim`` entries per sample.
    gamma: float
        Discount of the TD target.
    units: tuple[int, ...]
        Widths of the hidden layers of every Q-network.
    n_critics: int
        Number of Q-networks the target takes the minimum over.
    lr: float
        Learning rate of the Q-networks.
    tau: float
        Polyak coefficient of the target update.
    batch_size: int
        Number of transitions per gradient step.
    buffer_size: int
        Capacity of the replay buffer, in transitions.
    learning_starts: int
        Number of transitions collected before the first gradient step; until then
        the critic is not used as a terminal value either.
    updates_per_step: int
        Gradient steps per environment step.
    max_grad_norm: float
        Norm the critic gradient is clipped to.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        policy: nn.Module,
        gamma: float,
        units: tuple[int, ...] = (256, 256),
        n_critics: int = 2,
        lr: float = 3e-4,
        tau: float = 0.005,
        batch_size: int = 256,
        buffer_size: int = 100_000,
        learning_starts: int = 256,
        updates_per_step: int = 1,
        max_grad_norm: float = 10.0,
    ):
        super().__init__()
        self.q_nets = nn.ModuleList(
            mlp(obs_dim + action_dim, 1, units) for _ in range(n_critics)
        )
        self.q_targets = copy.deepcopy(self.q_nets)
        self.q_targets.requires_grad_(False)

        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

        # Not a submodule: the policy is trained and saved on its own
        self._policy = [policy]
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.updates_per_step = updates_per_step
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(self.q_nets.parameters(), lr=lr)

        device = action_low.device
        self.capacity = buffer_size
        self.obs_buf = torch.zeros(buffer_size, obs_dim, device=device)
        self.action_buf = torch.zeros(buffer_size, action_dim, device=device)
        self.reward_buf = torch.zeros(buffer_size, device=device)
        self.next_obs_buf = torch.zeros(buffer_size, obs_dim, device=device)
        self.done_buf = torch.zeros(buffer_size, device=device)
        self.n_stored = 0
        self.ptr = 0

    @property
    def policy(self) -> nn.Module:
        """The policy whose value is learned."""
        return self._policy[0]

    @property
    def ready(self) -> bool:
        """Whether enough transitions have been collected to train and use Q."""
        return self.n_stored >= max(self.learning_starts, self.batch_size)

    def q_values(
        self, obs: torch.Tensor, action: torch.Tensor, target: bool = False
    ) -> torch.Tensor:
        """All Q estimates of the flat batch ``(obs, action)``, ``[n_critics, B]``."""
        dtype = self.action_scale.dtype
        action = action.reshape(*obs.shape[:-1], -1).to(dtype)
        action = (action - self.action_bias) / self.action_scale
        x = torch.cat([obs.to(dtype), action], dim=-1)
        nets = self.q_targets if target else self.q_nets
        return torch.stack([q(x).squeeze(-1) for q in nets])

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """``min_i Q_i(s, pi(s))``, averaged over the batch, differentiable in both.

        This is the terminal value of a window in the units of the raw reward,
        matching the ``reward.mean()`` the window return is built from.
        """
        flat_obs = obs.reshape(-1, obs.shape[-1])
        action = self.policy(flat_obs)
        return self.q_values(flat_obs, action).min(dim=0).values.mean()

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        terminated: bool,
    ) -> None:
        """Store the transitions of one environment step, one per environment."""
        obs = obs.detach().reshape(-1, self.obs_buf.shape[1])
        n = obs.shape[0]
        idx = (self.ptr + torch.arange(n, device=self.obs_buf.device)) % self.capacity
        self.obs_buf[idx] = obs.to(self.obs_buf.dtype)
        self.action_buf[idx] = action.detach().reshape(n, -1).to(self.obs_buf.dtype)
        self.reward_buf[idx] = reward.detach().reshape(-1).to(self.obs_buf.dtype)
        self.next_obs_buf[idx] = (
            next_obs.detach().reshape(n, -1).to(self.obs_buf.dtype)
        )
        # A truncated episode is cut, not ended, so it is bootstrapped like any
        # other step
        self.done_buf[idx] = float(terminated)
        self.ptr = (self.ptr + n) % self.capacity
        self.n_stored = min(self.n_stored + n, self.capacity)

    def update(self) -> dict[str, float]:
        """Run ``updates_per_step`` TD steps, returning their mean metrics."""
        if not self.ready:
            return {}

        metrics: dict[str, list[float]] = defaultdict(list)
        for _ in range(self.updates_per_step):
            idx = torch.randint(
                0, self.n_stored, (self.batch_size,), device=self.obs_buf.device
            )
            obs = self.obs_buf[idx]
            action = self.action_buf[idx]
            reward = self.reward_buf[idx]
            next_obs = self.next_obs_buf[idx]
            done = self.done_buf[idx]

            with torch.no_grad():
                next_action = self.policy(next_obs)
                next_q = self.q_values(next_obs, next_action, target=True)
                target = reward + self.gamma * (1.0 - done) * next_q.min(dim=0).values

            q = self.q_values(obs, action)
            loss = 0.5 * (q - target[None]).pow(2).mean(dim=1).sum()

            # Also drops what the window backward left on the Q-networks through
            # the terminal value
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.q_nets.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

            with torch.no_grad():
                for p, p_targ in zip(
                    self.q_nets.parameters(), self.q_targets.parameters(), strict=True
                ):
                    p_targ.lerp_(p, self.tau)

            metrics["loss"].append(loss.item())
            metrics["q_mean"].append(q.mean().item())
            metrics["target_mean"].append(target.mean().item())
            metrics["grad_norm"].append(grad_norm.item())

        return {k: float(np.mean(v)) for k, v in metrics.items()}


class ReshapeAction(nn.Module):
    """Reshape the flat action of ``policy`` into the env's action shape."""

    def __init__(self, policy: nn.Module, action_shape: tuple[int, ...]):
        super().__init__()
        self.policy = policy
        self.action_shape = action_shape

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the action of the wrapped policy in the env's action shape."""
        action = self.policy(obs)
        return action.reshape(*action.shape[:-1], *self.action_shape)


def train_bptt_policy(
    env: FluidEnvLike,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    total_timesteps: int,
    discount_factor: float,
    horizon: int = 75,
    reward_scale: float = 10.0,
    max_grad_norm: float = 1.0,
    terminal_value_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    log_per_step: bool = True,
    ckpt_path: Path = Path("bptt_policy.pth"),
    enable_wandb: bool = True,
    detach_obs: bool = True,
    extra_metrics_fn: Callable[[], dict[str, float]] | None = None,
    obs_normalizer: NormalizeObservation | None = None,
    critic: QCritic | None = None,
):
    """Train a policy by truncated backpropagation through time.

    One update is one window of `horizon` environment steps: the discounted,
    scaled return of the window is accumulated along the rollout and
    backpropagated through the simulator in a single pass, after which the graph
    is cut and the episode continues from the detached simulation state. The
    environment is reset only at an episode boundary, and only once the backward
    of the last window has run -- a reset builds a new `Domain` and `Simulation`,
    and the checkpoint segments of a window are replayed against the one they
    were recorded on.

    Parameters
    ----------
    env: FluidEnvLike
        The differentiable, flattened training environment.
    policy: nn.Module
        The policy to train, e.g. a :class:`BPTTPolicy` or a
        :class:`ConstantPolicy`.
    optimizer: torch.optim.Optimizer
        The optimizer of the policy parameters.
    total_timesteps: int
        Number of environment steps to train for. Training stops at the first
        window boundary beyond this, so the last episode may be cut short.
    discount_factor: float
        Discount of the return that is differentiated.
    horizon: int
        Number of environment steps the gradient is carried over before the
        graph is cut and the policy updated. A value beyond the episode length
        makes one update per episode.
    reward_scale: float
        Factor on the reward before it enters the differentiated return. The
        window return is normalised by `horizon`, so this sets the size of the
        gradient independently of how long the window is.
    max_grad_norm: float
        Norm the policy gradient is clipped to.
    terminal_value_fn: Callable[[torch.Tensor], torch.Tensor] | None
        Value of the state the window ends in, in units of the raw reward. When
        given, the window return is bootstrapped with it in the manner of SHAC,
        so the return beyond the window is estimated instead of dropped; it has
        to stay differentiable in its argument for the bootstrap to reach the
        policy through the last observation as well as through the last action.
        When None, the window return is truncated.
    ckpt_path: Path
        File the policy is checkpointed into after every episode; the best one
        goes next to it with a ``_best`` suffix.
    enable_wandb: bool
        Whether to log to Weights & Biases.
    extra_metrics_fn: Callable[[], dict[str, float]] | None
        Metrics logged with every update, evaluated after the optimizer step,
        e.g. the parameters of a low-dimensional policy.
    obs_normalizer: NormalizeObservation | None
        The observation normalizer in `env`, if any. Its statistics are saved
        next to every policy checkpoint, with an ``_obs_norm`` suffix, since a
        policy only works on observations normalized the way it was trained on.
    critic: QCritic | None
        An off-policy Q-network trained on every environment step. Once it has
        seen enough transitions, its value of the last state of a window replaces
        `terminal_value_fn` as the bootstrap. It is checkpointed next to the policy
        with a ``_critic`` suffix.
    """
    training_logs: list[dict[str, float]] = []
    # The metrics of the steps of the window that is currently being unrolled; they
    # are averaged and logged when that window's update runs
    window_buffer: dict[str, list[float]] = defaultdict(list)

    # How one environment step decomposes, for step times that stay comparable when
    # the checkpoint layout changes: a PISO step is the same amount of physics
    # whatever `bptt_segment_size` is
    unwrapped = getattr(env, "unwrapped", env)
    n_piso_steps = max(1, int(getattr(unwrapped, "_n_sim_steps", 1)))
    n_segments = 1
    if getattr(unwrapped, "bptt_checkpoint", False):
        n_segments = max(1, len(unwrapped._segment_sizes()))

    # fluidgym runs the MHD env in float64, the policy in float32; the cast lives at
    # the boundary, so the gradient crosses back into float64 on its way into the
    # solver and the network does not have to know the environment's precision
    env_dtype = getattr(unwrapped, "_dtype", torch.float32)

    policy.train()
    global_step = 0
    episode = 0
    best_return: float | None = None

    def save_checkpoint(path: Path) -> None:
        torch.save(policy.state_dict(), path)
        if critic is not None:
            torch.save(critic.state_dict(), path.with_name(f"{path.stem}_critic.pth"))
        if obs_normalizer is not None:
            torch.save(
                obs_normalizer.state_dict(),
                path.with_name(f"{path.stem}_obs_norm.pth"),
            )

    while global_step < total_timesteps:
        obs, _ = env.reset()

        optimizer.zero_grad(set_to_none=True)
        window_return = torch.zeros((), device=env.cuda_device, dtype=env_dtype)
        gamma = 1.0
        episode_reward = 0.0
        episode_step = 0
        episode_metrics = defaultdict(list)
        grad_norms: list[float] = []
        done = False
        terminated_episode = False

        forward_time = 0.0
        window_forward_time = 0.0
        n_updates = 0
        t_episode = time.perf_counter()
        t_window = t_episode

        # Unroll the episode, keeping the graph of every step of the window
        while not done:
            # ------------------------------------------------
            # Collection Loop
            # ------------------------------------------------
            for window_step in range(horizon):
                prev_obs = obs
                action = policy(obs).to(dtype=env_dtype)

                # The solver is asynchronous, so both ends are synchronised: without
                # this the measured time is whenever the next blocking read falls
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(action)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_time = time.perf_counter() - t0
                forward_time += step_time
                window_forward_time += step_time

                window_return = window_return + gamma * reward_scale * reward.mean()
                gamma *= discount_factor

                episode_reward += reward.mean().item()
                episode_step += 1
                global_step += 1
                terminated_episode = bool(terminated)
                done = terminated_episode or bool(truncated)

                # The per-step keys are their own namespace: the window log below
                # runs at this very global_step, so shared keys would merge into
                # the same history row and replace the step by the window mean
                action_detached = action.detach()
                action_mean = action_detached.mean()
                step_metrics = {
                    "train/step_reward": reward.mean().item(),
                    "train/step_action_mean": action_mean.item(),
                    "train/step_action_std": (
                        (action_detached - action_mean).pow(2).mean().sqrt().item()
                    ),
                }
                for metric in env.metrics:
                    if metric in info:
                        value = info[metric].mean().item()
                        step_metrics[f"train/step_{metric}"] = value
                        episode_metrics[metric].append(value)

                # Step times on three grids: the env step is what training is counted
                # in, the PISO step is the one that stays comparable across step lengths
                # and checkpoint layouts, and the segment is what a backward replays at
                # a time
                step_metrics["train/step_time_env_step"] = step_time
                step_metrics["train/step_time_piso_step"] = step_time / n_piso_steps
                step_metrics["train/step_time_segment"] = step_time / n_segments

                # Off the graph: the critic sees detached copies of the transition
                if critic is not None:
                    critic.add(prev_obs, action, reward, obs, terminated_episode)
                    for key, value in critic.update().items():
                        step_metrics[f"train/step_critic_{key}"] = value

                if log_per_step and enable_wandb:
                    wandb.log(step_metrics, step=global_step)

                for key, value in step_metrics.items():
                    window_buffer[key].append(value)

                # A window never runs past the episode: when `horizon` does not
                # divide the episode length, the last window is a short one
                if done:
                    break

            # ------------------------------------------------
            # Terminal Bootstrap
            # ------------------------------------------------
            # A window is cut, not ended: what comes after it is worth
            # gamma^horizon * V(s_H), and `gamma` has been multiplied up to exactly
            # that factor by now. A terminated episode has no continuation to
            # value, so it keeps the truncated return
            terminal_value = None
            value_fn = terminal_value_fn
            if critic is not None and critic.ready:
                value_fn = critic.value
            if value_fn is not None and not terminated_episode:
                terminal_value = value_fn(obs)
                window_return = window_return + gamma * reward_scale * terminal_value

            # ------------------------------------------------
            # Update Step
            # ------------------------------------------------
            # Currently disabled: Normalize by horizon length
            loss = -window_return  # / horizon
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), max_grad_norm
            )
            grad_norms.append(grad_norm.item())

            grad_dead = float(grad_norm) == 0.0 or all(
                p.grad is None for p in policy.parameters()
            )
            if grad_dead:
                logger.error(
                    "Episode %d step %d: the policy gradient is exactly zero, so "
                    "this update does nothing. The adjoint solves are relative, so "
                    "the loss scale (reward_scale=%.3g) is not the cause; check "
                    "for excluded solve gradients or a graph cut between the "
                    "policy and the reward.",
                    episode,
                    episode_step,
                    reward_scale,
                )

            # A rollout that blew up in the solver would otherwise poison every
            # parameter, and with it every window after this one
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                logger.warning(
                    "Non-finite gradient in episode %d at step %d, skipping the "
                    "optimizer step.",
                    episode,
                    episode_step,
                )
            optimizer.zero_grad(set_to_none=True)
            n_updates += 1

            # One log per update: the means over the steps of the window, and the
            # quantities of the update itself
            window_time = time.perf_counter() - t_window
            metrics = {
                key.replace("train/step_", "train/mean_", 1): float(np.mean(values))
                for key, values in window_buffer.items()
            }
            metrics["train/window_return"] = -loss.item()
            metrics["train/grad_dead"] = float(grad_dead)
            metrics["train/window_length"] = float(window_step + 1)
            metrics["train/grad_norm"] = grad_norm.item()
            if terminal_value is not None:
                metrics["train/window_terminal_value"] = float(terminal_value.item())
            if extra_metrics_fn is not None:
                metrics.update(extra_metrics_fn())
            metrics["train/time_window"] = window_time
            metrics["train/time_window_backward"] = max(
                0.0, window_time - window_forward_time
            )
            window_buffer.clear()

            logger.info(
                "Episode %d, update %d (step %d): %s",
                episode,
                n_updates,
                global_step,
                ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
            )
            if enable_wandb:
                wandb.log(metrics, step=global_step)

            training_logs.append({"step": global_step, **metrics})
            pd.DataFrame(training_logs).to_csv("training_log.csv", index=False)

            # Cut the graph the window was recorded on. The simulation state
            # survives, so the episode continues from where it is
            env.detach()
            if detach_obs:
                obs = obs.detach()
            del loss, action, reward, prev_obs
            # The bootstrap holds the graph of the last step until it is dropped
            terminal_value = None

            window_return = torch.zeros((), device=env.cuda_device, dtype=env_dtype)
            window_step = 0
            window_forward_time = 0.0
            gamma = 1.0

            # The finished window is gone by now, so what it reserved can go back
            # to the driver. One call per window, against a rollout of minutes
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            t_window = time.perf_counter()

            # The budget is spent; the rest of the episode would not be trained on
            if global_step >= total_timesteps:
                break

        del window_return, obs

        episode_time = time.perf_counter() - t_episode
        backward_time = max(0.0, episode_time - forward_time)

        mean_grad_norm = float(np.mean(grad_norms)) if grad_norms else float("nan")

        logger.info(
            "Train episode %d | %d step(s) | %d update(s) | return %.3f | "
            "mean grad norm %.3e | forward %.1f s (%.2f s/PISO step), "
            "backward %.1f s (%.2f s/PISO step)",
            episode,
            episode_step,
            n_updates,
            episode_reward,
            mean_grad_norm,
            forward_time,
            forward_time / (episode_step * n_piso_steps),
            backward_time,
            backward_time / (episode_step * n_piso_steps),
        )

        if enable_wandb:
            wandb.log(
                {
                    "train/episode_reward": episode_reward,
                    "train/episode_length": float(episode_step),
                    "train/episode_grad_norm": mean_grad_norm,
                    "train/n_updates": float(n_updates),
                    "train_timings/actor_update": episode_time,
                    "train_timings/forward_env_step": forward_time / episode_step,
                    "train_timings/backward_env_step": backward_time / episode_step,
                    "train_timings/backward_segment": backward_time
                    / (episode_step * n_segments),
                    **{
                        f"train/episode_{metric}": float(np.mean(values))
                        for metric, values in episode_metrics.items()
                    },
                },
                step=global_step,
            )

        save_checkpoint(ckpt_path)
        if best_return is None or episode_reward > best_return:
            best_return = episode_reward
            save_checkpoint(ckpt_path.with_name(f"{ckpt_path.stem}_best.pth"))

        episode += 1

    # The steps of a window that never reached its update, e.g. one cut short by a
    # non-finite gradient earlier in the episode
    if window_buffer:
        metrics = {
            key.replace("train/step_", "train/mean_", 1): float(np.mean(values))
            for key, values in window_buffer.items()
        }
        training_logs.append({"step": global_step, **metrics})
    pd.DataFrame(training_logs).to_csv("training_log.csv", index=False)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_bptt")
def main(cfg: DictConfig):
    """Entry point: run the BPTT training defined by `cfg`."""
    logger.info("BPTT training script started.")
    logger.info("Initializing environments...")

    local_data_path = os.environ.get("FLUIDGYM_LOCAL_DATA_PATH", "./local_data")
    fluidgym.config.update("local_data_path", to_absolute_path(local_data_path))

    env = fluidgym.make(
        cfg.env_id,
        **cfg.env_kwargs,
        differentiable=True,
        use_marl=cfg.rl_mode == "marl",
    )
    env.adjoint_rtol = cfg.get("adjoint_rtol", None)

    if cfg.get("bptt_checkpoint", False):
        env.bptt_checkpoint = True
        env.bptt_offload_device = cfg.get("bptt_offload_device", "cpu")
        env.bptt_carry_spill_dir = cfg.get("bptt_carry_spill_dir", None)
        env.bptt_segment_size = cfg.get("bptt_segment_size", None)
        logger.info(
            "BPTT checkpointing on; %d PISO step(s) per segment, %d per env step; "
            "carries -> %s",
            env.bptt_segment_size or env._n_sim_steps,
            env._n_sim_steps,
            env.bptt_carry_spill_dir or env.bptt_offload_device,
        )
    else:
        logger.warning(
            "bptt_checkpoint is off: every step of the episode keeps its "
            "activations, which a full-episode backward does not fit into."
        )

    # Before the flatten, matching train_sb3.py. The penalty is differentiable in
    # the action, so it adds an *analytic* gradient term that does not pass through
    # the solver -- the one signal in the objective that the chaotic tangent cannot
    # corrupt, and the one that pushes the policy off the actuation bound
    if cfg.power_penalty > 0:
        env = PowerPenalty(env, penalty=cfg.power_penalty)
        logger.info("Charging actuation power at %.4g * ||a||^2.", cfg.power_penalty)

    env = FlattenObservation(env)
    # Per sensor, so the mean flow profile is removed and the small temporal
    # fluctuations, e.g. of the temperature, reach the policy at order one
    obs_normalizer = None
    if cfg.get("normalize_obs", False):
        env = obs_normalizer = NormalizeObservation(env)
        logger.info("Normalizing observations by running per-sensor statistics.")
    if cfg.get("render_training", True):
        env = VideoRecorder(env, filename="train", output_path=Path("train_render"))
    env.seed(cfg.seed)
    env.train()
    logger.info(
        "Backpropagating through %d-step windows of %d-step episodes.",
        int(cfg.horizon),
        env.episode_length,
    )
    logger.info("Done.")

    device = env.cuda_device
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    obs_dim = env.observation_space.shape[0]
    action_shape = tuple(env.action_space.shape)
    action_dim = int(np.prod(action_shape))
    action_low = torch.as_tensor(
        env.action_space.low, device=device, dtype=torch.float32
    ).flatten()
    action_high = torch.as_tensor(
        env.action_space.high, device=device, dtype=torch.float32
    ).flatten()

    policy_type = cfg.get("policy_type", "mlp")
    base_policy: nn.Module
    if policy_type == "mlp":
        base_policy = BPTTPolicy(obs_dim, action_dim, action_low, action_high)
    elif policy_type == "constant":
        base_policy = ConstantPolicy(action_dim, action_low, action_high)
    else:
        raise ValueError(f"Unknown policy_type={policy_type!r}.")
    policy: nn.Module = ReshapeAction(base_policy, action_shape).to(device)
    logger.info("Training a %s policy.", policy_type)

    def constant_action_metrics() -> dict[str, float]:
        # The learned constant action, one entry per component
        with torch.no_grad():
            action = base_policy(torch.zeros(1, obs_dim, device=device))[0]
        return {f"constant_action/{i}": float(a) for i, a in enumerate(action)}

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)

    critic: QCritic | None = None
    critic_cfg = cfg.get("critic", None)
    if critic_cfg is not None and critic_cfg.enable:
        critic = QCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            policy=policy,
            gamma=float(critic_cfg.get("gamma", cfg.discount_factor)),
            units=tuple(critic_cfg.units),
            n_critics=int(critic_cfg.n_critics),
            lr=float(critic_cfg.lr),
            tau=float(critic_cfg.tau),
            batch_size=int(critic_cfg.batch_size),
            buffer_size=int(float(critic_cfg.buffer_size)),
            learning_starts=int(critic_cfg.learning_starts),
            updates_per_step=int(critic_cfg.updates_per_step),
            max_grad_norm=float(critic_cfg.max_grad_norm),
        ).to(device)
        logger.info(
            "Bootstrapping every window with a twin Q-critic (gamma=%.4g) once it "
            "has seen %d transitions.",
            critic.gamma,
            max(critic.learning_starts, critic.batch_size),
        )

    now = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(
        id=f"BPTT{cfg.horizon}_{cfg.env_id}_{cfg.seed}_{now}",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=env.id,
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
        mode="online" if cfg.wandb.enable else "offline",
    )

    logger.info("Starting BPTT training...")
    try:
        train_bptt_policy(
            env=env,
            policy=policy,
            optimizer=optimizer,
            total_timesteps=int(float(cfg.total_timesteps)),
            discount_factor=cfg.discount_factor,
            horizon=int(cfg.horizon),
            reward_scale=float(cfg.get("reward_scale", 10.0)),
            max_grad_norm=cfg.get("max_grad_norm", 1.0),
            enable_wandb=cfg.wandb.enable,
            detach_obs=cfg.detach_obs,
            extra_metrics_fn=(
                constant_action_metrics if policy_type == "constant" else None
            ),
            obs_normalizer=obs_normalizer,
            critic=critic,
        )
    finally:
        wandb.finish()

    logger.info("Training finished. Policy saved to bptt_policy.pth.")


if __name__ == "__main__":
    main()
