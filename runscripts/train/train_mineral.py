"""Train a differentiable-physics agent (SHAC) from the mineral library.

The gradient of a short-horizon return is taken through the simulator, as in
train_bptt.py, and the return beyond the horizon is bootstrapped with a learned
critic. Single-agent (``rl_mode=sarl``) and multi-agent (``rl_mode=marl``) runs are
supported; in the latter every agent is one of SHAC's parallel actors.

Usage:
    python runscripts/train/train_mineral.py algorithm=shac \
        env_id=RBCMHD3D-easy-v0 rl_mode=marl seed=0
"""

import contextlib
import logging
import os
import time
from collections import defaultdict
from datetime import datetime as dt
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import wandb
from gymnasium import spaces
from hydra.utils import to_absolute_path
from mineral.agents.diffrl.shac import SHAC  # type: ignore
from omegaconf import DictConfig, OmegaConf, open_dict

import fluidgym
from fluidgym.types import FluidEnvLike
from fluidgym.wrappers import FlattenObservation, PowerPenalty, VideoRecorder

logger = logging.getLogger("fluidgym.mineral")

# The fluidgym namespace is pinned to WARNING on import, which would swallow every
# logger.info below as well as the solver logs
fluidgym.set_verbosity(logging.INFO)

OmegaConf.register_new_resolver("eval", lambda x: eval(x), replace=True)


class StepLogger:
    """Aggregates per-step metrics over `log_every` environment steps.

    Rows are indexed by agent steps, the x axis mineral logs its own metrics on,
    i.e. environment steps times the number of agents, as `num_timesteps` in SB3.

    Parameters
    ----------
    enable_wandb: bool
        Whether the flushed means are logged to Weights & Biases.
    csv_path: str | Path
        File the rows are written to.
    log_every: int
        Number of environment steps one row averages over.
    """

    def __init__(
        self,
        enable_wandb: bool,
        csv_path: str | Path = "training_log.csv",
        log_every: int = 1,
    ):
        self._enable_wandb = enable_wandb
        self._csv_path = Path(csv_path)
        self._log_every = max(1, int(log_every))
        self._buffer: dict[str, list[float]] = defaultdict(list)
        self._n_buffered = 0
        self._step = 0
        self.logs: list[dict[str, float]] = []

        # Wall-clock per environment step. The clock starts with the first buffered
        # step, so building the environment is not counted as training time
        self._t_last_flush: float | None = None

    def add(self, metrics: dict[str, float], step: int) -> None:
        """Buffer the metrics of one environment step, flushing when due."""
        if self._t_last_flush is None:
            self._t_last_flush = time.perf_counter()
        for key, value in metrics.items():
            self._buffer[key].append(float(value))
        self._n_buffered += 1
        self._step = int(step)
        if self._n_buffered >= self._log_every:
            self.flush()

    def flush(self) -> None:
        """Log the buffered means at the current step."""
        if not self._buffer:
            return

        metrics = {key: float(np.mean(values)) for key, values in self._buffer.items()}
        now = time.perf_counter()
        assert self._t_last_flush is not None
        metrics["train/time"] = (now - self._t_last_flush) / max(1, self._n_buffered)
        self._t_last_flush = now
        self._buffer.clear()
        self._n_buffered = 0

        logger.info(
            "Step %d: %s",
            self._step,
            ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
        )
        if self._enable_wandb and wandb.run is not None:
            wandb.log(metrics, step=self._step)

        self.logs.append({"step": self._step, **metrics})
        self.save()

    def restore(self, step: int) -> None:
        """Pick up the log of a run that is resumed at `step`.

        Rows beyond `step` describe steps that are run again -- the log is written
        more often than the checkpoint -- and are dropped, so the step column stays
        monotone across the restart.
        """
        self._step = step
        if not self._csv_path.exists():
            return

        existing = pd.read_csv(self._csv_path)
        existing.to_csv(
            self._csv_path.with_name(f"{self._csv_path.stem}_backup.csv"), index=False
        )
        self.logs = [
            row for row in existing.to_dict("records") if int(row["step"]) <= step
        ]

    def save(self) -> None:
        """Write everything logged so far to the csv."""
        pd.DataFrame(self.logs).to_csv(self._csv_path, index=False)


class MineralEnvAdapter:
    """Expose a FluidGym environment through mineral's diffrl env interface.

    SHAC's contract, as in DiffRL's dflex environments:

    - ``reset()`` returns the batched initial observation ``[num_envs, num_obs]``.
    - ``step(actions)`` takes ``[num_envs, num_actions]`` actions in ``[-1, 1]`` and
      returns ``(obs, reward, done, info)``. The environment auto-resets: when an
      episode ends, ``obs`` is already the first observation of the next one, and
      the last observation of the finished one is in ``info["obs_before_reset"]``,
      still attached to the graph, since SHAC differentiates the terminal value
      bootstrap through it.
    - ``initialize_trajectory()`` cuts the graph at the start of every rollout and
      returns the current observation, ``clear_grad()`` cuts it as well.

    A reset inside a rollout builds a new ``Domain`` and ``Simulation`` while the
    checkpoint segments recorded earlier in that rollout still refer to the old
    ones; ``FluidEnv._episode_binding`` replays each segment against the episode it
    was recorded in, so the auto-reset is exact for any horizon.

    In multi-agent mode every agent is one of SHAC's parallel actors, i.e.
    ``num_envs`` is the number of agents. They share one simulation, and so the
    episode: they are done together.

    Parameters
    ----------
    env: FluidEnvLike
        The differentiable, flattened environment to wrap.
    log_freq: int
        Number of environment steps between two logs of the running averages.
    log_single_steps: bool
        Whether to additionally log reward, metrics and action of every step.
    enable_wandb: bool
        Whether to log to Weights & Biases.
    """

    def __init__(
        self,
        env: FluidEnvLike,
        log_freq: int = 1,
        log_single_steps: bool = True,
        enable_wandb: bool = True,
    ):
        self.env = env
        self.log_single_steps = log_single_steps
        self.enable_wandb = enable_wandb

        unwrapped = env.unwrapped
        if unwrapped.vectorized:
            raise ValueError("Vectorized environments (n_envs) are not supported.")

        self._use_marl = bool(unwrapped.use_marl)
        self._num_envs = unwrapped.n_agents if self._use_marl else 1

        # The env takes one action per agent of the per-agent action space, which
        # may be more than one-dimensional; SHAC acts on a flat vector per actor
        action_space = env.action_space
        assert isinstance(action_space, spaces.Box)
        self._per_actor_action_shape = tuple(action_space.shape)
        self._env_action_shape = (
            (self._num_envs, *self._per_actor_action_shape)
            if self._use_marl
            else self._per_actor_action_shape
        )
        self._action_space = spaces.Box(
            low=action_space.low.reshape(-1),
            high=action_space.high.reshape(-1),
            dtype=np.float32,
        )

        # The tanh-squashed actions of the actor are mapped onto the environment's
        # own bounds here, so no action range is assumed
        device = env.cuda_device
        low = torch.as_tensor(self._action_space.low, device=device)
        high = torch.as_tensor(self._action_space.high, device=device)
        self._action_scale = ((high - low) / 2.0).to(torch.float32)
        self._action_bias = ((high + low) / 2.0).to(torch.float32)

        # How one environment step decomposes, for step times that stay comparable
        # when the checkpoint layout changes
        self._n_piso_steps = max(1, int(unwrapped._n_sim_steps))
        self._n_segments = 1
        if unwrapped.bptt_checkpoint:
            self._n_segments = max(1, len(unwrapped._segment_sizes()))

        self.step_count = 0
        self.episode_step = 0
        self.episode_return = 0.0
        self.episode_idx = 0
        self._episode_metrics: dict[str, list[float]] = defaultdict(list)
        self._last_obs: torch.Tensor | None = None

        # Rollout wall clock of the actor update in progress
        self._update_forward_time = 0.0
        self._update_steps = 0

        self.step_logger = StepLogger(
            enable_wandb=enable_wandb, log_every=max(1, int(log_freq))
        )

    @property
    def action_space(self) -> spaces.Box:
        """The flat per-actor action space."""
        return self._action_space

    @property
    def observation_space(self) -> spaces.Box:
        """The flat per-actor observation space."""
        assert isinstance(self.env.observation_space, spaces.Box)
        return self.env.observation_space

    @property
    def num_envs(self) -> int:
        """Number of SHAC actors: the agents in MARL, 1 otherwise."""
        return self._num_envs

    @property
    def num_obs(self) -> int:
        """Size of the flat observation vector of one actor."""
        return int(self.observation_space.shape[0])

    @property
    def num_actions(self) -> int:
        """Size of the flat action vector of one actor."""
        return int(self._action_space.shape[0])

    @property
    def max_episode_length(self) -> int:
        """Number of steps after which the environment truncates an episode."""
        return int(self.env.episode_length)

    @property
    def agent_steps(self) -> int:
        """Environment steps times actors, the step mineral counts and logs at."""
        return self.step_count * self._num_envs

    def _to_batch(self, obs: torch.Tensor) -> torch.Tensor:
        """``[num_envs, num_obs]`` float32 observation, differentiably."""
        return obs.reshape(self._num_envs, -1).to(dtype=torch.float32)

    def _to_env_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Map ``[num_envs, num_actions]`` actions in ``[-1, 1]`` onto the env."""
        actions = actions.reshape(self._num_envs, self.num_actions)
        actions = actions * self._action_scale + self._action_bias
        return actions.reshape(self._env_action_shape)

    def reset(self) -> torch.Tensor:
        """Reset the environment and return the batched initial observation."""
        obs, _ = self.env.reset()
        self.episode_step = 0
        self.episode_return = 0.0
        self._episode_metrics = defaultdict(list)
        self._last_obs = self._to_batch(obs)
        return self._last_obs

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Step the environment, auto-resetting at the end of an episode."""
        env_action = self._to_env_action(actions)

        # The solver is asynchronous, so both ends are synchronised: without this
        # the measured time is whenever the next blocking read happens to fall
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        torch.cuda.synchronize()
        step_time = time.perf_counter() - t0
        self._update_forward_time += step_time
        self._update_steps += 1
        self.step_count += 1

        obs = self._to_batch(obs)
        reward = reward.reshape(self._num_envs).to(dtype=torch.float32)
        is_done = bool(terminated) or bool(truncated)
        done = torch.full(
            (self._num_envs,), is_done, dtype=torch.bool, device=self.env.cuda_device
        )

        self._log(env_action, reward, info, step_time)

        extra_info: dict[str, torch.Tensor] = {}
        if is_done:
            # Not detached: SHAC bootstraps a truncated episode with the value of
            # this observation and differentiates the return through it
            extra_info["obs_before_reset"] = obs
            self._finish_episode()
            obs = self.reset()

        self._last_obs = obs
        return obs, reward, done, extra_info

    def _log(
        self,
        action: torch.Tensor,
        reward: torch.Tensor,
        info: dict[str, torch.Tensor],
        step_time: float,
    ) -> None:
        """Buffer the metrics of the step just taken, one row per `log_freq`."""
        mean_reward = reward.detach().mean().item()
        self.episode_return += mean_reward
        self.episode_step += 1

        metrics = {"train/mean_reward": mean_reward}
        for metric in self.env.metrics:
            if metric in info:
                value = info[metric].float().mean().item()
                metrics[f"train/mean_{metric}"] = value
                self._episode_metrics[metric].append(value)

        flat_action = action.detach().reshape(-1)
        metrics["train/mean_action"] = flat_action.mean().item()
        metrics["train/mean_action_abs"] = flat_action.abs().mean().item()

        # Step times on three grids. The env step is what an epoch is counted in,
        # the PISO step is the one that stays comparable across step lengths and
        # checkpoint layouts, and the segment is what a backward replays at a time
        metrics["train/time_env_step"] = step_time
        metrics["train/time_piso_step"] = step_time / self._n_piso_steps
        metrics["train/time_segment"] = step_time / self._n_segments

        if self.log_single_steps and self.enable_wandb and wandb.run is not None:
            step_data = {
                "train/step_reward": mean_reward,
                "train/step_return": self.episode_return,
                "train/episode_step": float(self.episode_step),
            }
            for metric in self.env.metrics:
                if metric in info:
                    step_data[f"train/step_{metric}"] = (
                        info[metric].float().mean().item()
                    )
            if flat_action.numel() <= 16:
                for i, value in enumerate(flat_action.tolist()):
                    step_data[f"train/step_action_{i}"] = value
            wandb.log(step_data, step=self.agent_steps)

        self.step_logger.add(metrics, step=self.agent_steps)

    def _finish_episode(self) -> None:
        """Log the return of the episode that just ended.

        mineral's ``train_scores/episode_rewards`` is a mean over a window that is
        prefilled with zeros, so this is the per-episode return to look at.
        """
        self.episode_idx += 1
        metrics = {
            "train/episode_reward": self.episode_return,
            "train/episode_length": float(self.episode_step),
            "train/n_episodes": float(self.episode_idx),
            **{
                f"train/episode_{metric}": float(np.mean(values))
                for metric, values in self._episode_metrics.items()
            },
        }
        logger.info(
            "Episode %d finished after %d step(s): return %.4f",
            self.episode_idx,
            self.episode_step,
            self.episode_return,
        )
        if self.enable_wandb and wandb.run is not None:
            wandb.log(metrics, step=self.agent_steps)

    def initialize_trajectory(self) -> torch.Tensor:
        """Cut the gradient graph between two rollouts and return the obs."""
        assert self._last_obs is not None, "reset() must be called before training."
        self.clear_grad()
        self._last_obs = self._last_obs.detach()

        # The previous rollout is gone by now, so what it reserved can go back to
        # the driver. One call per actor update, against a rollout of minutes
        torch.cuda.empty_cache()

        return self._last_obs

    def clear_grad(self) -> None:
        """Detach the simulation state from the computation graph.

        SHAC calls this once before the very first reset, when no simulation exists
        yet, so it is a no-op until the environment has been reset.
        """
        if self._last_obs is None:
            return
        self.env.detach()

    def resume_from(self, agent_steps: int) -> None:
        """Continue the step counting and the training log of an earlier run."""
        self.step_count = agent_steps // self._num_envs
        self.step_logger.restore(agent_steps)
        self.episode_idx = self.step_count // max(1, self.max_episode_length)
        logger.info(
            "Resuming at step %d, %d logged row(s) kept.",
            agent_steps,
            len(self.step_logger.logs),
        )

    def begin_actor_update(self) -> None:
        """Start the rollout accounting of one actor update."""
        self._update_forward_time = 0.0
        self._update_steps = 0

    def end_actor_update(self, total_time: float) -> None:
        """Report the forward/backward split of the actor update that just ran."""
        steps = max(1, self._update_steps)
        forward = self._update_forward_time
        backward = max(0.0, total_time - forward)

        timings = {
            "train_timings/actor_update": total_time,
            "train_timings/forward_env_step": forward / steps,
            "train_timings/backward_env_step": backward / steps,
            "train_timings/forward_piso_step": forward / (steps * self._n_piso_steps),
            "train_timings/backward_piso_step": backward / (steps * self._n_piso_steps),
            "train_timings/backward_segment": backward / (steps * self._n_segments),
        }
        logger.info(
            "Actor update over %d env step(s) in %.1f s: forward %.1f s "
            "(%.3f s/PISO step), backward %.1f s (%.3f s/PISO step).",
            steps,
            total_time,
            forward,
            timings["train_timings/forward_piso_step"],
            backward,
            timings["train_timings/backward_piso_step"],
        )
        if self.enable_wandb and wandb.run is not None:
            wandb.log(timings, step=self.agent_steps)

    def finish(self) -> None:
        """Flush the partial logging window at the end of training."""
        self.step_logger.flush()
        self.step_logger.save()


def time_actor_updates(model: SHAC, env: MineralEnvAdapter) -> None:
    """Report how an actor update splits into rollout forward and backward."""
    update_actor = model.update_actor

    def timed_update_actor():
        env.begin_actor_update()
        t0 = time.perf_counter()
        try:
            return update_actor()
        finally:
            env.end_actor_update(time.perf_counter() - t0)

    model.update_actor = timed_update_actor


def fix_episode_trackers(model: SHAC) -> None:
    """Empty mineral's episode trackers, which start prefilled with zeros.

    ``Tracker`` fills its window with ``tracker_len`` zeros, so until that many
    episodes have finished, ``train_scores/episode_rewards`` is the sum of the
    returns so far divided by the window length, a monotone curve that is not a
    learning curve, and the best checkpoint is picked on it. With the window
    emptied, the mean is over the episodes that actually finished.
    """
    for tracker in (
        model.episode_rewards_tracker,
        model.episode_lengths_tracker,
        model.episode_discounted_rewards_tracker,
    ):
        tracker.window.clear()


def fix_single_env_obs_rms(model: SHAC, env: MineralEnvAdapter) -> None:
    """Make the observation normaliser well-defined for a single actor.

    Unless ``network.tanh_clamp`` is set, mineral builds its `RunningMeanStd` with
    the Bessel correction, which is ``0 / 0`` over one actor and turns every
    normalised observation into NaN on the first step.
    """
    if env.num_envs > 1 or getattr(model, "obs_rms", None) is None:
        return
    for key, rms in model.obs_rms.items():
        if getattr(rms, "correction", 0) != 0:
            rms.correction = 0
            logger.info("Set obs_rms[%s].correction = 0 for the single actor.", key)


@contextlib.contextmanager
def patch_attr(obj, name, value):
    """Temporarily set `obj.name` to `value`."""
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def keep_optimizer_state(model: SHAC) -> None:
    """Extend mineral's checkpoints with the optimizer states it leaves out."""
    base_save, base_load = model.save, model.load

    def save(f):
        base_save(f)
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
        ckpt["actor_optim"] = model.actor_optim.state_dict()
        ckpt["critic_optim"] = model.critic_optim.state_dict()
        ckpt["best_stat"] = model.best_stat
        torch.save(ckpt, f)

    def load(f, ckpt_keys=""):
        # mineral's load uses torch.load's default weights_only=True, which refuses
        # the optimizer states; the checkpoints are written by this script
        base_torch_load = torch.load

        def trusted_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return base_torch_load(*args, **kwargs)

        with patch_attr(torch, "load", trusted_load):
            base_load(f, ckpt_keys)
        ckpt = torch.load(f, map_location=model.device, weights_only=False)
        for name in ("actor_optim", "critic_optim"):
            if name in ckpt:
                getattr(model, name).load_state_dict(ckpt[name])
            else:
                logger.warning("Checkpoint has no `%s`, keeping the new one.", name)
        if ckpt.get("best_stat") is not None:
            model.best_stat = ckpt["best_stat"]

    model.save = save
    model.load = load


def restore_checkpoint(model: SHAC, env: MineralEnvAdapter) -> None:
    """Resume `model` and the training log from the latest checkpoint."""
    ckpt_path = Path(model.ckpt_dir) / "latest.pth"
    if not ckpt_path.exists():
        logger.warning(
            "continue_training is set, but %s does not exist. Starting fresh.",
            ckpt_path,
        )
        return

    logger.info("Continuing training from %s...", ckpt_path)
    model.load(str(ckpt_path))
    env.resume_from(int(model.agent_steps))
    logger.info(
        "Restored epoch %d, %d agent step(s) of %d.",
        model.epoch,
        model.agent_steps,
        model.max_agent_steps,
    )


def make_env(cfg: DictConfig) -> FluidEnvLike:
    """Build the differentiable, flattened training environment."""
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
            "bptt_checkpoint is off: every step of the rollout keeps its activations."
        )

    # Before the flatten, matching train_sb3.py and train_bptt.py
    if cfg.power_penalty > 0:
        env = PowerPenalty(env, penalty=cfg.power_penalty)
        logger.info("Charging actuation power at %.4g * ||a||^2.", cfg.power_penalty)

    env = FlattenObservation(env)
    if cfg.get("render_training", False):
        env = VideoRecorder(env, filename="train", output_path=Path("train_render"))
    env.seed(cfg.seed)
    env.train()
    return env


def configure_agent(cfg: DictConfig, env: MineralEnvAdapter) -> None:
    """Set what SHAC's config cannot know before the environment exists.

    ``total_timesteps`` counts environment steps, as in train_sb3.py; SHAC counts
    agent steps, ``horizon_len * num_envs`` per epoch, so its budget is scaled by
    the number of actors, which is also SB3's ``total_timesteps * n_agents``.
    """
    total_timesteps = int(float(cfg.total_timesteps))
    horizon_len = int(cfg.agent.shac.horizon_len)
    with open_dict(cfg):
        cfg.agent.shac.num_actors = env.num_envs
        cfg.agent.shac.max_agent_steps = total_timesteps * env.num_envs
        cfg.agent.shac.max_epochs = -(-total_timesteps // horizon_len)

    if env.max_episode_length % horizon_len:
        logger.info(
            "horizon_len %d does not divide the episode length %d; episodes end "
            "inside rollouts and are auto-reset there.",
            horizon_len,
            env.max_episode_length,
        )


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_mineral")
def main(cfg: DictConfig):
    """Entry point: run the SHAC training defined by `cfg`."""
    try:
        run_experiment(cfg)
    except Exception as e:
        logger.exception("Job crashed with an exception")
        logger.error(e)
        raise


def run_experiment(cfg: DictConfig):
    """Build the environment and the SHAC agent, then train."""
    logger.info("Training script started.")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger.info("Initializing environments...")
    local_data_path = os.environ.get("FLUIDGYM_LOCAL_DATA_PATH", "./local_data")
    fluidgym.config.update("local_data_path", to_absolute_path(local_data_path))
    fluid_env = make_env(cfg)
    env = MineralEnvAdapter(
        fluid_env,
        log_freq=cfg.log_freq,
        log_single_steps=cfg.log_single_steps,
        enable_wandb=cfg.wandb.enable,
    )
    configure_agent(cfg, env)
    logger.info(
        "%d actor(s), %d-step rollouts of %d-step episodes, %d epoch(s).",
        env.num_envs,
        cfg.agent.shac.horizon_len,
        env.max_episode_length,
        cfg.agent.shac.max_epochs,
    )
    logger.info("Done.")

    now = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{cfg.agent.algo}_{cfg.env_id}_{cfg.seed}_{now}"
    run_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(run_config, dict), "Run config must be a dictionary."

    # mineral logs through wandb unconditionally, so a run always exists
    logger.info("Initializing Weights & Biases...")
    wandb.init(
        id=run_id,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=run_config,  # type: ignore
        group=fluid_env.id,
        mode="online" if cfg.wandb.enable else "disabled",
    )
    logger.info("Done.")

    logger.info("Initializing model...")
    model = SHAC(cfg, env=env, logdir=os.getcwd())
    time_actor_updates(model, env)
    fix_episode_trackers(model)
    fix_single_env_obs_rms(model, env)
    keep_optimizer_state(model)

    if cfg.get("continue_training", False):
        restore_checkpoint(model, env)
    logger.info("Done.")
    logger.info("Checkpoints every %s epoch(s) -> %s", model.ckpt_every, model.ckpt_dir)

    logger.info("Starting training...")
    try:
        model.train()
    finally:
        env.finish()
        fluid_env.close()
        wandb.finish()
    logger.info(
        "Training finished. Final model saved to %s.",
        os.path.join(model.ckpt_dir, "final.pth"),
    )


if __name__ == "__main__":
    main()
