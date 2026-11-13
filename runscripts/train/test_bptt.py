"""Run test episodes with a policy trained by `train_bptt.py`, as `test_sb3.py` does.

Runs in the training directory of the policy (the hydra run dir of `train_bptt`) and
loads ``policy_checkpoint`` and, if present, its observation statistics
(``<stem>_obs_norm.pth``). The results are written like those of `test_sb3.py`, so
`plot_experiments.py` reads them.

Without `domain_idx`, one episode is run on each of the domains
``range(n_test_episodes)``, otherwise only on the given domain. Every episode gets
its own directory ``<k>/`` and a fresh environment; the first episode is rendered
as gif, together with its initial and final frames, and its final domain is saved.

Usage:
    python runscripts/train/test_bptt.py env_id=RBCMHD3D-easy-v0 \
        rl_mode=marl seed=0 [domain_idx=0]
"""

import logging
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from phipict.io.domain_io import save_domain
from stable_baselines3.common.base_class import BaseAlgorithm

import fluidgym
from fluidgym.integration.gymnasium import GymFluidEnv
from fluidgym.integration.sb3 import VecFluidEnv
from fluidgym.wrappers import FlattenObservation, PowerPenalty, VideoRecorder

# The fluidgym namespace is pinned to WARNING on import, which would swallow the
# solver logs
fluidgym.set_verbosity(logging.INFO)

logger = logging.getLogger("bptt.testing")


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


def evaluate_episode(
    env: GymFluidEnv | VecFluidEnv,
    model: BaseAlgorithm,
    domain_idx: int,
    output_path: Path,
    save_name: str = "test_eval_episode_0",
    deterministic: bool = True,
    save_final_domain: bool = True,
    save_renders: bool = False,
    render_3d: bool = False,
) -> pd.DataFrame:
    """Run one episode on `domain_idx`, returning the per-step log.

    The gif is left to a `VideoRecorder` wrapping the environment, see
    `make_test_env`.

    Parameters
    ----------
    env: GymFluidEnv | VecFluidEnv
        The test environment.

    model: BaseAlgorithm
        The trained model.

    domain_idx: int
        Index of the initial domain to evaluate on.

    output_path: Path
        The path to save the final domain to.

    save_name: str
        The base name of the final domain. Defaults to "test_eval_episode_0".

    deterministic: bool
        Whether to use deterministic actions. Defaults to True.

    save_final_domain: bool
        Whether to save the final domain. Defaults to True.

    save_renders: bool
        Whether to save the initial and final frame as PNG to `output_path`.
        Defaults to False.

    render_3d: bool
        Whether to render the initial and final frame in 3d. Defaults to False.

    Returns
    -------
    pd.DataFrame
        The per-step reward, action and environment metrics.
    """
    is_marl = isinstance(env, VecFluidEnv)
    env_metrics = env.unwrapped.metrics

    obs = env.reset(domain_idx=domain_idx)
    if isinstance(obs, tuple):
        obs = obs[0]

    if save_renders:
        env.unwrapped.render(
            save=True,
            render_3d=render_3d,
            filename=f"{save_name}_initial",
            output_path=output_path,
        )

    rows = []
    step = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)  # type: ignore

        if is_marl:
            obs, rewards, dones, infos = env.step(action[:, None])
            info = infos[0]
            done = bool(np.any(dones))
            # The per-agent rewards are local, the shared objective is the global one
            reward = float(info["global_reward"])
            local_reward = float(np.mean(rewards))
        else:
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            reward = float(reward)
            local_reward = None

        row = {
            "domain_idx": domain_idx,
            "step": step,
            "reward": reward,
            **{
                metric: float(np.mean(info[metric]))
                for metric in env_metrics
                if metric in info
            },
        }
        if local_reward is not None:
            row["local_reward"] = local_reward
        for a_idx, value in enumerate(np.asarray(action).reshape(-1)):
            row[f"action_{a_idx}"] = float(value)

        rows.append(row)
        step += 1

    if save_renders:
        env.unwrapped.render(
            save=True,
            render_3d=render_3d,
            filename=f"{save_name}_final",
            output_path=output_path,
        )

    if save_final_domain:
        save_domain(
            env.unwrapped._domain, output_path / f"{save_name}_final_domain"
        )

    return pd.DataFrame(rows)


def make_test_env(
    cfg: DictConfig, output_path: Path, record: bool = False
) -> GymFluidEnv | VecFluidEnv:
    """Create the test environment, optionally recording it with a `VideoRecorder`.

    When recording, the episode is saved as ``<output_path>/test_ep<k>.gif`` when
    the environment is closed. With `cfg.save_frames`, every rendered frame is
    saved as PNG to `output_path` as well.

    Parameters
    ----------
    cfg: DictConfig
        The test configuration.

    output_path: Path
        The path to save the gif and frames to.

    record: bool
        Whether to wrap the environment in a `VideoRecorder`. Defaults to False.

    Returns
    -------
    GymFluidEnv | VecFluidEnv
        The test environment, seeded and in test mode.
    """
    fluid_test_env = fluidgym.make(
        cfg.test_env_id,
        use_marl=cfg.test_rl_mode == "marl",
        **cfg.test_env_kwargs,
    )

    if cfg.power_penalty > 0:
        fluid_test_env = PowerPenalty(fluid_test_env, penalty=cfg.power_penalty)

    fluid_test_env = FlattenObservation(fluid_test_env)
    if record:
        fluid_test_env = VideoRecorder(
            fluid_test_env,
            filename="test",
            output_path=output_path,
            render_kwargs={
                "render_3d": cfg.render_3d,
                "save": cfg.save_frames,
                "output_path": output_path,
            },
        )

    test_env: GymFluidEnv | VecFluidEnv
    if cfg.test_rl_mode == "marl":
        test_env = VecFluidEnv(fluid_test_env, auto_reset=False)
    elif cfg.test_rl_mode == "sarl":
        test_env = GymFluidEnv(fluid_test_env)
    else:
        raise ValueError(f"Unknown test rl_mode: {cfg.test_rl_mode}")
    test_env.seed(cfg.seed + 84)
    test_env.test()
    return test_env


def set_local_data_path() -> None:
    """Point fluidgym to the local data, resolved against the launch directory.

    Must run inside the hydra job: a Slurm worker does not execute the module-level
    code of this script, and the job changes into its run directory.
    """
    local_data_path = os.environ.get("FLUIDGYM_LOCAL_DATA_PATH", "./local_data")
    fluidgym.config.update("local_data_path", to_absolute_path(local_data_path))


class BPTTPolicyModel:
    """A drop-in replacement for an SB3 model that runs a BPTT policy.

    Normalizes the observation with the frozen training statistics, as the
    ``NormalizeObservation`` wrapper of `train_bptt.py` does in test mode.
    """

    def __init__(
        self,
        env: GymFluidEnv | VecFluidEnv,
        policy: torch.nn.Module,
        obs_norm: dict[str, Any] | None,
        device: torch.device,
        epsilon: float = 1e-8,
        clip: float | None = 10.0,
    ):
        self.policy = policy.eval()
        self.device = device
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        if obs_norm is not None:
            self.mean = torch.as_tensor(obs_norm["mean"], device=device)
            var = torch.as_tensor(obs_norm["var"], device=device)
            self.std = (var + epsilon).sqrt()
        self.clip = clip

    @torch.no_grad()
    def predict(self, obs: Any, deterministic: bool = True) -> tuple[np.ndarray, None]:
        """Return the policy's action for ``obs``."""
        x = torch.as_tensor(np.asarray(obs), device=self.device, dtype=torch.float64)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
            if self.clip is not None:
                x = x.clamp(-self.clip, self.clip)
        # Already in the env's action shape, with the agents leading in MARL
        return self.policy(x.float()).cpu().numpy(), None


def load_policy(
    cfg: DictConfig, env: GymFluidEnv | VecFluidEnv, device: torch.device
) -> torch.nn.Module:
    """Build the policy of `train_bptt.py` for ``env`` and load its weights."""
    obs_dim = int(np.prod(env.observation_space.shape))
    action_shape = tuple(env.action_space.shape)
    action_dim = int(np.prod(action_shape))
    low = torch.as_tensor(env.action_space.low, dtype=torch.float32).flatten()
    high = torch.as_tensor(env.action_space.high, dtype=torch.float32).flatten()

    base: torch.nn.Module
    if cfg.policy_type == "mlp":
        base = BPTTPolicy(obs_dim, action_dim, low, high)
    elif cfg.policy_type == "constant":
        base = ConstantPolicy(action_dim, low, high)
    else:
        raise ValueError(f"Unknown policy_type={cfg.policy_type!r}.")
    policy = ReshapeAction(base, action_shape)
    policy.load_state_dict(torch.load(cfg.policy_checkpoint, map_location="cpu"))
    return policy.to(device)


@hydra.main(version_base="1.3", config_path="../configs", config_name="test_bptt")
def main(cfg: DictConfig):
    try:
        run_experiment(cfg)
    except Exception as e:
        logger.exception("Job crashed with an exception")
        logger.error(e)
        raise


def run_experiment(cfg: DictConfig):
    logger.setLevel(logging.INFO)
    # Importing train_bptt turns the fluidgym and solver logs on
    fluidgym.set_verbosity(logging.WARNING)
    logger.info("BPTT testing script started.")
    # Importing train_bptt sets its own default, so the path is set again here
    set_local_data_path()

    if cfg.env_id != cfg.test_env_id:
        base_path = Path(f"./transfer/{cfg.test_env_id}/")
    else:
        base_path = Path("./test/")

    if cfg.domain_idx is None:
        domain_idxs = list(range(cfg.n_test_episodes))
    else:
        domain_idxs = [int(cfg.domain_idx)]

    model: BPTTPolicyModel | None = None
    for episode, domain_idx in enumerate(domain_idxs):
        output_path = base_path / str(domain_idx)
        output_path.mkdir(parents=True, exist_ok=True)
        # Only the first episode is rendered
        record = episode == 0

        logger.info(f"Initializing test environment for domain {domain_idx}...")
        test_env = make_test_env(cfg, output_path, record=record)
        logger.info("Done.")

        if model is None:
            device = test_env.unwrapped.cuda_device
            policy = load_policy(cfg, test_env, device)
            norm_path = Path(cfg.policy_checkpoint)
            norm_path = norm_path.with_name(f"{norm_path.stem}_obs_norm.pth")
            obs_norm = (
                torch.load(norm_path, map_location="cpu")
                if norm_path.exists()
                else None
            )
            if cfg.get("normalize_obs", False) and obs_norm is None:
                raise FileNotFoundError(
                    f"normalize_obs is set, but {norm_path} is missing."
                )
            model = BPTTPolicyModel(test_env, policy, obs_norm, device)
            logger.info(
                f"Loaded {cfg.policy_checkpoint} (obs norm: {obs_norm is not None})."
            )

        logger.info(f"Evaluating one episode on domain {domain_idx}...")
        results = evaluate_episode(
            env=test_env,
            model=model,  # type: ignore[arg-type]
            domain_idx=domain_idx,
            output_path=output_path,
            deterministic=True,
            save_final_domain=cfg.save_final_domain and record,
            save_renders=record,
            render_3d=cfg.render_3d,
        )
        # Saves the gif of a recorded episode
        test_env.close()

        results_path = output_path / "test_results.csv"
        results.to_csv(results_path, index=False)

        means = results.drop(columns=["domain_idx", "step"]).mean()
        logger.info(
            f"Test metrics (domain {domain_idx}): "
            + ", ".join([f"mean_{k}={v:.4f}" for k, v in means.items()])
        )
        logger.info(f"Wrote {len(results)} steps to {results_path.resolve()}.")


if __name__ == "__main__":
    main()
