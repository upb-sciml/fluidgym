"""Run test episodes with a SHAC agent trained by `train_mineral.py`, as `test_sb3.py`
does.

Runs in the training directory of the agent (the hydra run dir of `train_mineral`)
and loads the actor and the observation statistics from ``policy_checkpoint``. The
results are written like those of `test_sb3.py`, so `plot_experiments.py` reads
them.

Without `domain_idx`, one episode is run on each of the domains
``range(n_test_episodes)``, otherwise only on the given domain. Every episode gets
its own directory ``<k>/`` and a fresh environment; the first episode is rendered
as gif, together with its initial and final frames, and its final domain is saved.

Usage:
    python runscripts/train/test_mineral.py algorithm=shac \
        env_id=RBCMHD3D-easy-v0 rl_mode=marl seed=0 [domain_idx=0]
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
from mineral.agents.diffrl import models  # type: ignore
from mineral.common.normalizers import RunningMeanStd  # type: ignore
from omegaconf import DictConfig, OmegaConf
from phipict.io.domain_io import save_domain
from stable_baselines3.common.base_class import BaseAlgorithm

import fluidgym
from fluidgym.integration.gymnasium import GymFluidEnv
from fluidgym.integration.sb3 import VecFluidEnv
from fluidgym.wrappers import FlattenObservation, PowerPenalty, VideoRecorder

# Not below the fluidgym namespace, which fluidgym pins to WARNING
logger = logging.getLogger("mineral.testing")


OmegaConf.register_new_resolver("eval", lambda x: eval(x), replace=True)


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


class MineralPolicyModel:
    """A drop-in replacement for an SB3 model that runs a SHAC actor.

    Reproduces ``SHAC.get_actions`` on observations normalised with the frozen
    training statistics, and maps the action onto the environment's bounds as the
    ``MineralEnvAdapter`` of `train_mineral.py` does.
    """

    def __init__(
        self,
        env: GymFluidEnv | VecFluidEnv,
        actor: nn.Module,
        obs_rms: nn.Module | None,
        tanh_clamp: bool,
        device: torch.device,
    ):
        self.actor = actor.eval()
        self.obs_rms = obs_rms
        self.tanh_clamp = tanh_clamp
        self.device = device

        self.action_shape = tuple(env.action_space.shape)
        low = torch.as_tensor(env.action_space.low, device=device).flatten()
        high = torch.as_tensor(env.action_space.high, device=device).flatten()
        self.action_scale = ((high - low) / 2.0).to(torch.float32)
        self.action_bias = ((high + low) / 2.0).to(torch.float32)

    @torch.no_grad()
    def predict(self, obs: Any, deterministic: bool = True) -> tuple[np.ndarray, None]:
        """Return the actor's action for ``obs``."""
        x = torch.as_tensor(np.asarray(obs), device=self.device, dtype=torch.float32)
        if self.obs_rms is not None:
            x = self.obs_rms.normalize(x)

        mu, _, distr = self.actor(x)
        # The mean action for a deterministic rollout, a sample otherwise
        action = mu if deterministic else distr.sample()
        if self.tanh_clamp:
            action = torch.tanh(action)

        action = action * self.action_scale + self.action_bias
        # In the env's action shape, with the agents leading in MARL
        action = action.reshape(*action.shape[:-1], *self.action_shape)
        return action.cpu().numpy(), None


def load_model(
    cfg: DictConfig, env: GymFluidEnv | VecFluidEnv, device: torch.device
) -> MineralPolicyModel:
    """Build the actor of `train_mineral.py` for ``env`` and load its weights."""
    network_cfg = cfg.agent.network
    if network_cfg.get("encoder", None) is not None:
        raise ValueError("Agents with an observation encoder are not supported.")

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    ActorCls = getattr(models, network_cfg.actor)
    actor = ActorCls(obs_dim, action_dim, **network_cfg.get("actor_kwargs", {}))

    # The checkpoint is written by train_mineral.py and holds the optimizer states
    ckpt = torch.load(cfg.policy_checkpoint, map_location="cpu", weights_only=False)
    actor.load_state_dict(ckpt["actor"])

    # The normaliser as SHAC builds it, only its statistics are loaded
    tanh_clamp = bool(network_cfg.get("tanh_clamp", False))
    obs_rms: nn.Module | None = None
    if network_cfg.get("normalize_input", False):
        if tanh_clamp:
            rms_config = dict(
                eps=1e-5, correction=0, initial_count=1e-4, dtype=torch.float32
            )
        else:
            rms_config = dict(eps=1e-5, initial_count=1, dtype=torch.float64)
        obs_rms = nn.ModuleDict({"obs": RunningMeanStd((obs_dim,), **rms_config)})
        obs_rms.load_state_dict(ckpt["obs_rms"])
        obs_rms = obs_rms["obs"].to(device)

    return MineralPolicyModel(env, actor.to(device), obs_rms, tanh_clamp, device)


@hydra.main(version_base="1.3", config_path="../configs", config_name="test_mineral")
def main(cfg: DictConfig):
    try:
        run_experiment(cfg)
    except Exception as e:
        logger.exception("Job crashed with an exception")
        logger.error(e)
        raise


def run_experiment(cfg: DictConfig):
    logger.setLevel(logging.INFO)
    logger.info("Mineral testing script started.")
    set_local_data_path()

    if cfg.env_id != cfg.test_env_id:
        base_path = Path(f"./transfer/{cfg.test_env_id}/")
    else:
        base_path = Path("./test/")

    if cfg.domain_idx is None:
        domain_idxs = list(range(cfg.n_test_episodes))
    else:
        domain_idxs = [int(cfg.domain_idx)]

    model: MineralPolicyModel | None = None
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
            model = load_model(cfg, test_env, device)
            logger.info(
                f"Loaded {cfg.policy_checkpoint} "
                f"(obs norm: {model.obs_rms is not None})."
            )

        logger.info(f"Evaluating one episode on domain {domain_idx}...")
        results = evaluate_episode(
            env=test_env,
            model=model,  # type: ignore[arg-type]
            domain_idx=domain_idx,
            output_path=output_path,
            deterministic=cfg.deterministic,
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
