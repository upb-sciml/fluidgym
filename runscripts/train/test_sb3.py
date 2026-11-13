"""Run deterministic test episodes with an SB3 agent trained by `train_sb3.py`.

Without `domain_idx`, one episode is run on each of the domains
``range(n_test_episodes)``, otherwise only on the given domain. Every episode gets
its own directory ``<k>/`` and a fresh environment; the first episode is rendered
as gif, together with its initial and final frames, and its final domain is saved.

Usage:
    python runscripts/train/test_sb3.py algorithm=sb3_sac \
        env_id=RBCMHD3D-easy-v0 rl_mode=marl seed=0 [domain_idx=0]
"""

import logging
import os
import warnings
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import to_absolute_path
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from phipict.io.domain_io import save_domain
from stable_baselines3.common.base_class import BaseAlgorithm

import fluidgym
from fluidgym.integration.gymnasium import GymFluidEnv
from fluidgym.integration.sb3 import VecFluidEnv
from fluidgym.wrappers import FlattenObservation, PowerPenalty, VideoRecorder

# Not below the fluidgym namespace, which fluidgym pins to WARNING
logger = logging.getLogger("sb3.testing")


OmegaConf.register_new_resolver("eval", lambda x: eval(x))


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


@hydra.main(version_base="1.3", config_path="../configs", config_name="test_sb3")
def main(cfg: DictConfig):
    try:
        run_experiment(cfg)
    except Exception as e:
        logger.exception("Job crashed with an exception")
        logger.error(e)
        raise


def load_model(cfg: DictConfig, env: GymFluidEnv | VecFluidEnv) -> BaseAlgorithm:
    """Load the latest checkpoint of the trained model.

    Parameters
    ----------
    cfg: DictConfig
        The test configuration.

    env: GymFluidEnv | VecFluidEnv
        The test environment.

    Returns
    -------
    BaseAlgorithm
        The trained model.
    """
    with warnings.catch_warnings():
        # Ignore the warning 'len(rollout_buffer) % batch_size warning != 0'
        # when transfering a MARL policy to an env with a different number of agents
        warnings.simplefilter("ignore", category=UserWarning)
        model: BaseAlgorithm = hydra.utils.instantiate(cfg.algorithm.obj, env=env)
    return model.load("ckpt_latest", env=env, device=cfg.rl_device)


def set_local_data_path() -> None:
    """Point fluidgym to the local data, resolved against the launch directory.

    Must run inside the hydra job: a Slurm worker does not execute the module-level
    code of this script, and the job changes into its run directory.
    """
    local_data_path = os.environ.get("FLUIDGYM_LOCAL_DATA_PATH", "./local_data")
    fluidgym.config.update("local_data_path", to_absolute_path(local_data_path))


def run_experiment(cfg: DictConfig):
    logger.info("Testing script started.")
    set_local_data_path()

    if cfg.env_id != cfg.test_env_id:
        base_path = Path(f"./transfer/{cfg.test_env_id}/")
    else:
        base_path = Path("./test/")

    if cfg.domain_idx is None:
        domain_idxs = list(range(cfg.n_test_episodes))
    else:
        domain_idxs = [int(cfg.domain_idx)]

    model: BaseAlgorithm | None = None
    for episode, domain_idx in enumerate(domain_idxs):
        output_path = base_path / str(domain_idx)
        output_path.mkdir(parents=True, exist_ok=True)
        # Only the first episode is rendered
        record = episode == 0

        logger.info(f"Initializing test environment for domain {domain_idx}...")
        test_env = make_test_env(cfg, output_path, record=record)
        logger.info("Done.")

        if model is None:
            logger.info("Initializing model...")
            model = load_model(cfg, test_env)
            logger.info("Done.")

        logger.info(f"Evaluating one episode on domain {domain_idx}...")
        results = evaluate_episode(
            env=test_env,
            model=model,
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
