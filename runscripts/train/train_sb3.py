"""Train a model-free agent (PPO or SAC) from Stable Baselines 3.

Usage:
    python runscripts/train/train_sb3.py algorithm=sb3_sac \
        env_id=RBCMHD3D-easy-v0 rl_mode=marl seed=0
"""

import logging
import os
from datetime import datetime as dt
from pathlib import Path

import hydra
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", lambda x: eval(x))

@hydra.main(version_base="1.3", config_path="../configs", config_name="train_sb3")
def main(cfg: DictConfig):
    try:
        run_experiment(cfg)
    except Exception as e:
        logger.exception("Job crashed with an exception")
        logger.error(e)
        raise


def run_experiment(cfg: DictConfig):
    logger.info("Training script started.")

    import fluidgym
    from fluidgym.integration.gymnasium import GymFluidEnv
    from fluidgym.integration.sb3 import EvalCallback, VecFluidEnv, load_buffer
    from fluidgym.wrappers import FlattenObservation, PowerPenalty, VideoRecorder

    local_data_path = os.environ.get("FLUIDGYM_LOCAL_DATA_PATH", "./local_data")
    fluidgym.config.update("local_data_path", to_absolute_path(local_data_path))

    logger.info("Initializing environments...")
    fluid_env = fluidgym.make(
        cfg.env_id,
        use_marl=cfg.rl_mode == "marl",
        n_envs=cfg.get("n_envs", None),
        **cfg.env_kwargs,
    ) 

    fluid_env = FlattenObservation(fluid_env)

    if cfg.render_training:
        # Every training episode is saved as train_videos/train_ep<k>.gif
        fluid_env = VideoRecorder(
            fluid_env, filename="train", output_path=Path("train_videos")
        )

    env: GymFluidEnv | VecFluidEnv
    if cfg.rl_mode == "marl":
        env = VecFluidEnv(fluid_env)
    elif cfg.rl_mode == "sarl":
        if cfg.get("n_envs", None) is not None:
            env = VecFluidEnv(fluid_env)
        else:
            env = GymFluidEnv(fluid_env)
    else:
        raise ValueError(f"Unknown rl_mode: {cfg.rl_mode}")
    env.seed(cfg.seed)

    eval_env: GymFluidEnv | VecFluidEnv | None = None
    if cfg.n_eval_steps > 0:
        fluid_eval_env = fluidgym.make(
            cfg.env_id,
            use_marl=cfg.rl_mode == "marl",
            **cfg.eval_env_kwargs,
        )

        if cfg.power_penalty > 0:
            fluid_eval_env = PowerPenalty(fluid_eval_env, penalty=cfg.power_penalty)

        fluid_eval_env = FlattenObservation(fluid_eval_env)

        if cfg.rl_mode == "marl":
            eval_env = VecFluidEnv(fluid_eval_env, auto_reset=False)
        else:
            eval_env = GymFluidEnv(fluid_eval_env)
        eval_env.seed(cfg.seed + 42)
        eval_env.val()
    else:
        logger.info("Evaluation disabled (n_eval_steps <= 0), no eval env created.")
    logger.info("Done.")

    now = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{cfg.algorithm.name}_{cfg.env_id}_{cfg.seed}_{now}"
    run_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    assert isinstance(run_config, dict), "Run config must be a dictionary."

    eval_callback: EvalCallback = hydra.utils.instantiate(
        cfg.eval_callback,
        env=env,
        eval_env=eval_env,
    )

    logger.info("Initializing model...")
    model: BaseAlgorithm = hydra.utils.instantiate(cfg.algorithm.obj, env=env)
    if cfg.continue_training:
        logger.info("Continuing training from latest checkpoint...")
        model = model.load("ckpt_latest", env=env, device=cfg.rl_device)
        if isinstance(model, SAC):
            load_buffer(model, "ckpt_latest")
    logger.info("Done.")

    if cfg.wandb.enable:
        logger.info("Initializing Weights & Biases...")
        wandb.init(
            id=run_id,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=run_config,  # type: ignore
            group=fluid_env.id,
        )
        logger.info("Done.")

    logger.info("Starting training...")
    total_timesteps = cfg.total_timesteps * env.unwrapped.n_agents
    remaining_timesteps = total_timesteps - model.num_timesteps
    model.learn(
        total_timesteps=remaining_timesteps,
        progress_bar=True,
        callback=eval_callback,
        reset_num_timesteps=False,
    )
    # Saves the gif of the last, unfinished training episode
    env.close()
    if cfg.wandb.enable:
        wandb.finish()
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
