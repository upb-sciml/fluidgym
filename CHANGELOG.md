# Changelog

All notable changes to FluidGym are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-12

### Highlights

- **New solver: [phiPICT](https://github.com/upb-sciml/phiPICT).** The PICT-based
  solver moved to its own package, phiPICT, which ships as prebuilt wheels on PyPI.
  FluidGym itself is now pure Python, so `pip install fluidgym` no longer compiles CUDA code. So we keep the easy plug-and-play installation! 🙂
- **Batched environments.** `fluidgym.make(id, n_envs=N)` simulates `N` environments
  in a single simulation on one GPU. They share the grid, the solver setup and every
  kernel launch. `fluidgym.make_vec(id, n_envs=N, devices=[...])` also splits them
  across several GPUs. Observations, actions, rewards and flags get a leading env
  dim. See
  [`examples/advanced_usage/parallelization`](examples/advanced_usage/parallelization).
- **Much faster.** phiPICT and batching together make training on average **~20×**
  faster, and over **100×** faster for some environments.
- **Magnetohydrodynamics (MHD).** Adds the new `RBCMHD3D-{difficulty}-v0`
  environments: Rayleigh-Bénard convection of a liquid metal in a horizontal magnetic
  field, controlled by current electrodes in the Hartmann walls.
- **Improved differentiability.** Activation checkpointing for BPTT, with carries
  optionally offloaded to the CPU or spilled to disk. Also adds configurable adjoint
  tolerances and new examples for observation Jacobians and state VJPs. See
  [`examples/advanced_usage/differentiable`](examples/advanced_usage/differentiable).
- **New wrappers.** `VideoRecorder` records episodes as GIFs (one per env for
  vectorized envs). `PowerPenalty` charges the reward for actuation power.

### Added

- `fluidgym.make_vec()` and the vector API (`n_envs`, `num_envs`, `vectorized`).
  Each env in a batch can start from its own initial domain
  (`reset(domain_idx=[...])`).
- `GymVecFluidEnv` (a `gymnasium.vector.VectorEnv`). The SB3 `VecFluidEnv` and
  `TorchRLFluidEnv` support vectorized environments, and TorchRL gains
  `from_pixels=True`.
- Solver options for all environments: `advection_tol`, `pressure_tol`,
  `pressure_warm_start` and `linear_solve_max_iter`.
- BPTT options: `bptt_checkpoint`, `bptt_segment_size`, `bptt_offload_device`,
  `bptt_carry_spill_dir` and `adjoint_rtol`.
- Better rendering: slice axis selection, `render_resolution`, mirroring, and
  per-env frames (`env_ids`).
- Support for other PyTorch/CUDA versions through the
  [phiPICT wheel index](https://upb-sciml.github.io/phiPICT/whl/): install the
  matching phiPICT wheel first, then FluidGym (see the installation docs).
- `env.load_domain()` to load a saved domain.
- Fully differentiable resampling of the multiblock fields to a uniform grid.
- Logging below the `fluidgym` namespace, quiet by default. Use
  `fluidgym.set_verbosity()` to turn on solver and environment logs.
- `fluidgym.__version__`.
- SB3 training and evaluation example scripts
  (`examples/basic_usage/training_evaluation`).
- Documentation pages for differentiable simulation, batching and multi-GPU
  parallelization, and the RBC-MHD environments.

### Changed

- `step()` raises a `ValueError` for actions outside the action bounds, and when
  called after the episode has ended.
- `ActionNoise` clips the noisy actions to the action bounds, so the noise never
  produces actions that `step()` rejects.
- `ParallelFluidEnv` follows the vector API: MARL observations are
  `[E, n_agents, ...]` (previously flattened to `[E * n_agents, ...]`), and flags
  and infos are tensors (previously lists).
- Faster rendering with higher-quality output for all environments.
- Batched rendering resamples only the shown env; 3D rendering falls back to 2D
  slices when out of GPU memory.
- Observations are resampled at the sensors only (RBC).
- Examples were reorganized into `examples/basic_usage` and
  `examples/advanced_usage`.
- Upgraded PyTorch from 2.9 to 2.10 (via phiPICT, CUDA 12.8). Relaxed the
  `matplotlib` pin.
- Requires `phipict>=0.1.1,<0.2`. phiPICT 0.1.1 adds the wheel index and a clear
  error when the installed PyTorch does not match the phiPICT build.
- The build backend is now hatch. The version is derived from git tags, and the
  `experiments` extra and `dev`/`docs` extras became dependency groups.
- The license changed from MIT to Apache-2.0.
- The repository moved to [upb-sciml/fluidgym](https://github.com/upb-sciml/fluidgym).
  Docs are published by a GitHub workflow, and releases are built by CI.

### Removed

- The bundled PICT sources and CUDA extensions (`fluidgym.simulation`). They are
  replaced by phiPICT.
- `env.save_gif()` and the `auto_render` option. Use the `VideoRecorder` wrapper
  instead.
- `config.palette`. Use `fluidgym.DEFAULT_PALETTE` instead.
- Support for Python 3.10.
- The Docker images (`Dockerfile.runtime`, `Dockerfile.devel`). FluidGym is pure
  Python now, so `pip install fluidgym` is all that is needed.

## [0.1.2] - 2026-07-02

### Added

- Domain-generation index selection, with a fallback to the env default.

### Changed

- Improved the TorchRL integration and the environment docs.

### Fixed

- `step()` after a terminated episode now raises an error.
- fp64 dtype handling and the double-precision fallback.

## [0.1.1] - 2026-02-04

### Added

- `FlattenObservation` stores the original observation space in its info dict.
- Gymnasium integration checks for MARL environments.

### Fixed

- `render()` returns an RGB array for gymnasium environments.
- Removed the `wandb` import from the SB3 integration.

## [0.1.0] - 2026-01-30

### Added

- **Gymnasium spaces.** `FluidEnv` has `action_space` and `observation_space`.
  Observations are a `Dict` space with one key per field (e.g. velocity, pressure),
  shaped by the spatial layout of the sensors, which suits CNNs and equivariant
  networks.
- **Environment wrappers.** `FlattenObservation`, `ObsExtraction`, `ActionNoise` and
  `SensorNoise`, built on a general `FluidWrapper` interface.
- **Parallelization.** With the `FluidEnvLike` protocol, `ParallelFluidEnv` works
  with all FluidGym wrappers and integrations, across multiple GPUs.

### Changed

- **Unified SARL and MARL interface.** `reset_marl()` and `step_marl()` were merged
  into `reset()` and `step()`. Pass `use_marl=True` to `fluidgym.make()` to get
  per-agent batches of observations and rewards. Updated the PettingZoo and SB3
  integrations accordingly.

> **Note:** Use the `FlattenObservation` wrapper (or FluidGym v0.0.2) to use the
> models on HuggingFace, which were trained with v0.0.2.

[0.2.0]: https://github.com/upb-sciml/fluidgym/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/upb-sciml/fluidgym/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/upb-sciml/fluidgym/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/upb-sciml/fluidgym/releases/tag/v0.1.0
