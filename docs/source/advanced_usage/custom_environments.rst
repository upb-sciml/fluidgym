Adding Custom Environments to FluidGym
======================================

FluidGym allows users to create and integrate their own custom fluid dynamics
environments. This is particularly useful for researchers who want to experiment
with specific fluid scenarios not covered by the built-in environments.

To create a custom environment, subclass ``FluidEnv`` from the
``fluidgym.envs`` module. The simulation itself is set up with the solver
`phiPICT <https://github.com/upb-sciml/phiPICT>`_: your environment builds a
phiPICT ``Domain`` and ``Simulation``, and ``FluidEnv`` takes care of resets,
initial domains, batching (``n_envs``), MARL, rendering, differentiability and
the gymnasium-like API.

The built-in environments are the best reference for a concrete implementation,
e.g. ``fluidgym/envs/cylinder/rotating_cylinder_env_2d.py`` together with
``fluidgym/envs/cylinder/cylinder_env_base.py``.

Here is a skeleton with all methods a custom environment implements:

.. code-block:: python

    from pathlib import Path

    import numpy as np
    import torch
    from gymnasium import spaces
    from phipict import Domain, Hooks, Simulation

    from fluidgym.envs import FluidEnv


    class CustomFluidEnv(FluidEnv):
        # Whether the environment supports multi-agent RL. If True, also
        # implement _get_local_obs and _finish_marl_step.
        _supports_marl: bool = False

        def __init__(self, use_marl: bool = False, **kwargs):
            super().__init__(
                dt=0.01,
                adaptive_cfl=0.8,
                step_length=0.25,  # simulation time per environment step
                episode_length=200,  # environment steps per episode
                ndims=2,
                use_marl=use_marl,
                **kwargs,  # dtype, cuda_device, differentiable, n_envs, ...
            )

        # ---- Spaces and metadata -------------------------------------------

        def _get_action_space(self) -> spaces.Box:
            # Action space of a single agent (of a single environment)
            ...

        def _get_observation_space(self) -> spaces.Dict:
            # Observation space of a single agent, one key per field
            ...

        @property
        def n_agents(self) -> int:
            # The number of agents (1 without MARL support)
            ...

        @property
        def obs_resampling_shape(self) -> tuple[int, ...]:
            # Shape of the uniform grid the observations are resampled onto
            ...

        @property
        def id(self) -> str:
            # Unique identifier of the environment
            ...

        @property
        def initial_domain_id(self) -> str:
            # Identifier of the initial domains (directory name in the data)
            ...

        # ---- Simulation setup ----------------------------------------------

        def _get_domain(self) -> Domain:
            # Build the phiPICT domain: grid, blocks, boundaries, initial fields
            ...

        def _get_hooks(self, domain: Domain) -> Hooks:
            # Callbacks the simulation runs at its hooks, e.g. to update
            # boundary conditions; an empty Hooks() if none are needed
            ...

        def _get_simulation(self, domain: Domain, hooks: Hooks) -> Simulation:
            # Create the phiPICT simulation for the domain
            ...

        def _additional_initialization(self) -> None:
            # Called whenever a new domain and simulation are set up (reset,
            # load_domain, set_state), e.g. to cache boundary tensors
            ...

        def _randomize_domain(self) -> None:
            # Randomize the current domain, e.g. perturb the initial fields
            ...

        # ---- Stepping --------------------------------------------------------

        def _apply_action(self, action: torch.Tensor) -> None:
            # Write the action into the simulation, e.g. boundary velocities
            ...

        def _advance_impl(
            self, action: torch.Tensor, n_sim_steps: int
        ) -> dict[str, torch.Tensor]:
            # Run n_sim_steps solver steps (self._sim.single_step()), applying
            # the action, and return the metrics measured per step, stacked
            # along the first dim, e.g. {"drag": [n_sim_steps, n_envs]}
            ...

        def _finish_step(
            self, metrics: dict[str, torch.Tensor]
        ) -> tuple[dict[str, torch.Tensor], torch.Tensor, bool, dict[str, torch.Tensor]]:
            # Turn the metrics of one environment step into
            # (observation, reward, terminated, info), using _get_global_obs()
            ...

        def _get_global_obs(self) -> dict[str, torch.Tensor]:
            # Read the current observation from the flow state
            ...

        # ---- Multi-agent RL (only if _supports_marl) -------------------------

        def _get_local_obs(self) -> dict[str, torch.Tensor]:
            # Per-agent observations
            ...

        def _finish_marl_step(
            self, metrics: dict[str, torch.Tensor]
        ) -> tuple[dict[str, torch.Tensor], torch.Tensor, bool, dict[str, torch.Tensor]]:
            # Like _finish_step, but with per-agent observations and rewards
            ...

        # ---- Rendering and plotting ------------------------------------------

        def _get_render_data(
            self, render_3d: bool, output_path: Path | None = None
        ) -> dict[str, np.ndarray]:
            # Map slice names to the field data to render
            ...

        def plot(self, output_path: Path | None = None) -> None:
            # Plot the configuration of the environment (domain, sensors, ...)
            ...

A few things to keep in mind:

- **Stepping is split in two.** ``_advance_impl`` runs the solver and may be
  called several times per environment step, once per activation-checkpoint
  segment in differentiable mode. ``_finish_step`` runs once per environment
  step on the final state. Keep everything that depends on the full step
  (averaging, observation, reward) in ``_finish_step``.
- **Batching.** With ``n_envs``, all environments share one simulation, and
  actions, metrics and observations carry a leading environment dim. Write the
  methods for this batched layout, as the built-in environments do.
- **Initial domains.** With ``load_initial_domain=True`` (the default),
  ``FluidEnv`` loads the initial domains from the data identified by
  ``initial_domain_id``. For a new environment, create them with your own
  uncontrolled runs, or pass ``load_initial_domain=False`` and
  ``load_domain_statistics=False``.

Finally, register the environment to create it with ``fluidgym.make``:

.. code-block:: python

    import fluidgym

    fluidgym.register(
        id="CustomFluidEnv-v0",
        entry_point=CustomFluidEnv,
        defaults={},
    )

    env = fluidgym.make("CustomFluidEnv-v0")
