"""Rayleigh-Bénard convection under a horizontal magnetic field (RBC-MHD)."""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from phipict import _C, Face, Hook, Hooks, bc
from phipict.grid.helpers import get_cell_size
from phipict.grid.shapes import make_weights
from phipict.simulation.mhd import MHDSimulation
from phipict.simulation.simulation import Simulation
from phipict.solvers.tolerance import SolverTolerance, parse_tolerance

from fluidgym.envs.fluid_env import FluidEnv
from fluidgym.envs.rbc_mhd.grid import make_rbc_mhd_domain
from fluidgym.envs.util.mhd_grid import get_layer_thickness, make_mhd_wall_weights
from fluidgym.envs.util.visualization import render_3d_voxels

# Relative to the RHS, so that the criterion keeps its meaning across grid sizes
_DEFAULT_POTENTIAL_TOL = SolverTolerance(rtol=1e-4, atol=1e-14)

RBC_MHD_3D_DEFAULT_CONFIG = {
    # Ra = 50 Q, between the 4-3 roll transition and the no-roll regime of [1]
    "rayleigh_number": 5e4,
    "chandrasekhar_number": 1e3,
    "prandtl_number": 0.025,
    "aspect_ratio": 5.0,
    "aspect_ratio_x": None,
    "resolution": 32,
    "wall_grading": 2.0,
    "min_layer_cells": 3,
    "n_electrodes": 2,
    "n_stations": None,
    "electrode_fill": 0.5,
    "max_current": 1.0,
    # Joule heating in the Hartmann layers costs many times the buoyancy power, so
    # the weight is small; tuned on a 2 x 4 box to put the best current in bounds
    "power_penalty_weight": 0.01,
    "dt": 0.05,
    "adaptive_cfl": 0.8,
    "step_length": 1.5,
    "episode_length": 200,
    "local_obs_window": 3,
    "local_reward_weight": 0.0015,  # Based on beta in doi.org/10.1063/5.0153181
    "advection_scheme": "linear_upwind",
    "use_marl": False,
    "dtype": torch.float32,
    "load_initial_domain": True,
    "load_domain_statistics": True,
    "randomize_initial_state": True,
    "enable_actions": True,
    "differentiable": False,
    "pressure_warm_start": True,
    "potential_tol": _DEFAULT_POTENTIAL_TOL,
    "potential_solve_max_iter": 10000,
    "potential_use_preconditioner": True,
    "exclude_potential_solve_gradients": False,
}


class RBCMHDEnv3D(FluidEnv):
    """Rayleigh-Bénard convection in a horizontal magnetic field.

    Each Hartmann wall is divided into ``n_electrodes`` patches along y and
    ``n_stations`` stations along z. Each patch holds a raised-cosine electrode
    covering
    ``electrode_fill`` of the patch pitch in y and z, surrounded by insulating
    wall. Electrode ``k`` exists on both walls, and the pair carries the current
    ``I_k``, half through each wall (the convention of
    :mod:`fluidgym.envs.blanket.actuators`). The action sets ``I_k``; it is
    centred to zero net current (the box is otherwise insulating, so the
    potential problem needs it), clamped to the action bounds, centred again and
    scaled by ``max_current``. In MARL mode, each station along z is an agent
    that sets the currents of its column of electrodes.

    The reward is the relative change of the volume-averaged Nusselt number
    minus the electrical power fed into the fluid, relative to the power the
    buoyancy does on the uncontrolled flow::

        r = Nu / Nu_ref - 1 - power_penalty_weight * P_el / P_ref

    so the heat transfer is to be raised, both averaged over the PISO steps of
    the env step. ``Nu_ref`` and ``P_ref``
    are the means of the uncontrolled episodes (domain statistics). In MARL
    mode, an agent receives
    ``local_reward_weight`` times the same Nusselt term over the columns of its
    observation window plus the remainder of the global reward, as in
    :class:`~fluidgym.envs.rbc.RBCEnv3D`.

    Parameters
    ----------
    rayleigh_number: float
        Rayleigh number based on ``H`` and the plate temperature difference.

    chandrasekhar_number: float
        Chandrasekhar number ``Q = Ha^2`` based on ``H``. Must be positive.

    prandtl_number: float
        Prandtl number.

    aspect_ratio: float
        Extent of the box along z, in units of ``H``.

    aspect_ratio_x: float | None
        Extent of the box along x, i.e. along ``B`` between the Hartmann walls,
        in units of ``H``. Defaults to ``aspect_ratio``.

    resolution: int
        Number of cells along y; x and z have ``round(aspect_ratio_x *
        resolution)`` and ``round(aspect_ratio * resolution)``.

    wall_grading: float
        tanh grading strength towards the walls normal to y and z.

    min_layer_cells: int
        Minimum number of cells inside the Hartmann layer at each Hartmann wall;
        the x grading is chosen for it.

    n_electrodes: int
        Electrodes along y on each Hartmann wall.

    n_stations: int | None
        Electrode stations along z. Defaults to ``round(aspect_ratio *
        n_electrodes)``.

    electrode_fill: float
        Fraction of the patch pitch an electrode covers in y and z, in (0, 1].

    max_current: float
        Largest current of an electrode pair (both walls together), in units of
        ``sigma U B H^2``.

    power_penalty_weight: float
        Weight of the electrical power in the reward.

    dt: float
        The simulation time step; substeps adapt to ``adaptive_cfl``.

    adaptive_cfl: float
        Target CFL number for adaptive time stepping.

    step_length: float
        The physical time length of each environment step.

    episode_length: int
        The number of steps per episode.

    local_obs_window: int
        Number of electrode stations along z an agent observes in MARL mode
        (odd). Beyond the side walls, the window holds the quiescent
        conduction state.

    local_reward_weight: float | None
        Weighting factor for local rewards in multi-agent settings. Has to be set
        for multi-agent RL.

    advection_scheme: str
        Convective scheme of the momentum equation, ``"central"`` or
        ``"linear_upwind"``.

    use_marl: bool
        Whether to enable multi-agent reinforcement learning mode.

    potential_tol: float | SolverTolerance | Mapping | None
        Tolerance of the electric potential solve.

    potential_solve_max_iter: int
        Iteration limit of the electric potential solve.

    potential_use_preconditioner: bool
        Whether to solve the potential with AMG-preconditioned CG.

    exclude_potential_solve_gradients: bool
        Drop the gradient of the potential solve. Diagnostic only.

    **kwargs
        Parameters of :class:`~fluidgym.envs.fluid_env.FluidEnv`.

    References
    ----------
    [1] T. Yanagisawa, Y. Hamano, T. Miyagoshi, Y. Yamagishi, Y. Tasaka, and Y. Takeda,
    “Convection patterns in a liquid metal under an imposed horizontal magnetic field,”
    Phys. Rev. E, vol. 88, no. 6, p. 063020, Dec. 2013, doi: 10.1103/PhysRevE.88.063020.

    """

    _default_render_key: str = "x-z-temperature"
    _render_resolution: int = 2
    _ndims = 3
    _supports_marl = True

    # Same segment size as the 3D RBC env: the replay tape caps a differentiable
    # rollout of a 3D env (see ``FluidEnv.bptt_segment_size``)
    _default_bptt_segment_size: int | None = 5

    # 300 free-fall times at the default step length of 1.5
    _initial_domain_steps = 200
    _initial_domain_restart = True

    _H: float = 1.0
    _T_hot: float = 1.0
    _T_cold: float = 0.0
    _T_ref: float = 0.5

    _n_sensors_y: int = 8
    _n_sensors_per_actuator: int = 2
    # Resampling cells per H
    _n_obs_resampling: int = 32

    _metrics: list[str] = [
        "nusselt",
        "buoyancy_power",
        "electrical_power",
        "power_ratio",
    ]

    _sim: MHDSimulation

    def __init__(
        self,
        rayleigh_number: float,
        chandrasekhar_number: float,
        prandtl_number: float,
        aspect_ratio: float,
        resolution: int,
        wall_grading: float,
        min_layer_cells: int,
        n_electrodes: int,
        electrode_fill: float,
        max_current: float,
        power_penalty_weight: float,
        dt: float,
        adaptive_cfl: float,
        step_length: float,
        episode_length: int,
        aspect_ratio_x: float | None = None,
        n_stations: int | None = None,
        local_obs_window: int = 3,
        local_reward_weight: float | None = None,
        advection_scheme: str = "linear_upwind",
        use_marl: bool = False,
        potential_tol: float | SolverTolerance | Mapping[str, float] | None = None,
        potential_solve_max_iter: int = 10000,
        potential_use_preconditioner: bool = True,
        exclude_potential_solve_gradients: bool = False,
        **kwargs: Any,
    ):
        if chandrasekhar_number <= 0.0:
            raise ValueError("chandrasekhar_number must be positive.")
        if aspect_ratio <= 0.0:
            raise ValueError("aspect_ratio must be positive.")
        if aspect_ratio_x is not None and aspect_ratio_x <= 0.0:
            raise ValueError("aspect_ratio_x must be positive.")
        if n_stations is not None and n_stations < 1:
            raise ValueError("n_stations must be at least 1.")
        if n_electrodes < 1:
            raise ValueError("n_electrodes must be at least 1.")
        if not 0.0 < electrode_fill <= 1.0:
            raise ValueError("electrode_fill must be in (0, 1].")
        if max_current <= 0.0:
            raise ValueError("max_current must be positive.")
        if power_penalty_weight < 0.0:
            raise ValueError("power_penalty_weight must be non-negative.")
        if local_obs_window < 1 or local_obs_window % 2 == 0:
            raise ValueError("local_obs_window must be a positive odd number.")
        if advection_scheme.upper() not in ("CENTRAL", "LINEAR_UPWIND"):
            raise ValueError(f"Unknown advection_scheme {advection_scheme!r}.")

        self._rayleigh_number = float(rayleigh_number)
        self._chandrasekhar_number = float(chandrasekhar_number)
        self._prandtl_number = float(prandtl_number)
        self._hartmann_number = math.sqrt(self._chandrasekhar_number)
        self._aspect_ratio = float(aspect_ratio)
        self._aspect_ratio_x = (
            self._aspect_ratio if aspect_ratio_x is None else float(aspect_ratio_x)
        )
        self._Lx = self._aspect_ratio_x * self._H
        self._Lz = self._aspect_ratio * self._H
        self._resolution = int(resolution)
        self._nx = max(1, round(self._aspect_ratio_x * self._resolution))
        self._nz = max(1, round(self._aspect_ratio * self._resolution))
        self._wall_grading = float(wall_grading)
        self._min_layer_cells = int(min_layer_cells)
        self._n_electrodes = int(n_electrodes)
        self._n_stations_set = n_stations is not None
        self._n_stations = (
            int(n_stations)
            if n_stations is not None
            else max(1, round(self._aspect_ratio * self._n_electrodes))
        )
        self._electrode_fill = float(electrode_fill)
        self._max_current = float(max_current)
        self._power_penalty_weight = float(power_penalty_weight)
        self._local_obs_window = int(local_obs_window)
        self._local_reward_weight = local_reward_weight
        self._advection_scheme = advection_scheme
        self._potential_tol = parse_tolerance(
            _DEFAULT_POTENTIAL_TOL if potential_tol is None else potential_tol
        )
        self._potential_solve_max_iter = int(potential_solve_max_iter)
        self._potential_use_preconditioner = potential_use_preconditioner
        self._exclude_potential_solve_gradients = exclude_potential_solve_gradients

        self._e_b = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

        # The initial domain id resolved by the base class depends on the grid
        self._setup_grid()

        super().__init__(
            dt=dt,
            adaptive_cfl=adaptive_cfl,
            step_length=step_length,
            episode_length=episode_length,
            ndims=3,
            use_marl=use_marl,
            **kwargs,
        )

        to_env = {"device": self._cuda_device, "dtype": self._dtype}
        self._electrode_distribution = self._make_electrode_distribution().to(
            **to_env
        )  # [K, Z, Y]
        self._electrode_mask = self._electrode_distribution.sum(dim=0) > 0.0

        # Cells of each agent's local window along z
        self._footprint_z = self._window_footprint(self._z_vertices).to(**to_env)

        self._sensor_locations = self._get_sensor_locations()

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------
    def _setup_grid(self) -> None:
        """Vertex coordinates of the three axes; runs before the base initialisation."""
        delta = get_layer_thickness(
            self._hartmann_number, self._e_b, wall_axis=0, length=self._H
        )
        self._x_weights, self._x_grading = make_mhd_wall_weights(
            res=self._nx,
            delta=delta / self._Lx,
            min_layer_cells=self._min_layer_cells,
            grading_type="tanh",
        )
        self._y_weights = make_weights(
            grading_type="tanh",
            res=self._resolution,
            grading=self._wall_grading,
            refinement="BOTH",
        )
        self._z_weights = make_weights(
            grading_type="tanh",
            res=self._nz,
            grading=self._wall_grading,
            refinement="BOTH",
        )

        self._x_vertices = torch.tensor(
            [self._Lx * w for w in self._x_weights], dtype=torch.float64
        )
        self._y_vertices = torch.tensor(
            [self._H * (w - 0.5) for w in self._y_weights], dtype=torch.float64
        )
        self._z_vertices = torch.tensor(
            [self._Lz * w for w in self._z_weights], dtype=torch.float64
        )

    @staticmethod
    def _cell_centers(vertices: torch.Tensor) -> torch.Tensor:
        return 0.5 * (vertices[1:] + vertices[:-1])

    # ------------------------------------------------------------------
    # Actuator layout
    # ------------------------------------------------------------------
    def _make_electrode_distribution(self) -> torch.Tensor:
        """Share of each electrode's current per face cell, ``[K, Z, Y]``.

        Electrode ``k = iz * n_electrodes + iy`` is centred in its patch. Its
        cells get ``w * A / sum(w * A)``, with ``w`` the raised cosine over the
        electrode and ``A`` the cell area, so a total current spreads as the
        current density ``w``-shaped and each row sums to 1.
        """

        def profiles(vertices: torch.Tensor, n_e: int) -> list[torch.Tensor]:
            centers = self._cell_centers(vertices)
            pitch = float(vertices[-1] - vertices[0]) / n_e
            half_width = 0.5 * self._electrode_fill * pitch
            out = []
            for i in range(n_e):
                center = float(vertices[0]) + (i + 0.5) * pitch
                xi = (centers - center) / half_width
                profile = 0.5 * (1.0 + torch.cos(math.pi * xi))
                out.append(torch.where(xi.abs() < 1.0, profile, 0.0))
            return out

        profiles_y = profiles(self._y_vertices, self._n_electrodes)
        profiles_z = profiles(self._z_vertices, self._n_stations)
        area = self._face_cell_area()

        dist = []
        for iz, p_z in enumerate(profiles_z):
            for iy, p_y in enumerate(profiles_y):
                w = p_z[:, None] * p_y[None, :] * area
                total = w.sum()
                if not total > 0.0:
                    raise ValueError(
                        f"Electrode ({iz}, {iy}) covers no cell centre; increase "
                        "resolution or electrode_fill."
                    )
                dist.append(w / total)
        return torch.stack(dist)

    def _face_cell_area(self) -> torch.Tensor:
        """Areas of the Hartmann wall face cells, ``[Z, Y]`` in float64."""
        dy = self._y_vertices[1:] - self._y_vertices[:-1]
        dz = self._z_vertices[1:] - self._z_vertices[:-1]
        return dz[:, None] * dy[None, :]

    def _window_footprint(self, vertices: torch.Tensor) -> torch.Tensor:
        """Cells of each agent's observation window along one axis.

        Returns ``[cells, n_stations]``, 1 where the cell centre lies within
        ``local_obs_window // 2`` station pitches of the agent's own station.
        """
        centers = self._cell_centers(vertices)
        n = self._n_stations
        pitch = float(vertices[-1] - vertices[0]) / n
        patch = ((centers - float(vertices[0])) / pitch).floor().long().clamp(0, n - 1)
        offset = (patch[:, None] - torch.arange(n)[None, :]).abs()
        return (offset <= self._local_obs_window // 2).to(torch.float64)

    @property
    def electrode_mask(self) -> torch.Tensor:
        """Electrode cells of a Hartmann wall, ``[Z, Y]`` bool."""
        return self._electrode_mask

    @property
    def hartmann_number(self) -> float:
        """Hartmann number ``Ha = sqrt(Q)``."""
        return self._hartmann_number

    @property
    def stuart_number(self) -> float:
        """Lorentz force coefficient ``N = Q sqrt(Pr / Ra)``."""
        return self._chandrasekhar_number * math.sqrt(
            self._prandtl_number / self._rayleigh_number
        )

    # ------------------------------------------------------------------
    # Domain and simulation
    # ------------------------------------------------------------------
    def _get_domain(self) -> _C.Domain:
        # The electrodes are applied by _apply_potential_layout
        return make_rbc_mhd_domain(
            Lx=self._Lx,
            H=self._H,
            Lz=self._Lz,
            x_weights=self._x_weights,
            y_weights=self._y_weights,
            z_weights=self._z_weights,
            rayleigh_number=self._rayleigh_number,
            prandtl_number=self._prandtl_number,
            T_hot=self._T_hot,
            T_cold=self._T_cold,
            dtype=self._dtype,
            device=self._cuda_device,
            generator=self._torch_rng_cuda,
        )

    def _conduction_profile(self, y: torch.Tensor) -> torch.Tensor:
        """Temperature of the quiescent layer at heights ``y``."""
        return self._T_hot + (self._T_cold - self._T_hot) * (y / self._H + 0.5)

    def _get_hooks(self, domain: _C.Domain) -> Hooks:
        block = domain.getBlock(0)
        assert isinstance(block.passiveScalar, torch.Tensor)
        zero = torch.zeros_like(block.passiveScalar)

        # Free-fall scaling, so the buoyancy factor is 1
        def buoyancy_fn(domain, time_step, **kwargs):
            T = block.passiveScalar
            block.setVelocitySource(torch.cat([zero, T - self._T_ref, zero], dim=1))
            domain.UpdateDomainData()

        # The buoyancy overwrites the velocity source and add_lorentz_force adds
        # to it, so the buoyancy has to come first
        return Hooks().append(Hook.PRE_VELOCITY_SETUP, buoyancy_fn)

    def _apply_potential_layout(self, block: _C.Block) -> None:
        """Set this environment's potential conditions on both Hartmann walls.

        Current electrodes (all at zero) in insulating walls. Both faces are
        overwritten completely, so the result does not depend on the conditions a
        loaded domain brings. A zero current is an insulating wall, so all
        electrode layouts share the initial domains of the plain insulating box.

        Must run before the ``MHDSimulation`` is built: its constructor assembles
        the potential matrix from these conditions, and a later change would
        leave the matrix of the old layout in place.
        """
        for face in (Face.X_MINUS, Face.X_PLUS):
            # Clears any per-cell types the face carries
            bc.set_bc(block, face, bc.Potential.Insulating())
            bc.set_bc(
                block,
                face,
                bc.Potential.Current(value=0.0),
                where=self._electrode_mask,
            )

    def _get_simulation(self, domain: _C.Domain, hooks: Hooks) -> Simulation:
        self._apply_potential_layout(domain.getBlock(0))

        domain.setAdvectionScheme(
            getattr(_C.AdvectionScheme, self._advection_scheme.upper())
        )

        sim = MHDSimulation(
            domain=domain,
            hooks=hooks,
            stuart_number=torch.tensor(self.stuart_number),
            e_b=self._e_b.to(self._dtype),
            substeps="ADAPTIVE",
            dt=self._dt,
            corrector_steps=2,
            advection_tol=self._resolve_advection_tol(SolverTolerance(atol=1e-6)),
            pressure_tol=self._resolve_pressure_tol(SolverTolerance(atol=1e-6)),
            pressure_tol_intermediate=self._pressure_tol_intermediate,
            pressure_warm_start=self._pressure_warm_start,
            adaptive_CFL=self._adaptive_cfl,
            pressure_return_best_result=True,
            non_orthogonal=False,
            output_resampling_shape=self.obs_resampling_shape,
            output_resampling_fill_max_steps=16,
            differentiable=self._differentiable,
            potential_tol=self._potential_tol,
            potential_normalize=True,
            potential_use_BiCG=False,
            potential_solve_max_iter=self._potential_solve_max_iter,
            potential_solve_dtype=torch.float64,
            potential_return_best_result=True,
            potential_reuse_result=True,
            potential_use_preconditioner=self._potential_use_preconditioner,
            exclude_potential_solve_gradients=self._exclude_potential_solve_gradients,
        )

        # Retry a failed single-precision solve in double precision
        sim.solver_double_fallback = True

        sim.make_divergence_free(
            max_iter=10
            * self._resolve_linear_solve_max_iter(sim.linear_solve_max_iterations)
        )
        # A fresh domain starts with phi = 0, which the first step would turn into
        # a current that is not divergence-free. A loaded phi stays a valid start
        # under the new layout, whose zero currents are the insulating wall
        sim.make_current_divergence_free()

        return sim

    def _additional_initialization(self) -> None:
        self._block = self._domain.getBlock(0)

        # The layout _apply_potential_layout set must have reached the faces
        expected = torch.where(
            self._electrode_mask.cpu(),
            int(_C.PotentialBC.CURRENT),
            int(_C.PotentialBC.INSULATING),
        ).to(torch.int8)
        for face in (Face.X_MINUS, Face.X_PLUS):
            types = bc.get_bc(self._block, face, bc.Potential)
            if isinstance(types, torch.Tensor):
                matches = torch.equal(types.reshape(expected.shape).cpu(), expected)
            else:
                # A uniform face has no per-cell types
                matches = isinstance(types, bc.Potential.Current) and bool(
                    self._electrode_mask.all()
                )
            if not matches:
                raise RuntimeError(
                    f"The potential conditions of face {face.name} do not match the "
                    f"actuator layout of {self.id} after applying it."
                )

        # Wall distance of the first cell centres and the face cell areas, for the
        # electrical power
        x = self._cell_centers(self._x_vertices)
        self._h_minus = float(x[0] - self._x_vertices[0])
        self._h_plus = float(self._x_vertices[-1] - x[-1])
        self._face_area = self._face_cell_area().to(
            device=self._cuda_device, dtype=self._dtype
        )

        self._cell_size = get_cell_size(self._block)[0, 0]  # [Z, Y, X]

    def _randomize_domain(self) -> None:
        """Mirror the state per environment, add noise and run a short while.

        The box is symmetric under ``x -> L - x``, which leaves ``B = e_x``
        invariant (the Lorentz force is quadratic in ``B``), under
        ``z -> L - z`` and under the Boussinesq reflection ``y -> -y`` with
        ``T - T_ref`` changing sign.
        """
        # (flipped tensor dims, negated velocity components, T changes sign)
        symmetries = [([-1], [0], False), ([-3], [2], False), ([-2], [1], True)]

        T_envs, u_envs, p_envs = [], [], []
        for T, u, p in zip(
            self._block.passiveScalar.split(1),
            self._block.getVelocity(False).split(1),
            self._block.pressure.split(1),
            strict=True,
        ):
            u = u.clone()
            for dims, components, flip_T in symmetries:
                if self._np_rng.uniform() <= 0.5:
                    continue
                T, u, p = (torch.flip(f, dims=dims) for f in (T, u, p))
                u[:, components] *= -1.0
                if flip_T:
                    T = 2.0 * self._T_ref - T
            T_envs.append(T)
            u_envs.append(u)
            p_envs.append(p)

        T = torch.cat(T_envs)
        T = T + 0.01 * torch.normal(
            mean=0.0,
            std=1.0,
            size=T.shape,
            device=self._cuda_device,
            generator=self._torch_rng_cuda,
        ).to(self._dtype)
        T = torch.clamp(T, min=self._T_cold, max=self._T_hot)

        self._block.setPassiveScalar(T.contiguous())
        self._block.setVelocity(torch.cat(u_envs).contiguous())
        self._block.setPressure(torch.cat(p_envs).contiguous())
        self._domain.UpdateDomainData()
        self._sim.make_current_divergence_free()

        sim_time = self._np_rng.uniform(0.5, 2.0)
        for _ in range(max(1, int(sim_time / self._dt))):
            self._sim.single_step()

    def _checkpoint_accessors(self) -> list[Any]:
        """Extend the domain carry with ``u x e_b`` kept by the simulation.

        ``add_lorentz_force`` reads ``sim._u_cross_eb_flat`` from the previous
        step's magnetic step, which does not live in the ``Domain``.
        """
        return super()._checkpoint_accessors() + [
            (
                lambda: self._sim._u_cross_eb_flat,
                lambda t: setattr(self._sim, "_u_cross_eb_flat", t),
            )
        ]

    def detach(self) -> None:
        """Detach all tensors from the current computation graph."""
        super().detach()
        self._sim.detach()

    # ------------------------------------------------------------------
    # Spaces and observations
    # ------------------------------------------------------------------
    @property
    def n_agents(self) -> int:
        """The number of agents in the environment."""
        if not self._use_marl:
            return 1
        return self._n_stations

    @property
    def obs_resampling_shape(self) -> tuple[int, ...]:
        """The shape ``(x, y, z)`` of the observation resampling grid."""
        n = self._n_obs_resampling
        n_x = max(1, round(self._aspect_ratio_x * n))
        n_z = max(1, round(self._aspect_ratio * n))
        return (n_x, n, n_z)

    @property
    def _n_sensors(self) -> tuple[int, int, int]:
        """Sensors along x, y and z, a whole number per station along z.

        The electrodes have no actuators along x, so x gets the same sensor
        spacing as z.
        """
        n_z = self._n_sensors_per_actuator * self._n_stations
        n_x = max(1, round(n_z * self._Lx / self._Lz))
        return (n_x, self._n_sensors_y, n_z)

    def _get_action_space(self) -> spaces.Box:
        """Per-agent action space.

        One entry per electrode ``(iz, iy)``.
        """
        shape: tuple[int, ...]
        if self._use_marl:
            shape = (self._n_electrodes,)
        else:
            shape = (self._n_stations, self._n_electrodes)

        return spaces.Box(low=-1.0, high=1.0, shape=shape, dtype=np.float32)

    def _get_observation_space(self) -> spaces.Dict:
        """Per-agent observation space."""
        n_x, n_y, n_z = self._n_sensors
        if self._use_marl:
            n_z = self._local_obs_window * self._n_sensors_per_actuator
        shape = (n_z, n_y, n_x)

        return spaces.Dict(
            {
                "temperature": spaces.Box(
                    low=self._T_cold,
                    high=self._T_hot,
                    shape=shape,
                    dtype=np.float32,
                ),
                "velocity": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3, *shape),
                    dtype=np.float32,
                ),
            }
        )

    def _get_sensor_locations(self) -> torch.Tensor:
        """Sensor indices on the observation grid, ``[3, R]`` rows ``(x, y, z)``.

        Ordered with z slowest and x fastest, so the readings reshape to
        ``[Z, Y, X]``.
        """
        axes = [
            ((torch.arange(n) + 0.5) * n_res / n).floor().long()
            for n, n_res in zip(self._n_sensors, self.obs_resampling_shape, strict=True)
        ]
        z, y, x = torch.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
        return torch.stack([x.flatten(), y.flatten(), z.flatten()]).to(
            self._cuda_device
        )

    def _sensor_conduction_profile(self) -> torch.Tensor:
        """Conduction temperature at the sensor heights, ``[n_sensors_y, 1]``."""
        n_y = self._n_sensors_y
        n_res = self.obs_resampling_shape[1]
        idx = ((torch.arange(n_y) + 0.5) * n_res / n_y).floor()
        y = (idx + 0.5) * self._H / n_res - self._H / 2
        T = self._conduction_profile(y)
        return T[:, None].to(device=self._cuda_device, dtype=self._dtype)

    def _get_global_obs(self) -> dict[str, torch.Tensor]:
        n_x, n_y, n_z = self._n_sensors
        at = self._sensor_locations
        readings = torch.cat(
            [
                self._resample_blocks_at([self._block.passiveScalar], at),  # type: ignore[list-item]
                self._resample_blocks_at([self._block.velocity], at),  # type: ignore[list-item]
            ],
            dim=1,
        )  # [E, 4, R]
        readings = readings.reshape(readings.size(0), 4, n_z, n_y, n_x)
        return {
            "temperature": readings[:, 0],
            "velocity": readings[:, 1:],
        }

    def _sensor_windows(self, field: torch.Tensor, dim: int) -> torch.Tensor:
        """Windows of ``local_obs_window`` actuators of sensors along ``dim``.

        Beyond the walls, the windows are zero-padded. The result has the
        actuators at ``dim`` and the window right after it.
        """
        per = self._n_sensors_per_actuator
        pad = (self._local_obs_window // 2) * per
        padding = [0, 0] * (field.dim() - 1 - dim) + [pad, pad]
        windows = F.pad(field, padding).unfold(dim, self._local_obs_window * per, per)
        return windows.movedim(-1, dim + 1)

    def _get_local_obs(self) -> dict[str, torch.Tensor]:
        """Per-agent windows of the global observation, agents along dim 1.

        The temperature is padded with the conduction profile and the velocity
        with zeros, i.e. beyond the side walls the agents see quiescent fluid.
        """
        global_obs = self._get_global_obs()
        T_cond = self._sensor_conduction_profile()
        # [E, C, Z, Y, X], zero for the quiescent state
        fields = torch.cat(
            [(global_obs["temperature"] - T_cond)[:, None], global_obs["velocity"]],
            dim=1,
        )

        # [E, C, A, W_z, Y, X] -> [E, A, C, W_z, Y, X]
        local = self._sensor_windows(fields, dim=2).movedim(2, 1)

        return {
            "temperature": local[:, :, 0] + T_cond,
            "velocity": local[:, :, 1:],
        }

    def _temperature_fields(self) -> torch.Tensor:
        """The resampled temperature of every environment, ``[E, 1, *spatial]``."""
        return self._resample_blocks([self._block.passiveScalar])  # type: ignore[list-item]

    def get_temperature(self) -> torch.Tensor:
        """Return the current temperature field resampled to the render shape.

        Returns
        -------
        torch.Tensor
            Temperature field, with a leading env dim for a vectorized
            environment.
        """
        return self._squeeze_fields(self._temperature_fields())

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _action_bounds(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The action as ``[E, agents, per-agent entries]`` and its bounds."""
        low = torch.as_tensor(
            self.action_space.low, device=action.device, dtype=action.dtype
        ).reshape(1, 1, -1)
        high = torch.as_tensor(
            self.action_space.high, device=action.device, dtype=action.dtype
        ).reshape(1, 1, -1)
        return action.reshape(action.size(0), -1, low.size(-1)), low, high

    def _action_to_currents(self, action: torch.Tensor) -> torch.Tensor:
        """Electrode currents per environment, ``[E, K]``."""
        a, low, high = self._action_bounds(action)

        # Zero net current, again after the clamp, which can shift the mean
        a = a - a.mean(dim=(1, 2), keepdim=True)
        a = torch.clamp(a, low, high)
        a = a - a.mean(dim=(1, 2), keepdim=True)

        # Full current at the largest action magnitude the space allows
        scale = torch.maximum(low.abs(), high.abs())
        return (self._max_current * a / scale).reshape(a.size(0), -1)

    def _cell_currents(self, currents: torch.Tensor) -> torch.Tensor:
        """Current through each face cell of one Hartmann wall, ``[E, Z, Y]``.

        Half of every pair's current goes through each wall.
        """
        return 0.5 * torch.einsum("ek,kzy->ezy", currents, self._electrode_distribution)

    def _apply_action(self, action: torch.Tensor) -> None:
        """Set the electrode currents."""
        cells = self._cell_currents(self._action_to_currents(action))
        self._last_cell_currents = cells
        # Only the right-hand side depends on the values: no rebuild of the
        # potential matrix
        for face in (Face.X_MINUS, Face.X_PLUS):
            bc.set_potential_values(self._block, face, cells)

    # ------------------------------------------------------------------
    # Metrics and reward
    # ------------------------------------------------------------------
    def _volume_mean(self, field: torch.Tensor) -> torch.Tensor:
        """Volume mean of ``[E, Z, Y, X]`` per environment."""
        return (field * self._cell_size).sum(dim=(1, 2, 3)) / self._cell_size.sum()

    def _convective_flux(self) -> torch.Tensor:
        """``u_y (T - T_ref)`` per cell, ``[E, Z, Y, X]``."""
        T = self._block.passiveScalar[:, 0] - self._T_ref
        return self._block.velocity[:, 1] * T

    def _flux_to_nusselt(self, flux_mean: torch.Tensor) -> torch.Tensor:
        return 1.0 + math.sqrt(self._rayleigh_number * self._prandtl_number) * flux_mean

    def compute_nusselt(self) -> torch.Tensor:
        """Volume-averaged Nusselt number ``1 + sqrt(Ra Pr) <u_y (T - T_ref)>``.

        Returns
        -------
        torch.Tensor
            The Nusselt number of every environment, ``[E]``.
        """
        return self._flux_to_nusselt(self._volume_mean(self._convective_flux()))

    def _compute_local_nusselt(self) -> torch.Tensor:
        """Nusselt number over each agent's window columns, ``[E, n_agents]``."""
        weighted = self._convective_flux() * self._cell_size
        flux = torch.einsum("ezyx,zi->ei", weighted, self._footprint_z)
        volume = torch.einsum("zyx,zi->i", self._cell_size, self._footprint_z)
        return self._flux_to_nusselt(flux / volume)

    def _get_buoyancy_power(self) -> torch.Tensor:
        """Work done by the buoyancy per unit time, ``[E]``, in ``rho U^3 H^2``."""
        volume = self._Lx * self._Lz * self._H
        return self._volume_mean(self._convective_flux()) * volume

    def _get_electrical_power(self) -> torch.Tensor:
        """Electrical power fed into the fluid by the electrodes, ``[E]``.

        ``P = N * sum I * phi_w`` over both Hartmann walls. The no-slip wall has
        ``u x B = 0``, so the current density into the fluid is ``-dphi/dn`` and
        the wall potential is ``phi_w = phi_P + h * I / A`` from the first cell
        centre at distance ``h``. ``N`` converts ``sigma U^2 B^2 H^3`` to the
        ``rho U^3 H^2`` of the buoyancy power, as it does for the Lorentz force.
        """
        epot = self._block.epot
        assert epot is not None
        power = torch.zeros(epot.size(0), dtype=epot.dtype, device=epot.device)
        cells = self._last_cell_currents
        walls = (
            (epot[:, 0, :, :, 0], self._h_minus),
            (epot[:, 0, :, :, -1], self._h_plus),
        )
        for phi_p, h in walls:
            phi_w = phi_p + h * cells / self._face_area
            power = power + (cells * phi_w).sum(dim=(1, 2))
        return self.stuart_number * power

    def _metric_ref(self, key: str) -> float:
        """Mean of the uncontrolled episodes, or 1 without domain statistics."""
        if key in self._metrics_stats:
            return float(self._metrics_stats[key].mean)
        return 1.0

    @property
    def nu_ref(self) -> float:
        """Reference Nusselt number for the reward."""
        return self._metric_ref("nusselt")

    @property
    def power_ref(self) -> float:
        """Buoyancy power of the uncontrolled flow, the scale of the power penalty."""
        return self._metric_ref("buoyancy_power")

    def _advance_impl(
        self, action: torch.Tensor, n_sim_steps: int
    ) -> dict[str, torch.Tensor]:
        """Run ``n_sim_steps`` steps, recording the heat transfer and the powers."""
        # Without actions the actuators keep the zero action reset() applied
        if self._enable_actions:
            self._apply_action(action)

        nusselt, buoyancy, electrical = [], [], []
        for _ in range(n_sim_steps):
            self._sim.single_step()
            nusselt.append(self.compute_nusselt())
            buoyancy.append(self._get_buoyancy_power())
            electrical.append(self._get_electrical_power())

        return {
            "nusselt": torch.stack(nusselt),
            "buoyancy_power": torch.stack(buoyancy),
            "electrical_power": torch.stack(electrical),
        }

    def _finish_step(
        self, metrics: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, bool, dict[str, torch.Tensor]]:
        nu = metrics["nusselt"].mean(dim=0)
        power = metrics["electrical_power"].mean(dim=0)
        power_ratio = power / self.power_ref

        reward = nu / self.nu_ref - 1.0 - self._power_penalty_weight * power_ratio

        info = {
            "nusselt": nu,
            "buoyancy_power": metrics["buoyancy_power"].mean(dim=0),
            "electrical_power": power,
            "power_ratio": power_ratio,
        }
        return self._get_global_obs(), reward, False, info

    def _finish_marl_step(
        self, metrics: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, bool, dict[str, torch.Tensor]]:
        if self._local_reward_weight is None:
            raise ValueError("local_reward_weight must be set for multi-agent step.")

        _, global_reward, terminated, info = self._finish_step(metrics)

        local_obs = self._get_local_obs()

        if self._local_reward_weight > 0:
            local_nu = self._compute_local_nusselt()
            local_rewards = local_nu / self.nu_ref - 1.0
        else:
            local_rewards = torch.zeros(
                (self._n_envs, self.n_agents),
                dtype=self._dtype,
                device=self._cuda_device,
            )

        # [E, n_agents]
        agent_rewards = (
            self._local_reward_weight * local_rewards
            + (1 - self._local_reward_weight) * global_reward[:, None]
        )

        info["global_reward"] = global_reward

        return local_obs, agent_rewards, terminated, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _get_render_data(
        self,
        render_3d: bool,
        output_path: Path | None = None,
    ) -> dict[str, np.ndarray]:
        blocks = self._domain.getBlocks()

        def format_temperature(data: np.ndarray) -> np.ndarray:
            return self._format_render_data(
                data=data, v_min=self._T_cold, v_max=self._T_hot, cmap="rainbow"
            )

        def format_signed(data: np.ndarray) -> np.ndarray:
            v_max = max(float(np.abs(data).max()), 1e-12)
            return self._format_render_data(
                data=data, v_min=-v_max, v_max=v_max, cmap="RdBu_r"
            )

        T_blocks = [b.passiveScalar for b in blocks]
        T_xy = self._render_plane(T_blocks, axis="z")  # (ny, nx)
        T_xz = self._render_plane(T_blocks, axis="y")  # (nz, nx)
        T_yz = self._render_plane(T_blocks, axis="x")  # (nz, ny)
        # The vertical velocity across the rolls
        u_y = self._render_plane([b.velocity[:, 1:2] for b in blocks], axis="x")

        render_data = {
            "x-y-temperature": np.flipud(format_temperature(T_xy)),
            "x-z-temperature": format_temperature(T_xz),
            "y-z-temperature": format_temperature(T_yz).transpose(1, 0, 2),
            "y-z-velocity": format_signed(u_y).transpose(1, 0, 2),
        }

        # The potential on the first cell layer of the -x wall shows the
        # electrode pattern
        epot = self._render_plane([b.epot for b in blocks], axis="x", index=0)
        render_data["y-z-epot_wall"] = format_signed(epot).transpose(1, 0, 2)

        if render_3d:
            T = (
                self._render_env_field(self._temperature_fields())[0]
                .detach()
                .cpu()
                .numpy()
            )
            if output_path is not None:
                output_path_3d = (
                    output_path
                    / f"3d_temperature_fig_{self._n_episodes}_{self._n_steps}.png"
                )
            else:
                output_path_3d = None

            render_data["3d_temperature"] = render_3d_voxels(
                field=T,
                ds=4,
                field_range=(self._T_cold, self._T_hot),
                output_path=output_path_3d,
                colormap="rainbow",
                view_kwargs={"elev": 15, "azim": 45},
            )

        return render_data

    def plot(self, output_path: Path | None = None) -> None:
        """Plot the electrode current shares on a Hartmann wall.

        Parameters
        ----------
        output_path: Path | None
            Directory to save the plot in. If None, the current directory is
            used. Defaults to None.
        """
        if output_path is None:
            output_path = Path(".")

        fig, ax = plt.subplots(figsize=(5 * self._aspect_ratio, 5))
        weights = self._electrode_distribution.sum(dim=0).cpu().numpy()
        ax.pcolormesh(
            self._y_vertices.numpy(),
            self._z_vertices.numpy(),
            np.ma.masked_equal(weights, 0.0),
            cmap="viridis",
        )
        ax.set_xlabel("y")
        ax.set_title("Electrode current shares on the Hartmann walls")
        ax.set_ylabel("z")
        ax.set_aspect("equal")
        fig.savefig(output_path / f"{self.id}.pdf", bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Identifiers
    # ------------------------------------------------------------------
    def _physics_tag(self) -> str:
        tag = (
            f"Ra{self._rayleigh_number:g}_Q{self._chandrasekhar_number:g}"
            f"_Pr{self._prandtl_number:g}_A{self._aspect_ratio:g}"
        )
        # Only a box that is not square in plan gets a tag, so old ids stay valid
        if self._aspect_ratio_x != self._aspect_ratio:
            tag += f"_Ax{self._aspect_ratio_x:g}"
        return tag

    def _actuator_tag(self) -> str:
        tag = f"_el{self._n_electrodes}f{self._electrode_fill:g}"
        if self._n_stations_set:
            tag += f"s{self._n_stations}"
        return tag

    @property
    def id(self) -> str:
        """Unique identifier for the environment."""
        return (
            f"RBCMHD3d_{self._physics_tag()}_N{self._resolution}{self._actuator_tag()}"
        )

    @property
    def initial_domain_id(self) -> str:
        """Unique identifier for the initial domain.

        The uncontrolled flow is the same for every electrode layout, and the electrode
        conditions are applied after loading (see :meth:`_apply_potential_layout`),
        so all of them share the domains of the plain insulating box.
        """
        return (
            f"rbc_mhd_3d_{self._physics_tag()}_N{self._resolution}"
            f"_g{self._wall_grading:g}"
        )
