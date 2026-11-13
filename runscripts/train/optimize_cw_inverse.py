"""Recover a map of wall conductance ratios of a buoyant MHD cube from velocity probes.

Usage:
    python runscripts/train/optimize_cw_inverse.py optimizer=lbfgs seed=0
    python runscripts/train/optimize_cw_inverse.py optimizer=cma_es seed=0
"""

import csv
import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

import cma
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from phipict import _C, Face, bc
from phipict.grid import shapes
from phipict.grid.helpers import get_cell_centers, get_cell_size
from phipict.grid.shapes import make_weights
from phipict.io.domain_io import load_domain, save_domain
from phipict.simulation.mhd import MHDSimulation
from phipict.solvers.tolerance import SolverTolerance
from scipy.optimize import minimize

from fluidgym.envs.checkpointing import default_domain_accessors, restore, snapshot

logger = logging.getLogger("optimize_cw_inverse")

N_WALLS = len(Face)
WALL_NAMES = [face.name.lower() for face in Face]
LOG_FILENAME = "optimization_log.csv"
HISTORY_FILENAME = "log10_cw_history.csv"
LOG_COLUMNS = [
    "wall_time",
    "sim_steps",
    "fwd_steps",
    "bwd_steps",
    "evaluation",
    "objective",
    "field_error",
    "log10_cw_rmse",
    "grad_norm",
]
LINEARITY_TOL = 1e-10
# Largest error in a mode's log10(Cw) for it to count as determined by the observations
DETERMINED_LOG10_CW_ERROR = 0.1
MAX_COORDINATE_CHECKS = 6
RANDOM_DIRECTION_CHECKS = 6


class BudgetExhausted(Exception):
    """Raised when a run reaches its wall-time limit or the target objective."""


def make_domain(
    cfg: DictConfig, dtype: torch.dtype, device: torch.device
) -> tuple[_C.Domain, dict[str, Any]]:
    """Create the closed cube and load the steady flow of ``start_domain``."""
    reynolds_number = np.sqrt(cfg.grashof_number)

    x_weights = make_weights(
        grading_type=cfg.domain.grading_x_type,
        res=cfg.domain.nx,
        grading=cfg.domain.grading_x,
        refinement="BOTH",
    )
    y_weights = make_weights(
        grading_type=cfg.domain.grading_y_type,
        res=cfg.domain.ny,
        grading=cfg.domain.grading_y,
        refinement="BOTH",
    )
    z_weights = make_weights(
        grading_type=cfg.domain.grading_z_type,
        res=cfg.domain.nz,
        grading=cfg.domain.grading_z,
        refinement="BOTH",
    )

    half_width = cfg.domain.L / 2
    half_height = cfg.domain.H / 2
    grid = shapes.generate_grid_vertices_2D(
        [cfg.domain.ny + 1, cfg.domain.nx + 1],
        [
            (-half_width, -half_height),
            (half_width, -half_height),
            (-half_width, half_height),
            (half_width, half_height),
        ],
        None,
        x_weights=y_weights,
        y_weights=x_weights,
        dtype=dtype,
    )
    grid = shapes.extrude_grid_z(
        grid,
        res_z=cfg.domain.nz,
        start_z=-half_width,
        end_z=half_width,
        weights_z=z_weights,
    )
    grid = grid.to(device).contiguous()

    kinematic_viscosity = torch.tensor(
        [(cfg.domain.U * cfg.domain.L) / reynolds_number],
        dtype=dtype,
        device=device,
    )
    domain = _C.Domain(
        3,
        kinematic_viscosity.cpu(),
        name="Domain",
        device=device,
        dtype=dtype,
    )
    block = domain.CreateBlock(vertexCoordinates=grid, name="Block")

    for label in ("-x", "+x", "-y", "+y", "-z", "+z"):
        block.CloseBoundary(label)
    for face in Face:
        bc.set_bc(block, face, bc.Potential.ThinWall(cw=float(cfg.cw_init)))

    # alpha = nu/Pr, so it carries the same L as the viscosity above
    thermal_diffusivity = torch.tensor(
        [(cfg.domain.U * cfg.domain.L) / (reynolds_number * cfg.prandtl_number)],
        dtype=dtype,
        device=device,
    )
    domain.setScalarViscosity(thermal_diffusivity)

    block.getBoundary("-z").setPassiveScalar(
        torch.tensor([[cfg.domain.T_hot]], dtype=dtype, device=device)
    )
    block.getBoundary("+z").setPassiveScalar(
        torch.tensor([[cfg.domain.T_cold]], dtype=dtype, device=device)
    )
    for label in ("-x", "+x", "-y", "+y"):
        block.getBoundary(label).setPassiveScalarType(_C.BoundaryConditionType.NEUMANN)

    buoyancy_factor = cfg.grashof_number / reynolds_number**2
    source_padding = torch.zeros_like(block.passiveScalar)
    T_ref = (cfg.domain.T_hot + cfg.domain.T_cold) / 2

    def buoyancy_fn(domain, time_step, **kwargs):
        T = domain.getBlocks()[0].passiveScalar
        source = torch.cat(
            [source_padding, (T - T_ref) * buoyancy_factor, source_padding], dim=1
        )
        block.setVelocitySource(source)
        domain.UpdateDomainData()

    prep_fn: dict[str, Any] = {"PRE_VELOCITY_SETUP": [buoyancy_fn]}

    loaded = load_domain(
        Path(to_absolute_path(cfg.start_domain)), device=device, dtype=dtype
    )
    loaded_block = loaded.getBlocks()[0]
    block.setVelocity(loaded_block.velocity)
    block.setPressure(loaded_block.pressure)
    block.setPassiveScalar(loaded_block.passiveScalar)
    block.setEpot(loaded_block.epot)

    domain.PrepareSolve()
    return domain, prep_fn


def solver_tolerance(cfg: DictConfig, name: str) -> float | SolverTolerance | None:
    """Resolve ``sim.<name>_tol`` into the form the simulation expects.

    The mode is read from ``sim.<name>_tol_mode`` if set, otherwise from
    ``sim.tol_mode``. ``sim.tol_atol`` is the absolute floor on a relative tolerance.
    """
    value = cfg.sim.get(f"{name}_tol", None)
    if value is None:
        return None

    mode = cfg.sim.get(f"{name}_tol_mode", None) or cfg.sim.get("tol_mode", "absolute")
    mode = str(mode).lower()
    if mode == "absolute":
        return float(value)
    if mode != "relative":
        raise ValueError(
            f"sim.{name}_tol_mode must be 'absolute' or 'relative', got {mode!r}."
        )

    atol = cfg.sim.get("tol_atol", None)
    return SolverTolerance(
        rtol=float(value), atol=None if atol is None else float(atol)
    )


def make_mhd_simulation(
    cfg: DictConfig,
    domain: _C.Domain,
    prep_fn: dict[str, Any],
    differentiable: bool,
) -> MHDSimulation:
    if cfg.hartmann_number == 0.0:
        raise ValueError("Hartmann number must be > 0, Cw has no effect otherwise.")

    domain.setAdvectionScheme(getattr(_C.AdvectionScheme, cfg.advection_scheme.upper()))
    for name in ("advection", "pressure", "pressure_intermediate", "potential"):
        logger.info("%s_tol: %s", name, solver_tolerance(cfg, name))

    reynolds_number = np.sqrt(cfg.grashof_number)
    sim = MHDSimulation(
        domain=domain,
        dt=cfg.sim.dt,
        stuart_number=torch.tensor(cfg.hartmann_number**2 / reynolds_number),
        e_b=torch.tensor([1.0, 0.0, 0.0], dtype=domain.getDtype()),
        substeps="ADAPTIVE",
        corrector_steps=cfg.sim.corrector_steps,
        advection_tol=solver_tolerance(cfg, "advection"),
        pressure_tol=solver_tolerance(cfg, "pressure"),
        pressure_tol_intermediate=solver_tolerance(cfg, "pressure_intermediate"),
        pressure_warm_start=bool(cfg.sim.get("pressure_warm_start", False)),
        potential_tol=solver_tolerance(cfg, "potential"),
        non_orthogonal=False,
        potential_normalize=cfg.sim.potential_normalize,
        potential_use_BiCG=cfg.sim.potential_use_BiCG,
        potential_solve_max_iter=cfg.sim.potential_solve_max_iter,
        potential_use_preconditioner=bool(
            cfg.sim.get("potential_use_preconditioner", False)
        ),
        potential_amg_options=(
            OmegaConf.to_container(cfg.sim.potential_amg_options, resolve=True)
            if cfg.sim.get("potential_amg_options", None)
            else None
        ),
        prep_fn=prep_fn,
        adaptive_CFL=cfg.sim.adaptive_cfl,
        differentiable=differentiable,
    )
    sim.solver_double_fallback = True
    sim.linear_solve_max_iterations = cfg.sim.pressure_solve_max_iter
    with torch.no_grad():
        sim.make_divergence_free(max_iter=10000)
    return sim


def compute_nusselt(cfg: DictConfig, domain: _C.Domain) -> torch.Tensor:
    """Volume-averaged Nusselt number, ``sqrt(Ra Pr) <u_z T> - dT/dz``."""
    block = domain.getBlocks()[0]
    T = block.passiveScalar[0, 0]  # [nz, ny, nx]
    u_z = block.velocity[0, 2, ...]  # [nz, ny, nx]
    cell_size = get_cell_size(block).squeeze()  # [nz, ny, nx]

    rayleigh_number = cfg.grashof_number * cfg.prandtl_number
    dT_dz = (cfg.domain.T_cold - cfg.domain.T_hot) / cfg.domain.L
    inner_term = np.sqrt(rayleigh_number * cfg.prandtl_number) * u_z * T - dT_dz
    return (inner_term * cell_size).sum() / cell_size.sum()


class StepCounter:
    """Hook counting the PISO steps, so adaptive substeps count as well."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.count += 1


class WallConductanceMap:
    """Sets one Cw per wall patch through ``A(p) = A0 + sum_k p_k * B_k``."""

    def __init__(self, cfg: DictConfig, domain: _C.Domain, sim: MHDSimulation) -> None:
        self._cfg = cfg
        self._domain = domain
        self._sim = sim
        self._current: list[float] | None = None
        self.patches_per_wall = 0

        A_base = self._assemble([1.0] * N_WALLS)
        terms = []
        for i in range(N_WALLS):
            cw = [1.0] * N_WALLS
            cw[i] = 2.0
            terms.append(self._assemble(cw) - A_base)
        self._A_wall = torch.stack(terms)
        self._A0 = A_base - self._A_wall.sum(dim=0)

        # A direct assembly with a different Cw per wall, as a reference for any map
        self._check_cw = [0.01, 0.7, 0.03, 5.0, 0.2, 30.0]
        self._check_matrix = self._assemble(self._check_cw)

        # The solver's own Cw is unused from here on, as the matrix is set directly
        self._assemble([float(cfg.cw_init)] * N_WALLS)

        csr = domain.Epot
        row_pointer = csr.row.view(-1).long()
        self._columns = csr.index.view(-1).long()
        self._rows = torch.repeat_interleave(
            torch.arange(row_pointer.numel() - 1, device=row_pointer.device),
            row_pointer[1:] - row_pointer[:-1],
        )
        self._centers = get_cell_centers(domain.getBlocks()[0].vertexCoordinates)

    def _assemble(self, cw: list[float]) -> torch.Tensor:
        block = self._domain.getBlocks()[0]
        for face, value in zip(Face, cw, strict=True):
            bc.set_bc(block, face, bc.Potential.ThinWall(cw=float(value)))
        # setPotentialBC does not flag a change, and UpdateDomainData only copies the
        # new Cw to the GPU after one. Without this, the wall terms are silently zero
        self._domain.setEpotResult(self._domain.epotResult)
        self._domain.UpdateDomainData()
        _C.SetupEpotMatrix(self._domain)
        return self._domain.Epot.value.detach().clone()

    @property
    def n_patches(self) -> int:
        return N_WALLS * self.patches_per_wall**2

    def _patch_of(self, face: Face, cells: torch.Tensor) -> torch.Tensor:
        """Patch index of wall cells, from their position on the wall."""
        K = self.patches_per_wall
        nz, ny, nx = self._centers.shape[1:]
        z, y, x = cells // (nx * ny), (cells // nx) % ny, cells % nx

        patch = torch.zeros_like(cells)
        factor = 1
        for axis in range(3):
            if axis == face.axis:
                continue
            position = self._centers[axis, z, y, x] / self._cfg.domain.L + 0.5
            index = torch.clamp((position * K).long(), 0, K - 1)
            patch = patch + factor * index
            factor *= K
        return patch + int(face) * K**2

    def set_patches(self, patches_per_wall: int) -> None:
        """Split the wall terms of the matrix into links between patches."""
        self.patches_per_wall = patches_per_wall
        self._current = None

        diagonal = self._rows == self._columns
        diagonal_of_row = torch.zeros(
            int(self._rows.max()) + 1, dtype=torch.long, device=self._rows.device
        )
        diagonal_of_row[self._rows[diagonal]] = diagonal.nonzero().view(-1)

        positions, patches, coefficients = [], [], []
        for i, face in enumerate(Face):
            values = self._A_wall[i]
            links = ((values != 0) & ~diagonal).nonzero().view(-1)
            rows, columns, link_values = (
                self._rows[links],
                self._columns[links],
                values[links],
            )
            patch_row = self._patch_of(face, rows)
            patch_column = self._patch_of(face, columns)
            diagonals = diagonal_of_row[rows]
            positions += [links, links, diagonals, diagonals]
            patches += [patch_row, patch_column, patch_row, patch_column]
            half = 0.5 * link_values
            coefficients += [half, half, -half, -half]

        self._positions = torch.cat(positions)
        self._patches = torch.cat(patches)
        self._coefficients = torch.cat(coefficients)
        self._check_linearity()

    def _compose(self, p: torch.Tensor) -> torch.Tensor:
        return self._A0.index_add(
            0, self._positions, self._coefficients * p[self._patches]
        )

    def _check_linearity(self) -> None:
        """Compare the composed matrix with a direct assembly, Cw constant per wall."""
        p = torch.as_tensor(
            np.repeat(self._check_cw, self.patches_per_wall**2),
            dtype=self._A0.dtype,
            device=self._A0.device,
        )
        difference = self._check_matrix - self._compose(p)
        residual = (torch.norm(difference) / torch.norm(self._check_matrix)).item()
        logger.info(
            "Potential matrix check: %d patches, residual %.3e", p.numel(), residual
        )
        if residual > LINEARITY_TOL:
            raise RuntimeError(f"The potential matrix does not match: {residual}")

    def set(self, p: torch.Tensor) -> None:
        """Apply the patch values, keeping their graph if they have one."""
        values = p.tolist()
        if values != self._current:
            self._current = values
            self._domain.Epot.setValue(self._compose(p.detach()))
            self._domain.UpdateDomainData()

            # The cached hierarchy assumes constant matrix values. Rebuilt here to
            # keep it off the graph of the first differentiated solve
            self._sim._epot_amg_hierarchy = None
            if self._cfg.sim.potential_use_preconditioner:
                with torch.no_grad():
                    self._sim._get_epot_amg_hierarchy(self._domain.Epot)

        if p.requires_grad:
            self._domain.Epot.setValue(self._compose(p))
            self._domain.UpdateDomainData()


class InverseProblem:
    """Restarts from the steady state and compares the flow with the target."""

    def __init__(
        self,
        cfg: DictConfig,
        domain: _C.Domain,
        sim: MHDSimulation,
        counter: StepCounter,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._cfg = cfg
        self._domain = domain
        self._sim = sim
        self._counter = counter
        self._dtype = dtype
        self._device = device
        self._block = domain.getBlocks()[0]
        self._map = WallConductanceMap(cfg, domain, sim)
        self._map.set_patches(1)

        cw = self._tensor([float(cfg.cw_init)] * N_WALLS)
        with torch.no_grad():
            self._map.set(cw)
            # A loaded domain has no consistent u x e_b and phi, which offsets Nu
            self._sim.make_current_divergence_free()
        logger.info("Start state: Nu = %.6f", compute_nusselt(cfg, domain).item())

        # Cross-step state outside the domain: the lagged u x e_b
        self._accessors = default_domain_accessors(domain) + [
            (
                lambda: sim._u_cross_eb_flat,
                lambda t: setattr(sim, "_u_cross_eb_flat", t),
            ),
        ]
        self._state, self._mask = snapshot(self._accessors, clone=True)
        self._u_start = self._block.velocity.detach().clone()

        cell_size = get_cell_size(self._block).squeeze().to(dtype=dtype, device=device)
        self._sqrt_weights = torch.sqrt(cell_size / cell_size.sum())
        self._sensors: tuple[torch.Tensor, ...] | None = None

        self._horizon = 0
        self._target: torch.Tensor | None = None
        self._scale = 1.0
        self.bwd_steps = 0

    @property
    def fwd_steps(self) -> int:
        return self._counter.count

    @property
    def n_patches(self) -> int:
        return self._map.n_patches

    def reset_counters(self) -> None:
        self._counter.count = 0
        self.bwd_steps = 0

    def _tensor(self, values: Any) -> torch.Tensor:
        return torch.as_tensor(np.asarray(values), dtype=self._dtype, device=self._device)

    def configure(self, patches_per_wall: int, sensor_count: int | None) -> None:
        """Choose the map's resolution and the sensors, which are the same for all runs."""
        self._map.set_patches(patches_per_wall)

        if sensor_count is None:
            self._sensors = None
            return

        margin = float(self._cfg.sensors.margin)
        centers = get_cell_centers(self._block.vertexCoordinates).detach()
        half = self._cfg.domain.L / 2 - margin
        core = (centers.abs() < half).all(dim=0).nonzero()
        rng = np.random.default_rng(int(self._cfg.sensors.seed))
        chosen = rng.choice(core.shape[0], size=int(sensor_count), replace=False)
        sensors = core[torch.as_tensor(chosen, device=core.device)]
        self._sensors = (sensors[:, 0], sensors[:, 1], sensors[:, 2])

    def _observe(self, velocity: torch.Tensor) -> torch.Tensor:
        """Volume-weighted field, or the velocity at the sensors with equal weights."""
        if self._sensors is None:
            return (self._sqrt_weights * velocity).flatten()
        z, y, x = self._sensors
        return velocity[0, :, z, y, x].flatten() / np.sqrt(z.numel())

    def _step(self) -> None:
        if not self._sim.single_step():
            logger.error("Solver failed to converge.")

    def _rollout(self, log10_cw: torch.Tensor, differentiable: bool) -> torch.Tensor:
        """Run the horizon from the steady state and return the final velocity."""
        # Solvers write in place, so the saved state itself must never be handed out
        restore(
            self._accessors,
            [t.clone() for t in self._state],
            self._mask,
            on_restored=self._domain.UpdateDomainData,
        )
        # Every evaluation has to see the same solver state
        self._sim.clear_pressure_guess()

        cw = 10.0**log10_cw
        self._map.set(cw.detach())
        with torch.enable_grad() if differentiable else torch.no_grad():
            self._map.set(cw if differentiable else cw.detach())
            for _ in range(self._horizon):
                self._step()
            return self._block.velocity

    def set_target(self, log10_cw: np.ndarray, horizon: int) -> dict[str, float]:
        """Create the target. Its steps count towards no budget."""
        self._horizon = horizon
        steps = self._counter.count
        velocity = self._rollout(self._tensor(log10_cw), differentiable=False)
        self._counter.count = steps

        self._target = self._observe(velocity).detach().clone()
        start = self._observe(self._u_start)
        self._scale = torch.norm(self._target - start).item()
        if self._scale == 0.0:
            raise RuntimeError("The hidden Cw map does not change the observations.")

        relative_change = self._scale / torch.norm(start).item()
        logger.info(
            "Target: horizon = %d, ||obs_target - obs_start|| / ||obs_start|| = %.4e",
            horizon,
            relative_change,
        )
        return {"horizon": horizon, "relative_change": relative_change}

    def save_current_domain(self, name: str) -> None:
        """Save the domain, which holds the final state of the last rollout."""
        save_domain(self._domain, Path(name))

    def start_drift(self) -> float:
        """Change over the horizon at the unchanged ``cw_init``, relative to the target's.

        Candidates and target share this drift, so it is no error in itself, but it
        should be small for the start state to count as steady.
        """
        steps = self._counter.count
        log10_cw = np.full(self.n_patches, np.log10(float(self._cfg.cw_init)))
        velocity = self._rollout(self._tensor(log10_cw), differentiable=False)
        self._counter.count = steps
        difference = self._observe(velocity) - self._observe(self._u_start)
        return torch.norm(difference).item() / self._scale

    def _residual(self, velocity: torch.Tensor) -> torch.Tensor:
        return (self._observe(velocity) - self._target) / self._scale

    def residual(self, log10_cw: np.ndarray) -> torch.Tensor:
        """Normalized observation difference. Its squared norm is the objective."""
        return self._residual(self._rollout(self._tensor(log10_cw), False)).detach()

    def objective(self, log10_cw: np.ndarray) -> float:
        return torch.sum(self.residual(log10_cw) ** 2).item()

    def objective_and_grad(self, log10_cw: np.ndarray) -> tuple[float, np.ndarray]:
        x = self._tensor(log10_cw).clone().requires_grad_(True)
        start = self._counter.count
        velocity = self._rollout(x, differentiable=True)
        loss = torch.sum(self._residual(velocity) ** 2)
        window_steps = self._counter.count - start
        loss.backward()
        self.bwd_steps += window_steps

        assert x.grad is not None
        value = loss.item()
        grad = x.grad.detach().cpu().numpy()

        # Drop the graph, whose autograd cycles only the collector can free
        del velocity, loss
        self._domain.Detach()
        gc.collect()
        return value, grad


class OptimizationLog:
    """Writes the logs row by row, keeps the best domain and enforces the time limit."""

    def __init__(
        self, cfg: DictConfig, problem: InverseProblem, log10_cw_true: np.ndarray
    ) -> None:
        self._problem = problem
        self._log10_cw_true = log10_cw_true
        self._walltime_limit = float(cfg.walltime_limit)
        self._stop_objective = float(cfg.stop_objective)
        self._file = open(LOG_FILENAME, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=LOG_COLUMNS)
        self._writer.writeheader()
        self._file.flush()
        self._history_file = open(HISTORY_FILENAME, "w", newline="")
        self._history = csv.writer(self._history_file)
        self._history.writerow(
            ["evaluation", *[f"p{i}" for i in range(problem.n_patches)]]
        )
        self._evaluation = 0
        self._best_objective = float("inf")
        problem.reset_counters()
        self._start = time.perf_counter()

    def _save_best(self, objective: float, log10_cw: np.ndarray) -> None:
        """Keep the final domain of the best evaluation, outside the logged wall time."""
        start = time.perf_counter()
        self._best_objective = objective
        self._problem.save_current_domain("best_domain")
        with open("best_domain_info.json", "w") as f:
            json.dump(
                {
                    "evaluation": self._evaluation,
                    "objective": objective,
                    "log10_cw": np.asarray(log10_cw).tolist(),
                },
                f,
                indent=2,
            )
        self._start += time.perf_counter() - start

    def log(
        self, objective: float, log10_cw: np.ndarray, grad: np.ndarray | None = None
    ) -> None:
        log10_cw = np.asarray(log10_cw)
        wall_time = time.perf_counter() - self._start
        fwd_steps = self._problem.fwd_steps
        bwd_steps = self._problem.bwd_steps
        rmse = float(np.sqrt(np.mean((log10_cw - self._log10_cw_true) ** 2)))
        self._writer.writerow(
            {
                "wall_time": wall_time,
                "sim_steps": fwd_steps + bwd_steps,
                "fwd_steps": fwd_steps,
                "bwd_steps": bwd_steps,
                "evaluation": self._evaluation,
                "objective": objective,
                "field_error": float(np.sqrt(objective)),
                "log10_cw_rmse": rmse,
                "grad_norm": float("nan") if grad is None else np.linalg.norm(grad),
            }
        )
        self._file.flush()
        self._history.writerow([self._evaluation, *log10_cw.tolist()])
        self._history_file.flush()
        logger.info(
            "Evaluation %d: J = %.6e, log10(Cw) RMSE = %.4f, steps = %d + %d",
            self._evaluation,
            objective,
            rmse,
            fwd_steps,
            bwd_steps,
        )
        if objective < self._best_objective:
            self._save_best(objective, log10_cw)
        self._evaluation += 1

        if objective <= self._stop_objective:
            raise BudgetExhausted(f"Reached J = {objective:.3e}")
        if wall_time >= self._walltime_limit:
            raise BudgetExhausted(f"Used {wall_time:.0f} s of wall time")


def sample_maps(
    cfg: DictConfig, patches_per_wall: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the hidden map and the start. Same for all optimizers of a seed.

    The hidden map is a base value per wall plus defects on random patches. The start
    knows nothing about the defects: one other value per wall.
    """
    rng = np.random.default_rng(int(cfg.seed))
    low, high = (float(v) for v in cfg.log10_cw_sample_range)
    base = rng.uniform(low, high, N_WALLS)
    start = rng.uniform(low, high, N_WALLS)

    n_per_wall = patches_per_wall**2
    hidden = np.repeat(base, n_per_wall)
    if patches_per_wall > 1:
        low_defect, high_defect = (float(v) for v in cfg.defects.log10_offset)
        is_defect = rng.random(hidden.size) < float(cfg.defects.fraction)
        hidden += is_defect * rng.uniform(low_defect, high_defect, hidden.size)
        logger.info("Hidden map: %d of %d patches defect", is_defect.sum(), hidden.size)
    hidden = np.minimum(hidden, float(cfg.log10_cw_bounds[1]))
    return hidden, np.repeat(start, n_per_wall)


def run_lbfgs(
    cfg: DictConfig,
    problem: InverseProblem,
    log: OptimizationLog,
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
) -> None:
    """L-BFGS-B from SciPy, which evaluates objective and gradient together."""

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = problem.objective_and_grad(x)
        log.log(value, x, grad)
        return value, grad

    result = minimize(
        fun,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "ftol": float(cfg.lbfgs.ftol),
            "gtol": float(cfg.lbfgs.gtol),
            "maxiter": int(cfg.lbfgs.maxiter),
            "maxfun": int(cfg.lbfgs.maxfun),
        },
    )
    logger.info("L-BFGS-B stopped: %s", result.message)


def run_adam(
    cfg: DictConfig,
    problem: InverseProblem,
    log: OptimizationLog,
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
) -> None:
    """Adam, clamping to the bounds after each step."""
    x = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    optim = torch.optim.Adam(
        [x], lr=float(cfg.adam.lr), betas=(float(cfg.adam.beta1), 0.999)
    )
    lower = torch.tensor([b[0] for b in bounds], dtype=torch.float64)
    upper = torch.tensor([b[1] for b in bounds], dtype=torch.float64)

    while True:
        current = x.detach().numpy().copy()
        value, grad = problem.objective_and_grad(current)
        log.log(value, current, grad)

        optim.zero_grad()
        x.grad = torch.from_numpy(grad.copy()).to(dtype=torch.float64)
        optim.step()
        with torch.no_grad():
            x.copy_(torch.minimum(torch.maximum(x, lower), upper))


def run_cma_es(
    cfg: DictConfig,
    problem: InverseProblem,
    log: OptimizationLog,
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
) -> None:
    """CMA-ES at its defaults. Candidates run one by one."""
    es = cma.CMAEvolutionStrategy(
        x0.tolist(),
        float(cfg.cma_es.sigma0),
        {
            "bounds": [[b[0] for b in bounds], [b[1] for b in bounds]],
            "seed": int(cfg.seed) + 1,
            "verbose": -9,
        },
    )
    logger.info("CMA-ES population size: %d", es.popsize)
    while not es.stop():
        candidates = es.ask()
        values = []
        for candidate in candidates:
            value = problem.objective(candidate)
            values.append(value)
            log.log(value, candidate)
        es.tell(candidates, values)
    logger.info("CMA-ES stopped: %s", es.stop())


OPTIMIZERS = {
    "lbfgs": run_lbfgs,
    "adam": run_adam,
    "cma_es": run_cma_es,
}


def check_directions(n_patches: int) -> np.ndarray:
    """Directions for the gradient check: each wall if few, random signs otherwise."""
    if n_patches <= MAX_COORDINATE_CHECKS:
        return np.eye(n_patches)
    rng = np.random.default_rng(0)
    return rng.choice([-1.0, 1.0], size=(RANDOM_DIRECTION_CHECKS, n_patches))


def run_pilot_case(
    cfg: DictConfig, problem: InverseProblem, case: DictConfig
) -> dict[str, Any]:
    """Measure one combination of map resolution, sensors and horizon."""
    eps = float(cfg.pilot.eps)
    sensor_count = case.get("sensors", None)
    problem.configure(int(case.patches_per_wall), sensor_count)
    log10_cw_true, x0 = sample_maps(cfg, int(case.patches_per_wall))
    n_patches = problem.n_patches
    logger.info(
        "=== %d patches, sensors = %s, horizon = %d ===",
        n_patches,
        sensor_count,
        case.horizon,
    )

    start_time = time.perf_counter()
    result: dict[str, Any] = {
        "n_patches": n_patches,
        "sensors": sensor_count,
        **problem.set_target(log10_cw_true, int(case.horizon)),
    }

    result["start_drift"] = problem.start_drift()
    noise = torch.norm(problem.residual(log10_cw_true)).item()
    result["repeat_error"] = noise
    logger.info(
        "Drift relative to the target's change = %.4e, repeat error = %.4e",
        result["start_drift"],
        noise,
    )

    forward = time.perf_counter()
    problem.objective(x0)
    result["forward_seconds"] = time.perf_counter() - forward

    backward = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    value, grad = problem.objective_and_grad(x0)
    result["forward_backward_seconds"] = time.perf_counter() - backward
    result["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    result["objective_at_start"] = value
    logger.info("J(start) = %.6e", value)

    directions = check_directions(n_patches)
    bptt = directions @ grad
    fd = np.array(
        [
            (problem.objective(x0 + eps * v) - problem.objective(x0 - eps * v)) / (2 * eps)
            for v in directions
        ]
    )
    cosine = float(np.dot(bptt, fd) / (np.linalg.norm(bptt) * np.linalg.norm(fd)))
    result.update(bptt=bptt.tolist(), fd=fd.tolist(), grad_cosine=cosine)
    for i in range(len(directions)):
        logger.info(
            "  direction %d: BPTT = %+.6e, FD = %+.6e, BPTT/FD = %.4f",
            i,
            bptt[i],
            fd[i],
            bptt[i] / fd[i] if fd[i] != 0.0 else float("nan"),
        )
    logger.info("  cosine(BPTT, FD) = %.6f", cosine)

    # Sensitivity at the hidden map. A small singular value is a combination of patches
    # that the observations barely determine
    columns = []
    for i in range(n_patches):
        step = np.zeros(n_patches)
        step[i] = eps
        difference = problem.residual(log10_cw_true + step) - problem.residual(
            log10_cw_true - step
        )
        columns.append(difference / (2 * eps))
    sensitivity = torch.stack(columns, dim=1)
    patch_norms = torch.linalg.norm(sensitivity, dim=0).cpu().numpy()
    singular = torch.linalg.svdvals(sensitivity).cpu().numpy()
    determined = int(np.sum(noise / singular < DETERMINED_LOG10_CW_ERROR))

    result.update(
        patch_sensitivity=patch_norms.tolist(),
        singular_values=singular.tolist(),
        condition_number=float(singular[0] / singular[-1]),
        determined_modes=determined,
        seconds=time.perf_counter() - start_time,
    )
    logger.info(
        "Patch sensitivity per decade: min = %.3e, median = %.3e, max = %.3e",
        patch_norms.min(),
        np.median(patch_norms),
        patch_norms.max(),
    )
    logger.info("Singular values: %s", np.array2string(singular, precision=3))
    logger.info(
        "Condition number = %.3e, determined modes = %d of %d",
        result["condition_number"],
        determined,
        n_patches,
    )
    logger.info(
        "One forward = %.2f s, forward + backward = %.2f s, peak GPU = %.1f GiB, "
        "case took %.0f s",
        result["forward_seconds"],
        result["forward_backward_seconds"],
        result["peak_gpu_gib"],
        result["seconds"],
    )
    return result


def run_pilot(cfg: DictConfig, problem: InverseProblem) -> None:
    results = []
    for case in cfg.pilot.cases:
        results.append(run_pilot_case(cfg, problem, case))
        with open("pilot.json", "w") as f:
            json.dump(results, f, indent=2)


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="optimize_cw_inverse"
)
def main(cfg: DictConfig):
    try:
        run_optimization(cfg)
    except Exception as e:
        logger.exception("Job crashed with an exception")
        logger.error(e)
        raise
    finally:
        # A multirun keeps the process, and the next job should get a free GPU
        gc.collect()
        torch.cuda.empty_cache()


def run_optimization(cfg: DictConfig):
    if cfg.optimizer != "pilot" and cfg.optimizer not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{cfg.optimizer}'. Expected pilot or one of: "
            f"{', '.join(OPTIMIZERS)}."
        )

    dtype = torch.float64 if cfg.precision == "double" else torch.float32
    device = torch.device("cuda")

    domain, prep_fn = make_domain(cfg, dtype=dtype, device=device)

    counter = StepCounter()
    prep_fn["PRE_VELOCITY_SETUP"].append(counter)
    sim = make_mhd_simulation(
        cfg, domain, prep_fn=prep_fn, differentiable=cfg.optimizer != "cma_es"
    )
    problem = InverseProblem(cfg, domain, sim, counter, dtype, device)

    if cfg.optimizer == "pilot":
        run_pilot(cfg, problem)
        return

    patches_per_wall = int(cfg.patches_per_wall)
    problem.configure(patches_per_wall, cfg.sensors.count)
    log10_cw_true, x0 = sample_maps(cfg, patches_per_wall)
    lower, upper = (float(v) for v in cfg.log10_cw_bounds)
    bounds = [(lower, upper)] * problem.n_patches

    with open("maps.json", "w") as f:
        json.dump(
            {
                "patches_per_wall": patches_per_wall,
                "walls": WALL_NAMES,
                "log10_cw_true": log10_cw_true.tolist(),
                "log10_cw_start": x0.tolist(),
            },
            f,
            indent=2,
        )

    problem.set_target(log10_cw_true, int(cfg.horizon_steps))
    problem.save_current_domain("target_domain")
    log = OptimizationLog(cfg, problem, log10_cw_true)
    try:
        OPTIMIZERS[cfg.optimizer](cfg, problem, log, x0, bounds)
    except BudgetExhausted as stop:
        logger.info("Stopped: %s", stop)


if __name__ == "__main__":
    main()
