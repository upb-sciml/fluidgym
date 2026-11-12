"""Wall-normal grading for MHD environments.

An imposed magnetic field thins the boundary layers on the walls it touches, so
the grid has to be graded for the field rather than for the flow alone. The
helpers here turn a Hartmann number and a resolution into the vertex weights
that resolve those layers, and are shared by every MHD environment whose walls
are perpendicular to one grid axis.
"""

import math
from typing import Literal

import numpy as np
import torch
from phipict.grid.shapes import make_weights

from fluidgym.logging import get_logger

_LOG = get_logger("envs.util.mhd_grid")

GradingType = Literal["simple", "chebyshev_identity", "tanh"]

# Grading value that gives a uniform grid, and the range the automatic search
# covers, per grading type
_UNIFORM_GRADING: dict[str, float] = {
    "simple": 1.0,
    "chebyshev_identity": 0.0,
    "tanh": 0.0,
}
_GRADING_SEARCH_RANGE: dict[str, tuple[float, float]] = {
    "simple": (1.0, 50.0),
    "chebyshev_identity": (0.0, 1.0),
    "tanh": (0.0, 10.0),
}

# Largest ratio of neighbouring cell sizes accepted without a warning
_MAX_STRETCH_RATIO = 1.2


def get_layer_thickness(
    hartmann_number: float,
    e_b: torch.Tensor,
    wall_axis: int = 1,
    length: float = 1.0,
) -> float:
    """Return the thickness of the MHD boundary layer on a pair of walls.

    A field with a wall-normal component ``b_n`` forms a Hartmann layer of
    thickness ``length / (Ha |b_n|)``. This oblique scaling holds until it
    reaches the parallel-layer scale ``length / sqrt(Ha)``, which is the
    thickness for a field parallel to the walls. The layer is therefore the
    thinner of the two.

    Parameters
    ----------
    hartmann_number: float
        Hartmann number based on ``length``.

    e_b: torch.Tensor
        Unit field direction of shape ``(3,)`` in physical ``(x, y, z)`` order.

    wall_axis: int
        Axis the walls are perpendicular to, in the same ``(x, y, z)`` order.
        Defaults to 1, the wall-normal direction of a plate or channel.

    length: float
        Length scale the Hartmann number is defined on, e.g. the cell height of
        a Rayleigh-Bénard cell or the half-height of a channel. Defaults to 1.

    Returns
    -------
    float
        The layer thickness, ``inf`` for ``Ha = 0``.
    """
    if hartmann_number <= 0.0:
        return math.inf

    b_n = abs(float(e_b[wall_axis]))
    delta_parallel = length / math.sqrt(hartmann_number)
    if b_n == 0.0:
        return delta_parallel

    delta_hartmann = length / (hartmann_number * b_n)
    return min(delta_hartmann, delta_parallel)


def count_layer_cells(weights: list[float], delta: float) -> tuple[int, int]:
    """Count the cell centres within ``delta`` of the lower and upper wall.

    Parameters
    ----------
    weights: list[float]
        Normalised vertex positions in ``[0, 1]``, ``res + 1`` entries.

    delta: float
        Layer thickness relative to the wall-to-wall distance.

    Returns
    -------
    tuple[int, int]
        Number of cells in the layer at the lower and the upper wall.
    """
    w = np.asarray(weights)
    centers = 0.5 * (w[:-1] + w[1:])
    n_lower = int((centers < delta).sum())
    n_upper = int((centers > 1.0 - delta).sum())
    return n_lower, n_upper


def _max_stretch_ratio(weights: list[float]) -> float:
    sizes = np.diff(np.asarray(weights))
    if len(sizes) < 2:
        return 1.0
    ratios = sizes[1:] / sizes[:-1]
    return float(np.maximum(ratios, 1.0 / ratios).max())


def make_mhd_wall_weights(
    res: int,
    delta: float,
    min_layer_cells: int,
    grading_type: GradingType = "tanh",
    grading: float | None = None,
    uniform: bool = False,
    max_first_cell: float | None = None,
) -> tuple[list[float], float]:
    """Build the wall-normal vertex weights that resolve the MHD wall layers.

    If ``grading`` is None, the weakest grading that satisfies every constraint
    is found by bisection: at least ``min_layer_cells`` cell centres inside
    ``delta`` at both walls, and, when given, a first cell no larger than
    ``max_first_cell``. The layer check and the stretch check only warn; the
    report line is logged at INFO level.

    Parameters
    ----------
    res: int
        Number of cells between the walls.

    delta: float
        Layer thickness relative to the wall-to-wall distance, see
        :func:`get_layer_thickness`.

    min_layer_cells: int
        Minimum number of cells inside the layer at each wall.

    grading_type: str
        One of ``"simple"``, ``"chebyshev_identity"`` or ``"tanh"``.

    grading: float | None
        Grading strength. None (the default) selects it automatically.

    uniform: bool
        Force a uniform grid. Overrides ``grading``.

    max_first_cell: float | None
        Largest accepted size of the wall-adjacent cell, relative to the
        wall-to-wall distance. It lets a wall-turbulence resolution requirement
        (e.g. the first cell at ``y+ ~ 1``) be imposed alongside the MHD layer.
        None (the default) leaves the cell size to the layer constraint alone.

    Returns
    -------
    tuple[list[float], float]
        The weights (``res + 1`` entries from 0 to 1) and the grading used.
    """
    if grading_type not in _UNIFORM_GRADING:
        raise ValueError(
            f"Unknown grading type '{grading_type}'. Use 'simple', "
            "'chebyshev_identity', or 'tanh'."
        )

    def weights_for(g: float) -> list[float]:
        return make_weights(
            grading_type=grading_type, res=res, grading=g, refinement="BOTH"
        )

    def resolved(g: float) -> bool:
        w = weights_for(g)
        if min(count_layer_cells(w, delta)) < min_layer_cells:
            return False
        return max_first_cell is None or (w[1] - w[0]) <= max_first_cell

    if uniform:
        grading = _UNIFORM_GRADING[grading_type]
    elif grading is None:
        lo, hi = _GRADING_SEARCH_RANGE[grading_type]
        if resolved(lo):
            grading = lo
        elif not resolved(hi):
            grading = hi
        else:
            # Both constraints tighten with the grading strength
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if resolved(mid):
                    hi = mid
                else:
                    lo = mid
            grading = hi

    weights = weights_for(grading)
    n_lower, n_upper = count_layer_cells(weights, delta)
    stretch = _max_stretch_ratio(weights)
    first_cell = weights[1] - weights[0]

    _LOG.info(
        "Wall grading %s=%.4g with %d cells: layer thickness %.4g H, "
        "%d/%d cells in the layer at the lower/upper wall, first cell %.3g H, "
        "max stretch %.3f.",
        grading_type,
        grading,
        res,
        delta,
        n_lower,
        n_upper,
        first_cell,
        stretch,
    )

    if min(n_lower, n_upper) < min_layer_cells:
        _LOG.warning(
            "The MHD wall layer (%.4g H) holds only %d/%d cells, fewer than "
            "min_layer_cells=%d. Increase the wall-normal resolution.",
            delta,
            n_lower,
            n_upper,
            min_layer_cells,
        )
    if max_first_cell is not None and first_cell > max_first_cell:
        _LOG.warning(
            "The wall-adjacent cell is %.3g H, larger than the requested "
            "%.3g H. Increase the wall-normal resolution.",
            first_cell,
            max_first_cell,
        )
    if stretch > _MAX_STRETCH_RATIO:
        _LOG.warning(
            "The wall grading stretches neighbouring cells by up to %.3f "
            "(> %.2f). Increase the wall-normal resolution for a smoother grid.",
            stretch,
            _MAX_STRETCH_RATIO,
        )

    return weights, grading
