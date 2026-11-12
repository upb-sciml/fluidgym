"""Grid and domain setup of the RBC-MHD box."""

from typing import cast

import torch
from phipict import _C, Face, bc
from phipict.grid import shapes


def make_grid(
    Lx: float,
    H: float,
    Lz: float,
    x_weights: list[float],
    y_weights: list[float],
    z_weights: list[float],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Vertex coordinates of the box ``[0, Lx] x [-H/2, H/2] x [0, Lz]``.

    Parameters
    ----------
    Lx: float
        Extent along x, between the Hartmann walls.

    H: float
        Height along y, between the heated and the cooled plate.

    Lz: float
        Extent along z.

    x_weights: list[float]
        Normalised vertex positions along x in ``[0, 1]``, one more than cells.

    y_weights: list[float]
        Normalised vertex positions along y in ``[0, 1]``, one more than cells.

    z_weights: list[float]
        Normalised vertex positions along z in ``[0, 1]``, one more than cells.

    dtype: torch.dtype
        Data type of the coordinates. Defaults to ``torch.float32``.

    Returns
    -------
    torch.Tensor
        Vertex coordinates, ``[1, 3, Z + 1, Y + 1, X + 1]``, on the CPU.
    """
    h = H / 2
    grid = shapes.generate_grid_vertices_2D(
        [len(y_weights), len(x_weights)],
        [(0.0, -h), (Lx, -h), (0.0, h), (Lx, h)],
        None,
        # weights along the x-borders, i.e. along y, and vice versa
        x_weights=y_weights,
        y_weights=x_weights,
        dtype=dtype,
    )
    return shapes.extrude_grid_z(
        grid,
        res_z=len(z_weights) - 1,
        start_z=0.0,
        end_z=Lz,
        weights_z=z_weights,
    )


def make_rbc_mhd_domain(
    Lx: float,
    H: float,
    Lz: float,
    x_weights: list[float],
    y_weights: list[float],
    z_weights: list[float],
    rayleigh_number: float,
    prandtl_number: float,
    T_hot: float,
    T_cold: float,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None = None,
    noise_amplitude: float = 0.01,
) -> _C.Domain:
    """Build the closed RBC-MHD box, heated from below and cooled from above.

    All walls are no-slip. The plates at ``-y`` and ``+y`` are isothermal at
    ``T_hot`` and ``T_cold``, the side walls adiabatic. Every wall keeps the
    default insulating potential condition; the electrodes are applied by the
    environment, so every electrode layout shares the domain. The fluid starts at rest
    with the conduction profile plus noise to trigger the rolls.

    Parameters
    ----------
    Lx, H, Lz: float
        Extents of the box, see :func:`make_grid`.

    x_weights, y_weights, z_weights: list[float]
        Normalised vertex positions per axis, see :func:`make_grid`.

    rayleigh_number: float
        Rayleigh number; with the free-fall scaling the viscosity is
        ``sqrt(Pr / Ra)`` and the thermal diffusivity ``1 / sqrt(Ra Pr)``.

    prandtl_number: float
        Prandtl number.

    T_hot: float
        Temperature of the bottom plate.

    T_cold: float
        Temperature of the top plate.

    dtype: torch.dtype
        Data type of the domain fields.

    device: torch.device
        CUDA device of the domain.

    generator: torch.Generator or None
        Generator of the initial noise, on ``device``. Defaults to None.

    noise_amplitude: float
        Standard deviation of the initial temperature noise, relative to
        ``T_hot - T_cold``. Defaults to 0.01.

    Returns
    -------
    _C.Domain
        The domain, prepared for solving.
    """
    viscosity = torch.tensor([(prandtl_number / rayleigh_number) ** 0.5], dtype=dtype)
    diffusivity = torch.tensor(
        [(rayleigh_number * prandtl_number) ** -0.5], dtype=dtype, device=device
    )

    domain = _C.Domain(
        3,
        viscosity,
        passiveScalarChannels=1,
        name="RBCMHDDomain",
        device=device,
        dtype=dtype,
    )
    domain.setScalarViscosity(diffusivity)

    grid = make_grid(
        Lx=Lx,
        H=H,
        Lz=Lz,
        x_weights=x_weights,
        y_weights=y_weights,
        z_weights=z_weights,
        dtype=dtype,
    )
    block = domain.CreateBlock(
        vertexCoordinates=grid.to(device).contiguous(), name="Block"
    )
    for face_name in ("-x", "+x", "-y", "+y", "-z", "+z"):
        block.CloseBoundary(face_name)

    # Isothermal bottom and top plates, adiabatic side walls
    for face, value in ((Face.Y_MINUS, T_hot), (Face.Y_PLUS, T_cold)):
        cast(_C.FixedBoundary, block.getBoundary(face)).setPassiveScalar(
            torch.tensor([[value]], dtype=dtype, device=device)
        )
    for face in (Face.X_MINUS, Face.X_PLUS, Face.Z_MINUS, Face.Z_PLUS):
        bc.set_bc(block, face, bc.Scalar.Neumann())

    # Conduction profile at the cell centres plus noise
    y_w = torch.tensor(y_weights, dtype=torch.float64)
    y = (H * (0.5 * (y_w[1:] + y_w[:-1]) - 0.5)).to(device=device, dtype=dtype)
    T = T_hot + (T_cold - T_hot) * (y / H + 0.5)
    nx, nz = len(x_weights) - 1, len(z_weights) - 1
    T = T[None, None, None, :, None].expand(1, 1, nz, -1, nx)
    noise = torch.normal(
        mean=0.0,
        std=1.0,
        size=T.shape,
        device=device,
        generator=generator,
    ).to(dtype)
    T = T + noise_amplitude * (T_hot - T_cold) * noise
    block.setPassiveScalar(T.contiguous())

    domain.PrepareSolve()
    return domain
