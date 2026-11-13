"""Plot the potential on a Hartmann wall of the RBC-MHD env as a vector PDF.

Usage:
    python runscripts/evaluation/plot_rbc_mhd_actuation.py
    python runscripts/evaluation/plot_rbc_mhd_actuation.py \
        --policy output/training/marl/RBCMHD3D-easy-v0/BPTT10_mlp/0/bptt_policy_best.pth
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf
from scipy.interpolate import RegularGridInterpolator

import fluidgym
from fluidgym.wrappers import FlattenObservation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

from normalize_obs import NormalizeObservation  # noqa: E402
from test_bptt import load_policy  # noqa: E402

# Importing test_bptt sets its own default, so the path is set again here
fluidgym.config.update("local_data_path", "./local_data")

# The \textwidth of the L4DC template, whose body text is 11pt Computer Modern
FULL_WIDTH = 6.0
HALF_WIDTH = 0.49 * FULL_WIDTH
FIGHEIGHT = 0.85
CMAP = "RdBu_r"
# Resolution of the plotted fields, in nodes per unit length
RENDER_RESOLUTION = 80

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        # cmr10 has no standard minus sign
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)


def resample_wall_field(
    env, field: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a ``[Z, Y]`` wall field from the cell centres to a fine uniform grid.

    PCHIP does not overshoot, so the insulating wall between the electrodes stays
    exactly zero. The field is extrapolated over the half cell next to the walls.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The nodes along z and y, and the field on them, ``[Y, Z]``.
    """
    interpolator = RegularGridInterpolator(
        (
            env._cell_centers(env._z_vertices).numpy(),
            env._cell_centers(env._y_vertices).numpy(),
        ),
        field.astype(np.float64),
        method="pchip",
        bounds_error=False,
        fill_value=None,
    )
    z = np.linspace(
        float(env._z_vertices[0]),
        float(env._z_vertices[-1]),
        round(env._Lz * RENDER_RESOLUTION) + 1,
    )
    y = np.linspace(
        float(env._y_vertices[0]),
        float(env._y_vertices[-1]),
        round(env._H * RENDER_RESOLUTION) + 1,
    )
    return z, y, interpolator(tuple(np.meshgrid(z, y, indexing="ij"))).T


def add_electrode_outlines(ax: plt.Axes, env) -> None:
    """Outline the footprint of every electrode, including those at zero current."""
    pitch_z = env._Lz / env._n_stations
    pitch_y = env._H / env._n_electrodes
    fill = env._electrode_fill
    margin = 0.5 * (1.0 - fill)
    y_start = float(env._y_vertices[0])

    for iz in range(env._n_stations):
        for iy in range(env._n_electrodes):
            corner = ((iz + margin) * pitch_z, y_start + (iy + margin) * pitch_y)
            ax.add_patch(
                Rectangle(
                    corner,
                    fill * pitch_z,
                    fill * pitch_y,
                    fill=False,
                    edgecolor="0.45",
                    linewidth=0.6,
                )
            )


def plot_wall_field(ax: plt.Axes, env, field: np.ndarray, label: str) -> None:
    """Draw a ``[Z, Y]`` wall field, smoothly shaded and centred on zero."""
    z, y, resampled = resample_wall_field(env, field)
    v_max = max(float(np.abs(resampled).max()), 1e-12)
    # Gouraud shading is stored as vector gradients and has no seams between cells
    mesh = ax.pcolormesh(
        z, y, resampled, cmap=CMAP, vmin=-v_max, vmax=v_max, shading="gouraud"
    )
    add_electrode_outlines(ax, env)

    ax.set_aspect("equal")
    ax.set_ylabel("y")

    cbar = ax.figure.colorbar(mesh, ax=ax, label=label, fraction=0.025, pad=0.02)
    # Matplotlib rasterizes colorbars with many levels by default
    cbar.solids.set_rasterized(False)
    cbar.solids.set_edgecolor("face")


def run_random_step(args: argparse.Namespace) -> Any:
    """Return the env after one step under a random action."""
    env = fluidgym.make(args.env_id)
    env.test()
    env.reset(seed=args.seed, domain_idx=args.domain_idx)
    env.step(env.sample_action())
    return env


@torch.no_grad()
def run_policy_episode(args: argparse.Namespace) -> Any:
    """Return the env after one test episode with the policy of ``args.policy``."""
    run_cfg = OmegaConf.load(args.policy.parent / "config.yaml")
    env = FlattenObservation(
        fluidgym.make(args.env_id, use_marl=run_cfg.rl_mode == "marl")
    )

    # A policy only works on observations normalized the way it was trained on
    norm_path = args.policy.with_name(f"{args.policy.stem}_obs_norm.pth")
    if run_cfg.get("normalize_obs", False):
        env = NormalizeObservation(env)
        env.load_state_dict(torch.load(norm_path, map_location="cpu"))

    policy = load_policy(
        OmegaConf.create(
            {"policy_type": run_cfg.policy_type, "policy_checkpoint": str(args.policy)}
        ),
        env,
        env.cuda_device,
    ).eval()

    env.test()
    obs, _ = env.reset(seed=args.seed, domain_idx=args.domain_idx)
    n_steps = args.max_steps or env.episode_length
    for step in range(n_steps):
        obs, reward, terminated, truncated, info = env.step(policy(obs.float()))
        print(
            f"Step {step + 1}/{n_steps}: reward = {reward.mean().item():.4f}, "
            f"Nu = {info['nusselt'].mean().item():.4f}",
            flush=True,
        )
        if terminated or truncated:
            break
    return env.unwrapped


def plot_wall_potential(env: Any, output: Path) -> None:
    """Save the potential on the ``x = 0`` Hartmann wall to ``output``."""
    # phi_w = phi_P + h * I / A, as in the electrical power of the env
    current_density = env._last_cell_currents[0] / env._face_area
    phi_cell = env._block.epot[0, 0, :, :, 0]
    phi_wall = phi_cell + env._h_minus * current_density

    # No tight bounding box: the PDF has to stay exactly HALF_WIDTH wide, so that
    # LaTeX includes it unscaled and the font sizes hold
    fig, ax = plt.subplots(figsize=(HALF_WIDTH, FIGHEIGHT), constrained_layout=True)
    plot_wall_field(ax, env, phi_wall.detach().cpu().numpy(), r"$\phi_w$")
    ax.set_xlabel("z")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    plt.close(fig)
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="RBCMHD3D-easy-v0")
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--domain-idx", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop the policy episode early, defaults to the full episode.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("./paper/plots/training/rbc_mhd_actuation.pdf")
    )
    args = parser.parse_args()

    env = run_policy_episode(args) if args.policy else run_random_step(args)
    plot_wall_potential(env, args.output)


if __name__ == "__main__":
    main()
