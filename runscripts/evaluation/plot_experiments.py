"""Plot the training curves, test results and inverse-design runs for the paper.

Usage:
    python runscripts/evaluation/plot_experiments.py
"""

import json
import logging
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from phipict import _C
from phipict.grid.helpers import get_cell_centers
from phipict.io.domain_io import load_domain

import fluidgym
from fluidgym import DEFAULT_PALETTE
from fluidgym.envs.util.visualization import resample

local_data_path = os.environ.get("FLUIDGYM_LOCAL_DATA_PATH", "./local_data")
fluidgym.config.update("local_data_path", local_data_path)

EP_LENGTH = 150
BPTT_WINDOW = 100
# Forward-equivalent steps charged per env step of BPTT and SHAC
GRADIENT_STEP_COST = 3.0
# SHAC logs agent steps, one per agent and env step
N_AGENTS = 10
MIN_RUN_FRACTION = 0.5
TRAINING_WALLTIME_GRID_STEP = 600.0
# x column -> (label, divisor, resampling grid step in the column's unit)
TRAINING_X_AXES = {
    "wall_time": ("Wall-clock Time [h]", 3600.0, TRAINING_WALLTIME_GRID_STEP),
    "sim_steps": ("Forward-equivalent Steps", 1.0, 100.0),
}
SEEDS = range(5)
DOMAIN_IDXS = range(10)
DIFFICULTIES = ["easy", "medium"]
DIFFICULTY_LABELS = {
    "easy": r"$\mathrm{Ra} = 20\mathrm{Q}$",
    "medium": r"$\mathrm{Ra} = 50\mathrm{Q}$",
}
# LaTeX table rows, macros from the paper preamble
DIFFICULTY_TABLE_LABELS = {
    "easy": r"$20\chandr$",
    "medium": r"$50\chandr$",
}
ALGORITHMS = ["SAC", "BPTT", "SHAC"]
# Output folder of each algorithm, plots keep the names above
ALGORITHM_DIRS = {
    "SAC": "SAC",
    "BPTT": "BPTT10_mlp",
    "SHAC": "SHAC",
}
OPTIMIZERS = ["CMA-ES", "L-BFGS-B"]
WALLTIME_LIMIT = 1800.0
WALLTIME_GRID_STEP = 10.0
OPTIM_STOP_OBJECTIVE = 1e-5
OPTIM_FACES = ["-x", "+x", "-y", "+y", "-z", "+z"]
# Sequential for the fields, diverging for the signed errors centered on zero
OPTIM_TRUTH_CMAP = "viridis"
OPTIM_ERROR_CMAP = "RdBu"
OPTIM_METRICS = {
    "best_objective": "Objective",
    "best_log10_cw_rmse": "Wall Patch RMSE",
}
OPTIMIZER_NAMES = {
    "cma_es": "CMA-ES",
    "lbfgs": "L-BFGS-B"
}
ALGORITHM_COLOR_IDXS = {
    "Baseflow": 0,
    "SAC": 1,
    "BPTT": 2,
    "SHAC": 3,
}
OPTIMIZER_COLOR_IDXS = {
    "CMA-ES": 1,
    "L-BFGS-B": 2,
}

METRICS = {
    "action": "Action",
    "reward": "Mean Reward",
    "nu": r"$\mathrm{Nu}_k$",
    "probe_T": r"$T_\mathrm{Probe}$",
    "heat_flux_mean": r"$\overline q_\mathrm{out}$",
    "heat_flux_var": r"$\mathrm{Var}(q_\mathrm{out})$",
}

RENDER_KEYS_2D = ["temperature", "velocity", "vorticity"]
RENDER_KEYS_3D = ["u_z"]

TRAINING_DIR = Path("./output/training/marl")
INVERSE_DESIGN_DIR = Path("./output/optimize_cw_inverse")
PLOTS_DIR = Path("./paper/plots")
TRAINING_PLOTS_DIR = Path("./paper/plots/training")
OPTIM_PLOTS_DIR = Path("./paper/plots/optim")
TEST_PLOTS_DIR = Path("./paper/plots/test")
TABLES_PATH = Path("./paper/tables")
TEST_TABLES_DIR = Path("./paper/tables/test")

TRAINING_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OPTIM_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TEST_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_PATH.mkdir(parents=True, exist_ok=True)
TEST_TABLES_DIR.mkdir(parents=True, exist_ok=True)

COL_REPLACEMENT_MAP = {
    # TODO add algorithm-specific values
    "training/mean_reward": "training/mean_reward",
}

FULL_WIDTH = 6.0
HALF_WIDTH = 0.49 * FULL_WIDTH
FIGHEIGHT = 1.5
PANE_COLOR = "#F2F4F7"

sns.set_style(
    style="darkgrid",
    rc={
        "axes.facecolor": PANE_COLOR,
    },
)
sns.set_palette(DEFAULT_PALETTE)

mpl.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": (
            r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{times}"
        ),

        "pdf.fonttype": 42,  # TrueType in PDF
        "ps.fonttype": 42,  # TrueType in PS/EPS

        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],

        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 9,
    }
)


logger = logging.getLogger("plot_train_test")

# ---------------------------------------------------------------------------------
# RL DATA LOADING
# ---------------------------------------------------------------------------------

def iqm(data: pd.Series) -> float:
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqm_data = data[(data >= q1) & (data <= q3)]

    return iqm_data.mean()

def parse_training_csv(df: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    # 1) Remove train/ prefix
    for col in df.columns:
        df = df.rename(columns={col: col.replace("train/", "")})

    # 2) add algo column
    df["algorithm"] = algorithm

    if algorithm == "SAC":
        # Seconds since the start, `time` is the mean step time of the row's steps
        df["wall_time"] = (df["time"] * df["step"].diff().fillna(df["step"])).cumsum()
        # One forward simulation step per env step, the SB3 updates are not counted
        df["sim_steps"] = df["step"].astype(float)
        df = df.drop(
            columns=["time"]
        )
        for col in df.columns:
            if col.startswith("mean_"):
                df[col] = (
                    df[col].rolling(window=3, min_periods=1).mean()
                )
        return df
    elif "BPTT" in algorithm or algorithm == "SHAC":
        if algorithm == "SHAC":
            # Env steps, as BPTT logs them
            df["step"] = df["step"] // N_AGENTS
            # Seconds since the start, `time` is the mean step time of the row's steps
            # and the first row after each rollout also holds its backward pass
            df["wall_time"] = (
                df["time"] * df["step"].diff().fillna(df["step"])
            ).cumsum()
            df = df.drop(columns=["time"])
        else:
            # Seconds since the start, forward and backward of every update
            df["wall_time"] = df["time_window"].cumsum()

        # Constant training cost per env step, the same for BPTT and SHAC
        df["sim_steps"] = df["step"].astype(float) * GRADIENT_STEP_COST

        # Only complete windows, the last one may still be running
        # df = df[df["step"] <= (df["step"].max() // BPTT_WINDOW) * BPTT_WINDOW]
        window_id = (df["step"] - 1) // BPTT_WINDOW

        agg_dict = dict.fromkeys(df.select_dtypes(include="number").columns, "mean")
        for col in df.select_dtypes(exclude="number").columns:
            agg_dict[col] = "first"
        agg_dict["wall_time"] = "last"
        agg_dict["sim_steps"] = "last"

        df_aggregated = df.groupby(window_id).agg(agg_dict)

        # The step column holds the window endpoints (100, 200, 300, ...)
        df_aggregated["step"] = (df_aggregated.index + 1) * BPTT_WINDOW
        df_aggregated = df_aggregated.reset_index(drop=True)

        # The same smoothing as SAC, whose rows also span 100 steps
        for col in df_aggregated.columns:
            if col.startswith("mean_"):
                df_aggregated[col] = (
                    df_aggregated[col].rolling(window=3, min_periods=1).mean()
                )

        return df_aggregated
    else:
        raise ValueError("Invalid algorithm:", algorithm)


def load_training_runs(
    env_id: str,
) -> pd.DataFrame:
    all_runs: list[pd.DataFrame] = []

    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            training_path = (
                TRAINING_DIR
                / f"{env_id}/{ALGORITHM_DIRS[algorithm]}/{seed}/training_log.csv"
            )

            if not training_path.exists():
                logger.info(f"No training_log.csv for {algorithm} (seed {seed})")
                continue

            run_df = pd.read_csv(training_path)
            run_df = parse_training_csv(run_df, algorithm)

            run_df["seed"] = seed
            run_df["env_id"] = env_id

            all_runs += [run_df]

    if len(all_runs) == 0:
        return pd.DataFrame()

    runs = pd.concat(all_runs, ignore_index=True)

    # Runs still far behind are left out, they would cut the whole curve short
    # last_step = runs.groupby(["algorithm", "seed"])["step"].transform("max")
    # longest = last_step.groupby(runs["algorithm"]).transform("max")
    # runs = runs[last_step >= MIN_RUN_FRACTION * longest]

    # Each algorithm ends where its first seed ends, so all points use the same seeds
    # last_step = runs.groupby(["algorithm", "seed"])["step"].transform("max")
    # return runs[runs["step"] <= last_step.groupby(runs["algorithm"]).transform("min")]
    return runs

def load_test_sequence(
    env_id: str,
    algorithm: str,
    seed: int,
    domain_idx: int,
    test_env_id: str | None = None
) -> pd.DataFrame:
    if test_env_id is None:
        test_path = (
            TRAINING_DIR
            / f"{env_id}/{ALGORITHM_DIRS[algorithm]}/{seed}/test/{domain_idx}/"
            "test_results.csv"
        )
    else:
        test_path = (
            TRAINING_DIR
            / f"{env_id}/{ALGORITHM_DIRS[algorithm]}/{seed}/transfer/{test_env_id}/"
            f"{domain_idx}/"
            "test_results.csv"
        )

    if not test_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(test_path)
    df = df.rename(columns={"action_0": "action"})

    return df

def load_test_sequences(
    env_id: str,
    test_env_id: str | None = None
) -> pd.DataFrame:
    all_runs: list[pd.DataFrame] = []

    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            for domain_idx in DOMAIN_IDXS:
                test_df = load_test_sequence(
                    env_id=env_id,
                    algorithm=algorithm,
                    seed=seed,
                    domain_idx=domain_idx,
                    test_env_id=test_env_id
                )

                if test_df.empty:
                    logger.info(
                        f"No test_results.csv for {algorithm} "
                        f"(seed {seed}, idx {domain_idx})")
                    continue

                test_df["env_id"] = env_id
                test_df["algorithm"] = algorithm
                test_df["seed"] = seed
                test_df["domain_idx"] = domain_idx
                test_df["test_env_id"] = env_id if test_env_id is None else test_env_id

                all_runs += [test_df]

    if len(all_runs) == 0:
        return pd.DataFrame()

    return pd.concat(all_runs, ignore_index=True)

def load_test_results(env_name: str, y_metric: str = "reward") -> pd.DataFrame:
    env_ids = [f"{env_name}-{difficulty}-v0" for difficulty in DIFFICULTIES]
    test_runs = pd.concat(
        [load_test_sequences(env_id=env_id) for env_id in env_ids],
        ignore_index=True,
    )

    if test_runs.empty:
        return test_runs

    # One value per test episode, the steps of an episode are not independent
    return test_runs.groupby(
        ["env_id", "algorithm", "seed", "domain_idx"], as_index=False
    )[y_metric].mean()


# ---------------------------------------------------------------------------------
# OPTIM DATA LOADING
# ---------------------------------------------------------------------------------

def load_optim_results() -> pd.DataFrame:
    all_runs: list[pd.DataFrame] = []
    walltime_grid = pd.DataFrame(
        {
            "wall_time": np.arange(
                WALLTIME_GRID_STEP,
                WALLTIME_LIMIT + WALLTIME_GRID_STEP,
                WALLTIME_GRID_STEP,
            )
        }
    )

    for optim in OPTIMIZER_NAMES:
        for seed in SEEDS:
            optim_path = INVERSE_DESIGN_DIR / optim / str(seed) / "optimization_log.csv"

            if not optim_path.exists():
                logger.info(f"No optimization_log.csv for {optim} (seed {seed})")
                continue

            run_df = pd.read_csv(optim_path)

            # The values of the config with the lowest objective so far
            run_df["best_objective"] = run_df["objective"].cummin()
            is_best = run_df["objective"] == run_df["best_objective"]
            run_df["best_log10_cw_rmse"] = (
                run_df["log10_cw_rmse"].where(is_best).ffill()
            )

            # A shared time grid gives every seed a value at each time. A run that
            # stopped early keeps its final result, for the statistics over the seeds
            final_time = run_df["wall_time"].iloc[-1]
            run_df = pd.merge_asof(walltime_grid, run_df, on="wall_time")
            run_df["running"] = run_df["wall_time"] < final_time + WALLTIME_GRID_STEP

            run_df["seed"] = seed
            run_df["optimizer"] = OPTIMIZER_NAMES[optim]

            all_runs += [run_df]

    if len(all_runs) == 0:
        return pd.DataFrame()

    return pd.concat(all_runs, ignore_index=True)

def load_optim_domain(optimizer: str, seed: int) -> _C.Domain | None:
    """Load the final domain of a run's best evaluation, ``None`` if there is none."""
    domain_path = INVERSE_DESIGN_DIR / optimizer / str(seed) / "best_domain.json"

    if not domain_path.exists():
        logger.info(f"No domain found for {optimizer} (seed {seed})")
        return None

    return load_domain(domain_path)

def load_optim_target_domain(seed: int) -> _C.Domain | None:
    """Load the final domain of the hidden map, which all optimizers of a seed share."""
    for optim in OPTIMIZER_NAMES:
        domain_path = INVERSE_DESIGN_DIR / optim / str(seed) / "target_domain.json"

        if domain_path.exists():
            return load_domain(domain_path)

    logger.info(f"No target domain found (seed {seed})")
    return None

def resample_optim_velocity(domain: _C.Domain, resolution: int = 128) -> np.ndarray:
    """Resample the mid-x slice of the velocity along the magnetic field, u_x.

    Returns an array of shape [resolution, resolution] on a uniform grid, with y along
    the rows and z along the columns, so the hot wall is on the left with
    ``origin="lower"``.
    """
    block = domain.getBlocks()[0]
    centers = get_cell_centers(block.vertexCoordinates)  # [3, nz, ny, nx]
    z_centers = centers[2, :, 0, 0].detach().cpu().numpy()  # [nz]
    y_centers = centers[1, 0, :, 0].detach().cpu().numpy()  # [ny]

    u_x = block.velocity[0, 0].detach().cpu().numpy()  # [nz, ny, nx]
    return resample(
        x_centers=z_centers,
        y_centers=y_centers,
        data=u_x[:, :, u_x.shape[-1] // 2],
        render_shape=(resolution, resolution),
    ).T  # [res_y, res_z]


def load_optim_cw_patches(seed: int) -> dict[str, dict[str, np.ndarray]]:
    """Load the hidden and the best log10(Cw) of each wall patch.

    Returns a nested dict with keys: ground_truth, cma_es, lbfgs
    Second level contains keys: -x, +x, -y, +y, -z, +z

    The arrays have shape [K, K], with K = patches_per_wall, and index the wall's two
    in-plane axes in field order: [z, y] for x walls, [z, x] for y walls and [y, x] for
    z walls. Optimizers without results are left out. The dict is empty without any run.
    """

    def to_faces(log10_cw: list[float], K: int) -> dict[str, np.ndarray]:
        walls = np.reshape(log10_cw, (len(OPTIM_FACES), K, K))
        return dict(zip(OPTIM_FACES, walls, strict=True))

    patches: dict[str, dict[str, np.ndarray]] = {}

    for optim in OPTIMIZER_NAMES:
        run_dir = INVERSE_DESIGN_DIR / optim / str(seed)
        info_path = run_dir / "best_domain_info.json"

        if not info_path.exists():
            logger.info(f"No best_domain_info.json for {optim} (seed {seed})")
            continue

        with open(run_dir / "maps.json") as f:
            maps = json.load(f)
        with open(info_path) as f:
            info = json.load(f)

        # All optimizers of a seed share the hidden map
        K = maps["patches_per_wall"]
        if "ground_truth" not in patches:
            patches["ground_truth"] = to_faces(maps["log10_cw_true"], K)
        patches[optim] = to_faces(info["log10_cw"], K)

    return patches


# ---------------------------------------------------------------------------------
# TRAINING PLOTS
# ---------------------------------------------------------------------------------

def resample_runs(
    runs: pd.DataFrame, column: str, x_column: str = "wall_time"
) -> pd.DataFrame:
    """Interpolate a metric of each run onto a shared grid of ``x_column``."""
    step = TRAINING_X_AXES[x_column][2]
    resampled: list[pd.DataFrame] = []

    for algorithm, algorithm_runs in runs.groupby("algorithm"):
        # Only the times all seeds cover, so all points use the same seeds
        # seed_times = algorithm_runs.groupby("seed")["wall_time"]
        # start = np.ceil(seed_times.min().max() / step) * step
        # grid = np.arange(start, seed_times.max().min(), step)

        for seed, run in algorithm_runs.groupby("seed"):
            # Debugging: each seed on its own grid, up to where it currently is
            start = np.ceil(run[x_column].min() / step) * step
            grid = np.arange(start, run[x_column].max(), step)

            resampled += [
                pd.DataFrame(
                    {
                        x_column: grid,
                        column: np.interp(grid, run[x_column], run[column]),
                        "algorithm": algorithm,
                        "seed": seed,
                    }
                )
            ]

    return pd.concat(resampled, ignore_index=True)


def plot_training_runs_combined(
    env_name: str, y_metric: str, x_metric: str = "wall_time"
) -> None:
    x_label, x_divisor, _ = TRAINING_X_AXES[x_metric]

    fig, axs = plt.subplots(
        1, len(DIFFICULTIES), figsize=(FULL_WIDTH, 2.0), sharex=True
    )

    for ax, difficulty in zip(axs, DIFFICULTIES):
        env_id = f"{env_name}-{difficulty}-v0"
        all_runs = load_training_runs(env_id=env_id)

        if all_runs.empty:
            continue

        all_runs = resample_runs(all_runs, f"mean_{y_metric}", x_metric)
        all_runs[x_metric] = all_runs[x_metric] / x_divisor

        sns.lineplot(
            data=all_runs,
            x=x_metric,
            y=f"mean_{y_metric}",
            hue="algorithm",
            hue_order=ALGORITHMS,
            errorbar="sd",
            ax=ax,
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel(METRICS[y_metric] if ax is axs[0] else "")
        ax.legend().remove()
        ax.set_title(DIFFICULTY_LABELS[difficulty])

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.13),
        ncol=3,
        fancybox=False,
        shadow=False,
        frameon=False,
    )
    fig.subplots_adjust(
        top=0.89, bottom=0.32, left=0.1, right=0.96, wspace=0.33
    )

    x_suffix = "" if x_metric == "wall_time" else f"_{x_metric}"
    out_path = TRAINING_PLOTS_DIR / f"{env_name}_{y_metric}_combined{x_suffix}.pdf"

    fig.savefig(out_path, format="pdf")
    plt.close(fig)

# ---------------------------------------------------------------------------------
# TEST PLOTS
# ---------------------------------------------------------------------------------

def plot_test_results(env_name: str, y_metric: str = "reward") -> None:
    env_ids = [f"{env_name}-{difficulty}-v0" for difficulty in DIFFICULTIES]
    test_results = load_test_results(env_name, y_metric)

    if test_results.empty:
        logger.info(f"No test results for {env_name}")
        return

    algorithms =[alg for alg in ALGORITHMS if alg in set(test_results["algorithm"])]
    palette = {
        alg: sns.color_palette()[ALGORITHM_COLOR_IDXS[alg]] for alg in algorithms
    }

    fig, axs = plt.subplots(1, len(env_ids), figsize=(FULL_WIDTH, 2.0))

    for ax, env_id, difficulty in zip(axs, env_ids, DIFFICULTIES, strict=True):
        ax.set_title(DIFFICULTY_LABELS[difficulty])
        env_results = test_results[test_results["env_id"] == env_id]

        if env_results.empty:
            continue

        sns.boxplot(
            data=env_results,
            x="algorithm",
            y=y_metric,
            hue="algorithm",
            order=algorithms,
            hue_order=algorithms,
            palette=palette,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("")

    axs[0].set_ylabel(METRICS[y_metric])

    fig.subplots_adjust(top=0.89, bottom=0.2, left=0.1, right=0.98, wspace=0.33)

    out_path = TEST_PLOTS_DIR / f"{env_name}_test_{y_metric}.pdf"

    fig.savefig(out_path, format="pdf")
    plt.close(fig)

def write_test_table(
    test_results: pd.DataFrame,
    env_name: str,
    metric: str,
    out_path: Path,
    fmt: str = ".3f",
    unit: str = "",
    higher_is_better: bool = True,
) -> None:
    iqms = test_results.groupby(["env_id", "algorithm"])[metric].apply(iqm).unstack()
    iqms = iqms.reindex(
        index=[f"{env_name}-{difficulty}-v0" for difficulty in DIFFICULTIES],
        columns=ALGORITHMS,
    )
    iqms.index = pd.Index(
        [DIFFICULTY_TABLE_LABELS[difficulty] for difficulty in DIFFICULTIES],
        name=r"$\rayleigh$",
    )
    iqms.columns.name = None

    table = iqms.map(
        lambda value: "--" if pd.isna(value) else f"${value:{fmt}}{unit}$"
    )
    valid = iqms.dropna(how="all")
    best = valid.idxmax(axis=1) if higher_is_better else valid.idxmin(axis=1)
    for difficulty, algorithm in best.items():
        table.loc[difficulty, algorithm] = (
            rf"$\mathbf{{{iqms.loc[difficulty, algorithm]:{fmt}}}}{unit}$"
        )

    table.reset_index().to_latex(
        out_path, index=False, escape=False, column_format="l" + "c" * table.shape[1]
    )

def create_test_results_table(env_name: str, y_metric: str = "reward") -> None:
    test_results = load_test_results(env_name, y_metric)

    if test_results.empty:
        logger.info(f"No test results for {env_name}")
        return

    write_test_table(
        test_results,
        env_name,
        y_metric,
        TEST_TABLES_DIR / f"{env_name}_test_{y_metric}.tex",
    )

def create_test_nu_improvement_table(env_name: str) -> None:
    test_results = load_test_results(env_name, "nusselt")

    if test_results.empty:
        logger.info(f"No test results for {env_name}")
        return

    # Reference Nu of the uncontrolled flow from the env's domain statistics
    nu_refs = {
        env_id: fluidgym.make(env_id, load_initial_domain=False).nu_ref
        for env_id in test_results["env_id"].unique()
    }
    test_results["nu_improvement"] = 100.0 * (
        test_results["nusselt"] / test_results["env_id"].map(nu_refs) - 1.0
    )

    write_test_table(
        test_results,
        env_name,
        "nu_improvement",
        TEST_TABLES_DIR / f"{env_name}_test_nu_improvement.tex",
        fmt="+.2f",
        unit=r"\%",
    )

def create_test_power_table(env_name: str) -> None:
    test_results = load_test_results(env_name, "electrical_power")

    if test_results.empty:
        logger.info(f"No test results for {env_name}")
        return

    # Electrical power over the buoyancy power of the uncontrolled flow, the
    # power_ratio of the reward penalty
    power_refs = {
        env_id: fluidgym.make(env_id, load_initial_domain=False).power_ref
        for env_id in test_results["env_id"].unique()
    }
    test_results["power_increase"] = 100.0 * (
        test_results["electrical_power"] / test_results["env_id"].map(power_refs)
    )

    write_test_table(
        test_results,
        env_name,
        "power_increase",
        TEST_TABLES_DIR / f"{env_name}_test_power_increase.tex",
        fmt=".2f",
        unit=r"\%",
        higher_is_better=False,
    )

# ---------------------------------------------------------------------------------
# OPTIM PLOTS
# ---------------------------------------------------------------------------------

def load_optim_progress() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    all_runs = load_optim_results()

    # An optimizer's band ends with its last running seed
    last_running = all_runs[all_runs["running"]].groupby("optimizer")["wall_time"].max()
    band_runs = all_runs[
        all_runs["wall_time"] <= all_runs["optimizer"].map(last_running)
    ]
    band_groups = band_runs.groupby(["optimizer", "wall_time"])[list(OPTIM_METRICS)]
    band_min = band_groups.min()
    band_max = band_groups.max()

    # One dot per optimizer, where its median line first drops below the stop
    # objective. The line ends at the dot
    median_runs = band_groups.median().reset_index()
    stopped = (
        median_runs[median_runs["best_objective"] <= OPTIM_STOP_OBJECTIVE]
        .groupby("optimizer", as_index=False)
        .first()
    )
    line_end = pd.concat(
        [last_running, stopped.set_index("optimizer")["wall_time"]], axis=1
    ).min(axis=1)
    line_runs = band_runs[
        band_runs["wall_time"] <= band_runs["optimizer"].map(line_end)
    ]

    return line_runs, band_min, band_max, stopped

def plot_optimization_run() -> None:
    line_runs, band_min, band_max, stopped = load_optim_progress()

    fig, axs = plt.subplots(ncols=len(OPTIM_METRICS), figsize=(FULL_WIDTH, 1.8))

    for ax, (metric, label) in zip(axs, OPTIM_METRICS.items(), strict=True):
        sns.lineplot(
            data=line_runs,
            x="wall_time",
            y=metric,
            hue="optimizer",
            hue_order=OPTIMIZERS,
            estimator="median",
            errorbar=None,
            ax=ax,
        )

        ax.set_xlabel("Wall-clock Time [s]")
        ax.set_ylabel(label)
        ax.set_yscale("log")

    handles, labels = axs[0].get_legend_handles_labels()
    colors = {
        label: handle.get_color()
        for label, handle in zip(labels, handles, strict=True)
    }

    for ax, metric in zip(axs, OPTIM_METRICS, strict=True):
        # The min-max range over the seeds, as seaborn draws it for ("pi", 100)
        for optimizer in band_min.index.unique("optimizer"):
            ax.fill_between(
                band_min.loc[optimizer].index,
                band_min.loc[optimizer, metric],
                band_max.loc[optimizer, metric],
                color=colors[optimizer],
                alpha=0.2,
                linewidth=0,
            )

        ax.scatter(
            stopped["wall_time"],
            stopped[metric],
            facecolors="white",
            edgecolors=stopped["optimizer"].map(colors).tolist(),
            linewidths=1.2,
            s=16,
            zorder=3,
        )
        ax.legend().remove()

    stop_line = axs[0].axhline(
        OPTIM_STOP_OBJECTIVE, color="gray", linestyle="--", linewidth=1, zorder=1
    )

    fig.legend(
        [*handles, stop_line],
        [*labels, "Stop Objective"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.13),
        ncol=3,
        fancybox=False,
        shadow=False,
        frameon=False,
    )
    fig.subplots_adjust(
        top=0.98, bottom=0.32, left=0.1, right=0.98, wspace=0.3,
    )

    out_path = OPTIM_PLOTS_DIR / f"optimization_run.pdf"

    fig.savefig(out_path, format="pdf")
    plt.close(fig)

def plot_optimization_run_half_width() -> None:
    metric = "best_objective"
    line_runs, band_min, band_max, stopped = load_optim_progress()

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, 1.8))

    sns.lineplot(
        data=line_runs,
        x="wall_time",
        y=metric,
        hue="optimizer",
        hue_order=OPTIMIZERS,
        estimator="median",
        errorbar=None,
        ax=ax,
    )

    ax.set_xlabel("Wall-clock Time [s]")
    ax.set_ylabel(OPTIM_METRICS[metric])
    ax.set_yscale("log")

    handles, labels = ax.get_legend_handles_labels()
    colors = {
        label: handle.get_color()
        for label, handle in zip(labels, handles, strict=True)
    }

    # The min-max range over the seeds, as seaborn draws it for ("pi", 100)
    for optimizer in band_min.index.unique("optimizer"):
        ax.fill_between(
            band_min.loc[optimizer].index,
            band_min.loc[optimizer, metric],
            band_max.loc[optimizer, metric],
            color=colors[optimizer],
            alpha=0.2,
            linewidth=0,
        )

    ax.scatter(
        stopped["wall_time"],
        stopped[metric],
        facecolors="white",
        edgecolors=stopped["optimizer"].map(colors).tolist(),
        linewidths=1.2,
        s=16,
        zorder=3,
    )
    ax.legend().remove()

    stop_line = ax.axhline(
        OPTIM_STOP_OBJECTIVE, color="gray", linestyle="--", linewidth=1, zorder=1
    )

    fig.legend(
        [*handles, stop_line],
        [*labels, "Stop Objective"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.13),
        ncol=3,
        handlelength=1.5,
        columnspacing=1.0,
        fancybox=False,
        shadow=False,
        frameon=False,
    )
    fig.subplots_adjust(top=0.98, bottom=0.32, left=0.2, right=0.97)

    out_path = OPTIM_PLOTS_DIR / "optimization_run_half.pdf"

    fig.savefig(out_path, format="pdf")
    plt.close(fig)

def plot_optim_cw_patches(seed: int = 0) -> None:
    """Plot the hidden log10(Cw) of each wall patch and each optimizer's error."""
    patches = load_optim_cw_patches(seed=seed)

    if len(patches) == 0:
        logger.info(f"No optimization results for seed {seed}")
        return

    ground_truth = patches["ground_truth"]
    errors = {
        optim: {face: patches[optim][face] - ground_truth[face] for face in OPTIM_FACES}
        for optim in OPTIMIZER_NAMES
        if optim in patches
    }

    # Shared between the walls
    truth_min = np.min(list(ground_truth.values()))
    truth_max = np.max(list(ground_truth.values()))
    # Shared between the optimizers and walls, which makes the errors comparable
    error_max = np.abs([list(faces.values()) for faces in errors.values()]).max()

    nrows = len(patches)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=len(OPTIM_FACES),
        figsize=(FULL_WIDTH, 0.8 * nrows),
        squeeze=False,
    )

    def show(
        row: int,
        face_data: dict[str, np.ndarray],
        vmin: float,
        vmax: float,
        cmap: str,
    ) -> mpl.image.AxesImage:
        for ax, face in zip(axs[row], OPTIM_FACES, strict=True):
            img = ax.imshow(
                face_data[face], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        return img

    truth_img = show(0, ground_truth, truth_min, truth_max, OPTIM_TRUTH_CMAP)
    axs[0, 0].set_ylabel("Ground Truth")
    for row, (optim, faces) in enumerate(errors.items(), start=1):
        error_img = show(row, faces, -error_max, error_max, OPTIM_ERROR_CMAP)
        axs[row, 0].set_ylabel(OPTIMIZER_NAMES[optim])

    for ax, face in zip(axs[0], OPTIM_FACES, strict=True):
        ax.set_title(f"Wall at ${face}$")

    fig.subplots_adjust(
        top=0.9, bottom=0.03, left=0.03, right=1.05, wspace=0.15, hspace=0.08,
    )

    truth_bar = fig.colorbar(truth_img, ax=axs[0].tolist(), shrink=0.9, pad=0.02)
    truth_bar.set_label(r"$\log_{10} C_w$")
    # The error bar spans more rows, so it shrinks further to the same size
    row_height = axs[0, 0].get_position(original=True).height
    error_span = (
        axs[1, 0].get_position(original=True).y1
        - axs[-1, 0].get_position(original=True).y0
    )
    error_bar = fig.colorbar(
        error_img,
        ax=axs[1:].ravel().tolist(),
        shrink=0.9 * row_height / error_span,
        pad=0.02,
    )
    error_bar.set_label("Error")

    out_path = OPTIM_PLOTS_DIR / f"cw_patches_{seed}.pdf"

    fig.savefig(out_path, format="pdf")
    plt.close(fig)

def plot_optim_velocity(seed: int = 0) -> None:
    """Plot the mid-x u_x of the hidden map and the error of each optimizer's best map."""
    target_domain = load_optim_target_domain(seed=seed)

    if target_domain is None:
        return

    ground_truth = resample_optim_velocity(target_domain)
    truth_max = np.abs(ground_truth).max()

    # In percent of the largest |u_x|, since u_x itself crosses zero inside the slice
    errors = {}
    for optim, name in OPTIMIZER_NAMES.items():
        domain = load_optim_domain(optimizer=optim, seed=seed)
        if domain is not None:
            velocity = resample_optim_velocity(domain)
            errors[name] = 100.0 * (velocity - ground_truth) / truth_max

    # Shared between the optimizers, which makes the errors comparable
    error_max = np.abs(list(errors.values())).max()

    # The empty column leaves space for the colorbar of the ground truth
    fig = plt.figure(figsize=(FULL_WIDTH, 2.0))
    grid = fig.add_gridspec(
        ncols=len(errors) + 2,
        width_ratios=[1.0, 0.6, *[1.0] * len(errors)],
        top=0.88,
        bottom=0.1,
        left=0.04,
        right=0.85,
        wspace=0.08,
    )
    axs = [fig.add_subplot(grid[0, 0])]
    axs += [fig.add_subplot(grid[0, col + 2]) for col in range(len(errors))]

    truth_img = axs[0].imshow(
        ground_truth,
        origin="lower",
        cmap=OPTIM_TRUTH_CMAP,
        vmin=ground_truth.min(),
        vmax=ground_truth.max(),
    )
    axs[0].set_title("Ground Truth")
    axs[0].set_ylabel("$y$")

    for ax, (name, error) in zip(axs[1:], errors.items(), strict=True):
        error_img = ax.imshow(
            error,
            origin="lower",
            cmap=OPTIM_ERROR_CMAP,
            vmin=-error_max,
            vmax=error_max,
        )
        ax.set_title(f"{name} Error")

    for ax in axs:
        ax.set_xlabel("$z$")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    # Attached to the panels, so the colorbars have the height of the images
    truth_bar = fig.colorbar(truth_img, cax=axs[0].inset_axes([1.05, 0.0, 0.05, 1.0]))
    truth_bar.set_label("$u_x$")
    error_bar = fig.colorbar(error_img, cax=axs[-1].inset_axes([1.05, 0.0, 0.05, 1.0]))
    error_bar.set_label(r"Relative Error [\%]")

    out_path = OPTIM_PLOTS_DIR / f"velocity_{seed}.pdf"

    fig.savefig(out_path, format="pdf")
    plt.close(fig)

if __name__ == "__main__":
    plot_test_results("RBCMHD3D")
    create_test_results_table("RBCMHD3D")
    create_test_nu_improvement_table("RBCMHD3D")
    create_test_power_table("RBCMHD3D")
    plot_training_runs_combined("RBCMHD3D", "reward", x_metric="sim_steps")

    plot_optimization_run()
    plot_optimization_run_half_width()
    plot_optim_cw_patches(seed=0)
    plot_optim_velocity(seed=0)