"""Env steps per second of PICT vs. phiPICT: each improvement ablated, plus batching.

Usage:
    python runscripts/benchmark/benchmark.py env=cylinder_jet_2d_easy
    python runscripts/benchmark/benchmark.py -m "env=glob(*)"
    python runscripts/benchmark/benchmark.py aggregate_only=true
"""

import logging
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import fluidgym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from ablate_solver_improvements import (
    CONFIG_DIR,
    FORCED_ENV_KWARGS,
    apply_global_settings,
    apply_sim_settings,
    elapsed_since,
    free_memory,
    make_actions,
)
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("benchmark")

RESULTS_FILE = "benchmark.csv"
PLOT_FILE = "benchmark.pdf"
BATCHED = "phipict"  # mode of the batch size rows

FULL_WIDTH = 6.0
FIGHEIGHT = 1.5
PANE_COLOR = "#F2F4F7"

sns.set_style(style="darkgrid", rc={"axes.facecolor": PANE_COLOR})

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": (
            r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{times}"
        ),
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "figure.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)


def load_variant(name: str) -> DictConfig:
    """A variant from ``configs/solver_ablation``."""
    variant = OmegaConf.load(CONFIG_DIR / "solver_ablation" / f"{name}.yaml")
    assert isinstance(variant, DictConfig)
    return variant


def ablation_label(ablation: DictConfig) -> str:
    """The plot label of an ablation."""
    return rf"$\phi$-PICT {ablation.label}"


def make_variants(cfg: DictConfig) -> tuple[list[DictConfig], DictConfig]:
    """PICT and the leave-one-out ablations, and the full phiPICT."""
    phipict = load_variant("phipict")
    variants = [load_variant("pict")]
    for name, ablation in cfg.ablations.items():
        label = ablation_label(ablation)
        variants.append(
            OmegaConf.merge(phipict, ablation.switches, {"name": name, "label": label})
        )
    return variants, phipict


def make_env(cfg: DictConfig, variant: DictConfig, n_envs: int | None) -> Any:
    """Create and reset the env with the variant's switches applied."""
    apply_global_settings(variant)
    env = fluidgym.make(
        cfg.env_id, **FORCED_ENV_KWARGS, differentiable=False, n_envs=n_envs
    )
    env.seed(cfg.seed)
    env.reset()
    apply_sim_settings(cfg, env._sim, variant)
    return env


def steps_per_second(
    cfg: DictConfig, variant: DictConfig, n_envs: int | None
) -> tuple[float, float]:
    """Mean and std of env steps per second (summed over the batch)."""
    env = make_env(cfg, variant, n_envs)
    actions = make_actions(cfg, env, cfg.warmup_steps + cfg.repeats)
    rates = []
    for i, action in enumerate(actions):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            env.step(action)
        if i >= cfg.warmup_steps:
            rates.append(env.n_envs / elapsed_since(t0))
    return float(np.mean(rates)), float(np.std(rates, ddof=1))


def peak_memory_pct() -> float:
    """Peak GPU memory held by PyTorch since the last reset, in % of the device.

    Reserved rather than allocated memory: what the caching allocator holds is
    what other allocations cannot use.
    """
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    return 100.0 * torch.cuda.max_memory_reserved() / total


def measure(
    cfg: DictConfig, variant: DictConfig, mode: str, n_envs: int | None
) -> dict[str, Any]:
    """One CSV row; NaN steps/s on OOM."""
    free_memory()  # also resets the peak memory stats
    mean, std = float("nan"), float("nan")
    try:
        mean, std = steps_per_second(cfg, variant, n_envs)
    except torch.cuda.OutOfMemoryError:
        logger.warning("[%s] OOM with n_envs=%s.", mode, n_envs)
    memory_pct = peak_memory_pct()
    free_memory()
    logger.info(
        "[%s] n_envs=%s: %.2f +- %.2f steps/s, peak memory %.1f%%",
        mode,
        n_envs,
        mean,
        std,
        memory_pct,
    )
    return {
        "mode": mode,
        "label": str(variant.label),
        "n_envs": n_envs or 1,
        "steps_per_s": mean,
        "steps_per_s_std": std,
        "peak_memory_pct": memory_pct,
    }


def run(cfg: DictConfig) -> None:
    """Benchmark all modes and batch sizes of one env."""
    variants, phipict = make_variants(cfg)
    rows: list[dict[str, Any]] = []

    def record(row: dict[str, Any]) -> dict[str, Any]:
        rows.append(row)
        pd.DataFrame(rows).to_csv(RESULTS_FILE, index=False)
        return row

    pict, ablations = variants[0], variants[1:]
    record(measure(cfg, pict, str(pict.name), None))
    unbatched = record(
        measure(
            cfg,
            OmegaConf.merge(phipict, {"label": r"$\phi$-PICT w/o batching"}),
            "no_batching",
            None,
        )
    )

    # Stop at OOM, at a size over the memory limit (larger ones only need more),
    # or after `patience` sizes in a row without a new best
    best, best_n_envs, misses = unbatched["steps_per_s"], None, 0
    for n_envs in cfg.batch_sizes:
        row = record(measure(cfg, phipict, BATCHED, n_envs))
        rate = row["steps_per_s"]
        if np.isnan(rate):
            break
        if row["peak_memory_pct"] > cfg.max_memory_pct:
            logger.info(
                "n_envs=%d peaks at %.1f%% > %s%% of the GPU memory; stopping.",
                n_envs,
                row["peak_memory_pct"],
                cfg.max_memory_pct,
            )
            break
        if rate > best:
            best, best_n_envs, misses = rate, n_envs, 0
        else:
            misses += 1
            if misses >= cfg.patience:
                break

    # The ablations at the best batch size, so each one removes a single feature
    logger.info("Ablations with n_envs=%s.", best_n_envs)
    for variant in ablations:
        record(measure(cfg, variant, str(variant.name), best_n_envs))

    plot_env(pd.DataFrame(rows), cfg.env_id, cfg.max_memory_pct)


def summarize(df: pd.DataFrame, max_memory_pct: float) -> pd.DataFrame:
    """One row per mode, phiPICT at its best batch size, with speedup over PICT.

    The best batch size is the fastest one within ``max_memory_pct``, as in
    :func:`run`; rows from before the memory was recorded all count as within.
    """
    batched = df[df["mode"] == BATCHED].dropna(subset=["steps_per_s"])
    if "peak_memory_pct" in batched:
        batched = batched[~(batched["peak_memory_pct"] > max_memory_pct)]
    summary = df[df["mode"] != BATCHED]
    if not batched.empty:
        summary = pd.concat([summary, batched.loc[[batched["steps_per_s"].idxmax()]]])
    pict = summary.loc[summary["mode"] == "pict", "steps_per_s"].iloc[0]
    return summary.assign(speedup=summary["steps_per_s"] / pict).reset_index(drop=True)


def sort_modes(speedups: pd.Series) -> list[str]:
    """Modes by ascending speedup (largest impact first), PICT first, phiPICT last.

    ``speedups`` is indexed by mode; NaNs (OOM) go to the end of the ablations.
    """
    ablations = speedups.drop(["pict", BATCHED], errors="ignore")
    ordered = list(ablations.sort_values(na_position="last").index)
    return (
        [m for m in ["pict"] if m in speedups]
        + ordered
        + [m for m in [BATCHED] if m in speedups]
    )


def mode_colors(modes: Sequence[str]) -> dict[str, Any]:
    """A fixed color per mode, independent of the sorting."""
    return dict(zip(modes, sns.color_palette(n_colors=len(modes)), strict=True))


def annotate_speedups(
    ax: plt.Axes, speedups: Sequence[float], heights: Sequence[float]
) -> None:
    """Write the speedups above the bars."""
    for x, (s, h) in enumerate(zip(speedups, heights, strict=True)):
        if np.isfinite(s):
            ax.annotate(
                rf"$\times{s:.1f}$",
                (x, h),
                textcoords="offset points",
                xytext=(0, 2),
                ha="center",
                va="bottom",
                fontsize=7,
            )


def plot_env(df: pd.DataFrame, env_id: str, max_memory_pct: float) -> None:
    """Steps/s per mode (left); steps/s and peak memory over B (right)."""
    summary = summarize(df, max_memory_pct)
    colors = mode_colors(list(summary["mode"]))
    order = sort_modes(summary.set_index("mode")["speedup"])
    summary = summary.set_index("mode").loc[order].reset_index()
    fig, (ax, ax_b) = plt.subplots(
        1, 2, figsize=(9, 3.5), gridspec_kw={"width_ratios": [2, 1]}
    )

    x = np.arange(len(summary))
    ax.bar(
        x,
        summary["steps_per_s"],
        0.7,
        yerr=summary["steps_per_s_std"],
        color=[colors[m] for m in summary["mode"]],
        capsize=2,
        zorder=2,
    )
    annotate_speedups(
        ax, summary["speedup"], summary["steps_per_s"] + summary["steps_per_s_std"]
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["label"], rotation=30, ha="right", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("Env steps per second")
    batched = summary.loc[summary["mode"] == BATCHED, "n_envs"]
    ax.set_title(f"{env_id} ($B={batched.iloc[0]}$)" if len(batched) else env_id)

    sweep = df[df["mode"].isin(["no_batching", BATCHED])]
    ax_b.errorbar(
        sweep["n_envs"], sweep["steps_per_s"], yerr=sweep["steps_per_s_std"], marker="o"
    )
    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(sweep["n_envs"], [str(b) for b in sweep["n_envs"]])
    ax_b.minorticks_off()
    ax_b.set_xlabel("Batch size $B$")
    ax_b.set_ylabel("Env steps per second")
    if "peak_memory_pct" in sweep:
        ax_m = ax_b.twinx()
        ax_m.plot(sweep["n_envs"], sweep["peak_memory_pct"], "s--", color="gray", ms=3)
        ax_m.axhline(max_memory_pct, color="gray", linewidth=0.8, linestyle=":")
        ax_m.set_ylim(0, 100)
        ax_m.set_ylabel("Peak GPU memory [%]", color="gray")

    fig.tight_layout()
    fig.savefig(PLOT_FILE, format="pdf")
    plt.close(fig)


def env_label(env_id: str) -> str:
    """The env id without the version suffix."""
    return re.sub(r"-v\d+$", "", env_id)


def geomean(values: pd.Series) -> float:
    """Geometric mean, ignoring NaNs."""
    values = values.dropna()
    return float(np.exp(np.log(values).mean())) if len(values) else float("nan")


def aggregate(root: Path, max_memory_pct: float, ablations: DictConfig) -> None:
    """Speedup over PICT per env and mode, plus the mean over envs.

    Only the configured modes count, with their configured labels, so results
    from before a change of ``ablations`` can be reused. Envs missing one of them
    (an unfinished run) are skipped, so that every mean covers the same envs.
    """
    configured = {name: ablation_label(a) for name, a in ablations.items()}
    required = {"pict", "no_batching", BATCHED, *configured}
    summaries = []
    for path in sorted(root.glob(f"*/{RESULTS_FILE}")):
        results = pd.read_csv(path)
        missing = required - set(results["mode"])
        if missing:
            logger.warning(
                "Skipping %s: no results for %s.", path.parent.name, sorted(missing)
            )
            continue
        summaries.append(
            summarize(results, max_memory_pct).assign(env_id=path.parent.name)
        )
    if not summaries:
        return
    df = pd.concat(summaries)
    df = df[df["mode"].isin(required)]
    df = df.assign(label=df["mode"].map(configured).fillna(df["label"]))
    modes = list(dict.fromkeys(df["mode"]))
    labels = dict(zip(df["mode"], df["label"], strict=True))
    colors = mode_colors(modes)

    table = df.pivot(index="env_id", columns="mode", values="speedup")[modes]
    modes = sort_modes(table.apply(geomean))
    table = table[modes]
    best_b = df[df["mode"] == BATCHED].set_index("env_id")["n_envs"]
    table["best_batch_size"] = best_b
    mean = table[modes].apply(geomean)
    table.loc["mean"] = mean
    table.to_csv(root / "speedup.csv")
    write_table(table, modes, labels, root / "speedup.tex")
    plot_overall(df, modes, labels, colors, mean, root / "speedup.pdf")
    logger.info("Speedup over PICT (geometric mean over envs):\n%s", mean)


def write_table(
    table: pd.DataFrame, modes: list[str], labels: dict[str, str], out_path: Path
) -> None:
    """Booktabs table: one row per env, one column per mode."""
    cols = modes[1:]  # PICT is 1 by definition
    lines = [
        r"\begin{tabular}{l" + "c" * (len(cols) + 1) + "}",
        r"\toprule",
        " & ".join(["Environment", *(labels[m] for m in cols), "$B$"]) + r" \\",
        r"\midrule",
    ]

    def fmt(v: float) -> str:
        return rf"$\times{v:.2f}$" if np.isfinite(v) else "--"

    for env_id, row in table.iterrows():
        if env_id == "mean":
            lines.append(r"\midrule")
            name, b = "Mean", ""
        else:
            name, b = env_label(str(env_id)), row["best_batch_size"]
            b = "" if pd.isna(b) else f"{int(b)}"
        lines.append(" & ".join([name, *(fmt(row[m]) for m in cols), b]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n")


def plot_overall(
    df: pd.DataFrame,
    modes: list[str],
    labels: dict[str, str],
    colors: dict[str, Any],
    mean: pd.Series,
    out_path: Path,
) -> None:
    """Mean speedup per mode (bars) and per env (dots)."""
    x = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(0.49 * FULL_WIDTH, FIGHEIGHT))
    ax.bar(x, mean[modes], 0.7, color=[colors[m] for m in modes], zorder=2)
    for i, mode in enumerate(modes):
        speedups = df.loc[df["mode"] == mode, "speedup"]
        ax.scatter(
            np.full(len(speedups), i), speedups, s=8, color="black", alpha=0.5, zorder=3
        )
    # Above the highest dot, which would cover a number at the top of the bar
    tops = df.groupby("mode")["speedup"].max()
    annotate_speedups(ax, mean[modes], tops[modes])
    ax.axhline(1.0, color="black", linewidth=0.8, zorder=1)
    ax.set_yscale("log")
    ax.margins(y=0.25)
    ax.set_ylabel("Speedup")
    ax.set_xticks(x)
    # The full labels would take most of the height of the narrow figure
    ax.set_xticklabels(
        [
            labels[m] if m == BATCHED else labels[m].removeprefix(r"$\phi$-PICT ")
            for m in modes
        ],
        rotation=30,
        ha="right",
        fontsize=7,
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


@hydra.main(version_base="1.3", config_path="../configs", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    """Run the benchmark of one env, then update the overall table."""
    root = Path(hydra.utils.get_original_cwd())
    data_path = cfg.local_data_path or os.environ.get(
        "FLUIDGYM_LOCAL_DATA_PATH", "local_data"
    )
    fluidgym.config.update("local_data_path", str(root / data_path))

    if not cfg.aggregate_only:
        assert torch.cuda.is_available(), "The benchmark requires a CUDA device."
        run(cfg)
    aggregate(root / cfg.results_root, cfg.max_memory_pct, cfg.ablations)


if __name__ == "__main__":
    main()
