from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .problem import heteroscedastic_sigma, true_objective


def _prepare_output(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def plot_convergence(history: pd.DataFrame, output: str | Path) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=history,
        x="evaluation",
        y="best_true_f",
        hue="algorithm",
        estimator="median",
        errorbar=("pi", 50),
        ax=ax,
    )
    ax.set_title("Median convergence curve")
    ax.set_xlabel("Noisy objective evaluations")
    ax.set_ylabel("True f(x) of incumbent")
    fig.tight_layout()
    fig.savefig(_prepare_output(output), dpi=180)
    plt.close(fig)


def plot_box(summary: pd.DataFrame, output: str | Path) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=summary, x="algorithm", y="best_true_f", hue="algorithm", ax=ax)
    ax.set_title("Final true objective distribution")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Best true f(x)")
    ax.legend_.remove() if ax.legend_ else None
    fig.tight_layout()
    fig.savefig(_prepare_output(output), dpi=180)
    plt.close(fig)


def plot_final_points(summary: pd.DataFrame, output: str | Path) -> None:
    configure_style()
    grid = np.linspace(0.0, 1.0, 250)
    xx, yy = np.meshgrid(grid, grid)
    points = np.stack([xx, yy], axis=-1)
    zz = true_objective(points)
    zz = np.where(xx * xx + yy * yy <= 1.0, zz, np.nan)

    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contourf(xx, yy, zz, levels=30, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="true f(x)")
    sns.scatterplot(data=summary, x="x1", y="x2", hue="algorithm", ax=ax, s=50)
    theta = np.linspace(0.0, 0.5 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="white", linewidth=1.5)
    ax.set_title("Final solutions on feasible domain")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    fig.tight_layout()
    fig.savefig(_prepare_output(output), dpi=180)
    plt.close(fig)


def plot_noise_surface(output: str | Path) -> None:
    configure_style()
    grid = np.linspace(0.0, 1.0, 250)
    xx, yy = np.meshgrid(grid, grid)
    points = np.stack([xx, yy], axis=-1)
    zz = heteroscedastic_sigma(points)
    zz = np.where(xx * xx + yy * yy <= 1.0, zz, np.nan)

    fig, ax = plt.subplots(figsize=(7, 6))
    heat = ax.contourf(xx, yy, zz, levels=30, cmap="magma")
    fig.colorbar(heat, ax=ax, label="sigma(x)")
    theta = np.linspace(0.0, 0.5 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="white", linewidth=1.5)
    ax.set_title("Heteroscedastic noise level")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    fig.tight_layout()
    fig.savefig(_prepare_output(output), dpi=180)
    plt.close(fig)


def plot_sensitivity(sensitivity: pd.DataFrame, output: str | Path) -> None:
    configure_style()
    g = sns.catplot(
        data=sensitivity,
        x="value",
        y="best_true_f",
        col="parameter",
        row="algorithm",
        kind="box",
        sharex=False,
        sharey=True,
        height=3.4,
        aspect=1.25,
    )
    g.set_axis_labels("Parameter value", "Best true f(x)")
    g.fig.suptitle("Parameter sensitivity", y=1.02)
    g.fig.tight_layout()
    g.fig.savefig(_prepare_output(output), dpi=180)
    plt.close(g.fig)


def plot_noise_study(summary: pd.DataFrame, output: str | Path) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=summary, x="algorithm", y="best_true_f", hue="noise_label", ax=ax)
    ax.set_title("Heteroscedastic vs constant noise")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Best true f(x)")
    fig.tight_layout()
    fig.savefig(_prepare_output(output), dpi=180)
    plt.close(fig)
