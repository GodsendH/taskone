#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisy_opt.plots import (
    plot_box,
    plot_convergence,
    plot_final_points,
    plot_noise_study,
    plot_noise_surface,
    plot_sensitivity,
)


def _read_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        print(f"Skipping missing file: {path}")
        return None
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures from experiment CSV files.")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    args = parser.parse_args()

    summary = _read_if_exists(args.results / "main_summary.csv")
    history = _read_if_exists(args.results / "main_history.csv")
    if summary is not None:
        plot_box(summary, args.figures / "main_boxplot.png")
        plot_final_points(summary, args.figures / "final_points.png")
    if history is not None:
        plot_convergence(history, args.figures / "convergence.png")

    sensitivity = _read_if_exists(args.results / "sensitivity.csv")
    if sensitivity is not None:
        plot_sensitivity(sensitivity, args.figures / "sensitivity.png")

    noise_study = _read_if_exists(args.results / "noise_study_summary.csv")
    if noise_study is not None:
        plot_noise_study(noise_study, args.figures / "noise_study.png")

    plot_noise_surface(args.figures / "noise_surface.png")
    print(f"Wrote figures to {args.figures}")


if __name__ == "__main__":
    main()
