#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisy_opt.experiment import run_replicates, write_csv
from noisy_opt.problem import estimate_feasible_mean_sigma
from noisy_opt.statistics import save_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare heteroscedastic and constant-noise settings.")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--sigma-samples", type=int, default=100000)
    args = parser.parse_args()

    constant_sigma = estimate_feasible_mean_sigma(args.sigma_samples, seed=args.seed)
    all_summaries = []
    all_histories = []

    summaries, histories = run_replicates(
        ["SA", "GA", "ARS"],
        runs=args.runs,
        budget=args.budget,
        seed=args.seed,
        noise_mode="hetero",
        noise_label="hetero",
    )
    all_summaries.extend(summaries)
    all_histories.extend(histories)

    summaries, histories = run_replicates(
        ["SA", "GA", "ARS"],
        runs=args.runs,
        budget=args.budget,
        seed=args.seed + 500_000,
        noise_mode="constant",
        noise_label=f"constant_sigma_{constant_sigma:.4f}",
        constant_sigma=constant_sigma,
    )
    all_summaries.extend(summaries)
    all_histories.extend(histories)

    summary_path = args.out / "noise_study_summary.csv"
    history_path = args.out / "noise_study_history.csv"
    write_csv(summary_path, all_summaries)
    write_csv(history_path, all_histories)
    save_statistics(summary_path, args.out / "noise_study_stats")

    print(f"Constant-noise baseline sigma = {constant_sigma:.6f}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {history_path}")


if __name__ == "__main__":
    main()
