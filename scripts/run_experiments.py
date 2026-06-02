#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisy_opt.experiment import run_replicates, write_csv
from noisy_opt.statistics import save_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the main 50x2000 algorithm comparison.")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    summaries, histories = run_replicates(
        ["SA", "GA", "ARS"],
        runs=args.runs,
        budget=args.budget,
        seed=args.seed,
        noise_mode="hetero",
        noise_label="hetero",
    )
    summary_path = args.out / "main_summary.csv"
    history_path = args.out / "main_history.csv"
    write_csv(summary_path, summaries)
    write_csv(history_path, histories)
    stats_path, wilcoxon_path = save_statistics(summary_path, args.out)

    print(f"Wrote {summary_path}")
    print(f"Wrote {history_path}")
    print(f"Wrote {stats_path}")
    print(f"Wrote {wilcoxon_path}")


if __name__ == "__main__":
    main()
