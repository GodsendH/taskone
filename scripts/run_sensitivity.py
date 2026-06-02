#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noisy_opt.experiment import DEFAULT_PARAMS, run_single, write_csv


SCAN = {
    "SA": {
        "initial_temp": [0.2, 0.5, 1.0, 2.0],
        "cooling_rate": [0.985, 0.99, 0.995, 0.998],
        "step_size": [0.08, 0.14, 0.20, 0.30],
    },
    "GA": {
        "crossover_rate": [0.60, 0.75, 0.85, 0.95],
        "mutation_step": [0.08, 0.14, 0.20, 0.30],
        "elite_fraction": [0.05, 0.10, 0.20],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SA and GA parameter sensitivity scans.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    rows = []
    for alg_index, (algorithm, scan) in enumerate(SCAN.items()):
        for param_index, (parameter, values) in enumerate(scan.items()):
            for value_index, value in enumerate(values):
                params = dict(DEFAULT_PARAMS[algorithm])
                params[parameter] = value
                for run in range(args.runs):
                    run_seed = (
                        args.seed
                        + alg_index * 100_000
                        + param_index * 10_000
                        + value_index * 1000
                        + run * 1009
                    )
                    result = run_single(
                        algorithm,
                        run_seed,
                        args.budget,
                        noise_mode="hetero",
                        params=params,
                    )
                    rows.append(
                        {
                            "algorithm": result.algorithm,
                            "parameter": parameter,
                            "value": value,
                            "run": run,
                            "seed": run_seed,
                            "best_true_f": result.best_true_f,
                            "x1": float(result.best_x[0]),
                            "x2": float(result.best_x[1]),
                            "feasible": int(result.feasible),
                            "evaluations": result.evaluations,
                        }
                    )

    output = args.out / "sensitivity.csv"
    write_csv(output, rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
