from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


def summarize_by_algorithm(summary: pd.DataFrame) -> pd.DataFrame:
    required_raw_columns = {
        "raw_candidate_feasible_rate": float("nan"),
        "raw_candidate_count": 0,
        "raw_feasible_candidate_count": 0,
    }
    missing_raw_columns = [name for name in required_raw_columns if name not in summary.columns]
    if missing_raw_columns:
        summary = summary.copy()
        for name in missing_raw_columns:
            summary[name] = required_raw_columns[name]

    grouped = summary.groupby(["noise_label", "algorithm"], as_index=False)
    stats = grouped.agg(
        median_best_true_f=("best_true_f", "median"),
        q1_best_true_f=("best_true_f", lambda s: s.quantile(0.25)),
        q3_best_true_f=("best_true_f", lambda s: s.quantile(0.75)),
        feasible_rate=("feasible", "mean"),
        raw_candidate_count=("raw_candidate_count", "sum"),
        raw_feasible_candidate_count=("raw_feasible_candidate_count", "sum"),
        raw_candidate_feasible_rate_mean=("raw_candidate_feasible_rate", "mean"),
        raw_candidate_feasible_rate_median=("raw_candidate_feasible_rate", "median"),
        runs=("best_true_f", "count"),
    )
    stats["iqr_best_true_f"] = stats["q3_best_true_f"] - stats["q1_best_true_f"]
    stats["raw_candidate_feasible_rate_overall"] = (
        stats["raw_feasible_candidate_count"] / stats["raw_candidate_count"]
    ).fillna(0.0)
    return stats[
        [
            "noise_label",
            "algorithm",
            "runs",
            "median_best_true_f",
            "iqr_best_true_f",
            "q1_best_true_f",
            "q3_best_true_f",
            "feasible_rate",
            "raw_candidate_count",
            "raw_feasible_candidate_count",
            "raw_candidate_feasible_rate_overall",
            "raw_candidate_feasible_rate_mean",
            "raw_candidate_feasible_rate_median",
        ]
    ]


def pairwise_wilcoxon(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for noise_label, group in summary.groupby("noise_label"):
        algorithms = sorted(group["algorithm"].unique())
        for a, b in combinations(algorithms, 2):
            left = group[group["algorithm"] == a][["run", "best_true_f"]].rename(
                columns={"best_true_f": "a"}
            )
            right = group[group["algorithm"] == b][["run", "best_true_f"]].rename(
                columns={"best_true_f": "b"}
            )
            paired = left.merge(right, on="run", how="inner").sort_values("run")
            if len(paired) == 0:
                statistic = float("nan")
                p_value = float("nan")
            else:
                try:
                    test = wilcoxon(paired["a"], paired["b"], alternative="two-sided")
                    statistic = float(test.statistic)
                    p_value = float(test.pvalue)
                except ValueError:
                    statistic = 0.0
                    p_value = 1.0
            rows.append(
                {
                    "noise_label": noise_label,
                    "algorithm_a": a,
                    "algorithm_b": b,
                    "paired_runs": len(paired),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def save_statistics(summary_csv: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    summary = pd.read_csv(summary_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = summarize_by_algorithm(summary)
    tests = pairwise_wilcoxon(summary)
    stats_path = output_dir / "statistics.csv"
    tests_path = output_dir / "wilcoxon.csv"
    stats.to_csv(stats_path, index=False)
    tests.to_csv(tests_path, index=False)
    return stats_path, tests_path
