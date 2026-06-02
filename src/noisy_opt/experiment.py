from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from .algorithms import ALGORITHM_RUNNERS, AlgorithmResult
from .evaluation import NoisyObjective


DEFAULT_PARAMS = {
    "SA": {
        "initial_temp": 1.0,
        "cooling_rate": 0.995,
        "step_size": 0.18,
        "min_temp": 1e-4,
    },
    "GA": {
        "population_size": 40,
        "elite_fraction": 0.10,
        "crossover_rate": 0.85,
        "mutation_step": 0.20,
        "mutation_decay": 0.985,
        "min_mutation_step": 0.015,
        "tournament_size": 3,
    },
    "ARS": {
        "initial_temp": 0.8,
        "cooling_rate": 0.996,
        "initial_step": 0.20,
        "min_step": 0.015,
        "base_samples": 2,
        "max_samples": 8,
        "ambiguity_z": 1.0,
        "elite_pool_size": 6,
        "final_resample_fraction": 0.10,
        "final_resample_per_point": 6,
    },
}

RAW_CANDIDATE_KEYS = {
    "raw_candidate_count",
    "raw_feasible_candidate_count",
    "raw_candidate_feasible_rate",
}


def run_single(
    algorithm: str,
    seed: int,
    budget: int,
    *,
    noise_mode: str = "hetero",
    constant_sigma: float | None = None,
    params: dict | None = None,
) -> AlgorithmResult:
    if algorithm not in ALGORITHM_RUNNERS:
        raise ValueError(f"Unknown algorithm {algorithm!r}; choose from {sorted(ALGORITHM_RUNNERS)}.")
    rng = np.random.default_rng(seed)
    evaluator = NoisyObjective(
        budget=budget,
        rng=rng,
        noise_mode=noise_mode,
        constant_sigma=constant_sigma,
    )
    merged = dict(DEFAULT_PARAMS[algorithm])
    if params:
        merged.update(params)
    return ALGORITHM_RUNNERS[algorithm](evaluator=evaluator, rng=rng, seed=seed, **merged)


def result_to_summary_row(result: AlgorithmResult, run: int, noise_label: str = "hetero") -> dict:
    metadata = result.metadata
    row = {
        "algorithm": result.algorithm,
        "run": run,
        "seed": result.seed,
        "noise_label": noise_label,
        "best_true_f": result.best_true_f,
        "best_observed_f": result.best_observed_f,
        "x1": float(result.best_x[0]),
        "x2": float(result.best_x[1]),
        "feasible": int(result.feasible),
        "evaluations": int(result.evaluations),
        "raw_candidate_count": int(metadata.get("raw_candidate_count", 0)),
        "raw_feasible_candidate_count": int(metadata.get("raw_feasible_candidate_count", 0)),
        "raw_candidate_feasible_rate": float(metadata.get("raw_candidate_feasible_rate", 0.0)),
    }
    row.update({f"param_{k}": v for k, v in metadata.items() if k not in RAW_CANDIDATE_KEYS})
    return row


def result_to_history_rows(result: AlgorithmResult, run: int, noise_label: str = "hetero") -> Iterable[dict]:
    for item in result.history:
        yield {
            "algorithm": result.algorithm,
            "run": run,
            "seed": result.seed,
            "noise_label": noise_label,
            "evaluation": int(item["evaluation"]),
            "best_true_f": item["best_true_f"],
            "x1": item["x1"],
            "x2": item["x2"],
        }


def run_replicates(
    algorithms: list[str],
    runs: int,
    budget: int,
    *,
    seed: int = 20260602,
    noise_mode: str = "hetero",
    noise_label: str = "hetero",
    constant_sigma: float | None = None,
    params_by_algorithm: dict[str, dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    summaries: list[dict] = []
    histories: list[dict] = []
    params_by_algorithm = params_by_algorithm or {}

    for run in range(runs):
        for alg_index, algorithm in enumerate(algorithms):
            run_seed = int(seed + run * 1009 + alg_index * 100_003)
            result = run_single(
                algorithm,
                run_seed,
                budget,
                noise_mode=noise_mode,
                constant_sigma=constant_sigma,
                params=params_by_algorithm.get(algorithm),
            )
            summaries.append(result_to_summary_row(result, run, noise_label))
            histories.extend(result_to_history_rows(result, run, noise_label))
    return summaries, histories


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
