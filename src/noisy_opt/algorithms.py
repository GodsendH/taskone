from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Any

import numpy as np

from .evaluation import NoisyObjective
from .problem import is_feasible, project_to_feasible, sample_feasible


@dataclass
class AlgorithmResult:
    algorithm: str
    seed: int
    best_x: np.ndarray
    best_true_f: float
    best_observed_f: float
    feasible: bool
    evaluations: int
    history: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _append_history(history: list[dict[str, float]], evaluator: NoisyObjective, best_x: np.ndarray) -> None:
    history.append(
        {
            "evaluation": float(evaluator.evaluations),
            "best_true_f": evaluator.true(best_x),
            "x1": float(best_x[0]),
            "x2": float(best_x[1]),
        }
    )


def run_sa(
    evaluator: NoisyObjective,
    rng: np.random.Generator,
    seed: int,
    *,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.995,
    step_size: float = 0.18,
    min_temp: float = 1e-4,
) -> AlgorithmResult:
    """Standard simulated annealing with exponential cooling."""

    current_x = sample_feasible(rng)
    current_y = evaluator.evaluate(current_x)
    best_x = current_x.copy()
    best_observed = current_y
    history: list[dict[str, float]] = []
    _append_history(history, evaluator, best_x)

    iteration = 0
    while evaluator.can_evaluate():
        temp = max(min_temp, initial_temp * (cooling_rate**iteration))
        candidate_x = project_to_feasible(current_x + rng.normal(0.0, step_size, size=2))
        candidate_y = evaluator.evaluate(candidate_x)
        delta = candidate_y - current_y
        if delta <= 0.0 or rng.random() < exp(-delta / max(temp, 1e-12)):
            current_x = candidate_x
            current_y = candidate_y
        if current_y < best_observed:
            best_x = current_x.copy()
            best_observed = current_y
        _append_history(history, evaluator, best_x)
        iteration += 1

    return AlgorithmResult(
        algorithm="SA",
        seed=seed,
        best_x=best_x,
        best_true_f=evaluator.true(best_x),
        best_observed_f=float(best_observed),
        feasible=is_feasible(best_x),
        evaluations=evaluator.evaluations,
        history=history,
        metadata={
            "initial_temp": initial_temp,
            "cooling_rate": cooling_rate,
            "step_size": step_size,
            "min_temp": min_temp,
        },
    )


def _tournament_select(
    rng: np.random.Generator,
    population: np.ndarray,
    noisy_scores: np.ndarray,
    tournament_size: int,
) -> np.ndarray:
    size = min(tournament_size, len(population))
    idx = rng.choice(len(population), size=size, replace=False)
    return population[idx[np.argmin(noisy_scores[idx])]].copy()


def run_ga(
    evaluator: NoisyObjective,
    rng: np.random.Generator,
    seed: int,
    *,
    population_size: int = 40,
    elite_fraction: float = 0.10,
    crossover_rate: float = 0.85,
    mutation_step: float = 0.20,
    mutation_decay: float = 0.985,
    min_mutation_step: float = 0.015,
    tournament_size: int = 3,
) -> AlgorithmResult:
    """Real-coded GA with elitism and adaptive mutation step."""

    init_size = min(population_size, evaluator.remaining)
    population = sample_feasible(rng, init_size)
    noisy_scores = np.empty(init_size, dtype=float)
    history: list[dict[str, float]] = []

    best_x = population[0].copy()
    best_observed = float("inf")
    for i in range(init_size):
        noisy_scores[i] = evaluator.evaluate(population[i])
        if noisy_scores[i] < best_observed:
            best_observed = float(noisy_scores[i])
            best_x = population[i].copy()
        _append_history(history, evaluator, best_x)

    generation = 0
    stagnant_generations = 0
    last_best = best_observed
    while evaluator.can_evaluate():
        order = np.argsort(noisy_scores)
        population = population[order]
        noisy_scores = noisy_scores[order]
        elite_count = max(1, int(round(elite_fraction * len(population))))
        elite_count = min(elite_count, len(population))

        elites = population[:elite_count].copy()
        elite_scores = noisy_scores[:elite_count].copy()
        new_population = [x.copy() for x in elites]
        new_scores = [float(s) for s in elite_scores]

        diversity = float(np.mean(np.std(population, axis=0))) if len(population) > 1 else 0.0
        base_step = mutation_step * (mutation_decay**generation)
        if stagnant_generations >= 5:
            base_step *= 1.5
        adaptive_step = max(min_mutation_step, base_step * (0.75 + diversity))

        while len(new_population) < population_size and evaluator.can_evaluate():
            parent_a = _tournament_select(rng, population, noisy_scores, tournament_size)
            parent_b = _tournament_select(rng, population, noisy_scores, tournament_size)
            if rng.random() < crossover_rate:
                weight = rng.uniform(0.0, 1.0)
                child = weight * parent_a + (1.0 - weight) * parent_b
            else:
                child = parent_a.copy()
            child = project_to_feasible(child + rng.normal(0.0, adaptive_step, size=2))
            child_score = evaluator.evaluate(child)
            new_population.append(child)
            new_scores.append(float(child_score))
            if child_score < best_observed:
                best_observed = float(child_score)
                best_x = child.copy()
            _append_history(history, evaluator, best_x)

        population = np.asarray(new_population, dtype=float)
        noisy_scores = np.asarray(new_scores, dtype=float)
        if best_observed < last_best - 1e-12:
            stagnant_generations = 0
            last_best = best_observed
        else:
            stagnant_generations += 1
        generation += 1

    return AlgorithmResult(
        algorithm="GA",
        seed=seed,
        best_x=best_x,
        best_true_f=evaluator.true(best_x),
        best_observed_f=float(best_observed),
        feasible=is_feasible(best_x),
        evaluations=evaluator.evaluations,
        history=history,
        metadata={
            "population_size": population_size,
            "elite_fraction": elite_fraction,
            "crossover_rate": crossover_rate,
            "mutation_step": mutation_step,
            "mutation_decay": mutation_decay,
            "min_mutation_step": min_mutation_step,
            "tournament_size": tournament_size,
        },
    )


def _mean_and_var(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("inf"), float("inf")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(np.mean(values)), float(np.var(values, ddof=1))


def run_adaptive_resampling(
    evaluator: NoisyObjective,
    rng: np.random.Generator,
    seed: int,
    *,
    initial_temp: float = 0.8,
    cooling_rate: float = 0.996,
    initial_step: float = 0.20,
    min_step: float = 0.015,
    base_samples: int = 2,
    max_samples: int = 8,
    ambiguity_z: float = 1.0,
) -> AlgorithmResult:
    """Adaptive noise-tolerant search with resampling and SA-style acceptance."""

    current_x = sample_feasible(rng)
    current_values = evaluator.evaluate_many(current_x, base_samples)
    if not current_values:
        raise RuntimeError("No evaluation budget available for initialization.")
    current_mean, current_var = _mean_and_var(current_values)

    best_x = current_x.copy()
    best_estimate = current_mean
    history: list[dict[str, float]] = []
    _append_history(history, evaluator, best_x)

    iteration = 0
    while evaluator.can_evaluate():
        temp = max(1e-4, initial_temp * (cooling_rate**iteration))
        step = max(min_step, initial_step * np.sqrt(temp / max(initial_temp, 1e-12)))
        candidate_x = project_to_feasible(current_x + rng.normal(0.0, step, size=2))
        candidate_values = evaluator.evaluate_many(candidate_x, base_samples)
        if not candidate_values:
            break

        candidate_mean, candidate_var = _mean_and_var(candidate_values)
        current_mean, current_var = _mean_and_var(current_values)

        while evaluator.can_evaluate() and len(candidate_values) < max_samples:
            uncertainty = np.sqrt(
                candidate_var / max(len(candidate_values), 1)
                + current_var / max(len(current_values), 1)
            )
            unclear = abs(candidate_mean - current_mean) <= ambiguity_z * max(uncertainty, 1e-12)
            if not unclear:
                break
            candidate_values.extend(evaluator.evaluate_many(candidate_x, 1))
            if evaluator.can_evaluate() and len(current_values) < max_samples:
                current_values.extend(evaluator.evaluate_many(current_x, 1))
            candidate_mean, candidate_var = _mean_and_var(candidate_values)
            current_mean, current_var = _mean_and_var(current_values)

        delta = candidate_mean - current_mean
        if delta <= 0.0 or rng.random() < exp(-delta / max(temp, 1e-12)):
            current_x = candidate_x
            current_values = candidate_values
            current_mean = candidate_mean

        if current_mean < best_estimate:
            best_estimate = current_mean
            best_x = current_x.copy()

        _append_history(history, evaluator, best_x)
        iteration += 1

    return AlgorithmResult(
        algorithm="ARS",
        seed=seed,
        best_x=best_x,
        best_true_f=evaluator.true(best_x),
        best_observed_f=float(best_estimate),
        feasible=is_feasible(best_x),
        evaluations=evaluator.evaluations,
        history=history,
        metadata={
            "initial_temp": initial_temp,
            "cooling_rate": cooling_rate,
            "initial_step": initial_step,
            "min_step": min_step,
            "base_samples": base_samples,
            "max_samples": max_samples,
            "ambiguity_z": ambiguity_z,
        },
    )


ALGORITHM_RUNNERS = {
    "SA": run_sa,
    "GA": run_ga,
    "ARS": run_adaptive_resampling,
}
