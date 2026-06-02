import numpy as np

from noisy_opt.algorithms import run_adaptive_resampling, run_ga, run_sa
from noisy_opt.evaluation import NoisyObjective


def _run(runner, budget=100):
    rng = np.random.default_rng(42)
    evaluator = NoisyObjective(budget=budget, rng=rng)
    return runner(evaluator=evaluator, rng=rng, seed=42)


def test_sa_respects_budget_and_result_shape():
    result = _run(run_sa)
    assert result.evaluations == 100
    assert result.best_x.shape == (2,)
    assert result.feasible
    assert result.history


def test_ga_respects_budget_and_result_shape():
    result = _run(run_ga)
    assert result.evaluations == 100
    assert result.best_x.shape == (2,)
    assert result.feasible
    assert result.history


def test_adaptive_resampling_respects_budget_and_result_shape():
    result = _run(run_adaptive_resampling)
    assert result.evaluations <= 100
    assert result.evaluations > 0
    assert result.best_x.shape == (2,)
    assert result.feasible
    assert result.history
    assert result.metadata["elite_pool_size"] == 6
    assert result.metadata["final_resample_fraction"] == 0.10
    assert result.metadata["final_resample_per_point"] == 6
    assert result.metadata["final_pool_used"] > 0


def test_noisy_objective_resamples_each_call():
    rng = np.random.default_rng(1)
    evaluator = NoisyObjective(budget=5, rng=rng)
    values = [evaluator.evaluate(np.array([0.2, 0.3])) for _ in range(5)]
    assert len(set(np.round(values, 12))) > 1
    assert evaluator.evaluations == 5
