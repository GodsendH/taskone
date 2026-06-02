from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problem import heteroscedastic_sigma, true_objective


class EvaluationBudgetExceeded(RuntimeError):
    """Raised when an algorithm tries to call F(x) beyond the fixed budget."""


@dataclass
class NoisyObjective:
    """Budget-counted noisy objective F(x).

    Every call to evaluate draws a fresh independent Gaussian noise sample and
    increments the evaluation counter. Calls to true and sigma are diagnostic and
    do not count toward the noisy objective budget.
    """

    budget: int
    rng: np.random.Generator
    noise_mode: str = "hetero"
    constant_sigma: float | None = None
    evaluations: int = 0

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError("budget must be positive.")
        if self.noise_mode not in {"hetero", "constant"}:
            raise ValueError("noise_mode must be 'hetero' or 'constant'.")
        if self.noise_mode == "constant" and self.constant_sigma is None:
            raise ValueError("constant_sigma is required for constant noise mode.")

    @property
    def remaining(self) -> int:
        return self.budget - self.evaluations

    def can_evaluate(self, count: int = 1) -> bool:
        return self.remaining >= count

    def sigma(self, x: np.ndarray) -> float:
        if self.noise_mode == "constant":
            return float(self.constant_sigma)
        return float(heteroscedastic_sigma(x))

    def true(self, x: np.ndarray) -> float:
        return float(true_objective(x))

    def evaluate(self, x: np.ndarray) -> float:
        if not self.can_evaluate():
            raise EvaluationBudgetExceeded("No noisy objective evaluations remain.")
        self.evaluations += 1
        noise = self.rng.normal(0.0, self.sigma(x))
        return self.true(x) + float(noise)

    def evaluate_many(self, x: np.ndarray, count: int) -> list[float]:
        actual = min(int(count), self.remaining)
        if actual <= 0:
            return []
        return [self.evaluate(x) for _ in range(actual)]
