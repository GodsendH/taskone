from __future__ import annotations

import numpy as np


def _as_points(x: np.ndarray | list[float] | tuple[float, float]) -> tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=float)
    if arr.shape == (2,):
        return arr.reshape(1, 2), True
    if arr.ndim >= 2 and arr.shape[-1] == 2:
        return arr.reshape(-1, 2), False
    raise ValueError("Expected a point with shape (2,) or an array with final dimension 2.")


def true_objective(x: np.ndarray | list[float] | tuple[float, float]) -> float | np.ndarray:
    """The deterministic objective f(x), without noise."""

    points, scalar = _as_points(x)
    x1 = points[:, 0]
    x2 = points[:, 1]
    values = (x1 - 1.0) ** 2 + (x2 - 1.0) ** 2
    values += np.sin(3.0 * np.pi * x1) * np.cos(2.0 * np.pi * x2)
    if scalar:
        return float(values[0])
    return values.reshape(np.asarray(x).shape[:-1])


def heteroscedastic_sigma(x: np.ndarray | list[float] | tuple[float, float]) -> float | np.ndarray:
    """Noise standard deviation sigma(x) from the assignment."""

    points, scalar = _as_points(x)
    magnitude = np.abs(points[:, 0]) + np.abs(points[:, 1])
    values = 0.1 + 0.2 * magnitude / (1.0 + magnitude)
    if scalar:
        return float(values[0])
    return values.reshape(np.asarray(x).shape[:-1])


def constraints(x: np.ndarray | list[float] | tuple[float, float]) -> np.ndarray:
    """Return g1, g2, g3 where feasibility means every value is <= 0."""

    point = np.asarray(x, dtype=float)
    if point.shape != (2,):
        raise ValueError("constraints expects a single point with shape (2,).")
    x1, x2 = point
    return np.array([x1 * x1 + x2 * x2 - 1.0, -x1, -x2], dtype=float)


def is_feasible(x: np.ndarray | list[float] | tuple[float, float], tol: float = 1e-10) -> bool:
    return bool(np.all(constraints(x) <= tol))


def project_to_feasible(x: np.ndarray | list[float] | tuple[float, float]) -> np.ndarray:
    """Project to the first-quadrant unit disk.

    The assignment allows either penalties or projection. This project uses projection
    so every algorithm compares feasible candidates under the same rule.
    """

    arr = np.asarray(x, dtype=float)
    if arr.shape == (2,):
        original_shape = (2,)
        points = arr.reshape(1, 2).copy()
        scalar = True
    elif arr.ndim >= 2 and arr.shape[-1] == 2:
        original_shape = arr.shape
        points = arr.reshape(-1, 2).copy()
        scalar = False
    else:
        raise ValueError("Expected shape (2,) or an array with final dimension 2.")

    points = np.maximum(points, 0.0)
    norms = np.linalg.norm(points, axis=1)
    mask = norms > 1.0
    points[mask] = points[mask] / norms[mask, None]
    projected = points.reshape(original_shape)
    if scalar:
        return projected.reshape(2)
    return projected


def sample_feasible(rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    """Sample uniformly from the first-quadrant unit disk."""

    count = 1 if n is None else int(n)
    if count <= 0:
        raise ValueError("n must be positive.")
    radius = np.sqrt(rng.uniform(0.0, 1.0, size=count))
    theta = rng.uniform(0.0, 0.5 * np.pi, size=count)
    points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    if n is None:
        return points[0]
    return points


def estimate_feasible_mean_sigma(samples: int = 100_000, seed: int = 20260602) -> float:
    """Monte Carlo estimate used as the constant-noise baseline."""

    rng = np.random.default_rng(seed)
    points = sample_feasible(rng, samples)
    return float(np.mean(heteroscedastic_sigma(points)))
