"""
Macierze modelu filtru Kalmana — model stałej prędkości (CV) 2D.
"""

import numpy as np


def build_F(dt: float) -> np.ndarray:
    """Macierz przejścia stanu — model stałej prędkości (CV)."""
    return np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ], dtype=float)


def build_H() -> np.ndarray:
    """Macierz obserwacji — sensor mierzy tylko pozycję."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float)


def build_Q(q: float) -> np.ndarray:
    """Macierz kowariancji szumu procesowego."""
    return np.eye(4) * q


def build_R(sigma: float) -> np.ndarray:
    """Macierz kowariancji szumu pomiarowego."""
    return np.eye(2) * sigma**2
