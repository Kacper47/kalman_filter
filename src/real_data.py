"""
Loader dla datasetu TrajAir (Carnegie Mellon University).
Trajektorie samolotów ADS-B zarejestrowane wokół lotniska
Pittsburgh-Butler Regional Airport (KBTP), wrzesień 2020.

Format pliku processed_data: spacja-separated, 7 kolumn bez nagłówka:
  frame_id  aircraft_id  x  y  z  vx  vy

Współrzędne x,y są znormalizowane; 1 jednostka ≈ 2.5 km.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

_DATA_ROOT = Path("data/7days1/processed_data")


def list_trajectories(split: str = "test", min_len: int = 50) -> list[str]:
    """Zwraca posortowaną listę ścieżek do trajektorii w zbiorze train lub test."""
    if split not in ("train", "test"):
        raise ValueError("split musi być 'train' lub 'test'")
    paths = sorted(glob.glob(str(_DATA_ROOT / split / "*.txt")))
    if min_len > 0:
        paths = [p for p in paths if _count_rows(p) >= min_len]
    return paths


def load_trajectory(path: str, max_len: int | None = None) -> np.ndarray:
    """
    Wczytuje jedną trajektorię TrajAir.

    Zwraca tablicę (N, 2) z kolumnami [x, y] (pozycje w j. TrajAir).
    Opcjonalnie przycina do max_len pierwszych kroków.
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    positions = data[:, 2:4].copy()   # kolumny x, y
    if max_len is not None:
        positions = positions[:max_len]
    return positions


def load_all_trajectories(
    split: str = "test",
    min_len: int = 50,
    max_len: int | None = 500,
) -> list[np.ndarray]:
    """
    Wczytuje wszystkie trajektorie z danego zbioru.

    Zwraca listę tablic (N_i, 2).
    """
    return [load_trajectory(p, max_len=max_len)
            for p in list_trajectories(split, min_len=min_len)]


def compute_velocities(positions: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Oblicza prędkości metodą różnic centralnych.

    Przyjmuje (N, 2) pozycji, zwraca (N, 2) prędkości [vx, vy].
    Krawędzie obsługiwane różnicą jednostronną.
    """
    N = len(positions)
    vel = np.zeros_like(positions)
    if N < 2:
        return vel
    # Środkowe różnice
    vel[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
    # Krawędzie
    vel[0]  = (positions[1]  - positions[0])  / dt
    vel[-1] = (positions[-1] - positions[-2]) / dt
    return vel


def positions_to_states(positions: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Tworzy macierz stanów (N, 4) = [x, y, vx, vy] z samych pozycji (N, 2).
    Prędkości szacowane numerycznie.
    """
    vel = compute_velocities(positions, dt)
    return np.hstack([positions, vel])


def _count_rows(path: str) -> int:
    data = np.loadtxt(path)
    return len(data) if data.ndim == 2 else 1
