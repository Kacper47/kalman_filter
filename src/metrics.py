"""
Miary jakości predykcji i estymacji filtru Kalmana.
"""

import numpy as np
import pandas as pd


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Błąd średniokwadratowy (RMSE) pozycji 2D."""
    return float(np.sqrt(np.mean(np.sum((pred - true)**2, axis=1))))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Średni błąd bezwzględny (MAE) pozycji 2D."""
    return float(np.mean(np.linalg.norm(pred - true, axis=1)))


def sigma_analysis(
    sigmas: list[float],
    N: int,
    dt: float,
    q: float,
    horizon: int,
    seed: int = 42,
    q_filter: float | None = None,
) -> pd.DataFrame:
    """
    Analiza RMSE/MAE dla różnych wartości szumu pomiarowego σ.
    Zwraca DataFrame z kolumnami: sigma, rmse_est, mae_est, rmse_pred, mae_pred.
    """
    from .simulation import run_simulation

    rows = []
    for s in sigmas:
        r = run_simulation(N, dt, s, q, horizon, seed, q_filter=q_filter)
        rows.append({
            "sigma":     s,
            "rmse_est":  r["rmse_est"],
            "mae_est":   r["mae_est"],
            "rmse_pred": r["rmse_pred"],
            "mae_pred":  r["mae_pred"],
        })
    return pd.DataFrame(rows)


def horizon_analysis(
    horizons: list[int],
    N: int,
    dt: float,
    sigma: float,
    q: float,
    seed: int = 42,
    q_filter: float | None = None,
) -> pd.DataFrame:
    """
    Analiza RMSE/MAE estymacji i predykcji dla różnych horyzontów predykcji.

    Zwraca DataFrame z kolumnami: horizon, rmse_est, mae_est, rmse_pred, mae_pred.
    """
    from .simulation import run_simulation

    rows = []
    for h in horizons:
        r = run_simulation(N, dt, sigma, q, h, seed, q_filter=q_filter)
        rows.append({
            "horizon":   h,
            "rmse_est":  r["rmse_est"],
            "mae_est":   r["mae_est"],
            "rmse_pred": r["rmse_pred"],
            "mae_pred":  r["mae_pred"],
        })
    return pd.DataFrame(rows)
