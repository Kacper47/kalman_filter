"""
Generowanie trajektorii i pomiarów oraz główna pętla symulacji.
"""

import numpy as np

from .model import build_F, build_H, build_Q, build_R
from .kalman import KalmanFilter


def generate_trajectory(N: int, dt: float, q: float, seed: int = 42) -> np.ndarray:
    """
    Generuje rzeczywistą trajektorię zgodnie z modelem CV
    z małym szumem procesowym (zakłócenia dynamiki).

    Zwraca tablicę (N, 4): [x, y, vx, vy] dla każdego kroku.
    """
    rng = np.random.default_rng(seed)
    F = build_F(dt)
    Q = build_Q(q)
    L = np.linalg.cholesky(Q)

    states = np.zeros((N, 4))
    x = np.array([0.0, 0.0, 1.2, 0.5])  # stan początkowy

    for k in range(N):
        states[k] = x
        w = L @ rng.standard_normal(4)
        x = F @ x + w

    return states


def generate_measurements(
    states: np.ndarray,
    sigma: float,
    seed: int = 0,
    p_dropout: float = 0.0,
    p_outlier: float = 0.0,
    outlier_scale: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dodaje szum pomiarowy do pozycji z opcjonalnymi zaburzeniami.

    Zwraca (measurements, valid_mask, outlier_mask):
      measurements  (N, 2) — zaszumione pomiary (zawsze wypełnione, dla dropout krok losowy)
      valid_mask    (N,)   bool — True gdy pomiar dostępny (sensor nie milczał)
      outlier_mask  (N,)   bool — True gdy krok miał outlier (duży błąd)
    """
    rng = np.random.default_rng(seed)
    N = len(states)
    H = build_H()
    true_pos = (H @ states.T).T

    noise = rng.standard_normal((N, 2)) * sigma
    measurements = true_pos + noise

    # Outliers: zastąp szum dużo większym
    outlier_mask = np.zeros(N, dtype=bool)
    if p_outlier > 0:
        outlier_mask = rng.random(N) < p_outlier
        big_noise = rng.standard_normal((outlier_mask.sum(), 2)) * sigma * outlier_scale
        measurements[outlier_mask] = true_pos[outlier_mask] + big_noise

    # Dropout: oznacz kroki bez pomiaru
    valid_mask = np.ones(N, dtype=bool)
    if p_dropout > 0:
        valid_mask = rng.random(N) >= p_dropout

    return measurements, valid_mask, outlier_mask


def run_simulation(
    N: int,
    dt: float,
    sigma: float,
    q: float,
    horizon: int,
    seed: int = 42,
    p_dropout: float = 0.0,
    p_outlier: float = 0.0,
    q_filter: float | None = None,
    gate_threshold: float | None = None,
    freeze_from_step: int | None = None,
    resume_from_step: int | None = None,
) -> dict:
    """Jeden kompletny przebieg FK: generowanie trajektorii, pomiarów, filtracja, predykcja, ocena."""
    # Import wewnętrzny — unika cyklicznej zależności na poziomie modułu
    from .metrics import rmse, mae

    q_f = q_filter if q_filter is not None else q   # model mismatch: filter może używać innego Q

    F = build_F(dt)
    H = build_H()
    Q = build_Q(q_f)
    R = build_R(sigma)

    states = generate_trajectory(N, dt, q, seed=seed)
    measurements, valid_mask, outlier_mask = generate_measurements(
        states, sigma, seed=seed + 1,
        p_dropout=p_dropout, p_outlier=p_outlier,
    )

    # vx=vy=0 — filtr potrzebuje kilku kroków obserwacji by wyestymować prędkość
    x0 = np.array([measurements[0, 0], measurements[0, 1], 0.0, 0.0])
    P0 = np.diag([sigma**2, sigma**2, 1.0, 1.0])

    kf = KalmanFilter(F, H, Q, R, x0, P0)

    estimates   = np.zeros((N, 4))
    predictions = np.zeros((N, 2))

    for k in range(N):
        kf.predict()
        use_meas = valid_mask[k] and (
            freeze_from_step is None or
            k <= freeze_from_step or
            (resume_from_step is not None and k >= resume_from_step)
        )
        if use_meas:
            kf.update(measurements[k], gate=gate_threshold)
        estimates[k]   = kf.x
        predictions[k] = kf.predict_ahead(horizon)

    true_pos = states[:, :2]
    est_pos  = estimates[:, :2]

    # Predykcja oceniana na pozycji k+horizon (jeśli dostępna)
    pred_true = states[horizon:, :2]
    pred_est  = predictions[:N - horizon]

    return {
        "states":        states,
        "measurements":  measurements,
        "estimates":     estimates,
        "predictions":   predictions,
        "valid":         valid_mask,
        "outlier_mask":  outlier_mask,
        "rmse_est":      float(rmse(est_pos, true_pos)),
        "mae_est":       float(mae(est_pos, true_pos)),
        "rmse_pred":     float(rmse(pred_est, pred_true)),
        "mae_pred":      float(mae(pred_est, pred_true)),
    }


def run_simulation_on_track(
    true_positions: np.ndarray,
    dt: float,
    sigma: float,
    q: float,
    horizon: int,
    seed: int = 42,
    p_dropout: float = 0.0,
    p_outlier: float = 0.0,
    freeze_from_step: int | None = None,
    resume_from_step: int | None = None,
) -> dict:
    """
    Uruchamia FK na rzeczywistej trajektorii (np. z datasetu TrajAir).

    Parametry
    ----------
    true_positions : (N, 2) — prawdziwa trasa [x, y]
    Pozostałe parametry jak w run_simulation().

    Zwraca ten sam słownik co run_simulation().
    """
    from .metrics import rmse, mae
    from .real_data import positions_to_states

    N = len(true_positions)
    states = positions_to_states(true_positions, dt)   # (N, 4): [x, y, vx, vy]

    F = build_F(dt)
    H = build_H()
    Q = build_Q(q)
    R = build_R(sigma)

    # Pomiary = pozycje ADS-B bezpośrednio (bez dodatkowego szumu).
    # sigma trafia tylko do R — filtr wie jak bardzo ufać danym, ale nie dodajemy
    # sztucznego szumu do już zmierzonych pozycji.
    rng = np.random.default_rng(seed + 1)
    measurements = true_positions.astype(float).copy()
    valid_mask   = np.ones(N, dtype=bool)
    outlier_mask = np.zeros(N, dtype=bool)
    if p_dropout > 0:
        valid_mask = rng.random(N) >= p_dropout
    if p_outlier > 0:
        outlier_mask = rng.random(N) < p_outlier
        n_out = int(outlier_mask.sum())
        if n_out:
            big_noise = rng.standard_normal((n_out, 2)) * sigma * 8.0
            measurements[outlier_mask] = true_positions[outlier_mask] + big_noise

    # vx=vy=0 — filtr potrzebuje kilku kroków obserwacji by wyestymować prędkość
    x0 = np.array([measurements[0, 0], measurements[0, 1], 0.0, 0.0])
    P0 = np.diag([sigma**2, sigma**2, 1.0, 1.0])

    kf = KalmanFilter(F, H, Q, R, x0, P0)

    estimates   = np.zeros((N, 4))
    predictions = np.zeros((N, 2))

    for k in range(N):
        kf.predict()
        use_meas = valid_mask[k] and (
            freeze_from_step is None or
            k <= freeze_from_step or
            (resume_from_step is not None and k >= resume_from_step)
        )
        if use_meas:
            kf.update(measurements[k])
        estimates[k]   = kf.x
        predictions[k] = kf.predict_ahead(horizon)

    true_pos = states[:, :2]
    est_pos  = estimates[:, :2]
    pred_true = states[horizon:, :2]
    pred_est  = predictions[:N - horizon]

    return {
        "states":       states,
        "measurements": measurements,
        "estimates":    estimates,
        "predictions":  predictions,
        "valid":        valid_mask,
        "outlier_mask": outlier_mask,
        "rmse_est":     float(rmse(est_pos, true_pos)),
        "mae_est":      float(mae(est_pos, true_pos)),
        "rmse_pred":    float(rmse(pred_est, pred_true)),
        "mae_pred":     float(mae(pred_est, pred_true)),
    }
