"""Testy jednostkowe filtra Kalmana."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.kalman import KalmanFilter
from src.model import build_F, build_H, build_Q, build_R
from src.simulation import run_simulation


def _make_kf(sigma=0.5, q=0.1, dt=1.0):
    F = build_F(dt)
    H = build_H()
    Q = build_Q(q)
    R = build_R(sigma)
    x0 = np.array([0.0, 0.0, 1.0, 0.5])
    P0 = np.diag([1.0, 1.0, 1.0, 1.0])
    return KalmanFilter(F, H, Q, R, x0, P0)


def test_predict_moves_state():
    kf = _make_kf()
    x_before = kf.x.copy()
    kf.predict()
    assert not np.allclose(kf.x, x_before), "predict nie zmieniło stanu"


def test_predict_increases_uncertainty():
    kf = _make_kf()
    P_before = np.trace(kf.P)
    kf.predict()
    assert np.trace(kf.P) > P_before, "predict powinno zwiększyć niepewność"


def test_update_reduces_uncertainty():
    kf = _make_kf()
    kf.predict()
    P_before = np.trace(kf.P)
    z = np.array([0.5, 0.3])
    kf.update(z)
    assert np.trace(kf.P) < P_before, "update powinno zmniejszyć niepewność"


def test_P_stays_symmetric():
    kf = _make_kf()
    for _ in range(20):
        kf.predict()
        kf.update(np.array([1.0, 0.5]))
    assert np.allclose(kf.P, kf.P.T, atol=1e-10), "P powinna być symetryczna"


def test_P_stays_positive_definite():
    kf = _make_kf()
    for _ in range(20):
        kf.predict()
        kf.update(np.array([1.0, 0.5]))
    eigvals = np.linalg.eigvalsh(kf.P)
    assert np.all(eigvals > 0), f"P nie jest dodatnio określona: {eigvals}"


def test_update_with_exact_measurement():
    """Przy zerowym szumie pomiaru (R→0) estymacja powinna być bliska pomiarowi."""
    F = build_F(1.0)
    H = build_H()
    Q = build_Q(0.01)
    R = build_R(1e-6)
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    P0 = np.eye(4) * 10.0
    kf = KalmanFilter(F, H, Q, R, x0, P0)
    z = np.array([5.0, 3.0])
    kf.predict()
    kf.update(z)
    assert np.allclose(kf.x[:2], z, atol=1e-3), "przy małym R estymacja powinna odpowiadać pomiarowi"


def test_predict_ahead_n_steps():
    kf = _make_kf()
    pos1 = kf.predict_ahead(1)
    pos5 = kf.predict_ahead(5)
    # Stan: [0,0,1,0.5] → po 1 kroku x=1, y=0.5; po 5 krokach x=5, y=2.5
    assert np.allclose(pos1, [1.0, 0.5], atol=1e-9)
    assert np.allclose(pos5, [5.0, 2.5], atol=1e-9)


def test_predict_ahead_with_cov_larger_than_P():
    kf = _make_kf()
    _, P_pred = kf.predict_ahead_with_cov(5)
    assert np.trace(P_pred) > np.trace(kf.P), "niepewność predykcji n>1 powinna być większa niż bieżąca P"


def test_gating_rejects_outlier():
    kf = _make_kf(sigma=0.1)
    kf.predict()
    x_before = kf.x.copy()
    accepted = kf.update(np.array([100.0, 100.0]), gate=9.21)
    assert not accepted, "outlier powinien zostać odrzucony przez gating"
    assert np.allclose(kf.x, x_before), "stan nie powinien się zmienić po odrzuceniu outliera"


def test_run_simulation_returns_expected_keys():
    result = run_simulation(N=30, dt=1.0, sigma=0.5, q=0.1, horizon=5, seed=0)
    for key in ("states", "measurements", "estimates", "predictions", "valid",
                "outlier_mask", "rmse_est", "rmse_pred"):
        assert key in result, f"brak klucza: {key}"


def test_run_simulation_dropout():
    result = run_simulation(N=60, dt=1.0, sigma=0.5, q=0.1, horizon=5, seed=0, p_dropout=0.5)
    valid_count = result["valid"].sum()
    assert 10 < valid_count < 55, f"dropout 50% dał {valid_count}/60 ważnych pomiarów"


def test_run_simulation_freeze_gap():
    """Podczas przerwy błąd rośnie, po wznowieniu wraca do normy."""
    res_base   = run_simulation(60, 1.0, 0.3, 0.05, 5, seed=42)
    res_freeze = run_simulation(60, 1.0, 0.3, 0.05, 5, seed=42,
                                freeze_from_step=20, resume_from_step=30)
    err_base   = np.linalg.norm(res_base["estimates"][25, :2]   - res_base["states"][25, :2])
    err_gap    = np.linalg.norm(res_freeze["estimates"][25, :2] - res_freeze["states"][25, :2])
    err_after  = np.linalg.norm(res_freeze["estimates"][45, :2] - res_freeze["states"][45, :2])
    assert err_gap > err_base * 2, "podczas przerwy błąd powinien być wyraźnie większy"
    assert err_after < err_gap,    "po wznowieniu błąd powinien spaść"
