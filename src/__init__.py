"""Pakiet implementacji filtru Kalmana — predykcja trajektorii 2D."""

from .model import build_F, build_H, build_Q, build_R
from .kalman import KalmanFilter
from .simulation import generate_trajectory, generate_measurements, run_simulation
from .metrics import rmse, mae, sigma_analysis, horizon_analysis
