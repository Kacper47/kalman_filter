import numpy as np


class KalmanFilter:
    """Liniowy filtr Kalmana dla modelu CV 2D. Stan: [x, y, vx, vy]^T"""

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
    ) -> None:
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0.copy()
        self.P = P0.copy()

    def predict(self) -> None:
        """Faza predykcji: projekcja stanu i kowariancji naprzód."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray, gate: float | None = None) -> bool:
        """Faza korekcji. gate: próg chi² (None = brak bramkowania). Zwraca True jeśli pomiar przyjęty."""
        S  = self.H @ self.P @ self.H.T + self.R          # kowariancja innowacji
        nu = z - self.H @ self.x                           # innowacja

        if gate is not None:
            maha2 = float(nu @ np.linalg.solve(S, nu))    # odległość Mahalanobisa²
            if maha2 > gate:
                return False                                # odrzuć outlier

        K  = self.P @ self.H.T @ np.linalg.inv(S)         # wzmocnienie Kalmana
        self.x = self.x + K @ nu
        # Forma Josepha: (I-KH)P(I-KH)^T + KRK^T — numerycznie stabilna,
        # gwarantuje symetrię i dodatnią określoność P przy błędach numerycznych.
        I_KH = np.eye(len(self.x)) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return True

    def predict_ahead(self, n: int) -> np.ndarray:
        """Wielokrokowa predykcja pozycji na n kroków naprzód. Zwraca (2,): [x, y]."""
        Fn = np.linalg.matrix_power(self.F, n)
        return (Fn @ self.x)[:2]

    def predict_ahead_with_cov(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Wielokrokowa predykcja z propagacją kowariancji.
        Zwraca (pos (2,), P_pred (4,4)) gdzie P_pred = F^n P (F^n)^T + Q_sum.
        """
        Fn = np.linalg.matrix_power(self.F, n)
        P_pred = Fn @ self.P @ Fn.T
        # Akumulacja Q przez n kroków (aproksymacja: suma Q dla stałego Q)
        Q_sum = sum(
            np.linalg.matrix_power(self.F, i) @ self.Q @ np.linalg.matrix_power(self.F, i).T
            for i in range(n)
        )
        P_pred = P_pred + Q_sum
        return (Fn @ self.x)[:2], P_pred
