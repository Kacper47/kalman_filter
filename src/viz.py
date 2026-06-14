"""
Funkcje wizualizacji wyników filtru Kalmana (matplotlib).
Każda funkcja może zapisać wykres jako PNG i/lub wyświetlić go inline.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd


def plot_results(
    res: dict,
    sigma: float,
    horizon: int,
    output_dir: str | None = None,
    prefix: str = "simulation",
    show: bool = True,
) -> str | None:
    """Wykres trajektorii 2D + błąd estymacji i predykcji vs krok."""
    states       = res["states"]
    measurements = res["measurements"]
    estimates    = res["estimates"]
    predictions  = res["predictions"]
    N = len(states)

    fig = plt.figure(figsize=(14, 10), layout="constrained")
    fig.suptitle(
        f"Filtr Kalmana — predykcja trajektorii 2D   "
        f"[σ={sigma:.1f}, n={horizon}, N={N}]",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── Trajektoria 2D ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(states[:, 0],       states[:, 1],       "k-",  lw=1.5, label="Trajektoria rzeczywista")
    ax1.scatter(measurements[:, 0], measurements[:, 1], s=12, c="tab:blue", alpha=0.4, label="Pomiary (zaszumione)")
    ax1.plot(estimates[:, 0],    estimates[:, 1],    "tab:orange", lw=1.5, label="Estymacja FK")
    ax1.plot(predictions[:, 0],  predictions[:, 1],  "tab:red",   lw=1.2,
             linestyle="--", label=f"Predykcja +{horizon} kroków")

    # Zaznacz punkt startowy i końcowy
    ax1.scatter(*states[0, :2],  s=60, c="k",        zorder=5)
    ax1.scatter(*states[-1, :2], s=60, c="darkgreen", zorder=5, marker="*")

    ax1.set_xlabel("x [j]")
    ax1.set_ylabel("y [j]")
    ax1.set_title("Trajektoria 2D")
    ax1.legend(fontsize=9, loc="upper left")
    # Równe skale osi bez pustej przestrzeni
    all_x = np.concatenate([states[:, 0], measurements[:, 0], estimates[:, 0]])
    all_y = np.concatenate([states[:, 1], measurements[:, 1], estimates[:, 1]])
    cx, cy = (all_x.min() + all_x.max()) / 2, (all_y.min() + all_y.max()) / 2
    half   = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 * 1.08
    ax1.set_xlim(cx - half, cx + half); ax1.set_ylim(cy - half, cy + half)
    ax1.grid(True, alpha=0.3)

    # ── Błąd estymacji vs czas ──
    ax2 = fig.add_subplot(gs[1, 0])
    err_est  = np.linalg.norm(estimates[:, :2] - states[:, :2], axis=1)
    err_meas = np.linalg.norm(measurements    - states[:, :2], axis=1)
    ax2.plot(err_meas, color="tab:blue",   alpha=0.5, lw=1, label="Błąd pomiaru")
    ax2.plot(err_est,  color="tab:orange", lw=1.5,          label="Błąd estymacji FK")
    ax2.axhline(res["rmse_est"], color="tab:orange", lw=1, linestyle=":", label=f"RMSE={res['rmse_est']:.2f}")
    ax2.set_xlabel("Krok k")
    ax2.set_ylabel("Błąd pozycji [j]")
    ax2.set_title("Błąd estymacji vs czas")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Błąd predykcji vs czas ──
    ax3 = fig.add_subplot(gs[1, 1])
    pred_true_pos = states[horizon:, :2]
    pred_err      = np.linalg.norm(predictions[:N - horizon] - pred_true_pos, axis=1)
    ax3.plot(pred_err, color="tab:red", lw=1.5, label=f"Błąd predykcji +{horizon} kroków")
    ax3.axhline(res["rmse_pred"], color="tab:red", lw=1, linestyle=":",
                label=f"RMSE={res['rmse_pred']:.2f}")
    ax3.set_xlabel("Krok k")
    ax3.set_ylabel("Błąd pozycji [j]")
    ax3.set_title(f"Błąd predykcji +{horizon} kroków vs czas")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    out_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{prefix}_trajektoria.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()
    return out_path


def plot_sigma_analysis(
    sigmas: list[float],
    dt: float,
    q: float,
    horizon: int,
    N: int,
    seed: int = 42,
    output_dir: str | None = None,
    prefix: str = "sigma_analysis",
    show: bool = True,
    q_filter: float | None = None,
) -> str | None:
    """RMSE/MAE vs poziom szumu σ — uruchamia symulację dla każdej wartości z listy sigmas."""
    from .simulation import run_simulation

    rmse_est_list, mae_est_list   = [], []
    rmse_pred_list, mae_pred_list = [], []

    for s in sigmas:
        r = run_simulation(N, dt, s, q, horizon, seed, q_filter=q_filter)
        rmse_est_list.append(r["rmse_est"])
        mae_est_list.append(r["mae_est"])
        rmse_pred_list.append(r["rmse_pred"])
        mae_pred_list.append(r["mae_pred"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Analiza wpływu szumu pomiarowego σ na jakość predykcji", fontsize=13, fontweight="bold")

    axes[0].plot(sigmas, rmse_est_list,  "o-",  color="tab:orange",        label="RMSE estymacja")
    axes[0].plot(sigmas, mae_est_list,   "s--", color="tab:orange", alpha=0.6, label="MAE estymacja")
    axes[0].plot(sigmas, rmse_pred_list, "o-",  color="tab:red",            label="RMSE predykcja")
    axes[0].plot(sigmas, mae_pred_list,  "s--", color="tab:red",   alpha=0.6, label="MAE predykcja")
    axes[0].set_xlabel("σ (odch. std. szumu pomiarowego)")
    axes[0].set_ylabel("Błąd [j]")
    axes[0].set_title("RMSE i MAE vs σ")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Stosunek predykcja / estymacja (RMSE i MAE)
    rmse_ratio = [p / e if e > 0 else 0 for p, e in zip(rmse_pred_list, rmse_est_list)]
    mae_ratio  = [p / e if e > 0 else 0 for p, e in zip(mae_pred_list,  mae_est_list)]
    axes[1].plot(sigmas, rmse_ratio, "D-",  color="tab:purple",         label="RMSE_pred / RMSE_est")
    axes[1].plot(sigmas, mae_ratio,  "s--", color="tab:purple", alpha=0.6, label="MAE_pred / MAE_est")
    axes[1].set_xlabel("σ")
    axes[1].set_ylabel("Współczynnik degradacji")
    axes[1].set_title(f"Degradacja jakości przy predykcji +{horizon} kroków")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{prefix}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()
    return out_path


def plot_horizon_analysis(
    df: pd.DataFrame,
    output_dir: str | None = None,
    prefix: str = "horizon_analysis",
    show: bool = True,
) -> str | None:
    """RMSE/MAE vs horyzont predykcji. df: wynik metrics.horizon_analysis()."""
    horizons = df["horizon"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Analiza wpływu horyzontu predykcji na jakość", fontsize=13, fontweight="bold")

    axes[0].plot(horizons, df["rmse_est"],  "o-",  color="tab:orange",        label="RMSE estymacja")
    axes[0].plot(horizons, df["mae_est"],   "s--", color="tab:orange", alpha=0.6, label="MAE estymacja")
    axes[0].plot(horizons, df["rmse_pred"], "o-",  color="tab:red",            label="RMSE predykcja")
    axes[0].plot(horizons, df["mae_pred"],  "s--", color="tab:red",   alpha=0.6, label="MAE predykcja")
    axes[0].set_xlabel("Horyzont predykcji n [kroki]")
    axes[0].set_ylabel("Błąd [j]")
    axes[0].set_title("RMSE i MAE vs horyzont")
    axes[0].legend(fontsize=9)
    axes[0].set_xticks(horizons)
    axes[0].grid(True, alpha=0.3)

    # Stosunek predykcja / estymacja (RMSE i MAE)
    rmse_ratio = [p / e if e > 0 else 0 for p, e in zip(df["rmse_pred"], df["rmse_est"])]
    mae_ratio  = [p / e if e > 0 else 0 for p, e in zip(df["mae_pred"],  df["mae_est"])]
    axes[1].plot(horizons, rmse_ratio, "D-", color="tab:purple",        label="RMSE_pred / RMSE_est")
    axes[1].plot(horizons, mae_ratio,  "s--", color="tab:purple", alpha=0.6, label="MAE_pred / MAE_est")
    axes[1].set_xlabel("Horyzont n [kroki]")
    axes[1].set_ylabel("Współczynnik degradacji")
    axes[1].set_title("Degradacja jakości vs horyzont")
    axes[1].set_xticks(horizons)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{prefix}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()
    return out_path
