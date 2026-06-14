"""
Główna pętla interaktywnej symulacji filtru Kalmana (Pygame).
Sterowanie: SPACJA (play/pause), R (reset + widok), H (logi), M (tryb Synth/Real),
            F (freeze pomiarów od bieżącego kroku), ← → (trasa w trybie Real),
            scroll myszy (zoom mapy), drag lewym (pan mapy).
"""

import sys
import tkinter as tk
from pathlib import Path
import numpy as np
import pygame

sys.path.insert(0, ".")

from src.simulation import run_simulation, run_simulation_on_track
from src.metrics import rmse, mae

from .constants import (
    W, H, TOOLBAR_H, LOG_H, MARGIN_X, MARGIN_Y, FPS,
    BG, TOOLBAR_BG, LOG_BG, DIV,
    C_TRUE, C_TRUE_DIM, C_MEAS, C_EST, C_PRED, C_FUTURE, C_DRONE, C_OUTLIER,
    C_TEXT, C_ACCENT, C_DIM,
    N, DT, SEED, SIGMA_DEF, Q_DEF, HORIZON_DEF, SPEED_DEF, LOG_VISIBLE,
    P_DROPOUT_DEF, P_OUTLIER_DEF,
    SIGMA_REAL, Q_REAL,
)
from .ui import Slider, Button
from .renderer import (
    draw_grid, draw_axes,
    draw_drone, draw_diamond, draw_dashed, draw_polyline,
    draw_legend_overlay,
)

# ── Trasy TrajAir ─────────────────────────────────────────────────────────────
_DATA_ROOT = Path("data/7days1/processed_data")
_SELECTED_FILES = [
    "115.txt", "117.txt", "12.txt", "121.txt", "130.txt",
    "134.txt", "135.txt", "138.txt", "155.txt", "156.txt",
]

try:
    from src.real_data import load_trajectory
    _real_track_paths = []
    for name in _SELECTED_FILES:
        p = str(_DATA_ROOT / "test" / name)
        if Path(p).exists():
            _real_track_paths.append(p)
except Exception:
    _real_track_paths = []

_traj_cache: dict = {}   # path → positions; unika wielokrotnego czytania pliku


def _validate_data() -> str | None:
    """Sprawdza dostępność i format danych TrajAir. Zwraca komunikat błędu lub None."""
    if not _DATA_ROOT.exists():
        return (f"Brak katalogu danych:  {_DATA_ROOT}\n"
                f"Pobierz TrajAir i umieść w  data/7days1/processed_data/")
    if not (_DATA_ROOT / "test").exists():
        return f"Brak podkatalogu:  {_DATA_ROOT / 'test'}"
    if not _real_track_paths:
        return (f"Nie znaleziono plików tras w  {_DATA_ROOT / 'test'}\n"
                f"Oczekiwane: {', '.join(_SELECTED_FILES[:4])}, ...")
    try:
        sample = np.loadtxt(_real_track_paths[0])
        if sample.ndim < 2 or sample.shape[1] < 5:
            cols = sample.shape[1] if sample.ndim == 2 else "?"
            return (f"Nieprawidłowy format:  {Path(_real_track_paths[0]).name}\n"
                    f"Oczekiwano 7 kolumn (frame_id aircraft_id x y z vx vy), znaleziono {cols}")
    except Exception as exc:
        return f"Błąd odczytu danych:\n{exc}"
    return None


_data_error: str | None = _validate_data()


def _cached_load(path: str) -> np.ndarray:
    if path not in _traj_cache:
        _traj_cache[path] = load_trajectory(path, max_len=500)
    return _traj_cache[path]


def _copy_to_clipboard(text: str) -> None:
    try:
        root = tk.Tk(); root.withdraw()
        root.clipboard_clear(); root.clipboard_append(text)
        root.update(); root.destroy()
    except Exception:
        pass


# ── Kolory ────────────────────────────────────────────────────────────────────
_LC_STEP = ( 85,  90, 110)
_LC_TRUE = (140, 145, 160)
_LC_EST  = (190, 115,  35)
_LC_PRED = (185,  58,  58)
_LC_TGT  = ( 65, 180,  90)
_LC_NUM  = (195, 200, 215)
_LC_ERR  = ( 85,  90, 110)
C_MODE_REAL  = ( 80, 200, 120)
C_MODE_SYNTH = (120, 150, 220)
C_FREEZE_COL = (220,  80,  80)   # kolor linii open-loop po freeze


def _blit_segs(surf, font, segs, x, y):
    for text, color in segs:
        s = font.render(text, False, color)
        surf.blit(s, (x, y))
        x += s.get_width()


# ── Transformacja z kamerą ────────────────────────────────────────────────────

def _make_cam_conv(bounds, fr, cam_zoom, cam_pan):
    xmin, xmax, ymin, ymax = bounds
    rw = (xmax - xmin) or 1.0
    rh = (ymax - ymin) or 1.0
    base_scale = min(fr.w / rw, fr.h / rh)
    scale = base_scale * cam_zoom
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    ox = fr.centerx + cam_pan[0]
    oy = fr.centery + cam_pan[1]

    def conv(pt):
        return (int(ox + (pt[0] - cx) * scale),
                int(oy - (pt[1] - cy) * scale))

    def inv_conv(sx, sy):
        return np.array([(sx - ox) / scale + cx,
                         -(sy - oy) / scale + cy])

    return conv, inv_conv


# ── Uruchamianie symulacji ─────────────────────────────────────────────────────

def _run_sim(sigma, q, horizon, p_dropout=0.0, p_outlier=0.0,
             freeze_from_step=None, resume_from_step=None):
    res = run_simulation(N, DT, sigma, q, int(horizon), seed=SEED,
                         p_dropout=p_dropout, p_outlier=p_outlier,
                         freeze_from_step=freeze_from_step,
                         resume_from_step=resume_from_step)
    all_x = np.concatenate([res["states"][:, 0], res["measurements"][:, 0],
                             res["estimates"][:, 0], res["predictions"][:, 0]])
    all_y = np.concatenate([res["states"][:, 1], res["measurements"][:, 1],
                             res["estimates"][:, 1], res["predictions"][:, 1]])
    pad = max((all_x.max() - all_x.min()), (all_y.max() - all_y.min())) * 0.09
    bounds = (all_x.min() - pad, all_x.max() + pad,
              all_y.min() - pad, all_y.max() + pad)
    return res, bounds


def _run_sim_real(horizon, track_idx,
                  sigma=SIGMA_REAL, q=Q_REAL, p_dropout=0.0, p_outlier=0.0,
                  freeze_from_step=None, resume_from_step=None):
    pos = _cached_load(_real_track_paths[track_idx])
    res = run_simulation_on_track(pos, dt=DT, sigma=sigma, q=q,
                                  horizon=int(horizon), seed=SEED,
                                  p_dropout=p_dropout, p_outlier=p_outlier,
                                  freeze_from_step=freeze_from_step,
                                  resume_from_step=resume_from_step)
    all_x = np.concatenate([res["states"][:, 0], res["measurements"][:, 0],
                             res["estimates"][:, 0], res["predictions"][:, 0]])
    all_y = np.concatenate([res["states"][:, 1], res["measurements"][:, 1],
                             res["estimates"][:, 1], res["predictions"][:, 1]])
    pad = max((all_x.max() - all_x.min()), (all_y.max() - all_y.min())) * 0.09
    bounds = (all_x.min() - pad, all_x.max() + pad,
              all_y.min() - pad, all_y.max() + pad)
    return res, bounds


def _field_rect(show_logs):
    field_h = H - TOOLBAR_H - (LOG_H if show_logs else 0) - 2 * MARGIN_Y
    return pygame.Rect(MARGIN_X, TOOLBAR_H + MARGIN_Y,
                       W - 2 * MARGIN_X, field_h)


def _make_font(size, bold=False):
    for name in ("Consolas", "Courier New", "Lucida Console", None):
        try:
            return pygame.font.SysFont(name, size, bold=bold)
        except Exception:
            pass
    return pygame.font.SysFont(None, size, bold=bold)


# ── Główna pętla ──────────────────────────────────────────────────────────────

def main() -> None:
    pygame.init()
    surf  = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Filtr Kalmana — symulacja interaktywna")
    clock = pygame.time.Clock()

    f_big   = _make_font(19, bold=True)
    f_med   = _make_font(15)
    f_sm    = _make_font(13)
    f_log   = _make_font(13)
    f_label = _make_font(13)
    f_xs    = _make_font(11)

    # ── Suwaki — po prawej, większe odstępy (start x=625, co 130px) ──────────
    sl_sigma   = Slider( 625, 34, 105, "sigma",    0.3,  8.0, SIGMA_DEF,    fmt="{:.1f}")
    sl_q       = Slider( 755, 34, 105, "q",        0.01, 0.50, Q_DEF,       fmt="{:.3f}")
    sl_horizon = Slider( 885, 34, 105, "horyzont", 1,    20,   HORIZON_DEF, step=1, fmt="{:.0f}")
    sl_speed   = Slider(1015, 34, 105, "predkosc", 1,    30,   SPEED_DEF,   step=1, fmt="{:.0f}")
    sl_dropout = Slider(1150, 34,  95, "drop%",    0.0,  0.5,  P_DROPOUT_DEF, fmt="{:.2f}")
    sl_outlier = Slider(1265, 34,  95, "spike%",   0.0,  0.5,  P_OUTLIER_DEF, fmt="{:.2f}")
    sliders = [sl_sigma, sl_q, sl_horizon, sl_speed, sl_dropout, sl_outlier]

    # ── Przyciski ─────────────────────────────────────────────────────────────
    btn_toggle = Button( 12, 55, 110, 28, "Start [SPC]")
    btn_reset  = Button(132, 55, 100, 28, "Reset  [R]")
    btn_hud    = Button(242, 55, 110, 28, "Logi   [H]")
    btn_mode   = Button(362, 55, 110, 28, "Synth  [M]")
    btn_freeze = Button(482, 55, 130, 28, "Pomiary    [F]")
    btn_prev   = Button(624, 55,  62, 28, "< [←]")
    btn_next   = Button(696, 55,  62, 28, "[→] >")
    buttons = [btn_toggle, btn_reset, btn_hud, btn_mode, btn_freeze, btn_prev, btn_next]

    # ── Stan aplikacji ─────────────────────────────────────────────────────────
    mode          = "synth"
    real_idx      = 0
    freeze_state  = "none"   # "none" | "frozen" | "recovered"
    freeze_step   = 0        # krok w którym naciśnięto F (zamrożenie)
    recover_step  = 0        # krok w którym naciśnięto F ponownie (wznowienie)
    cam_pan       = np.array([0.0, 0.0])
    cam_zoom     = 1.0
    field_pan_active = False
    pan_start_pos = (0, 0)
    pan_start_cam = np.array([0.0, 0.0])

    # Selekcja wierszy logów
    sel_rows: set = set()
    sel_dragging  = False
    sel_row_start = -1

    n_steps = N

    # Trzy symulacje: pełna (base), po freeze (open-loop), po odzysku (przerwa+wznowienie)
    sim_base, bounds = _run_sim(sl_sigma.value, sl_q.value, sl_horizon.value,
                                sl_dropout.value, sl_outlier.value)
    sim_freeze     = None   # obliczany po naciśnięciu F (zamrożenie)
    sim_recovered  = None   # obliczany po naciśnięciu F po raz drugi (wznowienie)

    step       = 0
    playing    = False
    facc       = 0.0
    show_logs  = True
    log_lines: list = []
    prev_step  = -1
    log_scroll = 0

    def do_reset(reset_cam=True):
        nonlocal sim_base, bounds, sim_freeze, sim_recovered
        nonlocal freeze_state, freeze_step, recover_step
        nonlocal step, playing, facc, log_lines, prev_step, log_scroll, n_steps
        nonlocal cam_zoom
        if mode == "real" and not _data_error and _real_track_paths:
            sim_base, bounds = _run_sim_real(sl_horizon.value, real_idx,
                                             sigma=sl_sigma.value, q=sl_q.value,
                                             p_dropout=sl_dropout.value,
                                             p_outlier=sl_outlier.value)
        else:
            sim_base, bounds = _run_sim(sl_sigma.value, sl_q.value, sl_horizon.value,
                                        sl_dropout.value, sl_outlier.value)
        sim_freeze = None; sim_recovered = None
        freeze_state = "none"; freeze_step = 0; recover_step = 0
        n_steps = len(sim_base["states"])
        step = 0; playing = False; facc = 0.0
        log_lines = []; prev_step = -1; log_scroll = 0
        if reset_cam:
            cam_pan[:] = 0.0
            cam_zoom = 1.0

    def activate_freeze():
        """none → frozen: zamraża pomiary od bieżącego kroku."""
        nonlocal sim_freeze, freeze_step, freeze_state
        freeze_step  = step
        freeze_state = "frozen"
        if mode == "real" and not _data_error and _real_track_paths:
            sim_freeze, _ = _run_sim_real(sl_horizon.value, real_idx,
                                          sigma=sl_sigma.value, q=sl_q.value,
                                          p_dropout=sl_dropout.value,
                                          p_outlier=sl_outlier.value,
                                          freeze_from_step=freeze_step)
        else:
            sim_freeze, _ = _run_sim(sl_sigma.value, sl_q.value, sl_horizon.value,
                                     sl_dropout.value, sl_outlier.value,
                                     freeze_from_step=freeze_step)

    def activate_recover():
        """frozen → recovered: wznawia pomiary od NASTĘPNEGO kroku (step+1).
        Dzięki temu dron nie teleportuje się w momencie naciśnięcia F —
        sim_recovered[step] == sim_freeze[step] bo krok 'step' jeszcze bez pomiaru."""
        nonlocal sim_recovered, recover_step, freeze_state
        recover_step = step + 1
        if recover_step >= n_steps:   # naciśnięto F na ostatnim kroku — brak miejsca na recovery
            deactivate_all()
            return
        freeze_state = "recovered"
        if mode == "real" and not _data_error and _real_track_paths:
            sim_recovered, _ = _run_sim_real(sl_horizon.value, real_idx,
                                             sigma=sl_sigma.value, q=sl_q.value,
                                             p_dropout=sl_dropout.value,
                                             p_outlier=sl_outlier.value,
                                             freeze_from_step=freeze_step,
                                             resume_from_step=recover_step)
        else:
            sim_recovered, _ = _run_sim(sl_sigma.value, sl_q.value, sl_horizon.value,
                                        sl_dropout.value, sl_outlier.value,
                                        freeze_from_step=freeze_step,
                                        resume_from_step=recover_step)

    def deactivate_all():
        """recovered → none: powrót do stanu bazowego."""
        nonlocal sim_freeze, sim_recovered, freeze_state, freeze_step, recover_step
        sim_freeze = None; sim_recovered = None
        freeze_state = "none"; freeze_step = 0; recover_step = 0

    # ── Główna pętla ──────────────────────────────────────────────────────────
    running = True
    while running:
        dt_s = clock.tick(FPS) / 1000.0
        need_reset = False

        fr = _field_rect(show_logs)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            # ── Klawiatura ────────────────────────────────────────────────────
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    playing = not playing
                if ev.key == pygame.K_r:
                    do_reset(reset_cam=True)
                if ev.key == pygame.K_h:
                    show_logs = not show_logs
                if ev.key == pygame.K_m:
                    mode = "real" if mode == "synth" else "synth"
                    do_reset()
                if ev.key == pygame.K_f:
                    if freeze_state == "none":
                        activate_freeze()
                    elif freeze_state == "frozen":
                        activate_recover()
                    else:
                        deactivate_all()
                if ev.key == pygame.K_LEFT and mode == "real" and _real_track_paths:
                    real_idx = (real_idx - 1) % len(_real_track_paths)
                    do_reset()
                if ev.key == pygame.K_RIGHT and mode == "real" and _real_track_paths:
                    real_idx = (real_idx + 1) % len(_real_track_paths)
                    do_reset()

                # Ctrl+C: kopiuj logi
                if ev.key == pygame.K_c and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    if show_logs and log_lines:
                        start_i = max(0, len(log_lines) - LOG_VISIBLE - log_scroll)
                        if sel_rows:
                            to_copy = [log_lines[i] for i in sorted(sel_rows)
                                       if 0 <= i < len(log_lines)]
                        else:
                            to_copy = log_lines[start_i: start_i + LOG_VISIBLE]
                        lines_txt = []
                        for s_i, sp, ep, pp, tgt, err in to_copy:
                            tgt_str = f"  cel:({tgt[0]:.1f},{tgt[1]:.1f})" if tgt is not None else ""
                            lines_txt.append(
                                f"[k={s_i+1:03d}] true:({sp[0]:.1f},{sp[1]:.1f})"
                                f"  est:({ep[0]:.1f},{ep[1]:.1f})"
                                f"  pred:({pp[0]:.1f},{pp[1]:.1f})"
                                f"{tgt_str}  err:{err:.2f}"
                            )
                        _copy_to_clipboard("\n".join(lines_txt))

            # ── Scroll ────────────────────────────────────────────────────────
            if ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if fr.collidepoint(mx, my):
                    conv_now, inv_now = _make_cam_conv(bounds, fr, cam_zoom, cam_pan)
                    world_pt = inv_now(mx, my)
                    factor = 1.15 if ev.y > 0 else 1.0 / 1.15
                    cam_zoom = max(0.15, min(12.0, cam_zoom * factor))
                    conv_new, _ = _make_cam_conv(bounds, fr, cam_zoom, cam_pan)
                    new_sx, new_sy = conv_new(world_pt)
                    cam_pan[0] += mx - new_sx
                    cam_pan[1] += my - new_sy
                elif show_logs and my > H - LOG_H:
                    log_scroll += ev.y
                    max_sc = max(0, len(log_lines) - LOG_VISIBLE)
                    log_scroll = max(0, min(log_scroll, max_sc))

            # ── Mysz: przycisk w dół ──────────────────────────────────────────
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                if fr.collidepoint(mx, my):
                    field_pan_active = True
                    pan_start_pos = ev.pos
                    pan_start_cam = cam_pan.copy()
                elif show_logs and my > H - LOG_H:
                    log_y_ = H - LOG_H
                    row = (my - (log_y_ + 26)) // 15
                    start_i = max(0, len(log_lines) - LOG_VISIBLE - log_scroll)
                    abs_row = start_i + row
                    if 0 <= abs_row < len(log_lines):
                        sel_dragging = True
                        sel_row_start = abs_row
                        sel_rows = {abs_row}
                else:
                    sel_rows = set()

            # ── Mysz: ruch ────────────────────────────────────────────────────
            if ev.type == pygame.MOUSEMOTION:
                mx, my = ev.pos
                if field_pan_active:
                    cam_pan[0] = pan_start_cam[0] + (mx - pan_start_pos[0])
                    cam_pan[1] = pan_start_cam[1] + (my - pan_start_pos[1])
                if sel_dragging and show_logs and my > H - LOG_H:
                    log_y_ = H - LOG_H
                    row = (my - (log_y_ + 26)) // 15
                    start_i = max(0, len(log_lines) - LOG_VISIBLE - log_scroll)
                    abs_row = max(0, min(len(log_lines) - 1, start_i + row))
                    lo = min(sel_row_start, abs_row)
                    hi = max(sel_row_start, abs_row)
                    sel_rows = set(range(lo, hi + 1))

            # ── Mysz: przycisk w górę ─────────────────────────────────────────
            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                field_pan_active = False
                sel_dragging = False

            # ── Suwaki ────────────────────────────────────────────────────────
            for sl in sliders:
                changed = sl.handle_event(ev)
                if changed and sl is not sl_speed:
                    need_reset = True

            # ── Przyciski ─────────────────────────────────────────────────────
            if btn_toggle.handle_event(ev) and step < n_steps - 1:
                playing = not playing
            if btn_reset.handle_event(ev):
                do_reset(reset_cam=True)
            if btn_hud.handle_event(ev):
                show_logs = not show_logs
            if btn_mode.handle_event(ev):
                mode = "real" if mode == "synth" else "synth"
                do_reset()
            if btn_freeze.handle_event(ev):
                if freeze_state == "none":
                    activate_freeze()
                elif freeze_state == "frozen":
                    activate_recover()
                else:
                    deactivate_all()
            if btn_prev.handle_event(ev) and mode == "real" and _real_track_paths:
                real_idx = (real_idx - 1) % len(_real_track_paths)
                do_reset()
            if btn_next.handle_event(ev) and mode == "real" and _real_track_paths:
                real_idx = (real_idx + 1) % len(_real_track_paths)
                do_reset()

        if need_reset:
            do_reset(reset_cam=False)

        # Krok animacji
        if playing and step < n_steps - 1:
            facc += sl_speed.value * dt_s
            while facc >= 1.0 and step < n_steps - 1:
                step += 1; facc -= 1.0
        if step >= n_steps - 1:
            playing = False

        k   = step + 1
        hor = int(sl_horizon.value)

        # Aktywne dane do wyświetlenia
        states  = sim_base["states"]
        meas    = sim_base["measurements"]
        valid_m = sim_base["valid"]
        outlier_m = sim_base["outlier_mask"]

        # Estymaty i predykcje zależne od stanu freeze
        if freeze_state == "recovered" and sim_recovered is not None:
            est   = sim_recovered["estimates"]
            preds = sim_recovered["predictions"]
        elif freeze_state == "frozen" and sim_freeze is not None:
            est   = sim_freeze["estimates"]
            preds = sim_freeze["predictions"]
        else:
            est   = sim_base["estimates"]
            preds = sim_base["predictions"]

        # Log wiersz
        if step != prev_step:
            sp  = states[step, :2]
            ep  = est[step, :2]
            pp  = preds[step]
            tgt = states[step + hor, :2].copy() if step + hor < n_steps else None
            err = float(np.linalg.norm(ep - sp))
            log_lines.append((step, sp.copy(), ep.copy(), pp.copy(), tgt, err))
            prev_step = step

        # Etykiety przycisków
        btn_toggle.label = "Stop  [SPC]" if playing else "Start [SPC]"
        btn_mode.label   = "Real   [M]" if mode == "synth" else "Synth  [M]"
        if freeze_state == "frozen":
            btn_freeze.label = "Wznów pom. [F]"
        elif freeze_state == "recovered":
            btn_freeze.label = "Resetuj    [F]"
        else:
            btn_freeze.label = "Zamróź pom.[F]"

        fr = _field_rect(show_logs)
        conv, inv_conv = _make_cam_conv(bounds, fr, cam_zoom, cam_pan)

        # ── RYSOWANIE ─────────────────────────────────────────────────────────
        surf.fill(BG)

        # Toolbar
        pygame.draw.rect(surf, TOOLBAR_BG, pygame.Rect(0, 0, W, TOOLBAR_H))
        pygame.draw.line(surf, DIV, (0, TOOLBAR_H - 1), (W, TOOLBAR_H - 1))

        # Tytuł + tryb
        surf.blit(f_big.render("Filtr Kalmana", False, C_ACCENT), (12, 8))
        if freeze_state == "frozen":
            mode_col  = C_FREEZE_COL
            freeze_sfx = f"  [FREEZE od k={freeze_step+1}]"
        elif freeze_state == "recovered":
            mode_col  = (120, 200, 80)
            freeze_sfx = f"  [GAP k={freeze_step+1}..{recover_step-1}, wznowienie k={recover_step}]"
        else:
            mode_col  = C_MODE_REAL if mode == "real" else C_MODE_SYNTH
            freeze_sfx = ""
        mode_str   = ("REAL" if mode == "real" else "SYNTH") + freeze_sfx
        surf.blit(f_sm.render(f"[{mode_str}]", False, mode_col), (12, 34))

        if mode == "real" and _real_track_paths:
            track_name = Path(_real_track_paths[real_idx]).stem
            mode_w = f_sm.size(f"[{mode_str}]")[0]
            surf.blit(f_sm.render(f"{real_idx+1}/{len(_real_track_paths)}  ({track_name})",
                                  False, C_MODE_REAL), (12 + mode_w + 8, 34))

        # ── Panel logów ────────────────────────────────────────────────────────
        if show_logs:
            log_y = H - LOG_H
            pygame.draw.rect(surf, LOG_BG, pygame.Rect(0, log_y, W, LOG_H))
            pygame.draw.line(surf, DIV, (0, log_y), (W, log_y))
            total_logs = len(log_lines)
            max_sc = max(0, total_logs - LOG_VISIBLE)
            log_scroll = min(log_scroll, max_sc)

            lbl_logi = f"LOGI [{total_logs}]:" if total_logs > LOG_VISIBLE else "LOGI:"
            surf.blit(f_sm.render(lbl_logi, False, C_DIM), (12, log_y + 7))

            hint_parts = ["zaznacz myszą · Ctrl+C: kopiuj"]
            if max_sc > 0:
                hint_parts.append(f"kółko: przewiń [{max_sc - log_scroll}/{max_sc}]")
            hint = "   ".join(hint_parts)
            hint_surf = f_sm.render(hint, False, C_DIM)
            surf.blit(hint_surf, (W - hint_surf.get_width() - 12, log_y + 7))

            start_idx = max(0, total_logs - LOG_VISIBLE - log_scroll)
            visible   = log_lines[start_idx: start_idx + LOG_VISIBLE]

            for i, entry in enumerate(visible):
                s_i, sp, ep, pp, tgt, err = entry
                abs_row = start_idx + i
                hi_row  = (i == len(visible) - 1)

                if abs_row in sel_rows:
                    hl = pygame.Surface((W, 15), pygame.SRCALPHA)
                    hl.fill((70, 120, 210, 60))
                    surf.blit(hl, (0, log_y + 26 + i * 15))

                segs = [
                    (f"[k={s_i+1:03d}] ", C_ACCENT if hi_row else _LC_STEP),
                    ("true:(", _LC_TRUE),
                    (f"{sp[0]:6.1f},{sp[1]:5.1f})  ", _LC_NUM),
                    ("estymacja:(", _LC_EST),
                    (f"{ep[0]:6.1f},{ep[1]:5.1f})  ", _LC_NUM),
                    ("predykcja:(", _LC_PRED),
                    (f"{pp[0]:6.1f},{pp[1]:5.1f})", _LC_NUM),
                ]
                if tgt is not None:
                    segs += [("  cel:(", _LC_TGT),
                              (f"{tgt[0]:6.1f},{tgt[1]:5.1f})", _LC_NUM)]
                segs.append((f"  err:{err:.2f}", _LC_ERR))
                _blit_segs(surf, f_log, segs, 12, log_y + 26 + i * 15)

            re_val = rmse(est[:k, :2], states[:k, :2]) if k >= 2 else 0.0
            me_val = mae(est[:k, :2],  states[:k, :2]) if k >= 2 else 0.0
            rp     = sim_base["rmse_pred"]
            mp     = sim_base["mae_pred"]
            summary = (f"  RMSE_est:{re_val:.3f}   MAE_est:{me_val:.3f}"
                       f"   RMSE_pred:{rp:.3f}   MAE_pred:{mp:.3f}"
                       f"   krok: {step + 1} / {n_steps}")
            surf.blit(f_sm.render(summary, False, C_TEXT), (12, H - 18))

        # ── Pole symulacji ─────────────────────────────────────────────────────
        surf.set_clip(fr)
        draw_grid(surf, conv, bounds, fr)

        # Pełna (przyciemniona) trasa w tle
        draw_polyline(surf, C_TRUE_DIM,
                      [conv(states[i, :2]) for i in range(n_steps)], 1)
        # Odkryta trasa do bieżącego kroku
        if k >= 2:
            draw_polyline(surf, C_TRUE,
                          [conv(states[i, :2]) for i in range(k)], 2)

        # Pomiary — wg stanu freeze
        if freeze_state == "none":
            meas_ranges = [(0, k)]
        elif freeze_state == "frozen":
            meas_ranges = [(0, freeze_step)]
        else:  # recovered
            meas_ranges = [(0, freeze_step), (recover_step, k)]
        for r0, r1 in meas_ranges:
            for i in range(r0, r1):
                if valid_m[i]:
                    col = C_OUTLIER if outlier_m[i] else C_MEAS
                    pygame.draw.circle(surf, col, conv(meas[i]), 4, 1)

        # Estymacja FK — wg stanu freeze
        if freeze_state == "none":
            if k >= 2:
                draw_polyline(surf, C_EST, [conv(est[i, :2]) for i in range(k)], 2)
        elif freeze_state == "frozen" and sim_freeze is not None:
            normal_end = min(k, freeze_step + 1)
            if normal_end >= 2:
                draw_polyline(surf, C_EST,
                              [conv(sim_base["estimates"][i, :2])
                               for i in range(normal_end)], 2)
            if freeze_step < k:
                ol_pts = [conv(sim_freeze["estimates"][i, :2])
                          for i in range(freeze_step, k)]
                if len(ol_pts) >= 2:
                    draw_polyline(surf, C_FREEZE_COL, ol_pts, 2)
                pygame.draw.circle(surf, C_FREEZE_COL,
                                   conv(states[freeze_step, :2]), 7, 2)
        elif freeze_state == "recovered" and sim_recovered is not None:
            # 1. Pomarańczowa przed freeze_step
            normal_end = min(k, freeze_step + 1)
            if normal_end >= 2:
                draw_polyline(surf, C_EST,
                              [conv(sim_base["estimates"][i, :2])
                               for i in range(normal_end)], 2)
            # 2. Czerwona przerywana GAP: freeze_step..recover_step-1
            #    recover_step jest pierwszym krokiem Z pomiarem → gap kończy się na recover_step-1
            gap_draw_end = min(k, recover_step)   # range(freeze_step, gap_draw_end) → ..recover_step-1
            if freeze_step < gap_draw_end:
                p1 = conv(sim_freeze["estimates"][freeze_step, :2])
                p2 = conv(sim_freeze["estimates"][gap_draw_end - 1, :2])
                draw_dashed(surf, C_FREEZE_COL, p1, p2)
            # 3. Pomarańczowa po wznowieniu, zaczynając od końca open-loop (ciągłość wizualna)
            #    Łącznik: sim_freeze[recover_step-1] → sim_recovered[recover_step..]
            if recover_step <= k:
                conn = conv(sim_freeze["estimates"][recover_step - 1, :2])
                body = [conv(sim_recovered["estimates"][i, :2])
                        for i in range(recover_step, k)]
                rec_pts = [conn] + body
                if len(rec_pts) >= 2:
                    draw_polyline(surf, C_EST, rec_pts, 2)
            # Markery zdarzeń
            pygame.draw.circle(surf, C_FREEZE_COL,
                               conv(states[freeze_step, :2]), 7, 2)
            if recover_step < n_steps:
                pygame.draw.circle(surf, (80, 220, 90),
                                   conv(states[recover_step, :2]), 7, 2)

        # Predykcja (strzałka)
        est_now  = conv(est[step, :2])
        pred_now = conv(preds[step])
        draw_dashed(surf, C_PRED, est_now, pred_now)
        pygame.draw.circle(surf, C_PRED, pred_now, 8, 2)

        if step + hor < n_steps:
            draw_diamond(surf, conv(states[step + hor, :2]))

        if freeze_state == "none":
            drone_col = C_EST
        elif freeze_state == "frozen":
            drone_col = C_FREEZE_COL
        else:  # recovered — pomiary wróciły
            drone_col = C_EST
        draw_drone(surf, est_now, drone_col)

        # Etykieta freeze
        if freeze_state == "frozen":
            fr_lbl = f_med.render(f"FREEZE od k={freeze_step+1}  —  FK open-loop", False, C_FREEZE_COL)
            surf.blit(fr_lbl, (fr.centerx - fr_lbl.get_width() // 2, fr.top + 8))
        elif freeze_state == "recovered":
            fr_lbl = f_med.render(
                f"GAP k={freeze_step+1}..{recover_step-1}  |  wznowienie k={recover_step}",
                False, (80, 220, 90),
            )
            surf.blit(fr_lbl, (fr.centerx - fr_lbl.get_width() // 2, fr.top + 8))

        surf.set_clip(None)

        # ── Overlay błędu danych (tryb REAL bez danych) ───────────────────────
        if mode == "real" and _data_error:
            dim = pygame.Surface((fr.w, fr.h), pygame.SRCALPHA)
            dim.fill((18, 6, 6, 215))
            surf.blit(dim, fr.topleft)

            lines = [l for l in _data_error.split("\n") if l.strip()]
            box_w = fr.w - 120
            box_h = 56 + len(lines) * 22 + 28
            box_x = fr.centerx - box_w // 2
            box_y = fr.centery - box_h // 2

            pygame.draw.rect(surf, (50, 12, 12), (box_x, box_y, box_w, box_h), border_radius=8)
            pygame.draw.rect(surf, (210, 45, 45), (box_x, box_y, box_w, box_h), 2, border_radius=8)

            title = f_big.render("Brak danych TrajAir", False, (255, 65, 65))
            surf.blit(title, (fr.centerx - title.get_width() // 2, box_y + 12))

            for i, line in enumerate(lines):
                ls = f_sm.render(line, False, (215, 155, 155))
                surf.blit(ls, (fr.centerx - ls.get_width() // 2, box_y + 48 + i * 22))

            hint = f_xs.render(
                "Wróć do trybu SYNTH [M]  lub  pobierz dane — patrz README.md",
                False, (120, 100, 100),
            )
            surf.blit(hint, (fr.centerx - hint.get_width() // 2, box_y + box_h - 18))

        draw_axes(surf, conv, bounds, fr, f_xs)
        draw_legend_overlay(surf, fr, f_sm)

        # Zoom indicator
        if cam_zoom < 0.95 or cam_zoom > 1.05 or abs(cam_pan[0]) > 2 or abs(cam_pan[1]) > 2:
            zoom_lbl = f_xs.render(f"zoom {cam_zoom:.1f}x   R = reset widoku", False, C_DIM)
            surf.blit(zoom_lbl, (fr.left + 6, fr.top + 6))

        # ── Toolbar: suwaki i przyciski ────────────────────────────────────────
        for sl in sliders:
            sl.draw(surf, f_sm, f_label, compact=True)

        for btn in buttons:
            if btn in (btn_prev, btn_next) and mode == "synth":
                continue
            btn.draw(surf, f_med)

        if step >= n_steps - 1:
            surf.blit(f_med.render("KONIEC", False, C_DIM), (W - 80, 60))

        pygame.display.flip()

    pygame.quit()
    sys.exit()
