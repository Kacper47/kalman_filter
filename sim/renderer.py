"""Funkcje rysowania dla symulacji pygame — trajektoria, dron, legenda."""

import math
from typing import Callable

import numpy as np
import pygame

_AXIS_COL = (88, 96, 115)

from .constants import (
    C_GRID, C_TRUE_DIM, C_MEAS, C_EST, C_PRED, C_FUTURE, C_DRONE, C_OUTLIER,
    C_TRUE, DIV,
)


# ── Transformacja współrzędnych ───────────────────────────────────────────────

def make_transform(bounds: tuple, field_rect: pygame.Rect) -> Callable:
    """
    Zwraca funkcję conv(pt) mapującą współrzędne symulacji → piksele ekranu.
    Oś Y jest odwrócona (ekran: y rośnie w dół; symulacja: y rośnie w górę).
    """
    xmin, xmax, ymin, ymax = bounds
    rw = (xmax - xmin) or 1
    rh = (ymax - ymin) or 1
    scale = min(field_rect.w / rw, field_rect.h / rh)
    ox = field_rect.x + (field_rect.w - rw * scale) / 2
    oy = field_rect.bottom - (field_rect.h - rh * scale) / 2

    def conv(pt) -> tuple[int, int]:
        return (int(ox + (pt[0] - xmin) * scale),
                int(oy - (pt[1] - ymin) * scale))
    return conv


# ── Rysowanie pola symulacji ──────────────────────────────────────────────────

def draw_grid(surf: pygame.Surface, conv: Callable,
              bounds: tuple, field_rect: pygame.Rect) -> None:
    xmin, xmax, ymin, ymax = bounds
    step_x = max(1, round((xmax - xmin) / 10))
    step_y = max(1, round((ymax - ymin) /  8))
    for x in range(int(xmin) - 1, int(xmax) + step_x + 1, step_x):
        pygame.draw.line(surf, C_GRID, conv((x, ymin)), conv((x, ymax)))
    for y in range(int(ymin) - 1, int(ymax) + step_y + 1, step_y):
        pygame.draw.line(surf, C_GRID, conv((xmin, y)), conv((xmax, y)))
    pygame.draw.rect(surf, DIV, field_rect, 1)


def draw_polyline(surf: pygame.Surface, color: tuple,
                  pts: list, width: int = 2) -> None:
    if len(pts) >= 2:
        pygame.draw.lines(surf, color, False, pts, width)


def draw_dashed(surf: pygame.Surface, color: tuple,
                p1: tuple, p2: tuple,
                dash: int = 10, gap: int = 5, width: int = 2) -> None:
    x1, y1 = p1; x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    pos, on = 0.0, True
    while pos < length:
        seg = dash if on else gap
        end = min(pos + seg, length)
        if on:
            pygame.draw.line(surf, color,
                             (int(x1 + ux * pos), int(y1 + uy * pos)),
                             (int(x1 + ux * end), int(y1 + uy * end)), width)
        pos, on = end, not on


# ── Ikony obiektów ────────────────────────────────────────────────────────────

def draw_drone(surf: pygame.Surface, pos: tuple,
               color: tuple = C_DRONE, size: int = 15) -> None:
    """Dron: 4 ramiona skośne (układ X) + koła rotorów + korpus."""
    cx, cy = int(pos[0]), int(pos[1])
    d = int(size * 0.72)
    for dx, dy in [(d, d), (-d, d), (d, -d), (-d, -d)]:
        pygame.draw.line(surf, color, (cx, cy), (cx + dx, cy + dy), 2)
        pygame.draw.circle(surf, color, (cx + dx, cy + dy), 5, 2)
    pygame.draw.circle(surf, color, (cx, cy), 4)
    pygame.draw.circle(surf, (230, 240, 255), (cx, cy), 2)


def draw_diamond(surf: pygame.Surface, pos: tuple,
                 size: int = 9, color: tuple = C_FUTURE, width: int = 2) -> None:
    """Romb zaznaczający prawdziwą przyszłą pozycję (cel predykcji)."""
    cx, cy = int(pos[0]), int(pos[1])
    pygame.draw.polygon(surf, color,
        [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)],
        width)


# ── Osie numeryczne ───────────────────────────────────────────────────────────

def draw_axes(surf: pygame.Surface, conv: Callable,
              bounds: tuple, field_rect: pygame.Rect,
              font: pygame.font.Font) -> None:
    """Etykiety numeryczne osi X (pod polem) i Y (po lewej stronie pola)."""
    xmin, xmax, ymin, ymax = bounds
    step_x = max(1, round((xmax - xmin) / 10))
    step_y = max(1, round((ymax - ymin) / 8))

    start_x = math.ceil(xmin / step_x) * step_x
    start_y = math.ceil(ymin / step_y) * step_y

    # Etykiety osi X — tuż pod dolną krawędzią pola
    xi = start_x
    while xi <= xmax + step_x:
        px, _ = conv((xi, ymin))
        if field_rect.left + 2 <= px <= field_rect.right - 2:
            lbl = font.render(str(int(xi)), False, _AXIS_COL)
            surf.blit(lbl, (px - lbl.get_width() // 2, field_rect.bottom + 2))
        xi += step_x

    # Etykiety osi Y — tuż na lewo od lewej krawędzi pola
    yi = start_y
    while yi <= ymax + step_y:
        _, py = conv((xmin, yi))
        if field_rect.top + 2 <= py <= field_rect.bottom - 2:
            lbl = font.render(str(int(yi)), False, _AXIS_COL)
            surf.blit(lbl, (field_rect.left - lbl.get_width() - 4, py - lbl.get_height() // 2))
        yi += step_y


# ── Legenda overlay ───────────────────────────────────────────────────────────

def draw_legend_overlay(surf: pygame.Surface, field_rect: pygame.Rect,
                        font: pygame.font.Font) -> None:
    """Mini-legenda semi-transparent w prawym dolnym rogu pola symulacji."""
    items = [
        (C_TRUE,    "Trajektoria rzeczywista"),
        (C_MEAS,    "Pomiary sensora"),
        (C_OUTLIER, "Pomiar z outlierem"),
        (C_EST,     "Estymacja FK / dron"),
        (C_PRED,    "Predykcja FK"),
        (C_FUTURE,  "Cel predykcji (+n krokow)"),
    ]
    pad    = 8
    line_h = 20
    leg_w  = 196
    leg_h  = len(items) * line_h + 2 * pad

    leg = pygame.Surface((leg_w, leg_h), pygame.SRCALPHA)
    leg.fill((10, 12, 20, 178))

    for i, (color, label) in enumerate(items):
        y = pad + i * line_h
        pygame.draw.circle(leg, color, (10, y + 8), 5)
        leg.blit(font.render(label, False, (182, 188, 202)), (23, y + 1))

    surf.blit(leg, (field_rect.right - leg_w - 8,
                    field_rect.bottom - leg_h - 8))
