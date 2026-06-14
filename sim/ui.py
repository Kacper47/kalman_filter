"""Klasy interfejsu użytkownika: Slider i Button."""

import pygame
from .constants import C_ACCENT, C_DIM, C_TEXT, DIV


class Slider:
    """
    Poziomy suwak z etykietą i wartością liczbową.

    Tryb compact=True (toolbar): etykieta + wartość na jednej linii nad suwaczkiem.
    Tryb compact=False (panel boczny): etykieta nad, wartość po prawej stronie.
    """

    def __init__(self, x: int, y: int, w: int, label: str,
                 lo: float, hi: float, default: float,
                 step: float | None = None, fmt: str = "{:.2f}"):
        self.track = pygame.Rect(x, y, w, 5)
        self.label = label
        self.lo, self.hi = lo, hi
        self.step = step
        self.fmt  = fmt
        self._val = float(default)
        self.drag = False

    @property
    def value(self) -> float:
        return self._val

    def _clamp(self, v: float) -> float:
        v = max(self.lo, min(self.hi, v))
        if self.step:
            v = round(v / self.step) * self.step
        return v

    def _t(self) -> float:
        return (self._val - self.lo) / (self.hi - self.lo)

    def _handle(self) -> pygame.Rect:
        hx = int(self.track.x + self._t() * self.track.w)
        return pygame.Rect(hx - 8, self.track.y - 7, 16, 19)

    def handle_event(self, ev: pygame.event.Event) -> bool:
        """Zwraca True jeśli wartość suwaka się zmieniła."""
        changed = False
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self._handle().inflate(14, 4).collidepoint(ev.pos):
                self.drag = True
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.drag = False
        if ev.type == pygame.MOUSEMOTION and self.drag:
            t = (ev.pos[0] - self.track.x) / self.track.w
            new = self._clamp(self.lo + t * (self.hi - self.lo))
            if new != self._val:
                self._val = new
                changed = True
        return changed

    def draw(self, surf: pygame.Surface, fsm: pygame.font.Font,
             flabel: pygame.font.Font, compact: bool = False) -> None:
        # Track tło
        pygame.draw.rect(surf, DIV, self.track, border_radius=3)
        # Fill (wypełnienie do aktualnej wartości)
        fill = pygame.Rect(self.track.x, self.track.y,
                           int(self.track.w * self._t()), self.track.h)
        if fill.w > 0:
            pygame.draw.rect(surf, C_ACCENT, fill, border_radius=3)
        # Uchwyt
        hr = self._handle()
        pygame.draw.rect(surf, C_ACCENT, hr, border_radius=4)
        pygame.draw.rect(surf, C_TEXT, hr, 1, border_radius=4)
        # Etykieta i wartość
        if compact:
            lbl = f"{self.label}: {self.fmt.format(self._val)}"
            surf.blit(flabel.render(lbl, False, C_DIM),
                      (self.track.x, self.track.y - 20))
        else:
            surf.blit(flabel.render(self.label, False, C_DIM),
                      (self.track.x, self.track.y - 22))
            surf.blit(fsm.render(self.fmt.format(self._val), False, C_ACCENT),
                      (self.track.right + 10, self.track.y - 5))


class Button:
    """Klikalne prostokątne pole z etykietą tekstową."""

    def __init__(self, x: int, y: int, w: int, h: int, label: str):
        self.rect  = pygame.Rect(x, y, w, h)
        self.label = label
        self.hover = False

    def handle_event(self, ev: pygame.event.Event) -> bool:
        if ev.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(ev.pos)
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect.collidepoint(ev.pos):
                return True
        return False

    def draw(self, surf: pygame.Surface, font: pygame.font.Font) -> None:
        col = C_ACCENT if self.hover else DIV
        pygame.draw.rect(surf, col,   self.rect, border_radius=6)
        pygame.draw.rect(surf, C_DIM, self.rect, 1, border_radius=6)
        txt = font.render(self.label, False, C_TEXT)
        surf.blit(txt, txt.get_rect(center=self.rect.center))
