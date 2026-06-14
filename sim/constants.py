"""Stałe konfiguracyjne aplikacji symulacyjnej filtru Kalmana."""

# ── Okno ──────────────────────────────────────────────────────────────────────
W          = 1400
H          = 750
TOOLBAR_H  = 90      # pasek górny: suwaki + przyciski
LOG_H      = 185     # panel dolny: logi lotnicze
MARGIN_X   = 30      # margines poziomy pola symulacji
MARGIN_Y   = 14      # margines pionowy pola symulacji

FPS        = 60

# ── Kolory (ciemny motyw) ─────────────────────────────────────────────────────
BG         = ( 12,  14,  22)
TOOLBAR_BG = ( 17,  19,  31)
LOG_BG     = ( 14,  16,  26)
DIV        = ( 42,  46,  68)
C_GRID     = ( 26,  29,  46)

C_TRUE     = (200, 205, 220)   # trajektoria rzeczywista (odkryta)
C_TRUE_DIM = ( 35,  39,  55)   # trajektoria rzeczywista (tło)
C_MEAS     = ( 65, 125, 255)   # pomiary sensora
C_EST      = (255, 150,  40)   # estymacja FK
C_PRED     = (255,  65,  65)   # predykcja FK
C_FUTURE   = (100, 255, 130)   # prawdziwa przyszła pozycja (cel)
C_DRONE    = (  0, 230, 135)   # dron (bieżąca estymacja)
C_OUTLIER  = (255, 185,   0)   # marker pomiaru z outlierem

C_TEXT     = (195, 200, 215)
C_ACCENT   = ( 75, 165, 255)
C_DIM      = ( 85,  90, 110)
C_GREEN    = ( 80, 220,  90)
C_RED      = (255,  80,  80)

# ── Domyślne parametry symulacji ──────────────────────────────────────────────
N           = 100      # liczba kroków (dłuższa trajektoria)
DT          = 1.0      # krok czasowy [s]
SEED        = 42
SIGMA_DEF   = 2.0
Q_DEF       = 0.05
HORIZON_DEF = 10
SPEED_DEF      = 2        # kroków animacji na sekundę
P_DROPOUT_DEF  = 0.0
P_OUTLIER_DEF  = 0.0

LOG_VISIBLE = 9        # liczba widocznych wierszy logu

# ── Stałe parametry trybu Real (TrajAir: 1 j ≈ 2.5 km) ──────────────────────
SIGMA_REAL = 0.02     # ≈ 50 m szum ADS-B (nie zmienia się suwakiem)
Q_REAL     = 0.005    # mała dynamika - stabilny lot samolotu
