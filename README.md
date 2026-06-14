# Filtr Kalmana - śledzenie trajektorii lotniczych

Projekt dyplomowy implementujący liniowy filtr Kalmana (model CV 2D) do estymacji i predykcji pozycji statków powietrznych na podstawie danych ADS-B z datasetu TrajAir.

## Zawartość

```
Projekt/
├── src/                  # biblioteka FK
│   ├── kalman.py         # klasa KalmanFilter
│   ├── model.py          # macierze F, H, Q, R (model CV)
│   ├── simulation.py     # generowanie trajektorii i główna pętla FK
│   ├── real_data.py      # loader danych TrajAir
│   ├── metrics.py        # RMSE, MAE
│   └── viz.py            # funkcje wykresów do notebooka
├── sim/                  # interaktywna symulacja pygame
│   ├── app.py            # główna pętla aplikacji
│   ├── constants.py      # parametry domyślne, kolory
│   ├── ui.py             # suwaki, przyciski
│   └── renderer.py       # rysowanie trajektorii
├── tests/
│   └── test_kalman.py    # testy jednostkowe (pytest)
├── analysis.ipynb        # notebook z pełną analizą
├── start_simulation.py   # skrypt startowy symulacji
├── data/                 # dane TrajAir (pobierz osobno - patrz niżej)
└── results/              # wyeksportowane wykresy z notebooka
```

## Wymagania

Python 3.10+

```bash
pip install -r requirements.txt
```

## Dane TrajAir

Dataset **TrajAir** (Carnegie Mellon University) - trajektorie ADS-B samolotów wokół lotniska Pittsburgh-Butler Regional Airport (KBTP), wrzesień 2020.

1. Pobierz archiwum ze strony (wybierz `7days.zip`) 
   https://kilthub.cmu.edu/articles/dataset/TrajAir_A_General_Aviation_Trajectory_Dataset/14866251

2. Wypakuj i umieść folder `7days1` w katalogu `data/`:

```
data/
└── 7days1/
    └── processed_data/
        ├── train/
        │   ├── 1.txt
        │   └── ...
        └── test/
            ├── 1.txt
            └── ...
```

Każdy plik to trajektoria jednego samolotu w formacie:
```
frame_id  aircraft_id  x  y  z  vx  vy
```
Kolumny `x`, `y` to pozycja w jednostkach TrajAir (1 j ≈ 2,5 km).

## Uruchomienie symulacji interaktywnej

```bash
python start_simulation.py
```

### Sterowanie

| Klawisz / element | Akcja |
|---|---|
| `Spacja` | Start / pauza animacji |
| `R` | Reset symulacji |
| `M` | Przełącz tryb SYNTH ↔ REAL |
| `F` | Zamróź pomiary → wznów → resetuj |
| `←` / `→` | Poprzednia / następna trasa (tryb REAL) |
| `H` | Pokaż / ukryj panel logu |
| Scroll myszy | Zoom mapy |
| PPM + przeciągnij | Przesunięcie mapy |

### Suwaki (toolbar górny)

| Suwak | Opis |
|---|---|
| `sigma` | Odchylenie standardowe szumu pomiaru |
| `q` | Szum procesowy (dynamika modelu) |
| `horizon` | Horyzont predykcji (kroków) |
| `speed` | Prędkość animacji |
| `drop%` | Prawdopodobieństwo braku pomiaru (dropout) |
| `spike%` | Prawdopodobieństwo błędnego pomiaru (outlier) |

W trybie **REAL** wszystkie suwaki działają - sigma i q kontrolują parametry filtra na danych ADS-B, drop%/spike% symulują degradację łącza.

## Notebook analityczny

```bash
jupyter notebook analysis.ipynb
```

Lub w VS Code z rozszerzeniem Jupyter - otwórz `analysis.ipynb` i uruchom komórki sekwencyjnie (`Run All`).

Notebook zawiera:
- Sekcje 1–7: analiza wrażliwości FK na σ, q, horyzont predykcji, dropout, outliery, bramkowanie innowacji
- Sekcja 8: predykcja na trasie TrajAir (open-loop po N_OBS krokach)
- 

## Testy jednostkowe

```bash
python -m pytest tests/ -v
```

Testy sprawdzają poprawność FK: predykcję stanu, aktualizację P (symetria, dodatnia określoność), odrzucanie outlierów przez gating, zachowanie podczas przerwy w pomiarach.

## Model

Liniowy filtr Kalmana, model CV (constant velocity) 2D:

```
Stan:        x = [x, y, vx, vy]^T
Obserwacja:  z = [x_meas, y_meas]^T

Predykcja:   x̂⁻ = F x̂,    P⁻ = F P F^T + Q
Korekcja:    K   = P⁻ H^T (H P⁻ H^T + R)⁻¹
             x̂   = x̂⁻ + K (z - H x̂⁻)
             P   = (I - KH) P⁻ (I - KH)^T + K R K^T    [forma Josepha]
```
