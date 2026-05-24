# LCK Match Prediction Model

A machine learning model that predicts the winner of LCK (League of Legends Champions Korea) matches — including Bo3 and Bo5 series — based on historical team and player performance data.

## Overview

The model combines multiple data sources to generate win probabilities:

- **Team stats**: Gold diff, objectives, vision, kill participation (2025 + 2026)
- **Player stats**: KDA, DPM, CS diff per player — with Bayesian blending between player-specific and general LCK stats
- **Champion stats**: Win rate, scaling score, early/late game tendency per champion
- **Contextual features**: Recent form (last 5 games), Blue/Red side win rate, Head-to-Head history

Trained on 2026 LCK season data (Cup + Rounds 1–2) with 2025 data as time-decayed context (40% weight).

**Current accuracy: ~65.8%** per game | **~68.1%** per Bo3 series (5-fold cross-validation, XGBoost)

## Project Structure

```
lck-prediction/
├── run_pipeline.py       # Download data + run full pipeline (start here)
├── src/
│   ├── preprocessing.py  # Filter and clean raw data
│   ├── features.py       # Team-level features (form, side WR, H2H)
│   ├── player_features.py# Player + champion features with Bayesian blending
│   ├── champion_stats.py # LCK-wide champion stats (scaling, carry, early)
│   ├── model.py          # Train per-game model (XGBoost + CV)
│   ├── model_series.py   # Train Bo3/Bo5 series model
│   └── predict.py        # Predict a matchup
├── models/               # Saved models (.pkl) — generated, not in git
├── data/                 # CSV data — generated, not in git
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone git@github.com:Lukas2023h/lck-prediction.git
cd lck-prediction

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Setup & Data

Data is **not included in the repository** (large files). Run the pipeline once to download everything automatically:

```bash
python run_pipeline.py
```

This downloads the latest LCK data from Google Drive and runs the full pipeline:
1. Preprocessing → Team features → Player features → Champion stats
2. Trains the per-game model (XGBoost with hyperparameter tuning)
3. Trains the series model (Bo3/Bo5)

To update the model when new games are available, just run `run_pipeline.py` again.

## Predicting a Matchup

Edit the bottom of `src/predict.py` and run:

```bash
python src/predict.py
```

**Team-only prediction:**
```python
predict_matchup("T1", "Gen.G")
```

**With roster and champion picks:**
```python
predict_matchup(
    "T1", "Gen.G",
    roster_blue    = ["Doran", "Oner", "Faker", "Peyz", "Keria"],
    roster_red     = ["Doran", "Kanavi", "Chovy", "Peyz", "Delight"],
    champions_blue = ["Gnar", "Vi", "Ahri", "Jinx", "Thresh"],
    champions_red  = ["Renekton", "Xin Zhao", "Orianna", "Ezreal", "Nautilus"],
)
```

Output includes single game, Bo3 and Bo5 win probabilities:
```
============================================
  T1 (Blue)  vs  Gen.G (Red)
  [mit Spieler- & Champion-Daten]
============================================
  Einzelspiel:  T1 61.5%  |  Gen.G 38.5%
  Bo3-Serie:    T1 71.7%  |  Gen.G 28.3%
  Bo5-Serie:    T1 77.1%  |  Gen.G 22.9%
────────────────────────────────────────────
  Vorhersage: T1 gewinnt
============================================
```

**Available teams:** `T1`, `Gen.G`, `Hanwha Life Esports`, `KT Rolster`, `Dplus Kia`, `BNK FEARX`, `Kiwoom DRX`, `Nongshim RedForce`, `HANJIN BRION`, `DN SOOPers`

## Data Source

[OraclesElixir](https://oracleselixir.com) — 2025 & 2026 LCK season match data
