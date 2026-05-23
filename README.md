# LCK Match Prediction Model

A machine learning model that predicts the winner of LCK (League of Legends Champions Korea) matches based on historical team performance data.

## Overview

The model uses historical team stats (win rate, gold difference at 15 minutes, objective control, vision, etc.) to predict which team wins a given matchup. It is trained on game-by-game data from OraclesElixir covering the 2026 LCK season (Cup + Rounds 1-2).

**Current accuracy: ~73%** on held-out test data using Logistic Regression.

## Project Structure

```
lck-prediction/
├── data/
│   ├── raw/          # Original data from OraclesElixir (untouched)
│   └── processed/    # Cleaned and engineered data
├── src/
│   ├── preprocessing.py  # Filter and clean raw data
│   ├── features.py       # Build historical team matchup features
│   ├── model.py          # Train and evaluate the model
│   └── predict.py        # Predict a matchup
├── models/               # Saved model and scaler (.pkl)
├── notebooks/            # Exploratory notebooks
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/lck-prediction.git
cd lck-prediction

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the pipeline in order:

```bash
# 1. Clean and filter raw data
python src/preprocessing.py

# 2. Build historical matchup features
python src/features.py

# 3. Train the model
python src/model.py

# 4. Predict a matchup
python src/predict.py
```

To predict a custom matchup, edit the bottom of `src/predict.py`:

```python
predict_matchup("T1", "Gen.G")
```

Available teams: `T1`, `Gen.G`, `Hanwha Life Esports`, `KT Rolster`, `Dplus Kia`, `BNK FEARX`, `Kiwoom DRX`, `Nongshim RedForce`, `HANJIN BRION`, `DN SOOPers`

## Data Source

[OraclesElixir](https://oracleselixir.com) — 2026 LCK season match data
