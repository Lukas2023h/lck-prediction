import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

# Per-Game Features (chronologisch gebaut, kein Leakage)
matchups = pd.read_csv("data/processed/lck_matchups_v2.csv")
matchups["date"] = pd.to_datetime(matchups["date"])

# Rohdaten für Serien-Identifikation
raw = pd.read_csv("data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv", low_memory=False)
lck = raw[(raw["league"] == "LCK") & (raw["position"] == "team")].copy()
lck["date"] = pd.to_datetime(lck["date"])
lck["day"]  = lck["date"].dt.date

feature_cols = [c for c in matchups.columns if c.endswith("_diff")]

# --- Serien-Datensatz bauen ---
# Jede Zeile = eine Bo3/Bo5-Serie
# Features = Spiel-1-Features (alles was VOR der Serie bekannt ist)
# Label = wer hat die Serie gewonnen

series_index = []
for day, day_group in lck.groupby("day"):
    for gid, game_group in day_group.groupby("gameid"):
        if len(game_group) != 2:
            continue
        teams = sorted(game_group["teamname"].tolist())
        series_index.append({
            "day":    day,
            "game":   game_group["game"].iloc[0],
            "gameid": gid,
            "pair":   f"{teams[0]}|{teams[1]}",
        })

series_index = pd.DataFrame(series_index)

dataset_rows = []

for (day, pair), grp in series_index.groupby(["day", "pair"]):
    grp = grp.sort_values("game")
    if len(grp) < 2:
        continue  # Nur Serien mit mind. 2 Spielen

    # Spiel-1-Features als Serien-Features
    game1_id  = grp.iloc[0]["gameid"]
    game1_row = matchups[matchups["gameid"] == game1_id]
    if len(game1_row) == 0:
        continue
    game1_row = game1_row.iloc[0]

    blue_team = game1_row["team_blue"]
    red_team  = game1_row["team_red"]

    # Serien-Ergebnis: wer hat mehr Spiele gewonnen?
    all_gameids  = grp["gameid"].tolist()
    series_games = lck[lck["gameid"].isin(all_gameids)]
    wins         = series_games.groupby("teamname")["result"].sum()
    if len(wins) < 2:
        continue

    actual_winner = wins.idxmax()
    label = 1 if actual_winner == blue_team else 0

    row = {col: game1_row[col] for col in feature_cols}
    row.update({
        "series_label": label,
        "games_played": len(grp),
        "blue_team":    blue_team,
        "red_team":     red_team,
    })
    dataset_rows.append(row)

series_df = pd.DataFrame(dataset_rows)

print(f"Serien-Datensatz: {len(series_df)} Serien")
print(f"Features:         {len(feature_cols)}")
print(f"Label (Blue gewinnt Serie): {series_df['series_label'].sum()} / {len(series_df)}")

X = series_df[feature_cols]
y = series_df["series_label"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- XGBoost Hyperparameter-Tuning ---
print(f"\n{'='*44}")
print(f"  Serien-Modell: Hyperparameter-Tuning")
print(f"{'='*44}")
print("Suche läuft...")

param_grid = {
    "n_estimators":     [50, 100, 200],
    "max_depth":        [2, 3],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0],
    "reg_alpha":        [0, 0.1, 1.0],
    "reg_lambda":       [1.0, 5.0],
}

scaler_cv = StandardScaler()
X_scaled  = scaler_cv.fit_transform(X)

grid = GridSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42),
    param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0,
)
grid.fit(X_scaled, y)
beste_params  = grid.best_params_

# --- Modellvergleich ---
lr_scores  = cross_val_score(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    X, y, cv=cv, scoring="accuracy",
)
xgb_scores = cross_val_score(
    XGBClassifier(**beste_params, eval_metric="logloss", random_state=42),
    X_scaled, y, cv=cv, scoring="accuracy",
)

print(f"\n{'='*44}")
print(f"  Modellvergleich  (5-Fold CV auf Serien)")
print(f"{'='*44}")
print(f"  Logistic Regression:  {lr_scores.mean():.1%}  ±{lr_scores.std():.1%}")
print(f"  XGBoost (getuned):    {xgb_scores.mean():.1%}  ±{xgb_scores.std():.1%}")
print(f"\n  Zum Vergleich — Per-Game-Modell CV: ~65.8%")

# --- Finales Serien-Modell ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_final   = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train)
X_test_scaled  = scaler_final.transform(X_test)

bestes_modell = "XGBoost" if xgb_scores.mean() >= lr_scores.mean() else "Logistic Regression"
if bestes_modell == "XGBoost":
    final_model = XGBClassifier(**beste_params, eval_metric="logloss", random_state=42)
else:
    final_model = LogisticRegression(max_iter=1000)

final_model.fit(X_train_scaled, y_train)
y_pred   = final_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n  Bestes Modell:        {bestes_modell}")
print(f"  Test-Split Accuracy:  {accuracy:.1%}  ({len(X_test)} Serien)")
print(f"\n{classification_report(y_test, y_pred, target_names=['Red gewinnt Serie', 'Blue gewinnt Serie'])}")

joblib.dump(final_model,  "models/model_series.pkl")
joblib.dump(scaler_final, "models/scaler_series.pkl")
print(f"Gespeichert: models/model_series.pkl ({bestes_modell})")
