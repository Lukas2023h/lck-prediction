import pandas as pd
import joblib
import numpy as np

model  = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
df     = pd.read_csv("data/processed/lck_games.csv")

avg_cols = [
    "result",
    "golddiffat15", "xpdiffat15", "csdiffat15",
    "kills", "assists",
    "firstblood", "firsttower", "firstmidtower", "firsttothreetowers",
    "firstdragon", "firstherald", "firstbaron",
    "dragons", "elementaldrakes", "elders",
    "wpm", "wcpm", "vspm",
]

def get_team_stats(teamname):
    games = df[df["teamname"] == teamname]
    if len(games) == 0:
        print(f"Team '{teamname}' nicht gefunden.")
        print(f"Verfügbare Teams: {sorted(df['teamname'].unique())}")
        return None
    return games[avg_cols].mean()

def predict_matchup(team_blue, team_red):
    stats_blue = get_team_stats(team_blue)
    stats_red  = get_team_stats(team_red)

    if stats_blue is None or stats_red is None:
        return

    # Differenzen berechnen
    diff = pd.DataFrame(
        [(stats_blue - stats_red).values],
        columns=[f"{c}_diff" for c in avg_cols]
    )

    # Skalieren und vorhersagen
    diff_scaled = scaler.transform(diff)
    prob = model.predict_proba(diff_scaled)[0]

    prob_blue = prob[1]
    prob_red  = prob[0]

    print(f"\n{'='*40}")
    print(f"  {team_blue} (Blue)  vs  {team_red} (Red)")
    print(f"{'='*40}")
    print(f"  {team_blue}: {prob_blue:.1%}")
    print(f"  {team_red}:  {prob_red:.1%}")
    print(f"{'='*40}")
    winner = team_blue if prob_blue > prob_red else team_red
    print(f"  Vorhersage: {winner} gewinnt")
    print(f"{'='*40}\n")


# Verfügbare Teams:
# "T1"
# "Gen.G"
# "Hanwha Life Esports"
# "KT Rolster"
# "Dplus Kia"
# "BNK FEARX"
# "Kiwoom DRX"
# "Nongshim RedForce"
# "HANJIN BRION"
# "DN SOOPers"

predict_matchup("DN SOOPers", "Gen.G")
predict_matchup("HANJIN BRION", "T1")
