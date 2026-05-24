import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/lck_games.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"Geladen: {len(df)} Zeilen, sortiert nach Datum")

avg_cols = [
    "result",
    "golddiffat15", "xpdiffat15", "csdiffat15",
    "kills", "assists",
    "firstblood", "firsttower", "firstmidtower", "firsttothreetowers",
    "firstdragon", "firstherald", "firstbaron",
    "dragons", "elementaldrakes", "elders",
    "wpm", "wcpm", "vspm",
]

FORM_GAMES = 5   # Anzahl letzter Spiele für Formkurve
MIN_GAMES  = 3   # Mindest-History pro Team

rows = []

for idx, game in df.iterrows():
    gid  = game["gameid"]
    date = game["date"]
    side = game["side"]
    team = game["teamname"]

    past = df[(df["teamname"] == team) & (df["date"] < date)]

    if len(past) < MIN_GAMES:
        continue

    avg = past[avg_cols].mean()

    # --- Formkurve: Winrate der letzten N Spiele ---
    form_wr = past.tail(FORM_GAMES)["result"].mean()

    # --- Side-Winrate: Wie gut ist das Team auf dieser Seite? ---
    past_on_side = past[past["side"] == side]
    side_wr = past_on_side["result"].mean() if len(past_on_side) > 0 else 0.5

    row = {
        "gameid":   gid,
        "date":     date,
        "team":     team,
        "side":     side,
        "result":   game["result"],
        "form_wr":  form_wr,
        "side_wr":  side_wr,
    }
    for col in avg_cols:
        row[f"avg_{col}"] = avg[col]

    rows.append(row)

team_stats = pd.DataFrame(rows)
print(f"Team-Stats mit History: {len(team_stats)} Zeilen ({len(team_stats) // 2} Games)")

# Blue und Red zusammenführen
blue = team_stats[team_stats["side"] == "Blue"]
red  = team_stats[team_stats["side"] == "Red"]

merged = blue.merge(red, on="gameid", suffixes=("_blue", "_red"))
print(f"Games mit History (beide Teams): {len(merged)}")

# Basis-Matchup DataFrame
games = pd.DataFrame()
games["gameid"]    = merged["gameid"]
games["date"]      = merged["date_blue"]
games["team_blue"] = merged["team_blue"]
games["team_red"]  = merged["team_red"]
games["result"]    = merged["result_blue"]  # 1 = Blue gewinnt

# Differenzen: Blue - Red
for col in avg_cols:
    games[f"{col}_diff"] = merged[f"avg_{col}_blue"].values - merged[f"avg_{col}_red"].values

# Formkurve-Differenz
games["form_wr_diff"] = merged["form_wr_blue"].values - merged["form_wr_red"].values

# Side-Winrate-Differenz
games["side_wr_diff"] = merged["side_wr_blue"].values - merged["side_wr_red"].values

# --- Head-to-Head History ---
# Für jedes Matchup: Wie oft hat Blue dieses Duell historisch gewonnen?
h2h_diffs = []

for _, row in games.iterrows():
    team_a = row["team_blue"]
    team_b = row["team_red"]
    date   = row["date"]

    # Vergangene Duelle zwischen diesen zwei Teams (in beliebiger Seiten-Zuordnung)
    past_h2h = games[
        (games["date"] < date) &
        (
            ((games["team_blue"] == team_a) & (games["team_red"] == team_b)) |
            ((games["team_blue"] == team_b) & (games["team_red"] == team_a))
        )
    ]

    if len(past_h2h) == 0:
        h2h_diffs.append(0.0)  # Neutral wenn keine History
        continue

    # Siege von team_a berechnen
    a_as_blue = past_h2h[past_h2h["team_blue"] == team_a]["result"].sum()
    a_as_red  = (1 - past_h2h[past_h2h["team_red"] == team_a]["result"]).sum()
    total     = len(past_h2h)

    h2h_wr_a   = (a_as_blue + a_as_red) / total
    h2h_diffs.append(h2h_wr_a - 0.5)  # Diff von Neutral (0.5)

games["h2h_diff"] = h2h_diffs

new_features = ["form_wr_diff", "side_wr_diff", "h2h_diff"]
print(f"\nNeue Features: {new_features}")
print(f"Finaler Datensatz: {games.shape}")

games.to_csv("data/processed/lck_matchups.csv", index=False)
print("Gespeichert: data/processed/lck_matchups.csv")
