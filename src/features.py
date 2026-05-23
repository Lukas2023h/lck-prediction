import pandas as pd

df = pd.read_csv("data/processed/lck_games.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"Geladen: {len(df)} Zeilen, sortiert nach Datum")

# Spalten über die wir historische Durchschnitte berechnen
avg_cols = [
    "result",
    "golddiffat15", "xpdiffat15", "csdiffat15",
    "kills", "assists",
    "firstblood", "firsttower", "firstmidtower", "firsttothreetowers",
    "firstdragon", "firstherald", "firstbaron",
    "dragons", "elementaldrakes", "elders",
    "wpm", "wcpm", "vspm",
]

MIN_GAMES = 3  # Mindestanzahl vergangener Spiele pro Team

rows = []

# Alle Games chronologisch durchgehen
for idx, game in df.iterrows():
    gid   = game["gameid"]
    date  = game["date"]
    side  = game["side"]
    team  = game["teamname"]
    label = game["result"]

    # Vergangene Spiele dieses Teams (vor diesem Game)
    past = df[(df["teamname"] == team) & (df["date"] < date)]

    if len(past) < MIN_GAMES:
        continue

    # Durchschnitt berechnen
    avg = past[avg_cols].mean()

    row = {"gameid": gid, "date": date, "team": team, "side": side, "result": label}
    for col in avg_cols:
        row[f"avg_{col}"] = avg[col]

    rows.append(row)

team_stats = pd.DataFrame(rows)
print(f"Team-Stats mit History: {len(team_stats)} Zeilen ({len(team_stats)//2} Games)")

# Blue und Red auf gameid zusammenführen
blue = team_stats[team_stats["side"] == "Blue"]
red  = team_stats[team_stats["side"] == "Red"]

merged = blue.merge(red, on="gameid", suffixes=("_blue", "_red"))
print(f"Games wo beide Teams >= {MIN_GAMES} Spiele History haben: {len(merged)}")

# Matchup DataFrame bauen
games = pd.DataFrame()
games["gameid"]    = merged["gameid"]
games["date"]      = merged["date_blue"]
games["team_blue"] = merged["team_blue"]
games["team_red"]  = merged["team_red"]
games["result"]    = merged["result_blue"]  # 1 = Blue gewinnt

# Differenzen: Blue_avg - Red_avg
for col in avg_cols:
    games[f"{col}_diff"] = merged[f"avg_{col}_blue"].values - merged[f"avg_{col}_red"].values

print(f"\nFinaler Datensatz: {games.shape}")
print(f"\nErste Zeile:\n{games.iloc[0]}")

games.to_csv("data/processed/lck_matchups.csv", index=False)
print("\nGespeichert: data/processed/lck_matchups.csv")
