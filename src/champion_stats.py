import pandas as pd

df = pd.read_csv("data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv", low_memory=False)
players = df[(df["league"] == "LCK") & (df["position"] != "team")].copy()
players["gamelength_min"] = players["gamelength"] / 60

# Gamelength-Schwellen (untere/obere 33%)
SHORT_GAME = players["gamelength_min"].quantile(0.33)  # < ~29 min
LONG_GAME  = players["gamelength_min"].quantile(0.67)  # > ~34 min
MIN_SAMPLE = 3  # Mindestspiele pro Bucket für zuverlässige Stats

rows = []

for champion, group in players.groupby("champion"):
    games = len(group)

    short = group[group["gamelength_min"] < SHORT_GAME]
    long  = group[group["gamelength_min"] > LONG_GAME]

    # Scaling-Score: Winrate in langen minus Winrate in kurzen Spielen
    # Positiv = Late-Game-Stärke, Negativ = Early-Game-Stärke
    wr_short = short["result"].mean() if len(short) >= MIN_SAMPLE else None
    wr_long  = long["result"].mean()  if len(long)  >= MIN_SAMPLE else None

    if wr_short is not None and wr_long is not None:
        scaling_score = wr_long - wr_short
    else:
        scaling_score = 0.0  # neutral bei zu wenig Daten

    # Early-Game-Score: durchschnittliche Gold-Differenz bei Minute 15
    early_score = group["golddiffat15"].mean()

    # Carry-Score: durchschnittlicher Damage-Anteil am Team
    carry_score = group["damageshare"].mean()

    # Utility-Score: hohe Assists, niedrige Kills (Support/Engage)
    avg_kills   = group["kills"].mean()
    avg_assists = group["assists"].mean()
    avg_deaths  = group["deaths"].mean()
    utility_score = avg_assists / max(avg_kills, 0.1)

    rows.append({
        "champion":      champion,
        "games":         games,
        "winrate":       group["result"].mean(),
        "avg_gd15":      early_score,
        "avg_dpm":       group["dpm"].mean(),
        "avg_csd15":     group["csdiffat15"].mean(),
        "avg_cspm":      group["cspm"].mean(),
        "avg_damageshare": carry_score,
        "avg_kills":     avg_kills,
        "avg_deaths":    avg_deaths,
        "avg_assists":   avg_assists,
        "kda":           (avg_kills + avg_assists) / max(avg_deaths, 1),
        "wr_short_games": wr_short,
        "wr_long_games":  wr_long,
        "scaling_score": scaling_score,   # > 0 = Late, < 0 = Early
        "early_score":   early_score,     # > 0 = Lane-dominant früh
        "carry_score":   carry_score,     # hoch = Carry-Champion
        "utility_score": utility_score,   # hoch = Support/Engage
    })

champ = pd.DataFrame(rows).sort_values("games", ascending=False).reset_index(drop=True)

print(f"Champions gesamt:          {len(champ)}")
print(f"Mit Scaling-Daten (5+ Games): {len(champ[champ['games'] >= 5])}")

print("\n--- Top Late-Game-Scaler (scaling_score > 0) ---")
late = champ[champ["games"] >= 5].sort_values("scaling_score", ascending=False).head(10)
print(late[["champion", "games", "scaling_score", "wr_short_games", "wr_long_games"]].to_string(index=False))

print("\n--- Top Early-Game-Stomper (scaling_score < 0) ---")
early = champ[champ["games"] >= 5].sort_values("scaling_score").head(10)
print(early[["champion", "games", "scaling_score", "wr_short_games", "wr_long_games"]].to_string(index=False))

print("\n--- Höchster Carry-Score (damageshare) ---")
carry = champ[champ["games"] >= 5].sort_values("carry_score", ascending=False).head(10)
print(carry[["champion", "games", "carry_score", "avg_dpm", "winrate"]].to_string(index=False))

champ.to_csv("data/processed/lck_champion_stats.csv", index=False)
print("\nGespeichert: data/processed/lck_champion_stats.csv")
