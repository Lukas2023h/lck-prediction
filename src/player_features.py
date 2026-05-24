import pandas as pd

df = pd.read_csv("data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv", low_memory=False)
df = df[df["league"] == "LCK"]
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

players = df[df["position"] != "team"].copy()

stat_cols = [
    "kills", "deaths", "assists",
    "dpm", "damageshare",
    "golddiffat15", "csdiffat15", "cspm",
]

# Wie viele Spiele auf einem Champion bis wir den Spieler-Stats voll vertrauen.
# Formel: weight = player_games / (player_games + PRIOR)
# → 0 Spiele = 0% Spieler, PRIOR Spiele = 50% Spieler, 3*PRIOR = 75% Spieler
PRIOR = 5

MIN_GAMES = 3

# Gamelength-Schwellen für Scaling-Score (feste Werte, global berechnet)
players["gamelength_min"] = players["gamelength"] / 60
SHORT_GAME  = players["gamelength_min"].quantile(0.33)
LONG_GAME   = players["gamelength_min"].quantile(0.67)
MIN_SCALING = 3  # Mindestspiele pro Bucket für Scaling-Score

rows = []

for idx, game in players.iterrows():
    gid      = game["gameid"]
    date     = game["date"]
    player   = game["playername"]
    champion = game["champion"]
    side     = game["side"]
    position = game["position"]

    # Vergangene Spiele dieses Spielers (chronologisch, kein Leakage)
    past_player = players[(players["playername"] == player) & (players["date"] < date)]

    if len(past_player) < MIN_GAMES:
        continue

    # Allgemeine LCK-Champion-Stats auf diesem Champion (chronologisch)
    past_champ_lck = players[(players["champion"] == champion) & (players["date"] < date)]

    # Spieler-spezifische Stats auf diesem Champion
    past_player_on_champ = past_player[past_player["champion"] == champion]

    player_champ_games = len(past_player_on_champ)
    lck_champ_games    = len(past_champ_lck)

    # Bayesian Blending: je mehr Spiele auf dem Champion, desto mehr Gewicht auf Spieler-Stats
    weight = player_champ_games / (player_champ_games + PRIOR)

    def blend(player_val, general_val):
        return weight * player_val + (1 - weight) * general_val

    # Champion-Winrate (geblended)
    player_champ_wr = past_player_on_champ["result"].mean() if player_champ_games > 0 else 0.5
    lck_champ_wr    = past_champ_lck["result"].mean()       if lck_champ_games > 0   else 0.5
    champ_winrate   = blend(player_champ_wr, lck_champ_wr)

    # Allgemeine Spieler-Stats (über alle Champions)
    player_avg = past_player[stat_cols].mean()

    # Champion-spezifische Spieler-Stats (geblended mit allgemeinen LCK-Stats)
    player_on_champ_avg = past_player_on_champ[stat_cols].mean() if player_champ_games > 0 else player_avg
    lck_on_champ_avg    = past_champ_lck[stat_cols].mean()       if lck_champ_games > 0    else player_avg

    blended_stats = {
        col: blend(player_on_champ_avg[col], lck_on_champ_avg[col])
        for col in stat_cols
    }

    # --- Comp-Scores (Champion-Scaling, Early-Game, Carry) ---
    # Scaling-Score: Winrate in langen Spielen minus kurzen Spielen
    if lck_champ_games >= MIN_SCALING:
        past_short = past_champ_lck[past_champ_lck["gamelength_min"] < SHORT_GAME]
        past_long  = past_champ_lck[past_champ_lck["gamelength_min"] > LONG_GAME]
        if len(past_short) >= MIN_SCALING and len(past_long) >= MIN_SCALING:
            scaling_score = past_long["result"].mean() - past_short["result"].mean()
        else:
            scaling_score = 0.0
    else:
        scaling_score = 0.0

    early_score = past_champ_lck["golddiffat15"].mean() if lck_champ_games > 0 else 0.0
    carry_score = past_champ_lck["damageshare"].mean()  if lck_champ_games > 0 else 0.25

    row = {
        "gameid":         gid,
        "date":           date,
        "playername":     player,
        "champion":       champion,
        "position":       position,
        "side":           side,
        "result":         game["result"],
        "champ_winrate":  champ_winrate,
        "champ_games":    player_champ_games,
        "blend_weight":   round(weight, 2),
        "scaling_score":  scaling_score,
        "early_score":    early_score,
        "carry_score":    carry_score,
    }
    for col in stat_cols:
        row[f"avg_{col}"] = blended_stats[col]

    rows.append(row)

player_stats = pd.DataFrame(rows)
print(f"Player-Stats mit History: {len(player_stats)} Zeilen ({len(player_stats) // 10} Games)")

avg_weight = player_stats["blend_weight"].mean()
print(f"Durchschnittliches Blend-Gewicht (Spieler vs. Allgemein): {avg_weight:.0%} Spieler / {1-avg_weight:.0%} Allgemein")

# Pro Team pro Game aggregieren (Durchschnitt über alle 5 Spieler)
agg_cols = ["champ_winrate", "scaling_score", "early_score", "carry_score"] + [f"avg_{c}" for c in stat_cols]

team_agg = (
    player_stats
    .groupby(["gameid", "side"])[agg_cols]
    .mean()
    .reset_index()
)

blue = team_agg[team_agg["side"] == "Blue"].drop(columns="side")
red  = team_agg[team_agg["side"] == "Red"].drop(columns="side")

merged = blue.merge(red, on="gameid", suffixes=("_blue", "_red"))

for col in agg_cols:
    merged[f"{col}_diff"] = merged[f"{col}_blue"] - merged[f"{col}_red"]

player_matchups = merged[["gameid"] + [f"{col}_diff" for col in agg_cols]]

# Mit Team-Level-Features zusammenführen
team_matchups = pd.read_csv("data/processed/lck_matchups.csv")
combined = team_matchups.merge(player_matchups, on="gameid")

print(f"Team-Features:   {len([c for c in team_matchups.columns if c.endswith('_diff')])}")
print(f"Player-Features: {len(agg_cols)}")
print(f"Gesamt-Features: {len([c for c in combined.columns if c.endswith('_diff')])}")
print(f"Games im Datensatz: {len(combined)}")

combined.to_csv("data/processed/lck_matchups_v2.csv", index=False)
print("\nGespeichert: data/processed/lck_matchups_v2.csv")
