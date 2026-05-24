import pandas as pd
import numpy as np

# 2026 Daten (Trainingsziel)
df26 = pd.read_csv("data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv", low_memory=False)
df26 = df26[df26["league"] == "LCK"].copy()
df26["weight"] = 1.0

# 2025 Daten (historischer Kontext, geringer gewichtet wegen Patch-Unterschied)
df25 = pd.read_csv("data/raw/2025_LoL_esports_match_data_from_OraclesElixir.csv", low_memory=False)
df25 = df25[df25["league"] == "LCK"].copy()
df25["weight"] = 0.4

# Zusammenführen, chronologisch sortieren
all_data = pd.concat([df25, df26], ignore_index=True)
all_data["date"] = pd.to_datetime(all_data["date"])
all_data = all_data.sort_values("date").reset_index(drop=True)

# Nur Player-Rows (kein team-aggregat)
all_players = all_data[all_data["position"] != "team"].copy()
all_players["gamelength_min"] = all_players["gamelength"] / 60

# Für Champion-Scaling nur 2026 (patch-relevant)
players_2026 = all_players[all_players["weight"] == 1.0].copy()

stat_cols = [
    "kills", "deaths", "assists",
    "dpm", "damageshare",
    "golddiffat15", "csdiffat15", "cspm",
]

PRIOR       = 5    # Bayesian Blending: Player vs. allgemeine Champion-Stats
MIN_GAMES   = 3    # Mindest-Spieler-History (in 2026)
SHORT_GAME  = players_2026["gamelength_min"].quantile(0.33)
LONG_GAME   = players_2026["gamelength_min"].quantile(0.67)
MIN_SCALING = 3


def weighted_mean(series, weights):
    w = weights.values
    v = series.values
    mask = ~np.isnan(v)
    if mask.sum() == 0:
        return np.nan
    return np.average(v[mask], weights=w[mask])


rows = []

# Nur über 2026-Games iterieren (das sind die Games die wir vorhersagen wollen)
for idx, game in players_2026.iterrows():
    gid      = game["gameid"]
    date     = game["date"]
    player   = game["playername"]
    champion = game["champion"]
    side     = game["side"]
    position = game["position"]

    # Vergangene Spiele dieses Spielers aus BEIDEN Jahren (chronologisch, time-decay)
    past_player = all_players[
        (all_players["playername"] == player) & (all_players["date"] < date)
    ]

    # Mindestens MIN_GAMES 2026-Spiele als Basis
    past_player_2026 = past_player[past_player["weight"] == 1.0]
    if len(past_player_2026) < MIN_GAMES:
        continue

    # Spieler-Stats (gewichteter Durchschnitt über 2025 + 2026)
    player_avg = {col: weighted_mean(past_player[col], past_player["weight"])
                  for col in stat_cols}

    # Allgemeine LCK-Champion-Stats (nur 2026, patch-relevant)
    past_champ_lck = players_2026[
        (players_2026["champion"] == champion) & (players_2026["date"] < date)
    ]

    # Spieler-spezifische Champion-Stats (gewichtet)
    past_player_on_champ = past_player[past_player["champion"] == champion]
    player_champ_games   = len(past_player_on_champ)
    lck_champ_games      = len(past_champ_lck)

    # Bayesian Blending
    weight_blend = player_champ_games / (player_champ_games + PRIOR)

    def blend(player_val, general_val):
        return weight_blend * player_val + (1 - weight_blend) * general_val

    player_champ_wr = (
        weighted_mean(past_player_on_champ["result"], past_player_on_champ["weight"])
        if player_champ_games > 0 else 0.5
    )
    lck_champ_wr = past_champ_lck["result"].mean() if lck_champ_games > 0 else 0.5
    champ_winrate = blend(player_champ_wr, lck_champ_wr)

    # Champion-spezifische Spieler-Stats (geblended)
    player_on_champ_avg = (
        {col: weighted_mean(past_player_on_champ[col], past_player_on_champ["weight"])
         for col in stat_cols}
        if player_champ_games > 0 else player_avg
    )
    lck_on_champ_avg = (
        {col: past_champ_lck[col].mean() for col in stat_cols}
        if lck_champ_games > 0 else player_avg
    )

    blended_stats = {
        col: blend(player_on_champ_avg[col], lck_on_champ_avg[col])
        for col in stat_cols
    }

    # Comp-Scores (nur aus 2026 Champion-Daten)
    if lck_champ_games >= MIN_SCALING:
        past_short = past_champ_lck[past_champ_lck["gamelength_min"] < SHORT_GAME]
        past_long  = past_champ_lck[past_champ_lck["gamelength_min"] > LONG_GAME]
        scaling_score = (
            past_long["result"].mean() - past_short["result"].mean()
            if len(past_short) >= MIN_SCALING and len(past_long) >= MIN_SCALING
            else 0.0
        )
    else:
        scaling_score = 0.0

    early_score = past_champ_lck["golddiffat15"].mean() if lck_champ_games > 0 else 0.0
    carry_score = past_champ_lck["damageshare"].mean()  if lck_champ_games > 0 else 0.25

    row = {
        "gameid":        gid,
        "date":          date,
        "playername":    player,
        "champion":      champion,
        "position":      position,
        "side":          side,
        "result":        game["result"],
        "champ_winrate": champ_winrate,
        "champ_games":   player_champ_games,
        "scaling_score": scaling_score,
        "early_score":   early_score,
        "carry_score":   carry_score,
    }
    for col in stat_cols:
        row[f"avg_{col}"] = blended_stats[col]

    rows.append(row)

player_stats = pd.DataFrame(rows)
print(f"Player-Stats mit History: {len(player_stats)} Zeilen ({len(player_stats) // 10} Games)")

total = len(player_stats)
only_2025_boost = len(all_players[all_players["weight"] == 0.4]["playername"].unique())
print(f"Spieler mit 2025-Daten als Boost: {only_2025_boost}")

# Pro Team pro Game aggregieren
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

team_matchups = pd.read_csv("data/processed/lck_matchups.csv")
combined = team_matchups.merge(player_matchups, on="gameid")

print(f"Team-Features:   {len([c for c in team_matchups.columns if c.endswith('_diff')])}")
print(f"Player-Features: {len(agg_cols)}")
print(f"Gesamt-Features: {len([c for c in combined.columns if c.endswith('_diff')])}")
print(f"Games im Datensatz: {len(combined)}")

combined.to_csv("data/processed/lck_matchups_v2.csv", index=False)
print("\nGespeichert: data/processed/lck_matchups_v2.csv")
