import pandas as pd
import joblib

model  = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Team-Level-Daten
team_df = pd.read_csv("data/processed/lck_games.csv")

# Player-Level-Daten
raw = pd.read_csv("data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv", low_memory=False)
player_df = raw[(raw["league"] == "LCK") & (raw["position"] != "team")].copy()

team_avg_cols = [
    "result",
    "golddiffat15", "xpdiffat15", "csdiffat15",
    "kills", "assists",
    "firstblood", "firsttower", "firstmidtower", "firsttothreetowers",
    "firstdragon", "firstherald", "firstbaron",
    "dragons", "elementaldrakes", "elders",
    "wpm", "wcpm", "vspm",
]

player_stat_cols = [
    "kills", "deaths", "assists",
    "dpm", "damageshare",
    "golddiffat15", "csdiffat15", "cspm",
]


def get_team_stats(teamname):
    games = team_df[team_df["teamname"] == teamname]
    if len(games) == 0:
        print(f"Team '{teamname}' nicht gefunden.")
        print(f"Verfügbare Teams: {sorted(team_df['teamname'].unique())}")
        return None
    return games[team_avg_cols].mean()


def get_player_stats(playername, champion=None):
    games = player_df[player_df["playername"] == playername]
    if len(games) == 0:
        return None, None

    avg = games[player_stat_cols].mean()

    # Champion-Winrate: historisch auf diesem Pick, fallback 50%
    if champion:
        on_champ = games[games["champion"] == champion]
        champ_wr = on_champ["result"].mean() if len(on_champ) > 0 else 0.5
    else:
        champ_wr = 0.5

    return avg, champ_wr


def get_team_player_stats(roster, champions=None):
    """
    roster:   Liste von 5 Spielernamen [top, jng, mid, bot, sup]
    champions: Liste von 5 Champions (optional, gleiche Reihenfolge)
    Gibt den Durchschnitt aller Spieler-Stats + avg Champion-Winrate zurück.
    """
    if champions is None:
        champions = [None] * 5

    all_stats = []
    all_wr    = []

    for player, champ in zip(roster, champions):
        stats, wr = get_player_stats(player, champ)
        if stats is not None:
            all_stats.append(stats)
            all_wr.append(wr)

    if not all_stats:
        return None, None

    avg_stats = pd.concat(all_stats, axis=1).mean(axis=1)
    avg_wr    = sum(all_wr) / len(all_wr)
    return avg_stats, avg_wr


def predict_matchup(team_blue, team_red, roster_blue=None, roster_red=None,
                    champions_blue=None, champions_red=None):
    stats_blue = get_team_stats(team_blue)
    stats_red  = get_team_stats(team_red)

    if stats_blue is None or stats_red is None:
        return

    # Team-Level-Differenzen (immer vorhanden)
    team_diff = {f"{c}_diff": stats_blue[c] - stats_red[c] for c in team_avg_cols}

    # Player-Level-Differenzen (optional)
    player_cols = ["champ_winrate"] + [f"avg_{c}" for c in player_stat_cols]
    player_diff = {f"{c}_diff": 0.0 for c in player_cols}  # default: kein Vorteil

    has_player_data = False
    if roster_blue and roster_red:
        p_blue, wr_blue = get_team_player_stats(roster_blue, champions_blue)
        p_red,  wr_red  = get_team_player_stats(roster_red,  champions_red)

        if p_blue is not None and p_red is not None:
            has_player_data = True
            player_diff["champ_winrate_diff"] = wr_blue - wr_red
            for col in player_stat_cols:
                player_diff[f"avg_{col}_diff"] = p_blue[col] - p_red[col]

    all_diff = {**team_diff, **player_diff}
    diff_df  = pd.DataFrame([all_diff])
    diff_df  = diff_df[[c for c in diff_df.columns]]

    diff_scaled = scaler.transform(diff_df)
    prob        = model.predict_proba(diff_scaled)[0]

    prob_blue = prob[1]
    prob_red  = prob[0]
    winner    = team_blue if prob_blue > prob_red else team_red

    print(f"\n{'='*44}")
    print(f"  {team_blue} (Blue)  vs  {team_red} (Red)")
    if has_player_data:
        print(f"  [mit Spieler- & Champion-Daten]")
    print(f"{'='*44}")
    print(f"  {team_blue}: {prob_blue:.1%}")
    print(f"  {team_red}:  {prob_red:.1%}")
    print(f"{'─'*44}")
    print(f"  Vorhersage: {winner} gewinnt")
    print(f"{'='*44}\n")


# --- Beispiele ---

# Nur Teams (wie bisher)
predict_matchup("DN SOOPers", "Gen.G")

# Mit Spielern und Champions
predict_matchup(
    "T1", "Gen.G",
    roster_blue    = ["Zeus", "Oner", "Faker", "Gumayusi", "Keria"],
    roster_red     = ["Doran", "Kanavi", "Chovy", "Peyz", "Delight"],
    champions_blue = ["Garen", "Vi", "Ahri", "Jinx", "Thresh"],
    champions_red  = ["Gnar", "Xin Zhao", "Orianna", "Ezreal", "Nautilus"],
)
