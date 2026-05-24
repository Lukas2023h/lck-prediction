import pandas as pd
import numpy as np
import joblib

model  = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

team_df  = pd.read_csv("data/processed/lck_games.csv")
team_df["date"] = pd.to_datetime(team_df["date"])

raw       = pd.read_csv("data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv", low_memory=False)
player_df = raw[(raw["league"] == "LCK") & (raw["position"] != "team")].copy()

champ_stats = pd.read_csv("data/processed/lck_champion_stats.csv").set_index("champion")

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

FORM_GAMES = 5


def get_team_stats(teamname):
    games = team_df[team_df["teamname"] == teamname]
    if len(games) == 0:
        print(f"Team '{teamname}' nicht gefunden.")
        print(f"Verfügbare Teams: {sorted(team_df['teamname'].unique())}")
        return None
    avg     = games[team_avg_cols].mean()
    form_wr = games.tail(FORM_GAMES)["result"].mean()
    return avg, form_wr, games


def get_side_wr(games, side):
    on_side = games[games["side"] == side]
    return on_side["result"].mean() if len(on_side) > 0 else 0.5


def get_h2h(team_a, team_b):
    past = team_df[
        ((team_df["teamname"] == team_a) | (team_df["teamname"] == team_b))
    ]
    # Finde Games wo beide Teams gespielt haben (selbe gameid)
    game_ids = past.groupby("gameid").filter(lambda x: len(x) == 2)["gameid"].unique()
    if len(game_ids) == 0:
        return 0.0

    h2h = team_df[team_df["gameid"].isin(game_ids)]
    a_rows = h2h[h2h["teamname"] == team_a]
    wins_a = a_rows["result"].sum()
    total  = len(a_rows)
    return (wins_a / total) - 0.5  # Diff von neutral


def get_player_stats(playername, champion=None):
    games = player_df[player_df["playername"] == playername]
    if len(games) == 0:
        return None, None
    avg      = games[player_stat_cols].mean()
    champ_wr = (
        games[games["champion"] == champion]["result"].mean()
        if champion and len(games[games["champion"] == champion]) > 0
        else 0.5
    )
    return avg, champ_wr


def get_comp_scores(champions):
    scaling, early, carry = [], [], []
    for champ in champions:
        if champ and champ in champ_stats.index:
            scaling.append(champ_stats.loc[champ, "scaling_score"])
            early.append(champ_stats.loc[champ, "avg_gd15"])
            carry.append(champ_stats.loc[champ, "avg_damageshare"])
        else:
            scaling.append(0.0)
            early.append(0.0)
            carry.append(0.25)
    return np.mean(scaling), np.mean(early), np.mean(carry)


def get_team_player_stats(roster, champions=None):
    if champions is None:
        champions = [None] * 5
    all_stats, all_wr = [], []
    for player, champ in zip(roster, champions):
        stats, wr = get_player_stats(player, champ)
        if stats is not None:
            all_stats.append(stats)
            all_wr.append(wr)
    if not all_stats:
        return None, None
    return pd.concat(all_stats, axis=1).mean(axis=1), sum(all_wr) / len(all_wr)


def series_probability(p, format="bo3"):
    """Berechnet Serien-Winrate aus Einzelspiel-Winrate p."""
    if format == "bo3":
        return p**2 * (3 - 2*p)
    elif format == "bo5":
        return p**3 * (1 + 3*(1-p) + 6*(1-p)**2)
    raise ValueError("format muss 'bo3' oder 'bo5' sein")


def predict_matchup(team_blue, team_red, roster_blue=None, roster_red=None,
                    champions_blue=None, champions_red=None):

    result_blue = get_team_stats(team_blue)
    result_red  = get_team_stats(team_red)
    if result_blue is None or result_red is None:
        return

    avg_blue, form_blue, games_blue = result_blue
    avg_red,  form_red,  games_red  = result_red

    # Team-Differenzen
    diff = {f"{c}_diff": avg_blue[c] - avg_red[c] for c in team_avg_cols}
    diff["form_wr_diff"] = form_blue - form_red
    diff["side_wr_diff"] = get_side_wr(games_blue, "Blue") - get_side_wr(games_red, "Red")
    diff["h2h_diff"]     = get_h2h(team_blue, team_red)

    # Comp-Scores (aus Champion-Stats-CSV)
    champs_b = champions_blue or [None] * 5
    champs_r = champions_red  or [None] * 5
    sc_b, ea_b, ca_b = get_comp_scores(champs_b)
    sc_r, ea_r, ca_r = get_comp_scores(champs_r)
    diff["champ_winrate_diff"]  = 0.0
    diff["scaling_score_diff"]  = sc_b - sc_r
    diff["early_score_diff"]    = ea_b - ea_r
    diff["carry_score_diff"]    = ca_b - ca_r

    # Spieler-Stats
    player_defaults = {f"avg_{c}_diff": 0.0 for c in player_stat_cols}
    has_player_data = False
    if roster_blue and roster_red:
        p_blue, wr_blue = get_team_player_stats(roster_blue, champions_blue)
        p_red,  wr_red  = get_team_player_stats(roster_red,  champions_red)
        if p_blue is not None and p_red is not None:
            has_player_data = True
            diff["champ_winrate_diff"] = wr_blue - wr_red
            for col in player_stat_cols:
                player_defaults[f"avg_{col}_diff"] = p_blue[col] - p_red[col]

    diff.update(player_defaults)

    diff_df     = pd.DataFrame([diff])
    diff_scaled = scaler.transform(diff_df)
    prob        = model.predict_proba(diff_scaled)[0]

    prob_blue = prob[1]
    prob_red  = prob[0]
    winner    = team_blue if prob_blue > prob_red else team_red

    bo3_blue = series_probability(prob_blue, "bo3")
    bo5_blue = series_probability(prob_blue, "bo5")

    print(f"\n{'='*44}")
    print(f"  {team_blue} (Blue)  vs  {team_red} (Red)")
    if has_player_data:
        print(f"  [mit Spieler- & Champion-Daten]")
    print(f"{'='*44}")
    print(f"  Einzelspiel:  {team_blue} {prob_blue:.1%}  |  {team_red} {prob_red:.1%}")
    print(f"  Bo3-Serie:    {team_blue} {bo3_blue:.1%}  |  {team_red} {1-bo3_blue:.1%}")
    print(f"  Bo5-Serie:    {team_blue} {bo5_blue:.1%}  |  {team_red} {1-bo5_blue:.1%}")
    print(f"{'─'*44}")
    print(f"  Vorhersage: {winner} gewinnt")
    print(f"{'='*44}\n")


# T1 vs HANJIN BRION
# Reihenfolge: Top, Jungle, Mid, Bot, Support
predict_matchup(
    "T1", "HANJIN BRION",
    roster_blue    = ["Doran", "Oner", "Faker", "Peyz", "Keria"],
    roster_red     = ["Casting", "GIDEON", "Roamer", "Teddy", "Namgung"],
    champions_blue = ["Sion", "Naafiri", "Twisted Fate", "Caitlyn", "Lux"],
    champions_red  = ["Rumble", "Lee Sin", "Ryze", "Jhin", "Pyke"],
)
