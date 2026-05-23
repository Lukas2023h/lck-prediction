import pandas as pd

# Rohdaten laden
df = pd.read_csv("data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv", low_memory=False)

print(f"Gesamte Zeilen: {len(df)}")
print(f"Spalten: {df.shape[1]}")

# Nur LCK, nur Team-Zeilen
df = df[(df["league"] == "LCK") & (df["position"] == "team")]

print(f"\nNach Filter (LCK + Team-Zeilen): {len(df)} Zeilen")
print(f"Das entspricht {len(df) // 2} Games")
print(f"\nTeams:\n{df['teamname'].value_counts()}")

# Relevante Spalten auswählen
cols = [
    # Identifikation
    "gameid", "date", "game", "teamname", "side", "result",
    # Spielverlauf
    "gamelength", "ckpm",
    # Frühe Phase (Minute 15)
    "golddiffat15", "xpdiffat15", "csdiffat15",
    # Objectives
    "firstblood", "firstdragon", "firstherald", "firstbaron",
    "firsttower", "firstmidtower", "firsttothreetowers",
    # Kampf
    "kills", "deaths", "assists",
    # Vision
    "wpm", "wcpm", "vspm",
    # Drachen
    "dragons", "elementaldrakes", "elders",
]

df = df[cols]

print(f"\nSpalten nach Auswahl: {df.shape[1]}")
print(f"\nErste Zeile:\n{df.iloc[0]}")
print(f"\nFehlende Werte pro Spalte:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# Bereinigten Datensatz speichern
df.to_csv("data/processed/lck_games.csv", index=False)
print("\nGespeichert: data/processed/lck_games.csv")
