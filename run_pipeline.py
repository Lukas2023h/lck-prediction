"""
update_data.py — Lädt neue Daten von Google Drive und trainiert das Modell neu.

Verwendung:
    venv/bin/python update_data.py

Google Drive Links eintragen:
    Datei öffnen → Teilen → "Jeder mit dem Link" → Link kopieren
    Den Link unten bei DRIVE_FILES eintragen.
"""

import subprocess
import sys
import gdown

# --- Einzelne Dateien per Google Drive File-ID ---
# Rechtsklick auf Datei → "Link abrufen" → die ID aus dem Link kopieren
# https://drive.google.com/file/d/DIESE_ID_HIER/view
DRIVE_FILES = {
    "data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv": "1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm",
    "data/raw/2025_LoL_esports_match_data_from_OraclesElixir.csv": "1v6LRphp2kYciU4SXp0PCjEMuev1bDejc",
}


def download_files():
    print("=" * 50)
    print("  Daten von Google Drive herunterladen")
    print("=" * 50)

    for dest, file_id in DRIVE_FILES.items():
        if file_id == "ID_HIER_EINTRAGEN":
            print(f"  [SKIP] {dest} — keine File-ID eingetragen")
            continue

        print(f"\n  Lade: {dest}")
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, dest, quiet=False)
            print(f"  ✓ Gespeichert: {dest}")
        except Exception as e:
            print(f"  ✗ Fehler: {e}")
            sys.exit(1)

def run_pipeline():
    print("\n" + "=" * 50)
    print("  Pipeline neu durchlaufen")
    print("=" * 50)

    steps = [
        ("Preprocessing",    "src/preprocessing.py"),
        ("Team-Features",    "src/features.py"),
        ("Spieler-Features", "src/player_features.py"),
        ("Champion-Stats",   "src/champion_stats.py"),
        ("Modell trainieren","src/model.py"),
        ("Serien-Modell",    "src/model_series.py"),
    ]

    for name, script in steps:
        print(f"\n  [{name}]")
        result = subprocess.run(
            ["venv/bin/python", script],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ✗ Fehler in {script}:")
            print(result.stderr)
            sys.exit(1)

        # Letzte Zeile der Ausgabe anzeigen
        last_line = [l for l in result.stdout.strip().splitlines() if l]
        if last_line:
            print(f"  → {last_line[-1]}")
        print(f"  ✓ Fertig")


if __name__ == "__main__":
    download_files()
    run_pipeline()
    print("\n" + "=" * 50)
    print("  Update abgeschlossen — Modell ist aktuell!")
    print("=" * 50)
