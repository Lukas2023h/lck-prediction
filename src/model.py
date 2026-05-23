import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/processed/lck_matchups.csv")

print(f"Geladen: {len(df)} Games")

# Features (X) und Label (y) trennen
feature_cols = [col for col in df.columns if col.endswith("_diff")]
X = df[feature_cols]
y = df["result"]

print(f"Features: {len(feature_cols)}")
print(f"Label-Verteilung:\n{y.value_counts()}")

# Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining: {len(X_train)} Games")
print(f"Test:     {len(X_test)} Games")

# Features skalieren (gleiche Skala für alle Spalten)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Modell trainieren
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Modell evaluieren
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nGenauigkeit auf Test-Daten: {accuracy:.1%}")
print(f"\nDetaillierter Report:")
print(classification_report(y_test, y_pred, target_names=["Red gewinnt", "Blue gewinnt"]))

# Modell und Scaler speichern
joblib.dump(model,  "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Gespeichert: models/model.pkl, models/scaler.pkl")
