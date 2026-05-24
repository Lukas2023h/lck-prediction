import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

df = pd.read_csv("data/processed/lck_matchups_v2.csv")

print(f"Geladen: {len(df)} Games")

feature_cols = [col for col in df.columns if col.endswith("_diff")]
X = df[feature_cols]
y = df["result"]

print(f"Features: {len(feature_cols)}")
print(f"Label-Verteilung:\n{y.value_counts()}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- XGBoost Hyperparameter-Tuning ---
print(f"\n{'='*44}")
print(f"  XGBoost Hyperparameter-Tuning")
print(f"{'='*44}")

param_grid = {
    "n_estimators":    [50, 100, 200],
    "max_depth":       [2, 3, 4],
    "learning_rate":   [0.01, 0.05, 0.1],
    "subsample":       [0.7, 1.0],
    "colsample_bytree":[0.7, 1.0],
    "reg_alpha":       [0, 0.1, 1.0],   # L1 Regularisierung
    "reg_lambda":      [1.0, 5.0],       # L2 Regularisierung
}

grid_search = GridSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42),
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Suche läuft... (kann einen Moment dauern)")
grid_search.fit(X_scaled, y)

beste_params = grid_search.best_params_
bester_score = grid_search.best_score_

print(f"\nBeste Parameter:")
for k, v in beste_params.items():
    print(f"  {k}: {v}")
print(f"\nBester CV-Score: {bester_score:.1%}")

# --- Modellvergleich ---
print(f"\n{'='*44}")
print(f"  Modellvergleich (5-Fold Cross-Validation)")
print(f"{'='*44}")

lr_scores  = cross_val_score(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    X, y, cv=cv, scoring="accuracy"
)
xgb_scores = cross_val_score(
    XGBClassifier(**beste_params, eval_metric="logloss", random_state=42),
    X_scaled, y, cv=cv, scoring="accuracy"
)

print(f"\n  Logistic Regression:  {lr_scores.mean():.1%}  ±{lr_scores.std():.1%}")
print(f"  XGBoost (getuned):    {xgb_scores.mean():.1%}  ±{xgb_scores.std():.1%}")

bestes_modell = "XGBoost" if xgb_scores.mean() >= lr_scores.mean() else "Logistic Regression"
print(f"\n  → Bestes Modell: {bestes_modell}")
print(f"{'='*44}")

# --- Finales Modell trainieren und speichern ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_final = StandardScaler()
X_train_scaled = scaler_final.fit_transform(X_train)
X_test_scaled  = scaler_final.transform(X_test)

if bestes_modell == "XGBoost":
    model = XGBClassifier(**beste_params, eval_metric="logloss", random_state=42)
else:
    model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(f"\nGenauigkeit (Test-Split): {accuracy_score(y_test, y_pred):.1%}")
print(f"\nDetaillierter Report:")
print(classification_report(y_test, y_pred, target_names=["Red gewinnt", "Blue gewinnt"]))

joblib.dump(model,        "models/model.pkl")
joblib.dump(scaler_final, "models/scaler.pkl")
print(f"Gespeichert: models/model.pkl ({bestes_modell}), models/scaler.pkl")
