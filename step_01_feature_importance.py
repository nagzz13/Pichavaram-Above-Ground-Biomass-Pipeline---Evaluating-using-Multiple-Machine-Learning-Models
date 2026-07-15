"""
step_01_feature_importance.py
=============================
Train each classical ML model on the full training table and compute
feature importance:
  * tree models (RandomForest / XGBoost / CatBoost) -> SHAP mean |value|
  * KNN -> permutation importance (SHAP has no tree explainer for it)

For every model a sorted CSV and a bar-plot PNG are written to FEAT_DIR.
The CSV feature order is reused by later steps to stack rasters correctly.

Run:  python step_01_feature_importance.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import shap

import config


# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
df = pd.read_csv(config.DATA_PATH)
print("Dataset shape:", df.shape)

X = df.drop(config.TARGET_COLUMN, axis=1)
y = df[config.TARGET_COLUMN]
X = X.fillna(X.mean())
y = y.fillna(y.mean())
feature_names = X.columns

# StandardScaler copy used for KNN permutation importance.
X_scaled = StandardScaler().fit_transform(X)

config.ensure_dirs()


# -------------------------------------------------------------------
# MODELS
# -------------------------------------------------------------------
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost":      XGBRegressor(n_estimators=100, random_state=42),
    "CatBoost":     CatBoostRegressor(iterations=100, random_state=42, verbose=0),
    "KNN":          KNeighborsRegressor(n_neighbors=5),
}


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def save_sorted_df(imp_df, model_key):
    imp_sorted = imp_df.sort_values("importance", ascending=False)
    out = config.feature_importance_csv(model_key)
    imp_sorted.to_csv(out, index=False)
    print(f"  Saved CSV  -> {out}")
    return imp_sorted


def plot_and_save(imp_df, model_key):
    plt.figure(figsize=(16, 6))
    plt.bar(imp_df["feature"], imp_df["importance"])
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Importance")
    plt.title(f"{config.MODEL_DISPLAY[model_key]} Feature Importance")
    plt.tight_layout()
    out = config.feature_importance_csv(model_key).replace(".csv", ".png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved plot -> {out}")


# -------------------------------------------------------------------
# TRAIN + IMPORTANCE
# -------------------------------------------------------------------
for model_key, model in models.items():
    print(f"\nTraining {model_key} on full dataset...")

    if model_key in config.SCALED_MODELS:
        # KNN: permutation importance on scaled features.
        model.fit(X_scaled, y)
        perm = permutation_importance(
            model, X_scaled, y, n_repeats=10, random_state=42,
            scoring="r2", n_jobs=1,
        )
        imp_df = pd.DataFrame(
            {"feature": feature_names, "importance": perm.importances_mean}
        )
    else:
        # Tree models: SHAP mean absolute value, permutation as fallback.
        model.fit(X, y)
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            imp_df = pd.DataFrame(
                {"feature": feature_names,
                 "importance": np.abs(shap_vals).mean(axis=0)}
            )
        except Exception as err:
            print(f"  SHAP failed ({err}); falling back to permutation.")
            perm = permutation_importance(
                model, X, y, n_repeats=10, random_state=42, n_jobs=1
            )
            imp_df = pd.DataFrame(
                {"feature": feature_names, "importance": perm.importances_mean}
            )

    imp_sorted = save_sorted_df(imp_df, model_key)
    plot_and_save(imp_sorted, model_key)

print("\nAll feature importances computed and saved.")
