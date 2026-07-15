"""
step_02_train_models.py
=======================
Train each model on the FULL dataset using the feature order stored in the
step-01 importance CSV, then persist the fitted model (and scaler for KNN)
to MODEL_DIR with joblib.

Run:  python step_02_train_models.py
"""

import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import config


# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
df = pd.read_csv(config.DATA_PATH)
config.ensure_dirs()

shap_files = config.shap_files()

# Fresh model objects (tuned a little more than step-01's quick fit).
model_objects = {
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGBoost":      XGBRegressor(n_estimators=300, learning_rate=0.05,
                                 max_depth=6, random_state=42),
    "CatBoost":     CatBoostRegressor(verbose=0, random_state=42),
    "KNN":          KNeighborsRegressor(n_neighbors=5),
}


def load_feature_list(filepath):
    """First column of the importance CSV = ordered feature names."""
    feat = pd.read_csv(filepath)
    return feat[feat.columns[0]].tolist()


# -------------------------------------------------------------------
# TRAIN + SAVE
# -------------------------------------------------------------------
for model_key in config.MODEL_KEYS:
    shap_file = shap_files[model_key]
    if not os.path.exists(shap_file):
        print(f"Skipping {model_key} (importance file not found: {shap_file})")
        continue

    print(f"\nTraining {model_key} on full dataset...")
    feature_list = load_feature_list(shap_file)
    X_full = df[feature_list].values
    y_full = df[config.TARGET_COLUMN].values

    model_obj = model_objects[model_key]

    if model_key in config.SCALED_MODELS:
        scaler = MinMaxScaler()
        X_fit = scaler.fit_transform(X_full)
        joblib.dump(scaler, config.scaler_pkl(model_key))
        print(f"  Saved scaler -> {config.scaler_pkl(model_key)}")
    else:
        X_fit = X_full  # raw features for tree models

    model_obj.fit(X_fit, y_full)
    joblib.dump(model_obj, config.model_pkl(model_key))
    print(f"  Saved model  -> {config.model_pkl(model_key)}")

print("\nAll classical ML models trained and saved on the full dataset.")
