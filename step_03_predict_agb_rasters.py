"""
step_03_predict_agb_rasters.py
==============================
Apply each trained model wall-to-wall over the parameter rasters and write
a per-model AGB prediction GeoTIFF to AGB_DIR.

For every model the rasters are loaded in the SAME order as the model's
feature-importance CSV, aligned to the first raster's grid, and only fully
valid pixels are predicted.

Run:  python step_03_predict_agb_rasters.py
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import joblib

import config


config.ensure_dirs()
shap_files = config.shap_files()


# -------------------------------------------------------------------
# LOAD RASTERS IN FEATURE ORDER
# -------------------------------------------------------------------
def load_rasters(feature_list):
    rasters = []
    ref_profile = None
    print("Loading rasters in feature-importance order...")

    for i, feature in enumerate(feature_list):
        if feature not in config.RASTER_NAME_MAP:
            raise ValueError(f"No raster mapping for feature: {feature}")
        path = os.path.join(config.RASTER_DIR, config.RASTER_NAME_MAP[feature])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing raster: {path}")

        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            arr[~np.isfinite(arr)] = np.nan
            if i == 0:
                ref_profile = src.profile.copy()
            else:
                aligned = np.zeros(
                    (ref_profile["height"], ref_profile["width"]),
                    dtype=np.float32,
                )
                reproject(
                    source=arr,
                    destination=aligned,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_profile["transform"],
                    dst_crs=ref_profile["crs"],
                    resampling=Resampling.bilinear,
                )
                arr = aligned
            rasters.append(arr)
            print(f"  Loaded: {feature} -> {os.path.basename(path)}")

    stack = np.stack(rasters, axis=0)
    valid_mask = np.all(np.isfinite(stack), axis=0)
    X_all = stack[:, valid_mask].T
    print(f"  Valid pixels: {X_all.shape[0]}")
    return X_all, stack.shape[1:], valid_mask, ref_profile


# -------------------------------------------------------------------
# PREDICT ONE MODEL
# -------------------------------------------------------------------
def run_prediction(model_key):
    print(f"\n=== Predicting with {model_key} ===")
    feature_list = pd.read_csv(shap_files[model_key]).iloc[:, 0].tolist()
    X_all, (rows, cols), valid_mask, ref_profile = load_rasters(feature_list)

    model_path = config.model_pkl(model_key)
    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}")
        return
    model = joblib.load(model_path)

    scaler_path = config.scaler_pkl(model_key)
    X_input = np.clip(X_all, -1e6, 1e6)
    if os.path.exists(scaler_path):
        X_input = joblib.load(scaler_path).transform(X_input)

    y_pred = model.predict(X_input)

    out = np.full((rows, cols), np.nan, dtype=np.float32)
    out[valid_mask] = y_pred.astype(np.float32)

    profile = ref_profile.copy()
    profile.update(count=1, dtype="float32", nodata=np.nan, compress="lzw")

    out_raster = config.agb_raster(model_key)
    with rasterio.open(out_raster, "w", **profile) as dst:
        dst.write(out, 1)
    print(f"  Saved -> {out_raster}")


# -------------------------------------------------------------------
# RUN ALL MODELS
# -------------------------------------------------------------------
if __name__ == "__main__":
    for model_key in config.MODEL_KEYS:
        run_prediction(model_key)
    print("\nAll predictions complete!")
