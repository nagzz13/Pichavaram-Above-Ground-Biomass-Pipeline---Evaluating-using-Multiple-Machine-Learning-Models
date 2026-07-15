"""
step_04_weighted_ensemble.py
============================
Build an inverse-RMSE weighted ensemble AGB raster from the per-model
predictions, plus a pixel-wise ensemble uncertainty raster (std across
models).

Weights are derived from each model's RMSE against the field-measured AGB
points (sampled from the rasters). This replaces hard-coded RMSE values so
the weights always reflect the current rasters. If the validation CSV is
missing, it falls back to an equal-weight ensemble.

Run:  python step_04_weighted_ensemble.py
"""

import os
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from sklearn.metrics import mean_squared_error

import config


config.ensure_dirs()


# -------------------------------------------------------------------
# RMSE PER MODEL (for the weights)
# -------------------------------------------------------------------
def sample_raster(raster_path, lat, lon):
    with rasterio.open(raster_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        vals = [v[0] for v in src.sample(list(zip(x, y)))]
    return np.array(vals, dtype=np.float32)


def compute_weights():
    if not os.path.exists(config.VALID_CSV):
        print("Validation CSV not found -> using equal weights.")
        return {k: 1.0 / len(config.MODEL_KEYS) for k in config.MODEL_KEYS}

    df = pd.read_csv(config.VALID_CSV)
    lat = df[config.LAT_COL].values
    lon = df[config.LON_COL].values
    y_obs = df[config.TARGET_COLUMN].values

    rmse = {}
    for model_key in config.MODEL_KEYS:
        y_pred = sample_raster(config.agb_raster(model_key), lat, lon)
        mask = np.isfinite(y_pred)
        rmse[model_key] = np.sqrt(mean_squared_error(y_obs[mask], y_pred[mask]))

    inv = {k: 1.0 / v for k, v in rmse.items()}
    total = sum(inv.values())
    weights = {k: inv[k] / total for k in inv}

    print("Model     RMSE      Weight")
    for k in config.MODEL_KEYS:
        print(f"  {k:<12} {rmse[k]:8.3f}  {weights[k]:.4f}")
    return weights


# -------------------------------------------------------------------
# WEIGHTED ENSEMBLE + UNCERTAINTY
# -------------------------------------------------------------------
def build_ensemble(weights):
    arrays, w_list, profile = [], [], None
    for model_key in config.MODEL_KEYS:
        with rasterio.open(config.agb_raster(model_key)) as src:
            arr = src.read(1).astype(np.float32)
            arr[~np.isfinite(arr)] = np.nan
            if profile is None:
                profile = src.profile.copy()
            arrays.append(arr)
            w_list.append(weights[model_key])

    stack = np.stack(arrays, axis=0)
    w = np.array(w_list).reshape(-1, 1, 1)

    ensemble = np.nansum(stack * w, axis=0) / np.nansum(w, axis=0)
    ensemble = np.clip(ensemble, 0, config.AGB_MAX)          # biological bound
    uncertainty = np.nanstd(stack, axis=0)                   # spread across models

    profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")

    with rasterio.open(config.ENSEMBLE_AGB, "w", **profile) as dst:
        dst.write(ensemble.astype(np.float32), 1)
    print(f"\nSaved ensemble AGB     -> {config.ENSEMBLE_AGB}")

    with rasterio.open(config.ENSEMBLE_UNC, "w", **profile) as dst:
        dst.write(uncertainty.astype(np.float32), 1)
    print(f"Saved ensemble std/unc -> {config.ENSEMBLE_UNC}")


if __name__ == "__main__":
    build_ensemble(compute_weights())
