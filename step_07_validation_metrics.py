"""
step_07_validation_metrics.py
=============================
Validate every model raster (and the ensemble) against the field-measured
AGB points. Lat/Lon are reprojected into each raster's CRS before sampling,
so it works regardless of the raster projection.

Metrics: R2, RMSE, MAE, sample count.  Saved to AGB_DIR/Validation_Metrics.csv

Run:  python step_07_validation_metrics.py
"""

import os
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import config


# Models to validate: the four base models + the ensemble.
rasters = {k: config.agb_raster(k) for k in config.MODEL_KEYS}
rasters["ENSEMBLE"] = config.ENSEMBLE_AGB


df = pd.read_csv(config.VALID_CSV)
lat = df[config.LAT_COL].values
lon = df[config.LON_COL].values
y_obs = df[config.TARGET_COLUMN].values


def latlon_to_raster_coords(raster_path):
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
    transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return list(zip(x, y))


def extract_raster_values(raster_path, coords):
    with rasterio.open(raster_path) as src:
        vals = [v[0] for v in src.sample(coords)]
    return np.array(vals, dtype=np.float32)


def compute_metrics(y_true, y_pred):
    mask = np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae, len(y_true)


results = []
for model, raster in rasters.items():
    if not os.path.exists(raster):
        print(f"Skipping {model} (raster not found: {raster})")
        continue
    coords = latlon_to_raster_coords(raster)
    y_pred = extract_raster_values(raster, coords)
    r2, rmse, mae, n = compute_metrics(y_obs, y_pred)
    results.append({"Model": model, "R2": r2, "RMSE": rmse, "MAE": mae, "Samples": n})

results_df = pd.DataFrame(results).set_index("Model")
print(results_df)

out_csv = os.path.join(config.AGB_DIR, "Validation_Metrics.csv")
results_df.to_csv(out_csv)
print(f"\nSaved metrics -> {out_csv}")
