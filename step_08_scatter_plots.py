"""
step_08_scatter_plots.py
========================
Observed-vs-predicted scatter plots with a fitted trendline and R^2 for
every model raster and the ensemble. One PNG per model saved to PLOT_DIR.

Lat/Lon are reprojected into each raster's CRS before sampling.

Run:  python step_08_scatter_plots.py
"""

import os
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import stats
import matplotlib
matplotlib.use("Agg")          # headless-safe, no interactive window
import matplotlib.pyplot as plt

import config


os.makedirs(config.PLOT_DIR, exist_ok=True)

raster_files = {k: config.agb_raster(k) for k in config.MODEL_KEYS}
raster_files["Ensemble"] = config.ENSEMBLE_AGB


df_points = pd.read_csv(config.VALID_CSV)
lat = df_points[config.LAT_COL].values
lon = df_points[config.LON_COL].values


def extract_raster_values(raster_path):
    with rasterio.open(raster_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        vals = [v[0] for v in src.sample(list(zip(x, y)))]
    return np.array(vals, dtype=np.float32)


def plot_scatter_trend(y_true, y_pred, model_name):
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_true, y_pred = y_true[mask], y_pred[mask]

    slope, intercept, r_value, _, _ = stats.linregress(y_true, y_pred)
    line_eq = f"y = {slope:.3f}x + {intercept:.3f}"
    r2 = r_value ** 2

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, color="blue", label="Data points")
    order = np.argsort(y_true)
    plt.plot(
        y_true[order], slope * y_true[order] + intercept,
        color="red", label=f"Trendline\n{line_eq}\n$R^2$ = {r2:.3f}",
    )
    plt.xlabel("Observed AGB")
    plt.ylabel(f"Predicted AGB ({model_name})")
    plt.title(f"Observed vs Predicted: {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(config.PLOT_DIR, f"Observed_vs_{model_name}.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved -> {out}")


for model_name, raster_path in raster_files.items():
    if not os.path.exists(raster_path):
        print(f"Skipping {model_name} (raster not found: {raster_path})")
        continue
    y_pred = extract_raster_values(raster_path)
    plot_scatter_trend(df_points[config.TARGET_COLUMN].values, y_pred, model_name)

print("\nAll scatter plots complete.")
