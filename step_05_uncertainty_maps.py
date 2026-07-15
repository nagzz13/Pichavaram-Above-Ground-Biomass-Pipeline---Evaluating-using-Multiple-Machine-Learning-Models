"""
step_05_uncertainty_maps.py
===========================
Per-model spatial uncertainty:
  * ABSOLUTE uncertainty  = local 3x3 standard deviation of the AGB raster
  * RELATIVE uncertainty  = absolute / AGB * 100  (%) with a biological
                            floor (AGB >= AGB_MIN_REL) and a cap (REL_UNC_MAX)

Outputs go to UNC_DIR as
  AGB_<model>_Absolute_Uncertainty.tif
  AGB_<model>_Relative_Uncertainty.tif

Run:  python step_05_uncertainty_maps.py
"""

import os
import numpy as np
import rasterio
from scipy.ndimage import generic_filter

import config


os.makedirs(config.UNC_DIR, exist_ok=True)


def local_std(window):
    if np.all(~np.isfinite(window)):
        return np.nan
    return np.nanstd(window)


def abs_unc_path(model_key):
    return os.path.join(config.UNC_DIR, f"AGB_{model_key}_Absolute_Uncertainty.tif")


def rel_unc_path(model_key):
    return os.path.join(config.UNC_DIR, f"AGB_{model_key}_Relative_Uncertainty.tif")


# -------------------------------------------------------------------
# ABSOLUTE UNCERTAINTY  (local 3x3 std)
# -------------------------------------------------------------------
for model_key in config.MODEL_KEYS:
    print(f"\nAbsolute uncertainty for {model_key}...")
    with rasterio.open(config.agb_raster(model_key)) as src:
        agb = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    agb[~np.isfinite(agb)] = np.nan
    agb = np.clip(agb, 0, config.AGB_MAX)

    abs_unc = generic_filter(
        agb, function=local_std, size=3, mode="nearest"
    ).astype(np.float32)

    profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    with rasterio.open(abs_unc_path(model_key), "w", **profile) as dst:
        dst.write(abs_unc, 1)
    print(f"  Saved -> {abs_unc_path(model_key)}")


# -------------------------------------------------------------------
# RELATIVE UNCERTAINTY  (%)
# -------------------------------------------------------------------
for model_key in config.MODEL_KEYS:
    print(f"\nRelative uncertainty for {model_key}...")
    with rasterio.open(config.agb_raster(model_key)) as src:
        agb = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    with rasterio.open(abs_unc_path(model_key)) as src:
        abs_unc = src.read(1).astype(np.float32)

    agb = np.clip(agb, 0, config.AGB_MAX)

    rel_unc = np.full_like(agb, np.nan)
    valid = (
        np.isfinite(agb)
        & np.isfinite(abs_unc)
        & (agb >= config.AGB_MIN_REL)
    )
    rel_unc[valid] = (abs_unc[valid] / agb[valid]) * 100.0
    rel_unc = np.clip(rel_unc, 0, config.REL_UNC_MAX)

    profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    with rasterio.open(rel_unc_path(model_key), "w", **profile) as dst:
        dst.write(rel_unc.astype(np.float32), 1)
    print(f"  Saved -> {rel_unc_path(model_key)}")

print("\nPer-model uncertainty maps complete.")
