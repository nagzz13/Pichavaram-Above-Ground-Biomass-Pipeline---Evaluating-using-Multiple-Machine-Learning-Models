"""
step_06_ensemble_relative_uncertainty.py
========================================
Relative uncertainty of the ENSEMBLE product:
  relative (%) = ensemble_uncertainty / ensemble_AGB * 100

Uses the ensemble AGB (step 04) and the ensemble uncertainty/std raster
(step 04). Result written to REL_UNC_DIR.

Run:  python step_06_ensemble_relative_uncertainty.py
"""

import os
import numpy as np
import rasterio

import config


os.makedirs(config.REL_UNC_DIR, exist_ok=True)

out_rel = os.path.join(
    config.REL_UNC_DIR, "AGB_Relative_Uncertainty_Ensemble.tif"
)

with rasterio.open(config.ENSEMBLE_AGB) as src_agb, \
     rasterio.open(config.ENSEMBLE_UNC) as src_unc:
    agb = src_agb.read(1).astype(np.float32)
    unc = src_unc.read(1).astype(np.float32)
    profile = src_agb.profile.copy()

relative_unc = np.full_like(agb, np.nan, dtype=np.float32)
valid = (agb > 0) & np.isfinite(agb) & np.isfinite(unc)
relative_unc[valid] = (unc[valid] / agb[valid]) * 100.0

profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
with rasterio.open(out_rel, "w", **profile) as dst:
    dst.write(relative_unc, 1)

print(f"Ensemble relative uncertainty saved -> {out_rel}")
