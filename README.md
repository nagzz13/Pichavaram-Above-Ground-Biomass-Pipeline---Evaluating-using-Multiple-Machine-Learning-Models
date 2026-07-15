# Pichavaram AGB Mapping Pipeline

Your original notebook code, split into ordered, standalone steps. Every
script imports shared paths and settings from `config.py`, so you only edit
paths in one place.

## Run order

```
python step_01_feature_importance.py          # train + SHAP/permutation importance -> Feature_Imp/
python step_02_train_models.py                # fit models on full data -> All_models/
python step_03_predict_agb_rasters.py         # wall-to-wall AGB rasters -> AGB_Mangroves/
python step_04_weighted_ensemble.py           # inverse-RMSE ensemble + std -> AGB_Mangroves/
python step_05_uncertainty_maps.py            # per-model abs + rel uncertainty -> Uncertainty/
python step_06_ensemble_relative_uncertainty.py  # ensemble relative uncertainty -> Relative_Uncertainty/
python step_07_validation_metrics.py          # R2/RMSE/MAE vs field points -> Validation_Metrics.csv
python step_08_scatter_plots.py               # observed vs predicted plots -> Plots1/
```

## Files

- `config.py` — all paths, model list, biological constants, and the
  feature→raster name map. **Edit paths here.**
- `step_01` … `step_08` — one pipeline stage each.

## Changes vs the original scripts (bugs fixed / cleanups)

- **step_02**: the original had a broken `if/else` (an `else:` after an
  `else:`) so no model was actually trained/saved. Rewritten so every model
  fits and persists correctly.
- **step_03**: added the missing `import pandas as pd`; dropped the unused
  `.h5` Keras branch (pipeline is the four classical models).
- **Consistent names**: predicted rasters are now `<Model>_AGB.tif`
  (RandomForest/XGBoost/CatBoost/KNN) in `AGB_Mangroves/`, and every step
  reads that same convention. The originals mixed `AGB_Outputs/`,
  `AGB_{model}.tif`, and `RF_AGB.tif`, which broke the hand-off between
  prediction and validation.
- **step_04**: ensemble weights are now computed from live per-model RMSE
  against the field points instead of hard-coded numbers, so they stay
  correct if you retrain.
- **step_07 / step_08**: lat/lon are reprojected into each raster's CRS
  before sampling (the original scatter script sampled raw lat/lon, which
  fails for projected rasters). `plt.show()` replaced with `Agg` backend so
  it runs headless.

## Requirements

```
pip install numpy pandas scikit-learn xgboost catboost shap rasterio pyproj scipy matplotlib joblib
```
