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


## Requirements

```
pip install numpy pandas scikit-learn xgboost catboost shap rasterio pyproj scipy matplotlib joblib
```
