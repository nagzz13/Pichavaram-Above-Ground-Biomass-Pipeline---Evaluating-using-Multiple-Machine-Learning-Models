"""
config.py
=========
Shared configuration for the Pichavaram AGB (Above-Ground Biomass) mapping
pipeline. Every step script imports paths, model definitions and the
raster <-> feature name map from here so that names stay consistent across
the whole workflow.

Edit the paths in the USER PATHS block once and every step will follow.
"""

import os

# -------------------------------------------------------------------
# USER PATHS  (edit these once)
# -------------------------------------------------------------------
DATA_PATH   = r"C:/Users/mssrf/Downloads/Cubesat+S1SAR+Tandem+Indices.csv"
VALID_CSV   = r"D:/Pichavaram_Biomass_Mapping/Datasheets/All_Parameters_AGB_Model_Pichavaram_1.csv"

BASE_DIR    = r"D:/Pichavaram_Biomass_Mapping"
FEAT_DIR    = os.path.join(BASE_DIR, "Feature_Imp")
MODEL_DIR   = os.path.join(BASE_DIR, "All_models")
RASTER_DIR  = os.path.join(BASE_DIR, "Full_Parameters_Rasters")
AGB_DIR     = os.path.join(BASE_DIR, "AGB_Mangroves")
UNC_DIR     = os.path.join(BASE_DIR, "Uncertainty")
REL_UNC_DIR = os.path.join(BASE_DIR, "Relative_Uncertainty")
PLOT_DIR    = os.path.join(BASE_DIR, "Plots1")

TARGET_COLUMN = "AGB(kg/9 sq.m)"
LON_COL = "Longitude"
LAT_COL = "Latitude"

# -------------------------------------------------------------------
# MODELS
# -------------------------------------------------------------------
# Canonical model keys used everywhere (files, dicts, outputs).
MODEL_KEYS = ["RandomForest", "XGBoost", "CatBoost", "KNN"]

# Display names used for the feature-importance CSVs written in step 01.
MODEL_DISPLAY = {
    "RandomForest": "Random Forest",
    "XGBoost":      "XGBoost",
    "CatBoost":     "CatBoost",
    "KNN":          "KNN",
}

# Models that need feature scaling (MinMax) before fit / predict.
SCALED_MODELS = ["KNN"]

# -------------------------------------------------------------------
# BIOLOGICAL / UNCERTAINTY CONSTRAINTS
# -------------------------------------------------------------------
AGB_MAX     = 350.0    # kg / 9 sq.m  -> hard upper bound on biomass
AGB_MIN_REL = 5.0      # ignore pixels below this for relative uncertainty
REL_UNC_MAX = 200.0    # cap relative uncertainty (%)


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def feature_importance_csv(model_key):
    """Path of the feature-importance CSV written by step 01."""
    return os.path.join(FEAT_DIR, f"{MODEL_DISPLAY[model_key]}_feature_importance.csv")


def model_pkl(model_key):
    return os.path.join(MODEL_DIR, f"{model_key}.pkl")


def scaler_pkl(model_key):
    return os.path.join(MODEL_DIR, f"{model_key}_scaler.pkl")


def agb_raster(model_key):
    """Predicted AGB raster path for a model (e.g. RandomForest_AGB.tif)."""
    return os.path.join(AGB_DIR, f"{model_key}_AGB.tif")


ENSEMBLE_AGB = os.path.join(AGB_DIR, "Ensemble_AGB.tif")
ENSEMBLE_UNC = os.path.join(AGB_DIR, "Ensemble_Uncertainty.tif")


def shap_files():
    """Map of model key -> feature-importance CSV (feature order for rasters)."""
    return {k: feature_importance_csv(k) for k in MODEL_KEYS}


# -------------------------------------------------------------------
# RASTER NAME MAP  (feature name -> raster filename in RASTER_DIR)
# -------------------------------------------------------------------
RASTER_NAME_MAP = {
    "VV/VH": "vvbyvh.tif",
    "VH/VV": "vhbyvv.tif",
    "(VV+VH)/2": "vvaddvhby2.tif",
    "VH-VV": "vhsubvv.tif",
    "VV-VH": "vvsubvh.tif",
    "TanDEM Canopy height (m)": "tdm_3m.tif",
    "VV": "vv.tif",
    "VH": "vh.tif",
    "PC_RVI": "PC_RVI.tif",
    "Yellow": "Yellow.tif",
    "Ocean_blue": "Ocean_blue.tif",
    "NDRE": "NDRE.tif",
    "MSAVI": "MSAVI.tif",
    "PC_Green": "PC_Green.tif",
    "Green1": "Green1.tif",
    "PC_VARI": "PC_VARI.tif",
    "GNDVI": "GNDVI.tif",
    "SAVI": "SAVI.tif",
    "PC_GEMI": "PC_GEMI.tif",
    "GEMI": "GEMI.tif",
    "PC_SR": "PC_SR.tif",
    "Blue": "Blue.tif",
    "Green": "Green.tif",
    "Red": "Red.tif",
    "RedEdge": "RedEdge.tif",
    "NIR": "NIR.tif",
    "NDVI": "NDVI.tif",
    "EVI": "EVI.tif",
    "DVI": "DVI.tif",
    "RVI": "RVI.tif",
    "SR": "SR.tif",
    "ARVI": "ARVI.tif",
    "VARI": "VARI.tif",
    "TVI": "TVI.tif",
    "PC_Ocean_blue": "PC_Ocean_blue.tif",
    "PC_Blue": "PC_Blue.tif",
    "PC_Green1": "PC_Green1.tif",
    "PC_Yellow": "PC_Yellow.tif",
    "PC_Red": "PC_Red.tif",
    "PC_RedEdge": "PC_RedEdge.tif",
    "PC_NIR": "PC_NIR.tif",
    "PC_NDVI": "PC_NDVI.tif",
    "PC_GNDVI": "PC_GNDVI.tif",
    "PC_EVI": "PC_EVI.tif",
    "PC_SAVI": "PC_SAVI.tif",
    "PC_MSAVI": "PC_MSAVI.tif",
    "PC_DVI": "PC_DVI.tif",
    "PC_ARVI": "PC_ARVI.tif",
    "PC_TVI": "PC_TVI.tif",
    "PC_NDRE": "PC_NDRE.tif",
}


def ensure_dirs():
    """Create every output directory if it does not exist."""
    for d in (FEAT_DIR, MODEL_DIR, AGB_DIR, UNC_DIR, REL_UNC_DIR, PLOT_DIR):
        os.makedirs(d, exist_ok=True)
