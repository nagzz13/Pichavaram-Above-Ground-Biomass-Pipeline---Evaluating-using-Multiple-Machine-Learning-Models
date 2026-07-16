# -*- coding: utf-8 -*-
"""
Vegetation Indices + PCA + Extractor (individual single‑band raster inputs)

Adds:
✓ 8-band PCA  → outputs PC_B1, PC_B2... with band names
✓ Vegetation index PCA → outputs PC_NDVI, PC_GNDVI, etc.
✓ Extracts all PCA values + all VIs + all band values into the output CSV
✓ No band‑to‑band ratios (as requested)

Assumptions:
• All input bands have identical CRS, extent, resolution & alignment.
• CSV contains latitude & longitude in decimal degrees.
• PCA computed using sklearn.decomposition.PCA (installed in ArcGIS Python).
• PCA rasters are written as GeoTIFFs.
"""
import arcpy
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as rio_transform
from sklearn.decomposition import PCA

class Toolbox(object):
    def __init__(self):
        self.label = "Vegetation Indices + PCA Extractor"
        self.alias = "VI_PCA_Extractor"
        self.tools = [VegetationIndicesExtractor]


class VegetationIndicesExtractor(object):
    def __init__(self):
        self.label = "Calculate Vegetation Indices + PCA + Extract"
        self.description = (
            "Calculate vegetation indices, PCA for bands and indices, and extract values using CSV coordinates."
        )
        self.canRunInBackground = False

    def getParameterInfo(self):
        p_csv = arcpy.Parameter(
            displayName="Input CSV (latitude, longitude)",
            name="input_csv",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"
        )

        p_outfolder = arcpy.Parameter(
            displayName="Output Folder",
            name="output_folder",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )

        p_outcsv = arcpy.Parameter(
            displayName="Output CSV (extracted values)",
            name="output_csv",
            datatype="DEFile",
            parameterType="Required",
            direction="Output"
        )

        # Band inputs
        band_params = []
        band_list = [
            ("Ocean Blue Band Raster", "ocean_blue_raster"),
            ("Blue Band Raster", "blue_raster"),
            ("Green1 Band Raster", "green1_raster"),
            ("Green Band Raster", "green_raster"),
            ("Yellow Band Raster", "yellow_raster"),
            ("Red Band Raster", "red_raster"),
            ("Red Edge Band Raster", "rededge_raster"),
            ("NIR Band Raster", "nir_raster"),
        ]

        for label, name in band_list:
            p = arcpy.Parameter(
                displayName=label,
                name=name,
                datatype="DERasterDataset",
                parameterType="Required",
                direction="Input"
            )
            band_params.append(p)

        return [p_csv, p_outfolder, p_outcsv] + band_params

    # ---------------------------------------------------------------
    def execute(self, params, messages):
        try:
            input_csv = params[0].valueAsText
            output_folder = params[1].valueAsText
            output_csv = params[2].valueAsText
            os.makedirs(output_folder, exist_ok=True)

            # Band rasters
            band_paths = {
                "Ocean_blue": params[3].valueAsText,
                "Blue": params[4].valueAsText,
                "Green1": params[5].valueAsText,
                "Green": params[6].valueAsText,
                "Yellow": params[7].valueAsText,
                "Red": params[8].valueAsText,
                "RedEdge": params[9].valueAsText,
                "NIR": params[10].valueAsText,
            }

            # Read rasters
            messages.addMessage("Reading and validating band rasters...")
            reference_profile = None
            bands = {}

            for name, path in band_paths.items():
                with rasterio.open(path) as src:
                    if reference_profile is None:
                        reference_profile = src.profile.copy()
                        ref_crs = src.crs
                        ref_transform = src.transform
                        ref_h, ref_w = src.height, src.width
                    else:
                        if src.crs != ref_crs:
                            raise ValueError(f"CRS mismatch for {name}")
                        if src.transform != ref_transform:
                            raise ValueError(f"Transform mismatch for {name}")
                        if src.height != ref_h or src.width != ref_w:
                            raise ValueError(f"Size mismatch for {name}")

                    bands[name] = src.read(1).astype(np.float32)

            # Bands as variables
            Ocean_blue = bands["Ocean_blue"]
            Blue = bands["Blue"]
            Green1 = bands["Green1"]
            Green = bands["Green"]
            Yellow = bands["Yellow"]
            Red = bands["Red"]
            RedEdge = bands["RedEdge"]
            NIR = bands["NIR"]

            # -------------------------------------------------------
            # Compute vegetation indices
            # -------------------------------------------------------
            messages.addMessage("Computing vegetation indices...")

            def safe_div(a, b):
                with np.errstate(divide='ignore', invalid='ignore'):
                    out = np.true_divide(a, b)
                    out[~np.isfinite(out)] = np.nan
                return out

            indices = {
                "NDVI": safe_div((NIR - Red), (NIR + Red)),
                "GNDVI": safe_div((NIR - Green), (NIR + Green)),
                "EVI": 2.5 * safe_div((NIR - Red), (NIR + 6 * Red - 7.5 * Blue + 1)),
                "SAVI": 1.5 * safe_div((NIR - Red), (NIR + Red + 0.5)),
                "NDRE": safe_div((NIR - RedEdge), (NIR + RedEdge)),
                "MSAVI": (2 * NIR + 1 - np.sqrt(np.maximum((2 * NIR + 1) ** 2 - 8 * (NIR - Red), 0))) / 2,
                "DVI": (NIR - Red),
                "RVI": safe_div(NIR, Red),
                "ARVI": safe_div((NIR - (2 * Red - Blue)), (NIR + (2 * Red - Blue))),
                "VARI": safe_div((Green - Red), (Green + Red - Blue)),
                "TVI": 0.5 * (120 * (NIR - Green) - 200 * (Red - Green)),
            }

            # GEMI
            x = (2 * (NIR ** 2 - Red ** 2) + 1.5 * NIR + 0.5 * Red)
            y = (NIR + Red + 0.5)
            z = safe_div(x, y)
            indices["GEMI"] = (z * (1 - 0.25 * z)) - safe_div((NIR - Red), (NIR + Red))

            # -------------------------------------------------------
            # PCA FOR BANDS
            # -------------------------------------------------------
            messages.addMessage("Running PCA for 8 bands...")
            band_stack = np.stack(list(bands.values()), axis=-1)  # H,W,8
            flat = band_stack.reshape(-1, 8)

            pca_b = PCA(n_components=8)
            pcs_b = pca_b.fit_transform(np.nan_to_num(flat))
            pcs_b = pcs_b.reshape(ref_h, ref_w, 8)

            # -------------------------------------------------------
            # PCA FOR INDICES
            # -------------------------------------------------------
            messages.addMessage("Running PCA for vegetation indices...")
            idx_stack = np.stack(list(indices.values()), axis=-1)  # H,W,N
            flat_i = idx_stack.reshape(-1, idx_stack.shape[-1])

            pca_i = PCA(n_components=idx_stack.shape[-1])
            pcs_i = pca_i.fit_transform(np.nan_to_num(flat_i))
            pcs_i = pcs_i.reshape(ref_h, ref_w, idx_stack.shape[-1])

            # -------------------------------------------------------
            # Write rasters (bands PCA + indices + indices PCA)
            # -------------------------------------------------------
            messages.addMessage("Writing output rasters...")

            def write_raster(path, arr, profile):
                p = profile.copy()
                for k in ["tiled", "blockxsize", "blockysize", "compress", "interleave"]:
                    if k in p:
                        p.pop(k)
                p.update(driver="GTiff", dtype="float32", count=1, nodata=-9999, compress="lzw")
                with rasterio.open(path, "w", **p) as dst:
                    dst.write(arr.astype(np.float32), 1)

            raster_paths = {}

            # VI rasters
            for name, arr in indices.items():
                out = os.path.join(output_folder, f"{name}.tif")
                write_raster(out, arr, reference_profile)
                raster_paths[name] = out

            # PCA Bands rasters
            band_keys = list(bands.keys())
            for i in range(8):
                nm = f"PC_{band_keys[i]}"
                out = os.path.join(output_folder, f"{nm}.tif")
                write_raster(out, pcs_b[:, :, i], reference_profile)
                raster_paths[nm] = out

            # PCA Indices rasters
            idx_keys = list(indices.keys())
            for i in range(len(idx_keys)):
                nm = f"PC_{idx_keys[i]}"
                out = os.path.join(output_folder, f"{nm}.tif")
                write_raster(out, pcs_i[:, :, i], reference_profile)
                raster_paths[nm] = out

            # -------------------------------------------------------
            # EXTRACTION
            # -------------------------------------------------------
            messages.addMessage("Extracting all band, index and PCA values to CSV...")
            df = pd.read_csv(input_csv)

            if not {"latitude", "longitude"}.issubset(df.columns):
                raise ValueError("CSV must contain latitude & longitude columns.")

            # For each raster: extract
            lats = df["latitude"].astype(float).values
            lons = df["longitude"].astype(float).values

            for name, rpath in raster_paths.items():
                with rasterio.open(rpath) as src:
                    arr = src.read(1)
                    nodata = src.nodata
                    src_crs = src.crs

                    if src_crs.to_string() != "EPSG:4326":
                        xs, ys = rio_transform("EPSG:4326", src_crs, lons.tolist(), lats.tolist())
                    else:
                        xs, ys = lons.tolist(), lats.tolist()

                    vals = []
                    for x, y in zip(xs, ys):
                        try:
                            r, c = src.index(x, y)
                            if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
                                v = arr[r, c]
                                if np.isnan(v) or (nodata is not None and v == nodata):
                                    vals.append(np.nan)
                                else:
                                    vals.append(float(v))
                            else:
                                vals.append(np.nan)
                        except:
                            vals.append(np.nan)

                    df[name] = vals

            # Save CSV
            df.to_csv(output_csv, index=False)
            messages.addMessage(f"✓ Extraction completed → {output_csv}")

        except Exception as e:
            try:
                messages.addErrorMessage(str(e))
            except:
                pass
            raise
