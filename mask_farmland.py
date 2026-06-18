import os
import glob
import json
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np

SHP_PATH = os.path.join(os.path.dirname(__file__), "input", "field_boundary.shp")
INPUT_DIR = os.path.join(os.path.dirname(__file__), "SAR_timeseries_output")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "SAR_timeseries_masked")

def mask_raster(input_tif, shp_gdf, output_dir, orbit_direction):
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(input_tif) as src:
        src_nodata = src.nodata if src.nodata is not None else 0
        num_bands = src.count

        basename = os.path.splitext(os.path.basename(input_tif))[0]

        all_geoms = [row.geometry for _, row in shp_gdf.iterrows()]

        try:
            out_image, out_transform = mask(src, all_geoms, crop=False, invert=False)
        except Exception as e:
            print(f"  Warning: Could not mask: {e}")
            return

        vv = out_image[0]
        vh = out_image[1]
        data_mask = out_image[2]

        valid = data_mask == 1
        if not valid.any():
            print(f"  No valid pixels inside polygon, skipping")
            return

        has_lia = num_bands >= 4
        if has_lia:
            lia = out_image[3]
            print(f"  Has LIA band, saving raw backscatter")
        else:
            lia = None
            print(f"  Note: No localIncidenceAngle band ({num_bands} bands)")

        out_bands = np.stack([vv, vh, data_mask, np.where(valid, lia, 0).astype(np.float32)], axis=0) if has_lia else np.stack([vv, vh, data_mask], axis=0)

        out_meta = src.meta.copy()
        out_meta.update({
            "count": out_bands.shape[0],
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": src_nodata
        })

        out_filename = os.path.join(output_dir, f"{basename}_{orbit_direction}_field.tif")
        with rasterio.open(out_filename, "w", **out_meta) as dst:
            dst.write(out_bands)

        print(f"  Saved: {out_filename}")

def get_orbit_direction(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        orbit_state = data.get('properties', {}).get('sat:orbit_state', 'unknown')
        return orbit_state
    except:
        return 'unknown'

def main():
    tif_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tif")))
    tif_files = [f for f in tif_files if not f.endswith(".aux.xml")]

    if not tif_files:
        print("No TIFF files found!")
        return

    print("Loading shapefile...")
    shp_gdf = gpd.read_file(SHP_PATH)
    print(f"  Original CRS: {shp_gdf.crs}")

    with rasterio.open(tif_files[0]) as src:
        target_crs = src.crs

    print(f"  Target CRS (from TIF): {target_crs}")

    if shp_gdf.crs is None:
        print("  WARNING: Shapefile has no CRS, assuming EPSG:32631")
        shp_gdf = shp_gdf.set_crs("EPSG:32631")
    elif shp_gdf.crs != target_crs:
        print(f"  Reprojecting from {shp_gdf.crs} to {target_crs}")
        shp_gdf = shp_gdf.to_crs(target_crs)

    print(f"Found {len(shp_gdf)} feature(s) in shapefile")
    print(f"Found {len(tif_files)} TIFF files")

    for i, tif_path in enumerate(tif_files):
        print(f"Processing {i+1}/{len(tif_files)}: {os.path.basename(tif_path)}")
        
        json_path = tif_path.replace('.tif', '.json')
        orbit_direction = get_orbit_direction(json_path)
        print(f"  Orbit: {orbit_direction}")
        
        mask_raster(tif_path, shp_gdf, OUTPUT_DIR, orbit_direction)

    print("\nDone!")

if __name__ == "__main__":
    main()
