import os
import glob
import json
import geopandas as gpd
import rasterio
from rasterio.mask import mask

SHP_PATH = r"c:\Users\Adib\Documents\Kuliah\Term 2\Thesis\sentinelproc\input\neeroeteren2025.shp"
INPUT_DIR = r"c:\Users\Adib\Documents\Kuliah\Term 2\Thesis\sentinelproc\SAR_timeseries_output"
OUTPUT_DIR = r"c:\Users\Adib\Documents\Kuliah\Term 2\Thesis\sentinelproc\SAR_timeseries_masked"

FEATURE_NAMES = {
    2: "nonirrigated",
    3: "irrigated"
}

def mask_raster_with_features(input_tif, shp_gdf, output_dir, orbit_direction):
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(input_tif) as src:
        src_nodata = src.nodata if src.nodata is not None else 0

        basename = os.path.splitext(os.path.basename(input_tif))[0]

        for feat_idx, row in shp_gdf.iterrows():
            feat_id = row.get('id', feat_idx + 1)

            if feat_id not in FEATURE_NAMES:
                continue

            geom = [row.geometry]

            try:
                out_image, out_transform = mask(src, geom, crop=False, invert=False)
            except Exception as e:
                print(f"  Warning: Could not mask with feature {feat_idx}: {e}")
                continue

            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": src_nodata
            })

            feat_name = FEATURE_NAMES[feat_id]
            out_filename = os.path.join(output_dir, f"{basename}_{orbit_direction}_{feat_name}.tif")
            with rasterio.open(out_filename, "w", **out_meta) as dst:
                dst.write(out_image)

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
        print("  WARNING: Shapefile has no CRS, assuming EPSG:32632")
        shp_gdf = shp_gdf.set_crs("EPSG:32632")
    elif shp_gdf.crs != target_crs:
        print(f"  Reprojecting from {shp_gdf.crs} to {target_crs}")
        shp_gdf = shp_gdf.to_crs(target_crs)

    print(f"Found {len(shp_gdf)} feature(s) in shapefile")
    print(f"Processing features: {list(FEATURE_NAMES.keys())}")
    print(f"Found {len(tif_files)} TIFF files")

    for i, tif_path in enumerate(tif_files):
        print(f"Processing {i+1}/{len(tif_files)}: {os.path.basename(tif_path)}")
        
        json_path = tif_path.replace('.tif', '.json')
        orbit_direction = get_orbit_direction(json_path)
        print(f"  Orbit: {orbit_direction}")
        
        mask_raster_with_features(tif_path, shp_gdf, OUTPUT_DIR, orbit_direction)

    print("\nDone!")

if __name__ == "__main__":
    main()