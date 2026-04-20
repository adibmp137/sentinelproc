# process_all_scenes.py
# This script implements a two-stage workflow to download a time-series of Sentinel-1 images.
# Stage 1: Use the Catalog API to find all available acquisition dates.
# Stage 2: Loop through each date and use the Process API to download the processed image.

#%% 1. Import Libraries & Load Environment Variables
import os
import requests
import json
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import rasterio
import numpy as np
import io
from dotenv import load_dotenv
from datetime import datetime, timedelta

# This function looks for a .env file and loads the variables from it
load_dotenv()

# Get credentials from the environment.
CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")

# A quick check to make sure the credentials were found
if not CLIENT_ID or not CLIENT_SECRET:
    print("FATAL: CLIENT_ID or CLIENT_SECRET not found.")
    print("Please ensure you have a .env file with your credentials.")
else:
    print("Credentials successfully loaded from .env file.")


#%% 2. Authenticate and Create a Session (Same as before)
oauth = None

if CLIENT_ID and CLIENT_SECRET:
    client = BackendApplicationClient(client_id=CLIENT_ID)
    oauth = OAuth2Session(client=client)
    try:
        token = oauth.fetch_token(
            token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET
        )
        print("Authentication successful. Token obtained.")
    except Exception as e:
        print(f"Authentication failed: {e}")
        oauth = None
else:
    print("Skipping authentication because credentials are not loaded.")

#%% 3. STAGE 1: Search for Available Scenes with the Catalog API

# --- Define Your Search Parameters ---
# UTM Zone 31N (EPSG:32631) - used for Process API (easy meter-based resolution)
bbox_neeroeteren_utm = [
  693430.169618,
  5665633.359806,
  693679.88356,
  5665921.360572
]

# WGS84 lat/lon (EPSG:4326) - used for Catalog API
bbox_neeroeteren_wgs84 = [
    5.763344,
    51.109497,
    5.766912,
    51.112003
]
start_date = "2025-07-14T00:00:00Z"
end_date = "2025-09-18T23:59:59Z"

# --- Construct and Send the Catalog API Request ---
catalog_api_url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"

catalog_payload = {
    "collections": ["sentinel-1-grd"],
    "bbox": bbox_neeroeteren_wgs84,
    "datetime": f"{start_date}/{end_date}",
    "limit": 100 # Ask for up to 100 scenes, increase if needed.
}

all_features = []
if oauth and oauth.authorized:
    try:
        response = oauth.post(catalog_api_url, json=catalog_payload)
        response.raise_for_status()
        all_features = response.json().get('features', [])

        if not all_features:
            print("Catalog search returned 0 features.")
            print("Check: 1) time range, 2) bbox coordinates, 3) collection name")
        else:
            # Extract unique dates for summary
            unique_dates = sorted(list(set([f['properties']['datetime'] for f in all_features])))
            print(f"Catalog search successful. Found {len(all_features)} potential scenes.")
            print(f"Unique acquisition dates: {len(unique_dates)}")
            for date_str in unique_dates:
                print(f" - {date_str}")

    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error during catalog search: {err}")
        print(f"Response body: {err.response.text}")
else:
    print("Cannot perform catalog search: Authentication session not available.")


#%% 4. STAGE 2: Loop Through Dates and Download Each Scene

# Create a directory to save the output images
output_dir = "SAR_timeseries_output"
os.makedirs(output_dir, exist_ok=True)
print(f"Output images will be saved in the '{output_dir}' directory.")

downloaded_files = []

# Load evalscript from external file
with open('evalscript.js', 'r') as f:
    evalscript_linear_power = f.read()

# Loop through each unique acquisition date found in Stage 1
for i, feature in enumerate(all_features):
    # Extract metadata
    acquisition_time_str = feature["properties"]["datetime"]
    scene_id = feature["id"]
    
    print(f"\n--- Processing scene {i+1} of {len(all_features)}: {acquisition_time_str} ---")
    print(f"    Scene ID: {scene_id}")

    # Create a daily time window for the acquisition date to isolate this one scene.
    # Using the full day since Sentinel data isn't indexed at second-level precision.
    acquisition_time = datetime.strptime(acquisition_time_str, "%Y-%m-%dT%H:%M:%SZ")
    start_window = acquisition_time.strftime("%Y-%m-%dT00:00:00Z")
    end_window = acquisition_time.strftime("%Y-%m-%dT23:59:59Z")
    request_payload = {
        "input": {
            "bounds": {
                "bbox": bbox_neeroeteren_utm,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/32631"
                }
            },
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": { "from": start_window, "to": end_window },
                    "mosaickingOrder": "mostRecent",
                    "resolution": "HIGH",
                    "acquisitionMode": "IW",
                    "polarization": "DV",
                },
                "processing": {
                    "orthorectify": "true",
                    "demInstance": "COPERNICUS_30",
                    "backCoeff": "GAMMA0_TERRAIN"
                }
            }]
        },
        "output": {
            "resx": 10, # Resolution in meters
            "resy": 10,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
        },
        "evalscript": evalscript_linear_power
    }

    # --- Send the Process API request and save the file ---
    process_api_url = "https://sh.dataspace.copernicus.eu/api/v1/process"

    if oauth and oauth.authorized:
        try:
            response = oauth.post(process_api_url, json=request_payload)
            response.raise_for_status()
            image_bytes = response.content

            # Create a clean filename from the scene ID and date
            base_filename = scene_id.split('_')[0] + "_" + acquisition_time.strftime("%Y-%m-%d_%H%M%S")
            
            # --- SAVE THE IMAGE ---
            tif_filepath = os.path.join(output_dir, f"{base_filename}.tif")
            with open(tif_filepath, 'wb') as f:
                f.write(image_bytes)
            
            # --- SAVE THE METADATA ---
            json_filepath = os.path.join(output_dir, f"{base_filename}.json")
            with open(json_filepath, 'w') as f:
                json.dump(feature, f, indent=4)

            print(f"Successfully downloaded image: {tif_filepath}")
            print(f"Successfully saved metadata: {json_filepath}")
            downloaded_files.append(tif_filepath)

        except requests.exceptions.HTTPError as err:
            print(f"-> FAILED to download image for {acquisition_time_str}. Error: {err.response.text}")
    else:
        print("-> FAILED. Authentication session not available.")
        break # Exit the loop if not authenticated

print("\n--- Time-series download complete! ---")
#%%