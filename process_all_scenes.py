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

load_dotenv()

CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    print("FATAL: CLIENT_ID or CLIENT_SECRET not found.")
    print("Please ensure you have a .env file with your credentials.")
else:
    print("Credentials successfully loaded from .env file.")


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


# Configure with your study area bounding box
# UTM coordinates (e.g., EPSG:32631)
bbox_utm = [0.0, 0.0, 0.0, 0.0]
# WGS84 coordinates (lon, lat)
bbox_wgs84 = [0.0, 0.0, 0.0, 0.0]
start_date = "2025-04-24T00:00:00Z"
end_date = "2025-07-15T23:59:59Z"

catalog_api_url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"

catalog_payload = {
    "collections": ["sentinel-1-grd"],
    "bbox": bbox_wgs84,
    "datetime": f"{start_date}/{end_date}",
    "limit": 100
}

all_features = []
if oauth and oauth.authorized:
    try:
        response = oauth.post(catalog_api_url, json=catalog_payload)
        response.raise_for_status()
        all_features = response.json().get('features', [])

        if not all_features:
            print("Catalog search returned 0 features.")
        else:
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


output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SAR_timeseries_output")
os.makedirs(output_dir, exist_ok=True)
print(f"Output images will be saved in the '{output_dir}' directory.")

downloaded_files = []

with open('evalscript.js', 'r') as f:
    evalscript_linear_power = f.read()

for i, feature in enumerate(all_features):
    acquisition_time_str = feature["properties"]["datetime"]
    scene_id = feature["id"]
    
    print(f"\n--- Processing scene {i+1} of {len(all_features)}: {acquisition_time_str} ---")
    print(f"    Scene ID: {scene_id}")

    acquisition_time = datetime.strptime(acquisition_time_str, "%Y-%m-%dT%H:%M:%SZ")
    start_window = acquisition_time.strftime("%Y-%m-%dT00:00:00Z")
    end_window = acquisition_time.strftime("%Y-%m-%dT23:59:59Z")
    request_payload = {
        "input": {
            "bounds": {
                "bbox": bbox_utm,
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
            "resx": 10,
            "resy": 10,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
        },
        "evalscript": evalscript_linear_power
    }

    process_api_url = "https://sh.dataspace.copernicus.eu/api/v1/process"

    if oauth and oauth.authorized:
        try:
            response = oauth.post(process_api_url, json=request_payload)
            response.raise_for_status()
            image_bytes = response.content

            base_filename = scene_id.split('_')[0] + "_" + acquisition_time.strftime("%Y-%m-%d_%H%M%S")
            
            tif_filepath = os.path.join(output_dir, f"{base_filename}.tif")
            with open(tif_filepath, 'wb') as f:
                f.write(image_bytes)
            
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
        break

print("\n--- Time-series download complete! ---")
