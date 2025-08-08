#%% 1. Import Libraries & Configure Credentials
import os
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import io
from dotenv import load_dotenv
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import geopandas as gpd
from matplotlib.path import Path 

load_dotenv() 

CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    print("FATAL: CLIENT_ID or CLIENT_SECRET not found.")
    print("Please ensure you have a .env file with your credentials.")
else:
    print("Credentials successfully loaded from .env file.")

print("Configuration loaded and ready.")


#%% 2. Authenticate and Create a Session
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


#%% 3. Define the Sentinel Hub Process API Request
bbox_kinrooi = [
    694512.806609,
    5666404.740308,
    694896.980069,
    5666790.60716
]
time_range = "2023-06-01T00:00:00Z/2023-06-30T23:59:59Z"

# Load evalscript from external file
with open('evalscript.js', 'r') as f:
    evalscript_linear_power = f.read()

# Construct the full request payload for the Process API
request_payload = {
    "input": {
        "bounds": {
            "bbox": bbox_kinrooi,
            "properties": {
                "crs": "http://www.opengis.net/def/crs/EPSG/0/32631"
            }
        },
        "data": [
            {
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {
                        "from": time_range.split('/')[0],
                        "to": time_range.split('/')[1]
                    },
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
            }
        ]
    },
    "output": {
        "resx": 10, # Resolution in meters
        "resy": 10,
        "responses": [
        {
            "identifier": "default",
            "format": {
            "type": "image/tiff"
            }
        }
        ]
    },
    "evalscript": evalscript_linear_power
}

print("Request payload defined.")
print(f"Requesting data for BBox: {bbox_kinrooi} in time range: {time_range}")


#%% 4. Send the Request and Fetch Data
process_api_url = "https://sh.dataspace.copernicus.eu/api/v1/process"

response = oauth.post(process_api_url, json=request_payload)
if response.status_code == 200:
    image_bytes = response.content
    print(f"Successfully received image data ({len(image_bytes)} bytes).")
else:
    print(f"Request failed: {response.status_code}")
    image_bytes = None


#%% 5. Analyze and Visualize the Resulting Image
if not image_bytes:
    print("No image data available for visualization.")
    exit()

src = rasterio.open(io.BytesIO(image_bytes))
print(f"Opened image with {src.count} bands.")
print(f"CRS: {src.crs}")

vv_band_linear = src.read(1)
vh_band_linear = src.read(2)
datamask = src.read(3)

vv_band_db = 10 * np.log10(vv_band_linear + 1e-10)  # Avoid log(0) by adding a small constant
vh_band_db = 10 * np.log10(vh_band_linear + 1e-10)


# Create masks for different pixel types
nodata_mask = datamask == 0
vv_noise_mask = vv_band_db < -22  # Noise mask specific to VV
vh_noise_mask = vh_band_db < -22  # Noise mask specific to VH
vv_invalid_mask = nodata_mask | vv_noise_mask  # VV-specific invalid pixels
vh_invalid_mask = nodata_mask | vh_noise_mask  # VH-specific invalid pixels

# Create custom colormap for visualization
cividis_cmap = plt.cm.cividis_r.copy()  # cividis_r reversed: blue to yellow
cividis_cmap.set_bad(color='red')  # NaN values appear as red

fig, axs = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle('Sentinel-1 SAR Backscatter Analysis - Kinrooi Agricultural Field (June 2023)', fontsize=14, fontweight='bold')

# Prepare display arrays - set invalid pixels to NaN
vv_display = vv_band_db.copy()
vh_display = vh_band_db.copy()

# Set invalid pixels to NaN
vv_display[vv_invalid_mask] = np.nan  # VV-specific invalid pixels
vh_display[vh_invalid_mask] = np.nan  # VH-specific invalid pixels

# Display the data
im1 = axs[0].imshow(vv_display, cmap=cividis_cmap, vmin=-22, vmax=0, extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top])
axs[0].set_title('VV Polarization (dB)')
fig.colorbar(im1, ax=axs[0], orientation='horizontal', label='Backscatter (dB)')

im2 = axs[1].imshow(vh_display, cmap=cividis_cmap, vmin=-22, vmax=0, extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top])
axs[1].set_title('VH Polarization (dB)')
fig.colorbar(im2, ax=axs[1], orientation='horizontal', label='Backscatter (dB)')

# Load shapefile and reproject to UTM31N
shp = gpd.read_file('input/VL_kinrooi_2023.shp')
shp = shp.set_crs('EPSG:31370')  # Set BD72 / Belgian Lambert 72
shp_utm = shp.to_crs('EPSG:32631')

# Format axes
for ax in axs:
    ax.set_xlabel('Easting (m) - UTM Zone 31N (EPSG:32631)')
    ax.set_ylabel('Northing (m) - UTM Zone 31N (EPSG:32631)')
    ax.set_aspect('equal')
    
    # Fix Y-axis to show full numbers instead of scientific notation
    ax.ticklabel_format(style='plain', axis='y', useOffset=False)
    
    # Rotate X-axis labels for better readability  
    ax.tick_params(axis='x', rotation=45)
    
    # Force Y-axis to show full numbers without offset
    ax.yaxis.get_major_formatter().set_useOffset(False)
    
    # Create white layer with vector hole from shapefile
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Outer rectangle (white layer boundary)
    outer_coords = [(xlim[0], ylim[0]), (xlim[1], ylim[0]), (xlim[1], ylim[1]), (xlim[0], ylim[1]), (xlim[0], ylim[0])]
    outer_codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    
    # Inner boundary (shapefile hole) - get first geometry
    geom = shp_utm.geometry.iloc[0]
    if geom.geom_type == 'Polygon':
        inner_coords = list(geom.exterior.coords)
    else:
        inner_coords = list(geom.coords)
    
    inner_codes = [Path.MOVETO] + [Path.LINETO] * (len(inner_coords) - 2) + [Path.CLOSEPOLY]
    
    # Combine paths - outer with inner hole
    compound_coords = outer_coords + inner_coords
    compound_codes = outer_codes + inner_codes
    
    # Create and apply compound path
    compound_path = Path(compound_coords, compound_codes)
    patch = mpatches.PathPatch(compound_path, facecolor='white', alpha=1, zorder=10)
    ax.add_patch(patch)

# Add legend
red_patch = mpatches.Patch(color='red', label='No Data/Noise')
fig.legend(handles=[red_patch], loc='upper right')

plt.tight_layout()
plt.show()

src.close()
# %%
