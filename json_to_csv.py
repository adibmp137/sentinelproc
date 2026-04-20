import os
import json
import csv
from pathlib import Path

# Define the input and output paths
input_dir = "SAR_timeseries_output"
output_csv = "SAR_metadata.csv"

# Collect all JSON files
json_files = sorted(Path(input_dir).glob("*.json"))

if not json_files:
    print("No JSON files found!")
    exit()

print(f"Found {len(json_files)} JSON files")

# Read all JSON files and flatten the data
all_rows = []

for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Create a flattened row (exclude geometry.coordinates)
    row = {}

    # Top-level fields
    row['id'] = data.get('id', '')
    row['stac_version'] = data.get('stac_version', '')
    row['type'] = data.get('type', '')
    row['collection'] = data.get('collection', '')

    # BBox (keep it, it's just 4 numbers)
    bbox = data.get('bbox', [])
    if bbox:
        row['bbox_min_lon'] = bbox[0]
        row['bbox_min_lat'] = bbox[1]
        row['bbox_max_lon'] = bbox[2]
        row['bbox_max_lat'] = bbox[3]

    # Geometry type (but not coordinates)
    if 'geometry' in data:
        row['geometry_type'] = data['geometry'].get('type', '')

    # Properties (flatten all properties)
    properties = data.get('properties', {})
    for key, value in properties.items():
        # Handle list values (like polarizations)
        if isinstance(value, list):
            row[f'properties_{key}'] = ', '.join(map(str, value))
        else:
            row[f'properties_{key}'] = value

    # Assets (just the href and type)
    assets = data.get('assets', {})
    for asset_name, asset_data in assets.items():
        if asset_data:
            row[f'asset_{asset_name}_href'] = asset_data.get('href', '')
            row[f'asset_{asset_name}_type'] = asset_data.get('type', '')

    all_rows.append(row)

# Get all unique column names
all_columns = set()
for row in all_rows:
    all_columns.update(row.keys())

# Sort columns for better readability
column_order = sorted(all_columns)

# Write to CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=column_order)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Successfully created {output_csv} with {len(all_rows)} rows and {len(column_order)} columns")
print(f"Columns: {', '.join(column_order[:10])}..." if len(column_order) > 10 else f"Columns: {', '.join(column_order)}")
