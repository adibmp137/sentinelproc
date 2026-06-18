import cdsapi
import os
import numpy as np
from math import ceil, floor
from datetime import datetime
import pandas as pd
import zipfile
import netCDF4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(BASE_DIR, "era5_data")

loc = "Field"
# Configure with your study area coordinates
lat, lon = 0.0, 0.0
year = 2025
months = [4, 5, 6, 7, 8]

c = cdsapi.Client()

os.makedirs(SAVE_PATH, exist_ok=True)
os.chdir(SAVE_PATH)

for v in ["2m_temperature", "precipitation_flux", "10m_wind_speed",
          "vapour_pressure", "2m_dewpoint_temperature", "solar_radiation_flux"]:
    os.makedirs(os.path.join(SAVE_PATH, v), exist_ok=True)


def get_agERA5_data(variable, i, stat=["24_hour_mean"]):
    month = f"{i:02d}"
    zip_name = f"{loc}_download_{variable}_{month}_{year}.zip"

    c.retrieve(
        "sis-agrometeorological-indicators",
        {
            "version": "1_1",
            "format": "zip",
            "area": [ceil(lat), floor(lon), floor(lat), ceil(lon)],
            "variable": [variable],
            "statistic": stat,
            "year": [str(year)],
            "month": [month],
            "day": [f"{d:02d}" for d in range(1, 32)],
        },
        zip_name,
    )
    return zip_name


def get_agERA5_data2(variable, i):
    month = f"{i:02d}"
    zip_name = f"{loc}_download_{variable}_{month}_{year}.zip"

    c.retrieve(
        "sis-agrometeorological-indicators",
        {
            "version": "1_1",
            "format": "zip",
            "area": [ceil(lat), floor(lon), floor(lat), ceil(lon)],
            "variable": [variable],
            "year": [str(year)],
            "month": [month],
            "day": [f"{d:02d}" for d in range(1, 32)],
        },
        zip_name,
    )
    return zip_name


def unzip_data(zip_name, variable):
    with zipfile.ZipFile(zip_name, "r") as z:
        z.extractall(variable)


print("Downloading dewpoint temperature, wind speed, vapour pressure...")
for i in months:
    for v in ["2m_dewpoint_temperature", "10m_wind_speed", "vapour_pressure"]:
        z = get_agERA5_data(v, i)
        unzip_data(z, v)

print("Downloading precipitation, solar radiation...")
for i in months:
    for v in ["precipitation_flux", "solar_radiation_flux"]:
        z = get_agERA5_data2(v, i)
        unzip_data(z, v)

print("Downloading min/max temperature...")
for i in months:
    v = "2m_temperature"
    z = get_agERA5_data(v, i, stat=["24_hour_maximum", "24_hour_minimum"])
    unzip_data(z, v)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def read_netcdf(folder):
    os.chdir(SAVE_PATH)
    folder_path = os.path.join(SAVE_PATH, folder)
    os.chdir(folder_path)
    data_list = []
    dates = []

    for fname in os.listdir(folder_path):
        if not fname.endswith(".nc"):
            continue
        nc = netCDF4.Dataset(fname, mode="r")
        keys = [*nc.variables.keys()]
        longs = nc.variables["lon"][:]
        lats = nc.variables["lat"][:]
        date = nc.variables["time"][:].data[0]
        data = nc.variables[keys[0]][:].data * ~nc.variables[keys[0]][:].mask
        lon_i, _ = find_nearest(longs, lon)
        lat_i, _ = find_nearest(lats, lat)
        value = data[0, lat_i, lon_i]
        data_list.append(value)
        dates.append(date)

    return dates, data_list


def read_netcdf_minmaxT(folder):
    os.chdir(SAVE_PATH)
    folder_path = os.path.join(SAVE_PATH, folder)
    os.chdir(folder_path)
    data_list_max = []
    data_list_min = []
    dates_min = []
    dates_max = []

    for fname in os.listdir(folder_path):
        if not fname.endswith(".nc"):
            continue
        nc = netCDF4.Dataset(fname, mode="r")
        keys = [*nc.variables.keys()]
        longs = nc.variables["lon"][:]
        lats = nc.variables["lat"][:]
        date = nc.variables["time"][:].data[0]
        data = nc.variables[keys[0]][:].data * ~nc.variables[keys[0]][:].mask
        lon_i, _ = find_nearest(longs, lon)
        lat_i, _ = find_nearest(lats, lat)
        value = data[0, lat_i, lon_i]
        if "-Max-" in fname:
            data_list_max.append(value)
            dates_max.append(date)
        elif "-Min-" in fname:
            data_list_min.append(value)
            dates_min.append(date)

    return dates_min, dates_max, data_list_min, data_list_max


print("\nReading NetCDF files...")
dates, data_list = read_netcdf("2m_dewpoint_temperature")
df_dew_temp = pd.DataFrame({"Date": dates, "Tdew (K)": data_list})

dates, data_list = read_netcdf("10m_wind_speed")
df_wind = pd.DataFrame({"Date": dates, "u (m/s)": data_list})

dates, data_list = read_netcdf("solar_radiation_flux")
df_solrad = pd.DataFrame({"Date": dates, "Solar radiation (J m-2 day-1)": data_list})

dates, data_list = read_netcdf("vapour_pressure")
df_vap = pd.DataFrame({"Date": dates, "Vapor pressure (hPa)": data_list})

dates, data_list = read_netcdf("precipitation_flux")
df_prec = pd.DataFrame({"Date": dates, "Precipitation (mm)": data_list})

dates_min, dates_max, data_list_min, data_list_max = read_netcdf_minmaxT("2m_temperature")
df_temp_min = pd.DataFrame({"Date": dates_min, "Tmin (K)": data_list_min})
df_temp_max = pd.DataFrame({"Date": dates_max, "Tmax (K)": data_list_max})

os.chdir(SAVE_PATH)

for df, name in [(df_dew_temp, "df_dew_temp.csv"),
                  (df_wind, "df_wind.csv"),
                  (df_solrad, "df_solrad.csv"),
                  (df_vap, "df_vap.csv"),
                  (df_prec, "df_prec.csv"),
                  (df_temp_min, "df_temp_min.csv"),
                  (df_temp_max, "df_temp_max.csv")]:
    df.sort_values(by="Date", inplace=True)
    df.to_csv(os.path.join(SAVE_PATH, name), index=False)

print(f"\nERA5 data saved to {SAVE_PATH}")
print("Done!")
