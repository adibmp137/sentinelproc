import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import csv
import os
from datetime import datetime

import pyeto
from pyeto import fao, convert

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
ERA5_DIR = os.path.join(BASE_DIR, "era5_data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "input")

loc = "Field"
# Configure with your study area coordinates
lat, lon = 0.0, 0.0
alt = 77

os.makedirs(OUTPUT_DIR, exist_ok=True)


def ConvertToDate(serial_date):
    serial_date = int(serial_date)
    return datetime.fromordinal(datetime(1899, 12, 30).toordinal() + serial_date)


df_dew_temp = pd.read_csv(os.path.join(ERA5_DIR, "df_dew_temp.csv"))
df_wind = pd.read_csv(os.path.join(ERA5_DIR, "df_wind.csv"))
df_solrad = pd.read_csv(os.path.join(ERA5_DIR, "df_solrad.csv"))
df_vap = pd.read_csv(os.path.join(ERA5_DIR, "df_vap.csv"))
df_temp_min = pd.read_csv(os.path.join(ERA5_DIR, "df_temp_min.csv"))
df_temp_max = pd.read_csv(os.path.join(ERA5_DIR, "df_temp_max.csv"))
df_prec = pd.read_csv(os.path.join(ERA5_DIR, "df_prec.csv"))

tdew_list = df_dew_temp["Tdew (K)"].tolist()
ws_10_list = df_wind["u (m/s)"].tolist()
sol_rad_list = df_solrad["Solar radiation (J m-2 day-1)"].tolist()
ea_list = df_vap["Vapor pressure (hPa)"].tolist()
prec_list = df_prec["Precipitation (mm)"].tolist()
tmin_list = df_temp_min["Tmin (K)"].tolist()
tmax_list = df_temp_max["Tmax (K)"].tolist()

ea_list = np.array(ea_list) / 10.0
sol_rad_list = np.array(sol_rad_list) / 1e6

lat_rad = pyeto.deg2rad(lat)
dates_serial = df_prec["Date"].tolist()
dates = [ConvertToDate(d) for d in dates_serial]
doy_list = [d.timetuple().tm_yday for d in dates]
n_days = len(dates)

ws_list = [pyeto.wind_speed_2m(ws, 10) for ws in ws_10_list]
avp_list = ea_list

atmos_pres = fao.atm_pressure(alt)
psy = pyeto.psy_const(atmos_pres)
G = 0

ETo_list = np.array([])

for i in range(n_days):
    tmin = tmin_list[i]
    tmax = tmax_list[i]
    t = np.mean([tmin, tmax])
    ws = ws_list[i]

    svp = pyeto.mean_svp(convert.kelvin2celsius(tmin), convert.kelvin2celsius(tmax))
    svp_tmin = pyeto.svp_from_t(convert.kelvin2celsius(tmin))
    svp_tmax = pyeto.svp_from_t(convert.kelvin2celsius(tmax))

    avp = avp_list[i]
    delta_svp = pyeto.delta_svp(convert.kelvin2celsius(t))

    day_of_year = doy_list[i]
    sol_dec = pyeto.sol_dec(day_of_year)
    sha = pyeto.sunset_hour_angle(lat_rad, sol_dec)
    ird = pyeto.inv_rel_dist_earth_sun(day_of_year)
    et_rad = pyeto.et_rad(lat_rad, sol_dec, sha, ird)
    cs_rad = pyeto.cs_rad(alt, et_rad)
    sol_rad = sol_rad_list[i]
    ni_sw_rad = pyeto.net_in_sol_rad(sol_rad, albedo=0.23)
    no_lw_rad = pyeto.net_out_lw_rad(tmin, tmax, sol_rad, cs_rad, avp)
    Rn = pyeto.net_rad(ni_sw_rad, no_lw_rad)

    ETo = pyeto.fao56_penman_monteith(Rn, t, ws, svp, avp, delta_svp, psy, shf=G)
    ETo_list = np.append(ETo_list, ETo)

print(f"ETo computed for {n_days} days ({dates[0].date()} to {dates[-1].date()})")
print(f"Mean ETo: {np.nanmean(ETo_list):.2f} mm/day")
print(f"Min ETo: {np.nanmin(ETo_list):.2f} mm/day")
print(f"Max ETo: {np.nanmax(ETo_list):.2f} mm/day")
print(f"Total ETo: {np.nansum(ETo_list):.1f} mm")

csv_path = os.path.join(OUTPUT_DIR, "eto.csv")
with open(csv_path, mode="w", newline="") as f:
    w = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    w.writerow(["DOY", "Date", "Daily ETo (mm/day)"])
    w.writerows([doy_list[i], dates[i].strftime("%d/%m/%Y"), round(ETo_list[i], 4)]
                for i in range(n_days))
print(f"Saved: {csv_path}")

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor("white")
ax.plot(dates, ETo_list, color="darkorange", linewidth=1.5)
ax.set_xlim(dates[0], dates[-1])
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax.set_axisbelow(True)
ax.yaxis.grid(linewidth=0.5, color="lightgray")
ax.set_ylim(bottom=0)
ax.set_xlabel("Date", labelpad=10)
ax.set_ylabel("ETo [mm/day]", labelpad=10)
ax.set_title(f"Daily ETo 2025\n{loc}")
plt.tight_layout()

chart_path = os.path.join(OUTPUT_DIR, "eto.png")
plt.savefig(chart_path, dpi=150, facecolor="white")
plt.close()
print(f"Saved: {chart_path}")
print("Done!")
