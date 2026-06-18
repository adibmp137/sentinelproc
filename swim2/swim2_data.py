import os
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "..", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")

GROWING_START = datetime(2025, 4, 24).date()
GROWING_END = datetime(2025, 6, 15).date()
PLANTING_DATE = datetime(2025, 4, 1)
OBS_START_DATE = datetime(2025, 4, 24)
YEAR = '2025'
CASES = ['MZ1', 'MZ2']

SENSOR_CAL_A = -0.006
SENSOR_CAL_B = 1.26

EXCEL_EPOCH = datetime(1899, 12, 30)


def to_serial_date(dt):
    if isinstance(dt, datetime):
        return (dt - EXCEL_EPOCH).days
    return (pd.Timestamp(dt) - EXCEL_EPOCH).days


def load_eto():
    path = os.path.join(INPUT_DIR, "eto.csv")
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df[df['date'].dt.year == 2025].copy()
    df['year'] = YEAR
    df['Date'] = df['date'].apply(to_serial_date)
    for case in CASES:
        df[case] = df['Daily ETo (mm/day)']
    return df[['year', 'Date'] + CASES].reset_index(drop=True)


def load_precipitation():
    path = os.path.join(INPUT_DIR, "precipitation.csv")
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year == 2025].copy()
    full_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
    df = df.set_index('date').reindex(full_range).fillna(0)
    df.index.name = 'date'
    df = df.reset_index()
    df['year'] = YEAR
    df['Date'] = df['date'].apply(to_serial_date)
    df.rename(columns={'precipitation_mm': CASES[0]}, inplace=True)
    for case in CASES[1:]:
        df[case] = df[CASES[0]]
    return df[['year', 'Date'] + CASES].reset_index(drop=True)


def load_irrigation():
    path = os.path.join(INPUT_DIR, "irrigation.csv")
    df = pd.read_csv(path, sep=';')
    df['date'] = pd.to_datetime(df['date'], format='%d-%b-%y')
    df = df[df['date'].dt.year == 2025].copy()
    df['year'] = YEAR
    df['Date'] = df['date'].apply(to_serial_date)
    df.rename(columns={'irrig_mm': CASES[0]}, inplace=True)
    for case in CASES[1:]:
        df[case] = df[CASES[0]]
    return df[['year', 'Date'] + CASES].reset_index(drop=True)


def load_soildata():
    cases_data = []
    bd_map = {'MZ1': 1.69, 'MZ2': 1.56}
    soil_type_map = {'MZ1': 'Aba1', 'MZ2': 'Abp(c)'}

    for case in CASES:
        bd = bd_map[case]
        row = {
            'year': YEAR,
            'case': case,
            'note': f'Field {case}',
            'Irrigatiemethode': 'haspel',
            'Textuurklasse': 'Loam (A)',
            'Bodemtype': soil_type_map[case],
            # NOTE: Configure with actual lat/lon for your study area (removed for farm privacy)
            'Lat': None,
            'Long': None,
            'Diepte': '0-30',
            'bd_gem': bd,
            'bd_1': bd, 'bd_2': np.nan, 'bd_3': np.nan,
            'pF0_1': np.nan, 'pF0_2': np.nan, 'pF0_3': np.nan,
            'pF2_1': np.nan, 'pF2_2': np.nan, 'pF2_3': np.nan,
            'pF2.7_1': np.nan, 'pF2.7_2': np.nan, 'pF2.7_3': np.nan,
            'pF4.2_1': np.nan, 'pF4.2_2': np.nan, 'pF4.2_3': np.nan,
            'v0': np.nan,
            'Teelt1': 'ui',
            'Teelt2': np.nan,
            'Planting1': to_serial_date(PLANTING_DATE),
            'Planting2': np.nan,
            'TOC_1': np.nan, 'TOC_2': np.nan, 'TOC_3': np.nan,
            'pH-KCl_1': np.nan, 'pH-KCl_2': np.nan, 'pH-KCl_3': np.nan,
            'pH-H2O': np.nan,
            'klei (< 2\u00b5m)(%)_1': 15, 'klei (< 2\u00b5m)(%)_2': np.nan, 'klei (< 2\u00b5m)(%)_3': np.nan,
            'zand (50-2000 \u00b5m)(%)_1': 8, 'zand (50-2000 \u00b5m)(%)_2': np.nan, 'zand (50-2000 \u00b5m)(%)_3': np.nan,
            'leem (2-50 \u00b5m)(%)_1': 77, 'leem (2-50 \u00b5m)(%)_2': np.nan, 'leem (2-50 \u00b5m)(%)_3': np.nan,
            'Textuuranalyse_BDB': np.nan,
            'Textuuranalyse_BE': 'Leem (A)',
            'Textuuranalyse_USDA': 'Silt loam',
            'wcr': np.nan, 'wcs': np.nan, 'alpha': np.nan, 'n': np.nan, 'm': np.nan,
        }
        cases_data.append(row)

    df = pd.DataFrame(cases_data)
    df['year'] = df['year'].astype(str)
    return df


def _parse_ground_sample_date(date_str):
    try:
        dt = datetime.strptime(date_str, '%d/%m/%Y')
    except ValueError:
        try:
            dt = datetime.strptime(date_str, '%m/%d/%Y')
        except ValueError:
            dt = pd.Timestamp(date_str)
    return dt

def load_soilobs():
    path = os.path.join(INPUT_DIR, "ground_sample.csv")
    df = pd.read_csv(path, sep=';')
    df['date'] = df['date'].apply(_parse_ground_sample_date)
    df = df[(df['date'].dt.date >= GROWING_START) & (df['date'].dt.date <= datetime(2025, 7, 15).date())].copy()
    df['Date'] = df['date'].apply(to_serial_date)
    df.rename(columns={
        'gravimetric_sm': '0_30(grav%)',
        'location': 'Sensornr',
    }, inplace=True)
    df['30_60(grav%)'] = np.nan
    df['0_5(grav%)'] = np.nan
    df['60_90(grav%)'] = np.nan
    df['Naam'] = ''
    df['year'] = YEAR
    df['info'] = 'Field onion'
    return df[['Date', '0_30(grav%)', '30_60(grav%)', '0_5(grav%)',
               'Sensornr', 'Naam', 'year', 'info', '60_90(grav%)']].reset_index(drop=True)


def load_sensordata(case):
    path = os.path.join(INPUT_DIR, f"{case}.csv")
    df = pd.read_csv(path)
    df['Datetime'] = pd.to_datetime(df['time_parsed'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    df['ID'] = range(len(df))
    df['Sensor'] = case
    df.rename(columns={
        'adc0': 'Adc0 (mV)',
        'adc1': 'Adc1 (mV)',
        'adc2': 'Adc2 (mV)',
        'vmc0': 'vwc0 (m3/m3)',
        'vmc1': 'vwc1 (m3/m3)',
        'vmc2': 'vwc2 (m3/m3)',
        'relPrecipitation': 'pluvio',
        'temperature': 'temp',
    }, inplace=True)
    return df[['Datetime', 'ID', 'Sensor', 'Adc0 (mV)', 'Adc1 (mV)', 'Adc2 (mV)',
               'vwc0 (m3/m3)', 'vwc1 (m3/m3)', 'vwc2 (m3/m3)', 'pluvio', 'temp']]


def load_sensor_overview():
    rows = []
    for case in CASES:
        rows.append({
            'sensor': case,
            int(2025): 'Field',
            'Verantwoordelijke': 'Adib',
        })
    df = pd.DataFrame(rows)
    df.sort_values(by=int(2025), ignore_index=True, inplace=True)
    df.dropna(subset=[int(2025)], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_all():
    return {
        'soildata': load_soildata(),
        'eto': load_eto(),
        'precipitation': load_precipitation(),
        'irrigation': load_irrigation(),
        'soilobs': load_soilobs(),
        'sensor_overview': load_sensor_overview(),
        'crop': pd.read_csv(os.path.join(BASE_DIR, "crop_FAO.csv"), encoding='unicode_escape'),
    }


if __name__ == '__main__':
    data = load_all()
    for name, df in data.items():
        print(f"\n{'='*40}")
        print(f"{name}: {df.shape}")
        print(df.head(3).to_string())
