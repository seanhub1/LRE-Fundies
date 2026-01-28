import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import jwt
import time
import plotly.graph_objects as go
import json
from pathlib import Path
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
from io import BytesIO


st.set_page_config(page_title="Forecasts", layout="wide", initial_sidebar_state="collapsed")


st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    [data-testid="stHeader"] {
        background-color: #0e1117;
    }
    [data-testid="stSidebar"] {
        background-color: #0e1117;
    }
    </style>
""", unsafe_allow_html=True)

# Auto-refresh once per hour at 16:30 past each hour Central Time (e.g., 1:16:30, 2:16:30, etc.)
def get_refresh_info():
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo('America/Chicago'))
    target_minute = 16
    target_second = 30
    
    # Calculate seconds into current hour
    current_seconds_into_hour = now.minute * 60 + now.second
    target_seconds_into_hour = target_minute * 60 + target_second  # 990 seconds = 16:30
    
    if current_seconds_into_hour < target_seconds_into_hour:
        # Refresh point is later this hour
        seconds_until = target_seconds_into_hour - current_seconds_into_hour
        next_refresh = now.replace(minute=target_minute, second=target_second, microsecond=0)
    else:
        # Refresh point is next hour
        seconds_remaining_this_hour = 3600 - current_seconds_into_hour
        seconds_until = seconds_remaining_this_hour + target_seconds_into_hour
        next_refresh = (now + timedelta(hours=1)).replace(minute=target_minute, second=target_second, microsecond=0)
    
    return seconds_until, next_refresh.strftime('%I:%M:%S %p CT')

refresh_seconds, next_refresh_time = get_refresh_info()
st.markdown(f'<meta http-equiv="refresh" content="{refresh_seconds}">', unsafe_allow_html=True)

# Load from Streamlit secrets (no .env file needed)
baseurl = "https://api-markets.meteologica.com/api/v1/"

# Cache file location (use temp directory for cloud)
CACHE_FILE = Path("/tmp/historical_cache.json")

def make_get_request(endpoint, query_params):
    url = baseurl + endpoint
    return requests.get(url, params=query_params)

def make_post_request(endpoint, json_body):
    url = baseurl + endpoint
    return requests.post(url, json=json_body)

def get_new_token(user, password):
    response = make_post_request("login", {"user": user, "password": password})
    try:
        return response.json()["token"]
    except Exception as e:
        raise RuntimeError(f"Could not get token: {response.text}") from e

def refresh_token():
    response = make_get_request("keepalive", {"token": os.getenv("API_TOKEN")})
    try:
        return response.json()["token"]
    except Exception as e:
        raise RuntimeError(f"Could not refresh token: {response.text}") from e

def get_or_refresh_stored_token():
    token = os.getenv("API_TOKEN")
    if not token or time.time() > jwt.decode(token, options={"verify_signature": False})["exp"]:
        user = st.secrets["meteologica"]["API_USER"]
        password = st.secrets["meteologica"]["API_PASSWORD"]
        new_token = get_new_token(user, password)
        os.environ["API_TOKEN"] = new_token
        return new_token
    else:
        exp = jwt.decode(token, options={"verify_signature": False})["exp"]
        if exp - time.time() < 300:
            new_token = refresh_token()
            os.environ["API_TOKEN"] = new_token
            return new_token
        return token

def get_content_data(content_id):
    token = get_or_refresh_stored_token()
    endpoint = f"contents/{content_id}/data"
    response = make_get_request(endpoint, {"token": token})
    if response.status_code == 200:
        try:
            return response.json()
        except:
            return None
    return None

def ercot_token():
    uid = st.secrets["ercot"]["username"]
    pwd = st.secrets["ercot"]["password"]
    SUBSCRIPTION = st.secrets["ercot"]["subscription"]
    AUTH_URL = (
        f"https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/"
        f"token?username={uid}&password={pwd}"
        f"&scope=openid+fec253ea-0d06-4272-a5e6-b478baeecd70+offline_access"
        f"&client_id=fec253ea-0d06-4272-a5e6-b478baeecd70"
        f"&response_type=id_token"
        f"&grant_type=password"
    )
    auth_response = requests.post(AUTH_URL)
    if auth_response.ok:
        access_token = auth_response.json().get("access_token")
        headers = {"Authorization": "Bearer " + access_token, "Ocp-Apim-Subscription-Key": SUBSCRIPTION}
        return headers
    st.error(f"Error in Authentication: {auth_response.text}")
    return None

def ercot_token_outages():
    uid = st.secrets["ercot"]["username"]
    pwd = st.secrets["ercot"]["password"]
    SUBSCRIPTION = st.secrets["ercot"]["subscription"]
    AUTH_URL = (
        f"https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/"
        f"B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
    )
    data = {
        'username': uid,
        'password': pwd,
        'scope': 'openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access',
        'client_id': 'fec253ea-0d06-4272-a5e6-b478baeecd70',
        'response_type': 'id_token',
        'grant_type': 'password',
    }
    resp = requests.post(AUTH_URL, data=data)
    if resp.ok:
        access_token = resp.json().get("access_token")
        headers = {
            "Authorization": "Bearer " + access_token,
            "Ocp-Apim-Subscription-Key": SUBSCRIPTION
        }
        return headers
    raise Exception(f"Error in Authentication: {resp.text}")

def fetch_outage_data_robust(url, headers, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=(30, 60))
            if response.ok:
                return response.json()
        except (requests.exceptions.Timeout, requests.exceptions.RequestException):
            if attempt < max_retries - 1:
                time.sleep(2)
            continue
    return {}

# PJM API Functions
def pjm_api_call(url, max_retries=3):
    pjm_headers = {
        'Ocp-Apim-Subscription-Key': st.secrets["pjm"]["subscription_key"],
    }
    for attempt in range(max_retries):
        try:
            result = requests.get(url, headers=pjm_headers, timeout=(30, 60))
            if result.ok:
                return result.json()
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(5)
            continue
    return {}

@st.cache_data(ttl=3600)
def fetch_pjm_load_forecast(cache_time):
    url = (
        "https://api.pjm.com/api/v1/load_frcstd_7_day"
        "?download=true&rowCount=10000"
        "&sort=forecast_datetime_beginning_utc&order=Asc&startRow=1"
    )
    data = pjm_api_call(url)
    if data:
        df = pd.json_normalize(data)
        if not df.empty:
            df['forecast_datetime_beginning_ept'] = pd.to_datetime(df['forecast_datetime_beginning_ept'])
            df['deliveryDate'] = df['forecast_datetime_beginning_ept'].dt.date
            df['HE'] = df['forecast_datetime_beginning_ept'].dt.hour + 1
            df['value'] = pd.to_numeric(df['forecast_load_mw'], errors='coerce')
            return df, cache_time
    return None, cache_time

@st.cache_data(ttl=3600)
def fetch_pjm_outages(cache_time):
    url = (
        "https://api.pjm.com/api/v1/gen_outages_by_type"
        "?download=true&rowCount=10000"
        "&sort=forecast_execution_date_ept&order=Desc&startRow=1"
        "&fields=forecast_execution_date_ept,forecast_date,region,"
        "total_outages_mw,planned_outages_mw,maintenance_outages_mw,forced_outages_mw"
    )
    data = pjm_api_call(url)
    if data:
        df = pd.json_normalize(data)
        if not df.empty:
            df['forecast_execution_date_ept'] = pd.to_datetime(df['forecast_execution_date_ept'])
            df['forecast_date'] = pd.to_datetime(df['forecast_date']).dt.date
            df = df[df['region'].str.upper() == 'PJM RTO']
            df = df[df['forecast_date'] >= datetime.today().date()]
            df = df.sort_values('forecast_execution_date_ept', ascending=False)
            df = df.groupby('forecast_date', as_index=False).first()
            df = df.sort_values('forecast_date')
            df['operatingDate'] = df['forecast_date']
            df['totalOutages'] = pd.to_numeric(df['total_outages_mw'], errors='coerce')
            return df, cache_time
    return None, cache_time

@st.cache_data(ttl=3600)
def fetch_ercot_wind_by_region(cache_time):
    """Fetch ERCOT wind forecast by region"""
    auths = ercot_token()
    if not auths:
        return None, None

    today = datetime.today()
    dateStart = today.strftime("%Y-%m-%d")
    dateEnd = (today + timedelta(days=7)).strftime("%Y-%m-%d")

    url = f"https://api.ercot.com/api/public-reports/np4-742-cd/wpp_hrly_actual_fcast_geo?deliveryDateFrom={dateStart}&deliveryDateTo={dateEnd}"

    try:
        response = requests.get(url, headers=auths, timeout=(60, 120))
        if response.ok:
            results = response.json()
            data = results.get("data", [])
            fields = [x['name'] for x in results.get("fields", [])]
            df = pd.DataFrame(data, columns=fields)

            if not df.empty:
                df['deliveryDate'] = pd.to_datetime(df['deliveryDate']).dt.date
                
                # Handle hourEnding
                if df['hourEnding'].dtype == 'object':
                    df['HE'] = df['hourEnding'].str.split(':').str[0].astype(int)
                else:
                    df['HE'] = pd.to_numeric(df['hourEnding'], errors='coerce').fillna(0).astype(int)

                # CRITICAL FIX: Get only the latest forecast for each date/hour combination
                if 'postedDatetime' in df.columns:
                    df['postedDatetime'] = pd.to_datetime(df['postedDatetime'])
                    # Keep only the most recent posted forecast for each date/hour
                    df = df.sort_values('postedDatetime', ascending=False)
                    df = df.drop_duplicates(subset=['deliveryDate', 'HE'], keep='first')
                    df = df.sort_values(['deliveryDate', 'HE'])

                return df, cache_time
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
        return None, None
    except Exception as e:
        st.error(f"Error fetching regional wind data: {str(e)}")
        return None, None


region_mapping = {
    'Coastal': 'coastForecast',
    'South': 'southForecast',
    'West': 'westForecast',
    'North': 'northForecast',
    'Panhandle': 'panhandleForecast',
    'System Wide': 'systemWideForecast'
}
def get_cache_time():
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo('America/Chicago'))
    if now.minute >= 5:
        cache_hour = now.replace(minute=5, second=0, microsecond=0)
    else:
        cache_hour = (now - timedelta(hours=1)).replace(minute=5, second=0, microsecond=0)
    return cache_hour.strftime('%Y-%m-%d %H:%M:%S CT')

@st.cache_data(ttl=7300)
def fetch_forecast_data(cache_time):
    auths = ercot_token()
    if not auths:
        return None, None
    today = datetime.today()
    dateStart = today.strftime("%Y-%m-%d")
    dateEnd = (today + timedelta(days=7)).strftime("%Y-%m-%d")
    url = f"https://api.ercot.com/api/public-reports/np3-565-cd/lf_by_model_weather_zone?deliveryDateFrom={dateStart}&deliveryDateTo={dateEnd}&inUseFlag=True"
    try:
        lf_request = requests.get(url, headers=auths, timeout=(60, 120))
        lf_results = lf_request.json() if lf_request.ok else {}
        returned_data = lf_results.get("data", [])
        fields = [x['name'] for x in lf_results.get("fields", [])]
        df = pd.DataFrame(returned_data, columns=fields)
        if not df.empty:
            df['postedDatetime'] = pd.to_datetime(df['postedDatetime'])
            df['hourEnding'] = df['hourEnding'].str.split(':').str[0].astype(int)
            df['interval'] = pd.to_datetime(df['deliveryDate']) + pd.to_timedelta(df['hourEnding'] - 1, unit='h')
            df['islatest'] = df.groupby(['interval'])['postedDatetime'].transform('max') == df['postedDatetime']
            df = df[df['islatest']].sort_values(by='interval').reset_index(drop=True)
            df['deliveryDate'] = pd.to_datetime(df['deliveryDate']).dt.date
            df['HE'] = df['hourEnding']
            df['systemTotal'] = pd.to_numeric(df['systemTotal'], errors='coerce')
            return df, cache_time
        return None, None
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, None

@st.cache_data(ttl=3600)
def fetch_meteologica_data(cache_time):
    ercot_load_id = 1943
    ercot_wind_id = 1877
    ercot_solar_id = 1840
    pjm_load_id = 2706
    pjm_wind_id = 2604
    pjm_solar_id = 2553
    result = {
        'ercot_load': None,
        'ercot_wind': None,
        'ercot_solar': None,
        'pjm_load': None,
        'pjm_wind': None,
        'pjm_solar': None
    }
    for data_type, content_id in [
        ('ercot_load', ercot_load_id),
        ('ercot_wind', ercot_wind_id),
        ('ercot_solar', ercot_solar_id),
        ('pjm_load', pjm_load_id),
        ('pjm_wind', pjm_wind_id),
        ('pjm_solar', pjm_solar_id)
    ]:
        data = get_content_data(content_id)
        if data:
            rows = data.get("data", [])
            if rows:
                df = pd.DataFrame(rows)
                columns_to_drop = ['To yyyy-mm-dd hh:mm', 'UTC offset from (UTC+/-hhmm)', 'UTC offset to (UTC+/-hhmm)']
                df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
                df = df.rename(columns={'From yyyy-mm-dd hh:mm': 'timestamp', 'forecast': 'value'})
                if 'timestamp' in df.columns and 'value' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df['deliveryDate'] = df['timestamp'].dt.date
                    df['HE'] = df['timestamp'].dt.hour + 1
                    df = df.dropna(subset=['value'])
                    result[data_type] = df
    return result, cache_time

@st.cache_data(ttl=3600)
def fetch_outage_data(cache_time):
    headers = ercot_token_outages()
    today = datetime.today().strftime('%Y-%m-%d')
    url = f"https://api.ercot.com/api/public-reports/np3-233-cd/hourly_res_outage_cap?operatingDateFrom={today}"
    json_data = fetch_outage_data_robust(url, headers)
    column_names = [
        'postedDatetime', 'operatingDate', 'hourEnding',
        'totalResourceMWZoneSouth', 'totalResourceMWZoneNorth', 'totalResourceMWZoneWest',
        'totalResourceMWZoneHouston', 'totalIRRMWZoneSouth', 'totalIRRMWZoneNorth',
        'totalIRRMWZoneWest', 'totalIRRMWZoneHouston', 'totalNewEquipResourceMWZoneSouth',
        'totalNewEquipResourceMWZoneNorth', 'totalNewEquipResourceMWZoneWest',
        'totalNewEquipResourceMWZoneHouston'
    ]
    df = pd.DataFrame(json_data.get("data", []), columns=column_names[:15])
    if not df.empty and 'postedDatetime' in df.columns:
        df['postedDatetime'] = pd.to_datetime(df['postedDatetime'], errors='coerce')
        latest_posted = df['postedDatetime'].max()
        df = df[df['postedDatetime'] == latest_posted]
        df['totalOutages'] = (
            pd.to_numeric(df['totalResourceMWZoneSouth'], errors='coerce').fillna(0) +
            pd.to_numeric(df['totalResourceMWZoneNorth'], errors='coerce').fillna(0) +
            pd.to_numeric(df['totalResourceMWZoneWest'], errors='coerce').fillna(0) +
            pd.to_numeric(df['totalResourceMWZoneHouston'], errors='coerce').fillna(0)
        )
        df['operatingDate'] = pd.to_datetime(df['operatingDate']).dt.date
        if df['hourEnding'].dtype == 'object':
            df['HE'] = df['hourEnding'].str.split(':').str[0].astype(int)
        else:
            df['HE'] = pd.to_numeric(df['hourEnding'], errors='coerce').astype(int)
        return df, cache_time
    return None, cache_time

def load_historical_cache():
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def save_historical_cache(cache_data):
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
        return True
    except Exception as e:
        st.warning(f"Could not save cache: {e}")
        return False

def create_snapshot_data(met_load_df, met_wind_df, met_solar_df, df, outage_df,
                        pjm_met_load_df, pjm_met_wind_df, pjm_met_solar_df, pjm_load_df, pjm_outage_df):
    snapshot = {}

    if met_load_df is not None and not met_load_df.empty:
        met_dates = sorted(met_load_df['deliveryDate'].unique())[:14]
        met_peak_loads = [met_load_df[met_load_df['deliveryDate'] == d]['value'].max() for d in met_dates]
        snapshot['met_load'] = {str(d): float(v) for d, v in zip(met_dates, met_peak_loads)}

    if df is not None and not df.empty:
        ercot_dates = sorted(df['deliveryDate'].unique())[:7]
        peak_loads = [df[df['deliveryDate'] == d]['systemTotal'].max() for d in ercot_dates]
        snapshot['ercot_load'] = {str(d): float(v) for d, v in zip(ercot_dates, peak_loads)}

    if met_wind_df is not None and not met_wind_df.empty:
        wind_dates = sorted(met_wind_df['deliveryDate'].unique())[:14]
        wind_avgs = []
        for date in wind_dates:
            onpeak = met_wind_df[(met_wind_df['deliveryDate'] == date) &
                                 (met_wind_df['HE'] >= 7) &
                                 (met_wind_df['HE'] <= 22)]
            wind_avgs.append(float(onpeak['value'].mean()) if not onpeak.empty else 0)
        snapshot['wind'] = {str(d): v for d, v in zip(wind_dates, wind_avgs)}

    if met_solar_df is not None and not met_solar_df.empty:
        solar_dates = sorted(met_solar_df['deliveryDate'].unique())[:14]
        solar_peaks = [float(met_solar_df[met_solar_df['deliveryDate'] == d]['value'].max())
                       for d in solar_dates]
        snapshot['solar'] = {str(d): v for d, v in zip(solar_dates, solar_peaks)}

    if outage_df is not None and not outage_df.empty:
        outage_dates = sorted(outage_df['operatingDate'].unique())[:14]
        outage_peaks = [float(outage_df[outage_df['operatingDate'] == d]['totalOutages'].max())
                        for d in outage_dates]
        snapshot['outages'] = {str(d): v for d, v in zip(outage_dates, outage_peaks)}

    if (met_load_df is not None and met_wind_df is not None and met_solar_df is not None):
        load_dates_set = set(met_load_df['deliveryDate'].unique())
        wind_dates_set = set(met_wind_df['deliveryDate'].unique())
        solar_dates_set = set(met_solar_df['deliveryDate'].unique())
        common_dates = sorted(load_dates_set & wind_dates_set & solar_dates_set)[:14]
        net_peaks = []
        for date in common_dates:
            merged = met_load_df[met_load_df['deliveryDate'] == date].merge(
                met_wind_df[met_wind_df['deliveryDate'] == date],
                on=['deliveryDate', 'HE'], suffixes=('_load', '_wind')
            ).merge(
                met_solar_df[met_solar_df['deliveryDate'] == date],
                on=['deliveryDate', 'HE']
            )
            merged['net_load'] = merged['value_load'] - merged['value_wind'] - merged['value']
            net_peaks.append(float(merged['net_load'].max()))
        snapshot['net_load'] = {str(d): v for d, v in zip(common_dates, net_peaks)}

    if (met_load_df is not None and met_wind_df is not None and
        met_solar_df is not None and outage_df is not None and not outage_df.empty):
        outage_dates_set = set(outage_df['operatingDate'].unique())
        common_dates_eff = sorted(load_dates_set & wind_dates_set & solar_dates_set & outage_dates_set)[:14]
        eff_peaks = []
        for date in common_dates_eff:
            merged = met_load_df[met_load_df['deliveryDate'] == date].merge(
                met_wind_df[met_wind_df['deliveryDate'] == date],
                on=['deliveryDate', 'HE'], suffixes=('_load', '_wind')
            ).merge(
                met_solar_df[met_solar_df['deliveryDate'] == date],
                on=['deliveryDate', 'HE']
            ).merge(
                outage_df[outage_df['operatingDate'] == date][['operatingDate', 'HE', 'totalOutages']],
                left_on=['deliveryDate', 'HE'], right_on=['operatingDate', 'HE'], how='left'
            )
            merged['totalOutages'] = merged['totalOutages'].fillna(0)
            merged['eff_net'] = (merged['value_load'] - merged['value_wind'] -
                                 merged['value'] + merged['totalOutages'])
            eff_peaks.append(float(merged['eff_net'].max()))
        snapshot['eff_net'] = {str(d): v for d, v in zip(common_dates_eff, eff_peaks)}

    if pjm_met_load_df is not None and not pjm_met_load_df.empty:
        pjm_met_dates = sorted(pjm_met_load_df['deliveryDate'].unique())[:14]
        pjm_met_peak_loads = [pjm_met_load_df[pjm_met_load_df['deliveryDate'] == d]['value'].max() for d in pjm_met_dates]
        snapshot['pjm_met_load'] = {str(d): float(v) for d, v in zip(pjm_met_dates, pjm_met_peak_loads)}
        if pjm_load_df is not None and not pjm_load_df.empty:
            rto_df = pjm_load_df[pjm_load_df['forecast_area'] == 'RTO_COMBINED']
            if not rto_df.empty:
                rto_dates = sorted(rto_df['deliveryDate'].unique())[:7]
                rto_peak_loads = [float(rto_df[rto_df['deliveryDate'] == d]['value'].max()) for d in rto_dates]
                snapshot['pjm_rto'] = {str(d): v for d, v in zip(rto_dates, rto_peak_loads)}

    if pjm_met_wind_df is not None and not pjm_met_wind_df.empty:
        pjm_wind_dates = sorted(pjm_met_wind_df['deliveryDate'].unique())[:14]
        pjm_wind_avgs = []
        for date in pjm_wind_dates:
            onpeak = pjm_met_wind_df[(pjm_met_wind_df['deliveryDate'] == date) &
                                     (pjm_met_wind_df['HE'] >= 8) &
                                     (pjm_met_wind_df['HE'] <= 23)]
            pjm_wind_avgs.append(float(onpeak['value'].mean()) if not onpeak.empty else 0)
        snapshot['pjm_wind'] = {str(d): v for d, v in zip(pjm_wind_dates, pjm_wind_avgs)}

    if pjm_met_solar_df is not None and not pjm_met_solar_df.empty:
        pjm_solar_dates = sorted(pjm_met_solar_df['deliveryDate'].unique())[:14]
        pjm_solar_peaks = [float(pjm_met_solar_df[pjm_met_solar_df['deliveryDate'] == d]['value'].max())
                           for d in pjm_solar_dates]
        snapshot['pjm_solar'] = {str(d): v for d, v in zip(pjm_solar_dates, pjm_solar_peaks)}

    if pjm_outage_df is not None and not pjm_outage_df.empty:
        pjm_outage_dates = sorted(pjm_outage_df['operatingDate'].unique())[:14]
        pjm_outage_peaks = [float(pjm_outage_df[pjm_outage_df['operatingDate'] == d]['totalOutages'].max())
                            for d in pjm_outage_dates]
        snapshot['pjm_outages'] = {str(d): v for d, v in zip(pjm_outage_dates, pjm_outage_peaks)}

    if (pjm_met_load_df is not None and pjm_met_wind_df is not None and pjm_met_solar_df is not None):
        pjm_load_dates_set = set(pjm_met_load_df['deliveryDate'].unique())
        pjm_wind_dates_set = set(pjm_met_wind_df['deliveryDate'].unique())
        pjm_solar_dates_set = set(pjm_met_solar_df['deliveryDate'].unique())
        pjm_common_dates = sorted(pjm_load_dates_set & pjm_wind_dates_set & pjm_solar_dates_set)[:14]
        pjm_net_peaks = []
        for date in pjm_common_dates:
            merged = pjm_met_load_df[pjm_met_load_df['deliveryDate'] == date].merge(
                pjm_met_wind_df[pjm_met_wind_df['deliveryDate'] == date],
                on=['deliveryDate', 'HE'], suffixes=('_load', '_wind')
            ).merge(
                pjm_met_solar_df[pjm_met_solar_df['deliveryDate'] == date],
                on=['deliveryDate', 'HE']
            )
            merged['net_load'] = merged['value_load'] - merged['value_wind'] - merged['value']
            pjm_net_peaks.append(float(merged['net_load'].max()))
        snapshot['pjm_net_load'] = {str(d): v for d, v in zip(pjm_common_dates, pjm_net_peaks)}

    if (pjm_met_load_df is not None and pjm_met_wind_df is not None and
        pjm_met_solar_df is not None and pjm_outage_df is not None and not pjm_outage_df.empty):
        pjm_outage_dates_set = set(pjm_outage_df['operatingDate'].unique())
        pjm_common_dates_eff = sorted(pjm_load_dates_set & pjm_wind_dates_set & pjm_solar_dates_set & pjm_outage_dates_set)[:14]
        pjm_eff_peaks = []
        for date in pjm_common_dates_eff:
            merged = pjm_met_load_df[pjm_met_load_df['deliveryDate'] == date].merge(
                pjm_met_wind_df[pjm_met_wind_df['deliveryDate'] == date],
                on=['deliveryDate', 'HE'], suffixes=('_load', '_wind')
            ).merge(
                pjm_met_solar_df[pjm_met_solar_df['deliveryDate'] == date],
                on=['deliveryDate', 'HE']
            ).merge(
                pjm_outage_df[pjm_outage_df['operatingDate'] == date][['operatingDate', 'totalOutages']],
                left_on=['deliveryDate'], right_on=['operatingDate'], how='left'
            )
            merged['totalOutages'] = merged['totalOutages'].fillna(0)
            merged['eff_net'] = (merged['value_load'] - merged['value_wind'] -
                                 merged['value'] + merged['totalOutages'])
            pjm_eff_peaks.append(float(merged['eff_net'].max()))
        snapshot['pjm_eff_net'] = {str(d): v for d, v in zip(pjm_common_dates_eff, pjm_eff_peaks)}

    return snapshot

def get_or_update_historical_cache(met_load_df, met_wind_df, met_solar_df, df, outage_df,
                                   pjm_met_load_df, pjm_met_wind_df, pjm_met_solar_df, pjm_load_df, pjm_outage_df):
    cache = load_historical_cache()
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo('America/Chicago'))
    today = now.date()

    if cache is None:
        cache = {
            'HE17_snapshot': {
                'captured_date': None,
                'data': {},
                'last_updated': None
            },
            'HE1_snapshot': {
                'captured_date': None,
                'data': {},
                'last_updated': None
            }
        }

    current_hour = now.hour

    if 16 <= current_hour < 18:
        last_updated = cache['HE17_snapshot'].get('last_updated')
        if last_updated is None or datetime.fromisoformat(last_updated).date() < today:
            snapshot = create_snapshot_data(met_load_df, met_wind_df, met_solar_df, df, outage_df,
                               pjm_met_load_df, pjm_met_wind_df, pjm_met_solar_df, pjm_load_df, pjm_outage_df)
            cache['HE17_snapshot'] = {
                'captured_date': str(today),
                'data': snapshot,
                'last_updated': now.isoformat()
            }
            save_historical_cache(cache)
            st.success("Captured HE17 snapshot")

    if 0 <= current_hour < 2:
        last_updated = cache['HE1_snapshot'].get('last_updated')
        captured_date = cache['HE1_snapshot'].get('captured_date')
        if last_updated is None or captured_date != str(today):
            snapshot = create_snapshot_data(met_load_df, met_wind_df, met_solar_df, df, outage_df,
                               pjm_met_load_df, pjm_met_wind_df, pjm_met_solar_df, pjm_load_df, pjm_outage_df)
            cache['HE1_snapshot'] = {
                'captured_date': str(today),
                'data': snapshot,
                'last_updated': now.isoformat()
            }
            save_historical_cache(cache)
            st.success(f"Captured HE1 snapshot at {now.strftime('%I:%M %p')} CT")

    display_cache = {}
    yesterday = today - timedelta(days=1)

    he17_captured = cache['HE17_snapshot'].get('captured_date')
    if he17_captured and he17_captured == str(yesterday):
        display_cache['yesterday_HE17'] = {
            'date': yesterday,
            'data': cache['HE17_snapshot']['data']
        }
    else:
        display_cache['yesterday_HE17'] = {
            'date': yesterday,
            'data': {}
        }

    he1_captured = cache['HE1_snapshot'].get('captured_date')
    if he1_captured and he1_captured == str(today):
        display_cache['today_HE1'] = {
            'date': today,
            'data': cache['HE1_snapshot']['data']
        }
    else:
        display_cache['today_HE1'] = {
            'date': today,
            'data': {}
        }

    display_cache['fetch_time'] = now.strftime('%Y-%m-%d %H:%M:%S CT')

    return display_cache

def display_cache_status(cache):
    with st.sidebar:
        st.markdown("### Snapshot Status")
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo('America/Chicago'))
        current_hour = now.hour
        he17_captured = cache.get('HE17_snapshot', {}).get('captured_date')
        he17_updated = cache.get('HE17_snapshot', {}).get('last_updated')
        if he17_captured:
            st.success(f"HE17: Captured {he17_captured}")
            if he17_updated:
                update_time = datetime.fromisoformat(he17_updated).strftime('%I:%M %p')
                st.caption(f"Last updated: {update_time}")
        else:
            if 16 <= current_hour < 18:
                st.info("HE17: Capturing now...")
            else:
                st.warning("HE17: Waiting for 4-5pm window")
        he1_captured = cache.get('HE1_snapshot', {}).get('captured_date')
        he1_updated = cache.get('HE1_snapshot', {}).get('last_updated')
        if he1_captured:
            st.success(f"HE1: Captured {he1_captured}")
            if he1_updated:
                update_time = datetime.fromisoformat(he1_updated).strftime('%I:%M %p')
                st.caption(f"Last updated: {update_time}")
        else:
            if 0 <= current_hour < 2:
                st.info("HE1: Capturing now...")
            else:
                st.warning("HE1: Waiting for midnight-2am window")
        st.caption(f"Current time: {now.strftime('%I:%M %p')}")

def get_color_for_value(value, min_val, max_val, reverse=False):
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (value - min_val) / (max_val - min_val)
    if reverse:
        normalized = 1 - normalized
    if normalized < 0.5:
        ratio = normalized * 2
        r = int(0 + (255 * ratio))
        g = int(200 - (35 * ratio))
        b = 0
    else:
        ratio = (normalized - 0.5) * 2
        r = 255
        g = int(165 * (1 - ratio))
        b = 0
    return f"rgb({r}, {g}, {b})"

def main():
    st.title("Forecasts")
    try:
        tab1, tab2, tab3 = st.tabs(["ERCOT Weekly", "PJM Weekly", "Gas"])
        with st.spinner("Loading forecast data..."):
            cache_time = get_cache_time()
            result = fetch_forecast_data(cache_time)
            df, ercot_fetch_time = result if result[0] is not None else (None, cache_time)
            met_result = fetch_meteologica_data(cache_time)
            met_data, met_fetch_time = met_result
            met_load_df = met_data['ercot_load']
            met_wind_df = met_data['ercot_wind']
            met_solar_df = met_data['ercot_solar']
            pjm_met_load_df = met_data['pjm_load']
            pjm_met_wind_df = met_data['pjm_wind']
            pjm_met_solar_df = met_data['pjm_solar']
            outage_result = fetch_outage_data(cache_time)
            outage_df, outage_fetch_time = outage_result if outage_result[0] is not None else (None, cache_time)
            pjm_load_result = fetch_pjm_load_forecast(cache_time)
            pjm_load_df, pjm_load_fetch_time = pjm_load_result if pjm_load_result[0] is not None else (None, cache_time)
            pjm_outage_result = fetch_pjm_outages(cache_time)
            pjm_outage_df, pjm_outage_fetch_time = pjm_outage_result if pjm_outage_result[0] is not None else (None, cache_time)

            wind_regional_result = fetch_ercot_wind_by_region(cache_time)
            wind_regional_df, wind_regional_fetch_time = wind_regional_result if wind_regional_result[0] is not None else (None, cache_time)
        historical_cache = get_or_update_historical_cache(
            met_load_df, met_wind_df, met_solar_df,
            df, outage_df,
            pjm_met_load_df, pjm_met_wind_df, pjm_met_solar_df,
            pjm_load_df,
            pjm_outage_df
                              )

        raw_cache = load_historical_cache()
        if raw_cache:
            display_cache_status(raw_cache)
        
        # Clear popup states if dialog was dismissed (clicked outside)
        # This prevents the dialog from re-opening on the next interaction
        if 'ercot_popup_date' in st.session_state and 'ercot_dialog_active' not in st.session_state:
            del st.session_state['ercot_popup_date']
        if 'pjm_popup_date' in st.session_state and 'pjm_dialog_active' not in st.session_state:
            del st.session_state['pjm_popup_date']
        if 'wind_region_popup_date' in st.session_state and 'wind_region_dialog_active' not in st.session_state:
            del st.session_state['wind_region_popup_date']
        
        with tab1:
            st.caption(f"Data last fetched: {met_fetch_time} | Next refresh: {next_refresh_time}")
            if met_load_df is not None and not met_load_df.empty:
                min_load = met_load_df['value'].min()
                max_load = met_load_df['value'].max()
                met_dates = sorted(met_load_df['deliveryDate'].unique())[:14]
                if df is not None and not df.empty:
                    ercot_dates = sorted(df['deliveryDate'].unique())[:7]
                    peak_loads = [df[df['deliveryDate'] == d]['systemTotal'].max() for d in ercot_dates]
                    min_peak = min(peak_loads) if peak_loads else 0
                    max_peak = max(peak_loads) if peak_loads else 1
                    # Get peak hours for ERCOT
                    peak_hours_ercot = [df[df['deliveryDate'] == d]['systemTotal'].idxmax() for d in ercot_dates]
                    peak_hours_ercot = [df.loc[idx, 'HE'] if idx in df.index else 0 for idx in peak_hours_ercot]
                else:
                    ercot_dates = []
                    peak_loads = []
                    peak_hours_ercot = []
                    min_peak = 0
                    max_peak = 1
                wind_min = met_wind_df['value'].min() if met_wind_df is not None and not met_wind_df.empty else 0
                wind_max = met_wind_df['value'].max() if met_wind_df is not None and not met_wind_df.empty else 1
                solar_min = met_solar_df['value'].min() if met_solar_df is not None and not met_solar_df.empty else 0
                solar_max = met_solar_df['value'].max() if met_solar_df is not None and not met_solar_df.empty else 1
                met_peak_loads = [met_load_df[met_load_df['deliveryDate'] == d]['value'].max() for d in met_dates]
                met_min_peak = min(met_peak_loads) if met_peak_loads else 0
                met_max_peak = max(met_peak_loads) if met_peak_loads else 1
                # Get peak hours for Meteologica
                peak_hours_met = [met_load_df[met_load_df['deliveryDate'] == d]['value'].idxmax() for d in met_dates]
                peak_hours_met = [met_load_df.loc[idx, 'HE'] if idx in met_load_df.index else 0 for idx in peak_hours_met]
                wind_dates = sorted(met_wind_df['deliveryDate'].unique())[:14] if met_wind_df is not None and not met_wind_df.empty else []
                wind_avgs = []
                for date in wind_dates:
                    onpeak = met_wind_df[(met_wind_df['deliveryDate'] == date) & (met_wind_df['HE'] >= 7) & (met_wind_df['HE'] <= 22)]
                    wind_avgs.append(onpeak['value'].mean() if not onpeak.empty else 0)
                wind_min_pk = min(wind_avgs) if wind_avgs else 0
                wind_max_pk = max(wind_avgs) if wind_avgs else 1
                solar_dates = sorted(met_solar_df['deliveryDate'].unique())[:14] if met_solar_df is not None and not met_solar_df.empty else []
                solar_peaks = [met_solar_df[met_solar_df['deliveryDate'] == d]['value'].max() for d in solar_dates]
                solar_min_pk = min(solar_peaks) if solar_peaks else 0
                solar_max_pk = max(solar_peaks) if solar_peaks else 1
                outage_dates = sorted(outage_df['operatingDate'].unique())[:7] if outage_df is not None and not outage_df.empty else []
                outage_peaks = [outage_df[outage_df['operatingDate'] == d]['totalOutages'].max() for d in outage_dates] if outage_dates else []
                outage_min = min(outage_peaks) if outage_peaks else 0
                outage_max = max(outage_peaks) if outage_peaks else 1
                load_dates_set = set(met_load_df['deliveryDate'].unique())
                wind_dates_set = set(met_wind_df['deliveryDate'].unique()) if met_wind_df is not None else set()
                solar_dates_set = set(met_solar_df['deliveryDate'].unique()) if met_solar_df is not None else set()
                common_dates = sorted(load_dates_set & wind_dates_set & solar_dates_set)[:14]
                net_peaks = []
                for date in common_dates:
                    merged = met_load_df[met_load_df['deliveryDate'] == date].merge(
                        met_wind_df[met_wind_df['deliveryDate'] == date], on=['deliveryDate', 'HE'], suffixes=('_load', '_wind')
                    ).merge(
                        met_solar_df[met_solar_df['deliveryDate'] == date], on=['deliveryDate', 'HE']
                    )
                    merged['net_load'] = merged['value_load'] - merged['value_wind'] - merged['value']
                    net_peaks.append(merged['net_load'].max())
                net_min = min(net_peaks) if net_peaks else 0
                net_max = max(net_peaks) if net_peaks else 1
                outage_dates_set = set(outage_df['operatingDate'].unique()) if outage_df is not None and not outage_df.empty else set()
                common_dates_eff = sorted(load_dates_set & wind_dates_set & solar_dates_set & outage_dates_set)[:7]
                eff_peaks = []
                for date in common_dates_eff:
                    merged = met_load_df[met_load_df['deliveryDate'] == date].merge(
                        met_wind_df[met_wind_df['deliveryDate'] == date], on=['deliveryDate', 'HE'], suffixes=('_load', '_wind')
                    ).merge(
                        met_solar_df[met_solar_df['deliveryDate'] == date], on=['deliveryDate', 'HE']
                    ).merge(
                        outage_df[outage_df['operatingDate'] == date][['operatingDate', 'HE', 'totalOutages']],
                        left_on=['deliveryDate', 'HE'], right_on=['operatingDate', 'HE'], how='left'
                    )
                    merged['totalOutages'] = merged['totalOutages'].fillna(0)
                    merged['eff_net'] = (merged['value_load'] - merged['value_wind'] -
                                         merged['value'] + merged['totalOutages'])
                    eff_peaks.append(merged['eff_net'].max())
                eff_min = min(eff_peaks) if eff_peaks else 0
                eff_max = max(eff_peaks) if eff_peaks else 1
                yesterday = (datetime.today() - timedelta(days=1)).date()
                today = datetime.today().date()
                dropdown_options = [
                    f"{yesterday.strftime('%m/%d')} HE17",
                    f"{today.strftime('%m/%d')} HE1"
                ]
                selected_option = st.selectbox("Compare with historical forecast:", dropdown_options)
                selected_key = 'yesterday_HE17' if 'HE17' in selected_option else 'today_HE1'
                st.markdown("### Meteologica")
                st.markdown("<div style='display: flex; justify-content: space-around; margin-bottom: 5px;'>" +
                            "".join([f"<div style='text-align: center; flex: 1; font-size: 11px; color: #888888;'>{pd.to_datetime(d).strftime('%a')}</div>" for d in met_dates]) +
                            "</div>", unsafe_allow_html=True)
                cols = st.columns(14)
                met_deltas = []
                for idx, date in enumerate(met_dates):
                    date_obj = pd.to_datetime(date)
                    peak_load = met_peak_loads[idx]
                    peak_hour = peak_hours_met[idx]
                    peak_color = get_color_for_value(peak_load, met_min_peak, met_max_peak)
                    with cols[idx]:
                        if st.button(f" {date_obj.strftime('%m/%d')}", key=f"ercot_met_{date}", use_container_width=True):
                            st.session_state['ercot_popup_date'] = date
                            st.session_state['ercot_dialog_active'] = True
                        st.markdown(f"""
                            <div style='text-align: center; padding: 10px 3px; background-color: {peak_color}; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                                <div style='font-size: 13px; font-weight: bold; margin-bottom: 4px; color: #000000;'>HE{peak_hour}</div>
                                <div style='font-size: 18px; font-weight: bold; color: #000000;'>{peak_load:,.0f}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    cached_value = historical_cache[selected_key]['data'].get('met_load', {}).get(str(date), None)
                    delta = peak_load - cached_value if cached_value is not None else None
                    met_deltas.append(delta)
                st.markdown("<div style='font-size: 12px; font-weight: bold; margin-top: 5px;'>Delta (Current - Historical):</div>", unsafe_allow_html=True)
                cols = st.columns(14)
                for idx, delta in enumerate(met_deltas):
                    with cols[idx]:
                        if delta is not None:
                            st.markdown(f"""
                                <div style='text-align: center; padding: 5px 3px;'>
                                    <div style='font-size: 12px; color: #ffffff; font-weight: bold;'>{delta:+,.0f}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='text-align: center; padding: 5px 3px; font-size: 12px;'>N/A</div>", unsafe_allow_html=True)
                st.markdown("<hr style='border: none; border-top: 3px solid white; margin: 20px 0;'>", unsafe_allow_html=True)
                st.markdown("### ERCOT")
                cols = st.columns(14)
                ercot_deltas = []
                for idx, date in enumerate(ercot_dates):
                    date_obj = pd.to_datetime(date)
                    peak_load = peak_loads[idx]
                    peak_hour = peak_hours_ercot[idx]
                    peak_color = get_color_for_value(peak_load, min_peak, max_peak)
                    with cols[idx]:
                        st.markdown(f"""
                            <div style='text-align: center; padding: 10px 3px; background-color: {peak_color}; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                                <div style='font-size: 13px; font-weight: bold; margin-bottom: 4px; color: #000000;'>{date_obj.strftime('%m/%d')} (HE{peak_hour})</div>
                                <div style='font-size: 18px; font-weight: bold; color: #000000;'>{peak_load:,.0f}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    cached_value = historical_cache[selected_key]['data'].get('ercot_load', {}).get(str(date), None)
                    delta = peak_load - cached_value if cached_value is not None else None
                    ercot_deltas.append(delta)
                st.markdown("<div style='font-size: 12px; font-weight: bold; margin-top: 5px;'>Delta (Current - Historical):</div>", unsafe_allow_html=True)
                cols = st.columns(14)
                for idx, delta in enumerate(ercot_deltas):
                    with cols[idx]:
                        if delta is not None:
                            st.markdown(f"""
                                <div style='text-align: center; padding: 5px 3px;'>
                                    <div style='font-size: 12px; color: #ffffff; font-weight: bold;'>{delta:+,.0f}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='text-align: center; padding: 5px 3px; font-size: 12px;'>N/A</div>", unsafe_allow_html=True)
                st.markdown("<hr style='border: none; border-top: 3px solid white; margin: 20px 0;'>", unsafe_allow_html=True)
                st.markdown("### Wind - Onpeak (HE 7-22)")
                if met_wind_df is not None and not met_wind_df.empty:
                    st.markdown("<div style='display: flex; justify-content: space-around; margin-bottom: 5px;'>" +
                                "".join([f"<div style='text-align: center; flex: 1; font-size: 11px; color: #888888;'>{pd.to_datetime(d).strftime('%a')}</div>" for d in wind_dates]) +
                                "</div>", unsafe_allow_html=True)
                    cols = st.columns(14)
                    wind_deltas = []
                    for idx, date in enumerate(wind_dates):
                        date_obj = pd.to_datetime(date)
                        peak_wind = wind_avgs[idx]
                        peak_color = get_color_for_value(peak_wind, wind_min_pk, wind_max_pk, reverse=True)
                        with cols[idx]:
                            st.markdown(f"""
                                <div style='text-align: center; padding: 10px 3px; background-color: {peak_color}; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                                    <div style='font-size: 13px; font-weight: bold; margin-bottom: 4px; color: #000000;'>{date_obj.strftime('%m/%d')}</div>
                                    <div style='font-size: 18px; font-weight: bold; color: #000000;'>{peak_wind:,.0f}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        cached_value = historical_cache[selected_key]['data'].get('wind', {}).get(str(date), None)
                        delta = peak_wind - cached_value if cached_value is not None else None
                        wind_deltas.append(delta)
                    st.markdown("<div style='font-size: 12px; font-weight: bold; margin-top: 5px;'>Delta (Current - Historical):</div>", unsafe_allow_html=True)
                    cols = st.columns(14)
                    for idx, delta in enumerate(wind_deltas):
                        with cols[idx]:
                            if delta is not None:
                                st.markdown(f"""
                                    <div style='text-align: center; padding: 5px 3px;'>
                                        <div style='font-size: 12px; color: #ffffff; font-weight: bold;'>{delta:+,.0f}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("<div style='text-align: center; padding: 5px 3px; font-size: 12px;'>N/A</div>", unsafe_allow_html=True)
                st.markdown("---")

                # Regional Wind with Date Buttons
                if wind_regional_df is not None and not wind_regional_df.empty:
                    st.markdown("### Wind Forecast by Region")

                    # These are the actual column names from the API
                    region_mapping = {
                        'COASTAL': 'STWPFCoastal',
                        'EAST': 'STWPFEast',
                        'FAR_WEST': 'STWPFFarWest',
                        'NORTH': 'STWPFNorth',
                        'NORTH_C': 'STWPFNorthC',
                        'PANHANDLE': 'STWPFPanhandle',
                        'SOUTH': 'STWPFSouth',
                        'SOUTHERN': 'STWPFSouthern',
                        'WEST': 'STWPFWest'
                    }

                    # Check which columns actually exist
                    available_regions = {k: v for k, v in region_mapping.items() if v in wind_regional_df.columns}
                    
                    # Store mapping in session state for dialog
                    st.session_state['wind_region_mapping'] = available_regions

                    if available_regions:
                        wind_regional_dates = sorted(wind_regional_df['deliveryDate'].unique())[:7]
                        
                        # Calculate OnPk averages for display boxes
                        regional_onpk_totals = []
                        for date in wind_regional_dates:
                            date_data = wind_regional_df[wind_regional_df['deliveryDate'] == date]
                            onpeak_data = date_data[(date_data['HE'] >= 7) & (date_data['HE'] <= 22)]
                            total = sum(onpeak_data[col].mean() for col in available_regions.values() if col in onpeak_data.columns and not onpeak_data[col].isna().all())
                            regional_onpk_totals.append(total)
                        
                        regional_min = min(regional_onpk_totals) if regional_onpk_totals else 0
                        regional_max = max(regional_onpk_totals) if regional_onpk_totals else 1
                        
                        # Day of week row
                        st.markdown("<div style='display: flex; justify-content: space-around; margin-bottom: 5px;'>" +
                                    "".join([f"<div style='text-align: center; flex: 1; font-size: 11px; color: #888888;'>{pd.to_datetime(d).strftime('%a')}</div>" for d in wind_regional_dates]) +
                                    "</div>", unsafe_allow_html=True)
                        
                        # Date buttons and value boxes
                        cols = st.columns(7)
                        for idx, date in enumerate(wind_regional_dates):
                            date_obj = pd.to_datetime(date)
                            total_wind = regional_onpk_totals[idx]
                            wind_color = get_color_for_value(total_wind, regional_min, regional_max, reverse=True)
                            with cols[idx]:
                                if st.button(f" {date_obj.strftime('%m/%d')}", key=f"wind_region_{date}", use_container_width=True):
                                    st.session_state['wind_region_popup_date'] = date
                                    st.session_state['wind_region_dialog_active'] = True
                                st.markdown(f"""
                                    <div style='text-align: center; padding: 10px 3px; background-color: {wind_color}; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                                        <div style='font-size: 12px; font-weight: bold; margin-bottom: 4px; color: #000000;'>Total</div>
                                        <div style='font-size: 16px; font-weight: bold; color: #000000;'>{total_wind:,.0f}</div>
                                    </div>
                                """, unsafe_allow_html=True)

                # Regional Wind Popup
                if 'wind_region_popup_date' in st.session_state and 'wind_region_mapping' in st.session_state:
                    @st.dialog("Regional Wind Forecast", width="large")
                    def show_wind_region_dialog():
                        popup_date = st.session_state['wind_region_popup_date']
                        available_regions = st.session_state['wind_region_mapping']
                        date_obj = pd.to_datetime(popup_date)
                        st.markdown(f"#### {date_obj.strftime('%A, %B %d, %Y')}")
                        
                        date_data = wind_regional_df[wind_regional_df['deliveryDate'] == popup_date].sort_values('HE')
                        
                        if date_data.empty:
                            st.warning("No data available for this date")
                            if st.button("Close", key="close_wind_region", type="primary", use_container_width=True):
                                if 'wind_region_dialog_active' in st.session_state:
                                    del st.session_state['wind_region_dialog_active']
                                del st.session_state['wind_region_popup_date']
                                st.rerun()
                            return
                        
                        # Calculate min/max for each region for color scaling
                        region_mins = {}
                        region_maxs = {}
                        for region_name, col_name in available_regions.items():
                            if col_name in date_data.columns:
                                vals = date_data[col_name].dropna()
                                region_mins[region_name] = vals.min() if len(vals) > 0 else 0
                                region_maxs[region_name] = vals.max() if len(vals) > 0 else 1
                        
                        # Build header row
                        region_names = list(available_regions.keys())
                        num_regions = len(region_names)
                        header_cols = "40px " + " ".join(["1fr"] * num_regions)
                        
                        header_html = f"<div style='display: grid; grid-template-columns: {header_cols}; gap: 4px; margin-bottom: 4px; font-size: 10px; font-weight: bold; color: #888;'>"
                        header_html += "<div style='text-align: center;'>HE</div>"
                        for region in region_names:
                            header_html += f"<div style='text-align: center;'>{region}</div>"
                        header_html += "</div>"
                        st.markdown(header_html, unsafe_allow_html=True)
                        
                        # Build data rows
                        rows_html = ""
                        for he in range(1, 25):
                            row_data = date_data[date_data['HE'] == he]
                            rows_html += f"<div style='display: grid; grid-template-columns: {header_cols}; gap: 4px; margin-bottom: 2px;'>"
                            rows_html += f"<div style='text-align: center; font-size: 11px; font-weight: bold; padding: 4px 0;'>{he}</div>"
                            
                            for region_name in region_names:
                                col_name = available_regions[region_name]
                                if not row_data.empty and col_name in row_data.columns:
                                    val = row_data[col_name].iloc[0]
                                    if pd.notna(val):
                                        color = get_color_for_value(val, region_mins.get(region_name, 0), region_maxs.get(region_name, 1), reverse=True)
                                        rows_html += f"<div style='background: {color}; color: #000; padding: 4px; border-radius: 4px; text-align: center; font-size: 11px; font-weight: bold;'>{val:,.0f}</div>"
                                    else:
                                        rows_html += "<div style='background: #333; color: #666; padding: 4px; border-radius: 4px; text-align: center; font-size: 11px;'>-</div>"
                                else:
                                    rows_html += "<div style='background: #333; color: #666; padding: 4px; border-radius: 4px; text-align: center; font-size: 11px;'>-</div>"
                            
                            rows_html += "</div>"
                        
                        st.markdown(rows_html, unsafe_allow_html=True)
                        
                        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                        if st.button("Close", key="close_wind_region", type="primary", use_container_width=True):
                            if 'wind_region_dialog_active' in st.session_state:
                                del st.session_state['wind_region_dialog_active']
                            del st.session_state['wind_region_popup_date']
                            st.rerun()
                    
                    show_wind_region_dialog()
                    # Clear the active flag after showing dialog
                    if 'wind_region_dialog_active' in st.session_state:
                        del st.session_state['wind_region_dialog_active']


                st.markdown("### Solar")
                if met_solar_df is not None and not met_solar_df.empty:
                    st.markdown("<div style='display: flex; justify-content: space-around; margin-bottom: 5px;'>" +
                                "".join([f"<div style='text-align: center; flex: 1; font-size: 11px; color: #888888;'>{pd.to_datetime(d).strftime('%a')}</div>" for d in solar_dates]) +
                                "</div>", unsafe_allow_html=True)
                    cols = st.columns(14)
                    solar_deltas = []
                    for idx, date in enumerate(solar_dates):
                        date_obj = pd.to_datetime(date)
                        peak_solar = solar_peaks[idx]
                        peak_color = get_color_for_value(peak_solar, solar_min_pk, solar_max_pk, reverse=True)
                        with cols[idx]:
                            st.markdown(f"""
                                <div style='text-align: center; padding: 10px 3px; background-color: {peak_color}; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                                    <div style='font-size: 13px; font-weight: bold; margin-bottom: 4px; color: #000000;'>{date_obj.strftime('%m/%d')}</div>
                                    <div style='font-size: 18px; font-weight: bold; color: #000000;'>{peak_solar:,.0f}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        cached_value = historical_cache[selected_key]['data'].get('solar', {}).get(str(date), None)
                        delta = peak_solar - cached_value if cached_value is not None else None
                        solar_deltas.append(delta)
                    st.markdown("<div style='font-size: 12px; font-weight: bold; margin-top: 5px;'>Delta (Current - Historical):</div>", unsafe_allow_html=True)
                    cols = st.columns(14)
                    for idx, delta in enumerate(solar_deltas):
                        with cols[idx]:
                            if delta is not None:
                                st.markdown(f"""
                                    <div style='text-align: center; padding: 5px 3px;'>
                                        <div style='font-size: 12px; color: #ffffff; font-weight: bold;'>{delta:+,.0f}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("<div style='text-align: center; padding: 5px 3px; font-size: 12px;'>N/A</div>", unsafe_allow_html=True)
                st.markdown("<hr style='border: none; border-top: 3px solid white; margin: 20px 0;'>", unsafe_allow_html=True)
                st.markdown("### Outages")
                if outage_df is not None and not outage_df.empty:
                    cols = st.columns(14)
                    outage_deltas = []
                    for idx, date in enumerate(outage_dates):
                        date_obj = pd.to_datetime(date)
                        peak_outage = outage_peaks[idx]
                        outage_color = get_color_for_value(peak_outage, outage_min, outage_max)
                        with cols[idx]:
                            st.markdown(f"<div style='text-align: center; font-size: 11px; color: #888888; margin-bottom: 5px;'>{date_obj.strftime('%a')}</div>", unsafe_allow_html=True)
                            st.markdown(f"""
                                <div style='text-align: center; padding: 10px 3px; background-color: {outage_color}; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                                    <div style='font-size: 13px; font-weight: bold; margin-bottom: 4px; color: #000000;'>{date_obj.strftime('%m/%d')}</div>
                                    <div style='font-size: 18px; font-weight: bold; color: #000000;'>{peak_outage:,.0f}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        cached_value = historical_cache[selected_key]['data'].get('outages', {}).get(str(date), None)
                        delta = peak_outage - cached_value if cached_value is not None else None
                        outage_deltas.append(delta)
                    st.markdown("<div style='font-size: 12px; font-weight: bold; margin-top: 5px;'>Delta (Current - Historical):</div>", unsafe_allow_html=True)
                    cols = st.columns(14)
                    for idx, delta in enumerate(outage_deltas):
                        with cols[idx]:
                            if delta is not None:
                                st.markdown(f"""
                                    <div style='text-align: center; padding: 5px 3px;'>
                                        <div style='font-size: 12px; color: #ffffff; font-weight: bold;'>{delta:+,.0f}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("<div style='text-align: center; padding: 5px 3px; font-size: 12px;'>N/A</div>", unsafe_allow_html=True)
                st.markdown("<hr style='border: none; border-top: 3px solid white; margin: 20px 0;'>", unsafe_allow_html=True)
                st.markdown("### Peak Net Load")
                if met_load_df is not None and met_wind_df is not None and met_solar_df is not None:
                    st.markdown("<div style='display: flex; justify-content: space-around; margin-bottom: 5px;'>" +
                                "".join([f"<div style='text-align: center; flex: 1; font-size: 11px; color: #888888;'>{pd.to_datetime(d).strftime('%a')}</div>" for d in common_dates]) +
                                "</div>", unsafe_allow_html=True)
                    cols = st.columns(14)
                    net_deltas = []
                    for idx, date in enumerate(common_dates):
                        date_obj = pd.to_datetime(date)
                        peak_net = net_peaks[idx]
                        net_color = get_color_for_value(peak_net, net_min, net_max)
                        with cols[idx]:
                            st.markdown(f"""
                                <div style='text-align: center; padding: 10px 3px; background-color: {net_color}; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                                    <div style='font-size: 13px; font-weight: bold; margin-bottom: 4px; color: #000000;'>{date_obj.strftime('%m/%d')}</div>
                                    <div style='font-size: 18px; font-weight: bold; color: #000000;'>{peak_net:,.0f}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        cached_value = historical_cache[selected_key]['data'].get('net_load', {}).get(str(date), None)
                        delta = peak_net - cached_value if cached_value is not None else None
                        net_deltas.append(delta)
                    st.markdown("<div style='font-size: 12px; font-weight: bold; margin-top: 5px;'>Delta (Current - Historical):</div>", unsafe_allow_html=True)
                    cols = st.columns(14)
                    for idx, delta in enumerate(net_deltas):
                        with cols[idx]:
                            if delta is not None:
                                st.markdown(f"""
                                    <div style='text-align: center; padding: 5px 3px;'>
                                        <div style='font-size: 12px; color: #ffffff; font-weight: bold;'>{delta:+,.0f}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("<div style='text-align: center; padding: 5px 3px; font-size: 12px;'>N/A</div>", unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("### Peak Effective Net Load")
                if met_load_df is not None and met_wind_df is not None and met_solar_df is not None and outage_df is not None:
                    cols = st.columns(14)
                    eff_deltas = []
                    for idx, date in enumerate(common_dates_eff):
                        date_obj = pd.to_datetime(date)
                        peak_eff = eff_peaks[idx]
                        eff_color = get_color_for_value(peak_eff, eff_min, eff_max)
                        with cols[idx]:
                            st.markdown(f"<div style='text-align: center; font-size: 11px; color: #888888; margin-bottom: 5px;'>{date_obj.strftime('%a')}</div>", unsafe_allow_html=True)
                            st.markdown(f"""
                                <div style='text-align: center; padding: 10px 3px; background-color: {eff_color}; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                                    <div style='font-size: 13px; font-weight: bold; margin-bottom: 4px; color: #000000;'>{date_obj.strftime('%m/%d')}</div>
                                    <div style='font-size: 18px; font-weight: bold; color: #000000;'>{peak_eff:,.0f}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        cached_value = historical_cache[selected_key]['data'].get('eff_net', {}).get(str(date), None)
                        delta = peak_eff - cached_value if cached_value is not None else None
                        eff_deltas.append(delta)
                    st.markdown("<div style='font-size: 12px; font-weight: bold; margin-top: 5px;'>Delta (Current - Historical):</div>", unsafe_allow_html=True)
                    cols = st.columns(14)
                    for idx, delta in enumerate(eff_deltas):
                        with cols[idx]:
                            if delta is not None:
                                st.markdown(f"""
                                    <div style='text-align: center; padding: 5px 3px;'>
                                        <div style='font-size: 12px; color: #ffffff; font-weight: bold;'>{delta:+,.0f}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("<div style='text-align: center; padding: 5px 3px; font-size: 12px;'>N/A</div>", unsafe_allow_html=True)

                # Popup Dialog for ERCOT date details
                if 'ercot_popup_date' in st.session_state:
                    @st.dialog("Load Details", width="large")
                    def show_ercot_dialog():
                        popup_date = st.session_state['ercot_popup_date']
                        date_obj = pd.to_datetime(popup_date)
                        st.markdown(f"#### {date_obj.strftime('%A, %B %d, %Y')}")
                        hours_data = met_load_df[met_load_df['deliveryDate'] == popup_date]
                        wind_date_data = met_wind_df[met_wind_df['deliveryDate'] == popup_date] if met_wind_df is not None else None
                        solar_date_data = met_solar_df[met_solar_df['deliveryDate'] == popup_date] if met_solar_df is not None else None
                        
                        # Pre-calculate net loads for color scaling
                        net_loads_all = []
                        if wind_date_data is not None and solar_date_data is not None and not wind_date_data.empty and not solar_date_data.empty:
                            for idx in range(24):
                                he = idx + 1
                                hour_row = hours_data[hours_data['HE'] == he]
                                wind_row = wind_date_data[wind_date_data['HE'] == he]
                                solar_row = solar_date_data[solar_date_data['HE'] == he]
                                if not hour_row.empty and not wind_row.empty and not solar_row.empty:
                                    net_load = hour_row['value'].iloc[0] - (wind_row['value'].iloc[0] + solar_row['value'].iloc[0])
                                    net_loads_all.append(net_load)
                        popup_net_min = min(net_loads_all) if net_loads_all else 0
                        popup_net_max = max(net_loads_all) if net_loads_all else 1
                        
                        if not hours_data.empty:
                            # Compact table header
                            st.markdown("""
                                <div style='display: grid; grid-template-columns: 40px 1fr 1fr 1fr 1fr; gap: 4px; margin-bottom: 4px; font-size: 11px; font-weight: bold; color: #888;'>
                                    <div style='text-align: center;'>HE</div>
                                    <div style='text-align: center;'>Load</div>
                                    <div style='text-align: center;'>Wind</div>
                                    <div style='text-align: center;'>Solar</div>
                                    <div style='text-align: center;'>Net</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Build all rows as compact grid
                            rows_html = ""
                            for idx in range(24):
                                he = idx + 1
                                hour_row = hours_data[hours_data['HE'] == he]
                                wind_row = wind_date_data[wind_date_data['HE'] == he] if wind_date_data is not None else None
                                solar_row = solar_date_data[solar_date_data['HE'] == he] if solar_date_data is not None else None
                                
                                load_val = hour_row['value'].iloc[0] if not hour_row.empty else 0
                                wind_val = wind_row['value'].iloc[0] if wind_row is not None and not wind_row.empty else 0
                                solar_val = solar_row['value'].iloc[0] if solar_row is not None and not solar_row.empty else 0
                                net_val = load_val - (wind_val + solar_val)
                                
                                load_color = get_color_for_value(load_val, min_load, max_load)
                                wind_color = get_color_for_value(wind_val, wind_min, wind_max, reverse=True)
                                solar_color = '#333333' if solar_val == 0 else get_color_for_value(solar_val, solar_min, solar_max, reverse=True)
                                solar_text_color = '#666666' if solar_val == 0 else '#000'
                                net_color = get_color_for_value(net_val, popup_net_min, popup_net_max)
                                
                                rows_html += f"""
                                    <div style='display: grid; grid-template-columns: 40px 1fr 1fr 1fr 1fr; gap: 4px; margin-bottom: 2px;'>
                                        <div style='text-align: center; font-size: 11px; font-weight: bold; padding: 4px 0;'>{he}</div>
                                        <div style='background: {load_color}; color: #000; padding: 4px; border-radius: 4px; text-align: center; font-size: 12px; font-weight: bold;'>{load_val:,.0f}</div>
                                        <div style='background: {wind_color}; color: #000; padding: 4px; border-radius: 4px; text-align: center; font-size: 12px; font-weight: bold;'>{wind_val:,.0f}</div>
                                        <div style='background: {solar_color}; color: {solar_text_color}; padding: 4px; border-radius: 4px; text-align: center; font-size: 12px; font-weight: bold;'>{solar_val:,.0f}</div>
                                        <div style='background: {net_color}; color: #000; padding: 4px; border-radius: 4px; text-align: center; font-size: 12px; font-weight: bold;'>{net_val:,.0f}</div>
                                    </div>
                                """
                            st.markdown(rows_html, unsafe_allow_html=True)
                            
                            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                            if st.button(f"View Chart for {date_obj.strftime('%m/%d/%Y')}", key=f"ercot_chart_{popup_date}"):
                                chart_col1, chart_col2 = st.columns(2)
                                with chart_col1:
                                    fig_load = go.Figure()
                                    load_hours = hours_data['HE'].tolist()
                                    load_values = hours_data['value'].tolist()
                                    fig_load.add_trace(go.Scatter(
                                        x=load_hours,
                                        y=load_values,
                                        mode='lines+markers',
                                        line=dict(color='#ff6b6b', width=3),
                                        marker=dict(size=6),
                                        hovertemplate='HE%{x}<br>%{y:,.0f}<extra></extra>'
                                    ))
                                    fig_load.update_layout(
                                        title=f"Load Forecast - {date_obj.strftime('%m/%d/%Y')}",
                                        xaxis=dict(title='Hour Ending', dtick=2),
                                        yaxis=dict(title='Load (MW)', range=[0, max(load_values) * 1.1]),
                                        template='plotly_dark',
                                        height=400,
                                        showlegend=False
                                    )
                                    st.plotly_chart(fig_load, use_container_width=True)

                                with chart_col2:
                                    fig_ren = go.Figure()
                                    if wind_date_data is not None and not wind_date_data.empty:
                                        wind_hours = wind_date_data['HE'].tolist()
                                        wind_values = wind_date_data['value'].tolist()
                                        fig_ren.add_trace(go.Scatter(
                                            x=wind_hours,
                                            y=wind_values,
                                            mode='lines+markers',
                                            name='Wind',
                                            line=dict(color='#51cf66', width=3),
                                            marker=dict(size=6),
                                            hovertemplate='HE%{x}<br>%{y:,.0f}<extra></extra>'
                                        ))
                                    if solar_date_data is not None and not solar_date_data.empty:
                                        solar_hours = solar_date_data['HE'].tolist()
                                        solar_values = solar_date_data['value'].tolist()
                                        fig_ren.add_trace(go.Scatter(
                                            x=solar_hours,
                                            y=solar_values,
                                            mode='lines+markers',
                                            name='Solar',
                                            line=dict(color='#ffd43b', width=3),
                                            marker=dict(size=6),
                                            hovertemplate='HE%{x}<br>%{y:,.0f}<extra></extra>'
                                        ))
                                    max_y = 20000
                                    if wind_date_data is not None and not wind_date_data.empty and solar_date_data is not None and not solar_date_data.empty:
                                        max_y = max(wind_values + solar_values) * 1.1
                                    fig_ren.update_layout(
                                        title=f"Renewable Forecast - {date_obj.strftime('%m/%d/%Y')}",
                                        xaxis=dict(title='Hour Ending', dtick=2),
                                        yaxis=dict(title='Generation (MW)', range=[0, max_y]),
                                        template='plotly_dark',
                                        height=400,
                                        showlegend=True,
                                        legend=dict(x=0.02, y=0.98),
                                        hovermode='x unified'
                                    )
                                    st.plotly_chart(fig_ren, use_container_width=True)
                            if st.button("Close", key="close_ercot", type="primary", use_container_width=True):
                                if 'ercot_dialog_active' in st.session_state:
                                    del st.session_state['ercot_dialog_active']
                                del st.session_state['ercot_popup_date']
                                st.rerun()
                    show_ercot_dialog()
                    # Clear the active flag after showing dialog so clicking outside will clean up
                    if 'ercot_dialog_active' in st.session_state:
                        del st.session_state['ercot_dialog_active']

                st.markdown("---")
            else:
                st.warning("No Meteologica load data available")

        # Tab 2 - PJM Weekly
        with tab2:
            st.caption(f"Data last fetched: {met_fetch_time} | Next refresh: {next_refresh_time}")
            if pjm_met_load_df is not None and not pjm_met_load_df.empty:
                pjm_min_load = pjm_met_load_df['value'].min()
                pjm_max_load = pjm_met_load_df['value'].max()
                pjm_met_dates = sorted(pjm_met_load_df['deliveryDate'].unique())[:14]
                pjm_load_df_filtered = None
                if pjm_load_df is not None and not pjm_load_df.empty:
                    pjm_load_df_filtered = pjm_load_df[pjm_load_df['forecast_area'] == 'RTO_COMBINED']
                if pjm_load_df_filtered is not None and not pjm_load_df_filtered.empty:
                    pjm_rto_dates = sorted(pjm_load_df_filtered['deliveryDate'].unique())[:7]
                    pjm_rto_peak_loads = [pjm_load_df_filtered[pjm_load_df_filtered['deliveryDate'] == d]['value'].max() for d in pjm_rto_dates]
                    pjm_rto_min_peak = min(pjm_rto_peak_loads) if pjm_rto_peak_loads else 0
                    pjm_rto_max_peak = max(pjm_rto_peak_loads) if pjm_rto_peak_loads else 1
                    # Get peak hours for PJM RTO
                    peak_hours_pjm = [pjm_load_df_filtered[pjm_load_df_filtered['deliveryDate'] == d]['value'].idxmax() for d in pjm_rto_dates]
                    peak_hours_pjm = [pjm_load_df_filtered.loc[idx, 'HE'] if idx in pjm_load_df_filtered.index else 0 for idx in peak_hours_pjm]
                else:
                    pjm_rto_dates = []
                    pjm_rto_peak_loads = []
                    peak_hours_pjm = []
                    pjm_rto_min_peak = 0
                    pjm_rto_max_peak = 1
                pjm_wind_min = pjm_met_wind_df['value'].min() if pjm_met_wind_df is not None and not pjm_met_wind_df.empty else 0
                pjm_wind_max = pjm_met_wind_df['value'].max() if pjm_met_wind_df is not None and not pjm_met_wind_df.empty else 1
                pjm_solar_min = pjm_met_solar_df['value'].min() if pjm_met_solar_df is not None and not pjm_met_solar_df.empty else 0
                pjm_solar_max = pjm_met_solar_df['value'].max() if pjm_met_solar_df is not None and not pjm_met_solar_df.empty else 1
                pjm_met_peak_loads = [pjm_met_load_df[pjm_met_load_df['deliveryDate'] == d]['value'].max() for d in pjm_met_dates]
                pjm_met_min_peak = min(pjm_met_peak_loads) if pjm_met_peak_loads else 0
                pjm_met_max_peak = max(pjm_met_peak_loads) if pjm_met_peak_loads else 1
                # Get peak hours for PJM Meteologica
                peak_hours_pjm_met = [pjm_met_load_df[pjm_met_load_df['deliveryDate'] == d]['value'].idxmax() for d in pjm_met_dates]
                peak_hours_pjm_met = [pjm_met_load_df.loc[idx, 'HE'] if idx in pjm_met_load_df.index else 0 for idx in peak_hours_pjm_met]
                pjm_wind_dates = sorted(pjm_met_wind_df['deliveryDate'].unique())[:14] if pjm_met_wind_df is not None and not pjm_met_wind_df.empty else []
                pjm_wind_avgs = []
                for date in pjm_wind_dates:
                    onpeak = pjm_met_wind_df[(pjm_met_wind_df['deliveryDate'] == date) & (pjm_met_wind_df['HE'] >= 8) & (pjm_met_wind_df['HE'] <= 23)]
                    pjm_wind_avgs.append(onpeak['value'].mean() if not onpeak.empty else 0)
                pjm_wind_min_pk = min(pjm_wind_avgs) if pjm_wind_avgs else 0
                pjm_wind_max_pk = max(pjm_wind_avgs) if pjm_wind_avgs else 1
                pjm_solar_dates = sorted(pjm_met_solar_df['deliveryDate'].unique())[:14] if pjm_met_solar_df is not None and not pjm_met_solar_df.empty else []
                pjm_solar_peaks = [pjm_met_solar_df[pjm_met_solar_df['deliveryDate'] == d]['value'].max() for d in pjm_solar_dates]
                pjm_solar_min_pk = min(pjm_solar_peaks) if pjm_solar_peaks else 0
                pjm_solar_max_pk = max(pjm_solar_peaks) if pjm_solar_peaks else 1
                pjm_outage_dates = sorted(pjm_outage_df['operatingDate'].unique())[:7] if pjm_outage_df is not None and not pjm_outage_df.empty else []
                pjm_outage_peaks = [pjm_outage_df[pjm_outage_df['operatingDate'] == d]['totalOutages'].max() for d in pjm_outage_dates] if pjm_outage_dates else []
                pjm_outage_min = min(pjm_outage_peaks) if pjm_outage_peaks else 0
                pjm_outage_max = max(pjm_outage_peaks) if pjm_outage_peaks else 1
                pjm_load_dates_set = set(pjm_met_load_df['deliveryDate'].unique())
                pjm_wind_dates_set = set(pjm_met_wind_df['deliveryDate'].unique()) if pjm_met_wind_df is not None else set()
                pjm_solar_dates_set = set(pjm_met_solar_df['deliveryDate'].unique()) if pjm_met_solar_df is not None else set()
                pjm_common_dates = sorted(pjm_load_dates_set & pjm_wind_dates_set & pjm_solar_dates_set)[:14]
                pjm_net_peaks = []
                for date in pjm_common_dates:
                    merged = pjm_met_load_df[pjm_met_load_df['deliveryDate'] == date].merge(
                        pjm_met_wind_df[pjm_met_wind_df['deliveryDate'] == date], on=['deliveryDate', 'HE'], suffixes=('_load', '_wind')
                    ).merge(
                        pjm_met_solar_df[pjm_met_solar_df['deliveryDate'] == date], on=['deliveryDate', 'HE']
                    )
                    merged['net_load'] = merged['value_load'] - merged['value_wind'] - merged['value']
                    pjm_net_peaks.append(merged['net_load'].max())
                pjm_net_min = min(pjm_net_peaks) if pjm_net_peaks else 0
                pjm_net_max = max(pjm_net_peaks) if pjm_net_peaks else 1
                pjm_outage_dates_set = set(pjm_outage_df['operatingDate'].unique()) if pjm_outage_df is not None and not pjm_outage_df.empty else set()
                pjm_common_dates_eff = sorted(pjm_load_dates_set & pjm_wind_dates_set & pjm_solar_dates_set & pjm_outage_dates_set)[:7]
                pjm_eff_peaks = []
                for date in pjm_common_dates_eff:
                    merged = pjm_met_load_df[pjm_met_load_df['deliveryDate'] == date].merge(
                        pjm_met_wind_df[pjm_met_wind_df['deliveryDate'] == date], on=['deliveryDate', 'HE'], suffixes=('_load', '_wind')
                    ).merge(
                        pjm_met_solar_df[pjm_met_solar_df['deliveryDate'] == date], on=['deliveryDate', 'HE']
                    ).merge(
                        pjm_outage_df[pjm_outage_df['operatingDate'] == date][['operatingDate', 'totalOutages']],
                        left_on=['deliveryDate'], right_on=['operatingDate'], how='left'
                    )
                    merged['totalOutages'] = merged['totalOutages'].fillna(0)
                    merged['eff_net'] = (merged['value_load'] - merged['value_wind'] -
                                         merged['value'] + merged['totalOutages'])
                    pjm_eff_peaks.append(merged['eff_net'].max())
                pjm_eff_min = min(pjm_eff_peaks) if pjm_eff_peaks else 0
                pjm_eff_max = max(pjm_eff_peaks) if pjm_eff_peaks else 1
                yesterday = (datetime.today() - timedelta(days=1)).date()
                today = datetime.today().date()
                dropdown_options = [
                    f"{yesterday.strftime('%m/%d')} HE17",
                    f"{today.strftime('%m/%d')} HE1"
                ]
                selected_option = st.selectbox("Compare with historical forecast:", dropdown_options, key="pjm_dropdown")
                selected_key = 'yesterday_HE17' if 'HE17' in selected_option else 'today_HE1'
                st.markdown("### Meteologica")
                st.markdown("<div style='display: flex; justify-content: space-around; margin-bottom: 5px;'>" +
                            "".join([f"<div style='text-align: center; flex: 1; font-size: 11px; color: #888888;'>{pd.to_datetime(d).strftime('%a')}</div>" for d in pjm_met_dates]) +
                            "</div>", unsafe_allow_html=True)
                cols = st.columns(14)
                pjm_met_deltas = []
                for idx, date in enumerate(pjm_met_dates):
                    date_obj = pd.to_datetime(date)
                    peak_load = pjm_met_peak_loads[idx]
                    peak_hour = peak_hours_pjm_met[idx]
                    peak_color = get_color_for_value(peak_load, pjm_met_min_peak, pjm_met_max_peak)
                    with cols[idx]:
                        if st.button(f" {date_obj.strftime('%m/%d')}", key=f"pjm_met_{date}", use_container_width=True):
                            st.session_state['pjm_popup_date'] = date
                            st.session_state['pjm_dialog_active'] = True
                        st.markdown(f"""
                            <div style='text-align: center; padding: 10px 3px; background-color: {peak_color}; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                                <div style='font-size: 13px; font-weight: bold; margin-bottom: 4px; color: #000000;'>HE{peak_hour}</div>
                                <div style='font-size: 18px; font-weight: bold; color: #000000;'>{peak_load:,.0f}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    cached_value = historical_cache[selected_key]['data'].get('pjm_met_load', {}).get(str(date), None)
                    delta = peak_load - cached_value if cached_value is not None else None
                    pjm_met_deltas.append(delta)
                st.markdown("<div style='font-size: 12px; font-weight: bold; margin-top: 5px;'>Delta (Current - Historical):</div>", unsafe_allow_html=True)
                cols = st.columns(14)
                for idx, delta in enumerate(pjm_met_deltas):
                    with cols[idx]:
                        if delta is not None:
                            st.markdown(f"""
                                <div style='text-align: center; padding: 5px 3px;'>
                                    <div style='font-size: 12px; color: #ffffff; font-weight: bold;'>{delta:+,.0f}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='text-align: center; padding: 5px 3px; font-size: 12px;'>N/A</div>", unsafe_allow_html=True)
                st.markdown("<hr style='border: none; border-top: 3px solid white; margin: 20px 0;'>", unsafe_allow_html=True)
                st.markdown("### PJM RTO")
                cols = st.columns(14)
                pjm_rto_deltas = []
                for idx, date in enumerate(pjm_rto_dates):
                    date_obj = pd.to_datetime(date)
                    peak_load = pjm_rto_peak_loads[idx]
                    peak_hour = peak_hours_pjm[idx]
                    peak_color = get_color_for_value(peak_load, pjm_rto_min_peak, pjm_rto_max_peak)
                    with cols[idx]:
                        st.markdown(f"""
                            <div style='text-align: center; padding: 10px 3px; background-color: {peak_color}; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                                <div style='font-size: 13px; font-weight: bold; margin-bottom: 4px; color: #000000;'>{date_obj.strftime('%m/%d')} (HE{peak_hour})</div>
                                <div style='font-size: 18px; font-weight: bold; color: #000000;'>{peak_load:,.0f}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    cached_value = historical_cache[selected_key]['data'].get('pjm_rto', {}).get(str(date), None)
                    delta = peak_load - cached_value if cached_value is not None else None
                    pjm_rto_deltas.append(delta)
                st.markdown("<div style='font-size: 12px; font-weight: bold; margin-top: 5px;'>Delta (Current - Historical):</div>", unsafe_allow_html=True)
                cols = st.columns(14)
                for idx, delta in enumerate(pjm_rto_deltas):
                    with cols[idx]:
                        if delta is not None:
                            st.markdown(f"""
                                <div style='text-align: center; padding: 5px 3px;'>
                                    <div style='font-size: 12px; color: #ffffff; font-weight: bold;'>{delta:+,.0f}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='text-align: center; padding: 5px 3px; font-size: 12px;'>N/A</div>", unsafe_allow_html=True)


                # Zone dropdown - only show if zone is selected
                if pjm_load_df is not None and not pjm_load_df.empty:
                    zone_options = sorted([z for z in pjm_load_df['forecast_area'].unique().tolist() if z != 'RTO_COMBINED'])
                    if zone_options:
                        selected_zone = st.selectbox("Select PJM Zone", ['None'] + zone_options, key="pjm_zone_select")

                        if selected_zone != 'None':
                            zone_df = pjm_load_df[pjm_load_df['forecast_area'] == selected_zone].copy()
                            if not zone_df.empty:
                                zone_dates = sorted(zone_df['deliveryDate'].unique())[:7]
                                zone_peak_loads = [zone_df[zone_df['deliveryDate'] == d]['value'].max() for d in zone_dates]
                                zone_min_peak = min(zone_peak_loads) if zone_peak_loads else 0
                                zone_max_peak = max(zone_peak_loads) if zone_peak_loads else 1

                                st.markdown(f"### PJM {selected_zone} Load Forecast")
                                cols = st.columns(14)
                                for idx, date in enumerate(zone_dates):
                                    date_obj = pd.to_datetime(date)
                                    peak_load = zone_peak_loads[idx]
                                    peak_color = get_color_for_value(peak_load, zone_min_peak, zone_max_peak)
                                    with cols[idx]:
                                        st.markdown(f"<div style='text-align: center; font-size: 11px; color: #888888; margin-bottom: 5px;'>{date_obj.strftime('%a')}</div>", unsafe_allow_html=True)
                                        st.markdown(f"""
                                            <div style='text-align: center; padding: 10px 3px; background-color: {peak_color}; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                                                <div style='font-size: 13px; font-weight: bold; margin-bottom: 4px; color: #000000;'>{date_obj.strftime('%m/%d')}</div>
                                                <div style='font-size: 18px; font-weight: bold; color: #000000;'>{peak_load:,.0f}</div>
                                            </div>
                                        """, unsafe_allow_html=True)

                                st.markdown("<hr style='border: none; border-top: 3px solid white; margin: 20px 0;'>", unsafe_allow_html=True)

                st.markdown("<hr style='border: none; border-top: 3px solid white; margin: 20px 0;'>", unsafe_allow_html=True)
                st.markdown("### Wind - Onpeak (HE 8-23)")
                if pjm_met_wind_df is not None and not pjm_met_wind_df.empty:
                    st.markdown("<div style='display: flex; justify-content: space-around; margin-bottom: 5px;'>" +
                                "".join([f"<div style='text-align: center; flex: 1; font-size: 11px; color: #888888;'>{pd.to_datetime(d).strftime('%a')}</div>" for d in pjm_wind_dates]) +
                                "</div>", unsafe_allow_html=True)
                    cols = st.columns(14)
                    pjm_wind_deltas = []
                    for idx, date in enumerate(pjm_wind_dates):
                        date_obj = pd.to_datetime(date)
                        peak_wind = pjm_wind_avgs[idx]
                        peak_color = get_color_for_value(peak_wind, pjm_wind_min_pk, pjm_wind_max_pk, reverse=True)
                        with cols[idx]:
                            st.markdown(f"""
                                <div style='text-align: center; padding: 10px 3px; background-color: {peak_color}; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                                    <div style='font-size: 13px; font-weight: bold; margin-bottom: 4px; color: #000000;'>{date_obj.strftime('%m/%d')}</div>
                                    <div style='font-size: 18px; font-weight: bold; color: #000000;'>{peak_wind:,.0f}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        cached_value = historical_cache[selected_key]['data'].get('pjm_wind', {}).get(str(date), None)
                        delta = peak_wind - cached_value if cached_value is not None else None
                        pjm_wind_deltas.append(delta)
                    st.markdown("<div style='font-size: 12px; font-weight: bold; margin-top: 5px;'>Delta (Current - Historical):</div>", unsafe_allow_html=True)
                    cols = st.columns(14)
                    for idx, delta in enumerate(pjm_wind_deltas):
                        with cols[idx]:
                            if delta is not None:
                                st.markdown(f"""
                                    <div style='text-align: center; padding: 5px 3px;'>
                                        <div style='font-size: 12px; color: #ffffff; font-weight: bold;'>{delta:+,.0f}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("<div style='text-align: center; padding: 5px 3px; font-size: 12px;'>N/A</div>", unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("### Solar")
                if pjm_met_solar_df is not None and not pjm_met_solar_df.empty:
                    st.markdown("<div style='display: flex; justify-content: space-around; margin-bottom: 5px;'>" +
                                "".join([f"<div style='text-align: center; flex: 1; font-size: 11px; color: #888888;'>{pd.to_datetime(d).strftime('%a')}</div>" for d in pjm_solar_dates]) +
                                "</div>", unsafe_allow_html=True)
                    cols = st.columns(14)
                    pjm_solar_deltas = []
                    for idx, date in enumerate(pjm_solar_dates):
                        date_obj = pd.to_datetime(date)
                        peak_solar = pjm_solar_peaks[idx]
                        peak_color = get_color_for_value(peak_solar, pjm_solar_min_pk, pjm_solar_max_pk, reverse=True)
                        with cols[idx]:
                            st.markdown(f"""
                                <div style='text-align: center; padding: 10px 3px; background-color: {peak_color}; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                                    <div style='font-size: 13px; font-weight: bold; margin-bottom: 4px; color: #000000;'>{date_obj.strftime('%m/%d')}</div>
                                    <div style='font-size: 18px; font-weight: bold; color: #000000;'>{peak_solar:,.0f}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        cached_value = historical_cache[selected_key]['data'].get('pjm_solar', {}).get(str(date), None)
                        delta = peak_solar - cached_value if cached_value is not None else None
                        pjm_solar_deltas.append(delta)
                    st.markdown("<div style='font-size: 12px; font-weight: bold; margin-top: 5px;'>Delta (Current - Historical):</div>", unsafe_allow_html=True)
                    cols = st.columns(14)
                    for idx, delta in enumerate(pjm_solar_deltas):
                        with cols[idx]:
                            if delta is not None:
                                st.markdown(f"""
                                    <div style='text-align: center; padding: 5px 3px;'>
                                        <div style='font-size: 12px; color: #ffffff; font-weight: bold;'>{delta:+,.0f}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("<div style='text-align: center; padding: 5px 3px; font-size: 12px;'>N/A</div>", unsafe_allow_html=True)
                st.markdown("<hr style='border: none; border-top: 3px solid white; margin: 20px 0;'>", unsafe_allow_html=True)
                st.markdown("### Outages")
                if pjm_outage_df is not None and not pjm_outage_df.empty:
                    cols = st.columns(14)
                    pjm_outage_deltas = []
                    for idx, date in enumerate(pjm_outage_dates):
                        date_obj = pd.to_datetime(date)
                        peak_outage = pjm_outage_peaks[idx]
                        outage_color = get_color_for_value(peak_outage, pjm_outage_min, pjm_outage_max)
                        with cols[idx]:
                            st.markdown(f"<div style='text-align: center; font-size: 11px; color: #888888; margin-bottom: 5px;'>{date_obj.strftime('%a')}</div>", unsafe_allow_html=True)
                            st.markdown(f"""
                                <div style='text-align: center; padding: 10px 3px; background-color: {outage_color}; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                                    <div style='font-size: 13px; font-weight: bold; margin-bottom: 4px; color: #000000;'>{date_obj.strftime('%m/%d')}</div>
                                    <div style='font-size: 18px; font-weight: bold; color: #000000;'>{peak_outage:,.0f}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        cached_value = historical_cache[selected_key]['data'].get('pjm_outages', {}).get(str(date), None)
                        delta = peak_outage - cached_value if cached_value is not None else None
                        pjm_outage_deltas.append(delta)
                    st.markdown("<div style='font-size: 12px; font-weight: bold; margin-top: 5px;'>Delta (Current - Historical):</div>", unsafe_allow_html=True)
                    cols = st.columns(14)
                    for idx, delta in enumerate(pjm_outage_deltas):
                        with cols[idx]:
                            if delta is not None:
                                st.markdown(f"""
                                    <div style='text-align: center; padding: 5px 3px;'>
                                        <div style='font-size: 12px; color: #ffffff; font-weight: bold;'>{delta:+,.0f}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("<div style='text-align: center; padding: 5px 3px; font-size: 12px;'>N/A</div>", unsafe_allow_html=True)
                st.markdown("<hr style='border: none; border-top: 3px solid white; margin: 20px 0;'>", unsafe_allow_html=True)
                st.markdown("### Peak Net Load")
                if pjm_met_load_df is not None and pjm_met_wind_df is not None and pjm_met_solar_df is not None:
                    st.markdown("<div style='display: flex; justify-content: space-around; margin-bottom: 5px;'>" +
                                "".join([f"<div style='text-align: center; flex: 1; font-size: 11px; color: #888888;'>{pd.to_datetime(d).strftime('%a')}</div>" for d in pjm_common_dates]) +
                                "</div>", unsafe_allow_html=True)
                    cols = st.columns(14)
                    pjm_net_deltas = []
                    for idx, date in enumerate(pjm_common_dates):
                        date_obj = pd.to_datetime(date)
                        peak_net = pjm_net_peaks[idx]
                        net_color = get_color_for_value(peak_net, pjm_net_min, pjm_net_max)
                        with cols[idx]:
                            st.markdown(f"""
                                <div style='text-align: center; padding: 10px 3px; background-color: {net_color}; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                                    <div style='font-size: 13px; font-weight: bold; margin-bottom: 4px; color: #000000;'>{date_obj.strftime('%m/%d')}</div>
                                    <div style='font-size: 18px; font-weight: bold; color: #000000;'>{peak_net:,.0f}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        cached_value = historical_cache[selected_key]['data'].get('pjm_net_load', {}).get(str(date), None)
                        delta = peak_net - cached_value if cached_value is not None else None
                        pjm_net_deltas.append(delta)
                    st.markdown("<div style='font-size: 12px; font-weight: bold; margin-top: 5px;'>Delta (Current - Historical):</div>", unsafe_allow_html=True)
                    cols = st.columns(14)
                    for idx, delta in enumerate(pjm_net_deltas):
                        with cols[idx]:
                            if delta is not None:
                                st.markdown(f"""
                                    <div style='text-align: center; padding: 5px 3px;'>
                                        <div style='font-size: 12px; color: #ffffff; font-weight: bold;'>{delta:+,.0f}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("<div style='text-align: center; padding: 5px 3px; font-size: 12px;'>N/A</div>", unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("### Peak Effective Net Load")
                if pjm_met_load_df is not None and pjm_met_wind_df is not None and pjm_met_solar_df is not None and pjm_outage_df is not None:
                    cols = st.columns(14)
                    pjm_eff_deltas = []
                    for idx, date in enumerate(pjm_common_dates_eff):
                        date_obj = pd.to_datetime(date)
                        peak_eff = pjm_eff_peaks[idx]
                        eff_color = get_color_for_value(peak_eff, pjm_eff_min, pjm_eff_max)
                        with cols[idx]:
                            st.markdown(f"<div style='text-align: center; font-size: 11px; color: #888888; margin-bottom: 5px;'>{date_obj.strftime('%a')}</div>", unsafe_allow_html=True)
                            st.markdown(f"""
                                <div style='text-align: center; padding: 10px 3px; background-color: {eff_color}; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
                                    <div style='font-size: 13px; font-weight: bold; margin-bottom: 4px; color: #000000;'>{date_obj.strftime('%m/%d')}</div>
                                    <div style='font-size: 18px; font-weight: bold; color: #000000;'>{peak_eff:,.0f}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        cached_value = historical_cache[selected_key]['data'].get('pjm_eff_net', {}).get(str(date), None)
                        delta = peak_eff - cached_value if cached_value is not None else None
                        pjm_eff_deltas.append(delta)
                    st.markdown("<div style='font-size: 12px; font-weight: bold; margin-top: 5px;'>Delta (Current - Historical):</div>", unsafe_allow_html=True)
                    cols = st.columns(14)
                    for idx, delta in enumerate(pjm_eff_deltas):
                        with cols[idx]:
                            if delta is not None:
                                st.markdown(f"""
                                    <div style='text-align: center; padding: 5px 3px;'>
                                        <div style='font-size: 12px; color: #ffffff; font-weight: bold;'>{delta:+,.0f}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("<div style='text-align: center; padding: 5px 3px; font-size: 12px;'>N/A</div>", unsafe_allow_html=True)

                # Popup Dialog for PJM date details
                if 'pjm_popup_date' in st.session_state:
                    @st.dialog("PJM Load Details", width="large")
                    def show_pjm_dialog():
                        popup_date = st.session_state['pjm_popup_date']
                        date_obj = pd.to_datetime(popup_date)
                        st.markdown(f"#### {date_obj.strftime('%A, %B %d, %Y')}")
                        hours_data = pjm_met_load_df[pjm_met_load_df['deliveryDate'] == popup_date]
                        wind_date_data = pjm_met_wind_df[pjm_met_wind_df['deliveryDate'] == popup_date] if pjm_met_wind_df is not None else None
                        solar_date_data = pjm_met_solar_df[pjm_met_solar_df['deliveryDate'] == popup_date] if pjm_met_solar_df is not None else None
                        
                        # Pre-calculate net loads for color scaling
                        net_loads_all = []
                        if wind_date_data is not None and solar_date_data is not None and not wind_date_data.empty and not solar_date_data.empty:
                            for idx in range(24):
                                he = idx + 1
                                hour_row = hours_data[hours_data['HE'] == he]
                                wind_row = wind_date_data[wind_date_data['HE'] == he]
                                solar_row = solar_date_data[solar_date_data['HE'] == he]
                                if not hour_row.empty and not wind_row.empty and not solar_row.empty:
                                    net_load = hour_row['value'].iloc[0] - (wind_row['value'].iloc[0] + solar_row['value'].iloc[0])
                                    net_loads_all.append(net_load)
                        pjm_popup_net_min = min(net_loads_all) if net_loads_all else 0
                        pjm_popup_net_max = max(net_loads_all) if net_loads_all else 1
                        
                        if not hours_data.empty:
                            # Compact table header
                            st.markdown("""
                                <div style='display: grid; grid-template-columns: 40px 1fr 1fr 1fr 1fr; gap: 4px; margin-bottom: 4px; font-size: 11px; font-weight: bold; color: #888;'>
                                    <div style='text-align: center;'>HE</div>
                                    <div style='text-align: center;'>Load</div>
                                    <div style='text-align: center;'>Wind</div>
                                    <div style='text-align: center;'>Solar</div>
                                    <div style='text-align: center;'>Net</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Build all rows as compact grid
                            rows_html = ""
                            for idx in range(24):
                                he = idx + 1
                                hour_row = hours_data[hours_data['HE'] == he]
                                wind_row = wind_date_data[wind_date_data['HE'] == he] if wind_date_data is not None else None
                                solar_row = solar_date_data[solar_date_data['HE'] == he] if solar_date_data is not None else None
                                
                                load_val = hour_row['value'].iloc[0] if not hour_row.empty else 0
                                wind_val = wind_row['value'].iloc[0] if wind_row is not None and not wind_row.empty else 0
                                solar_val = solar_row['value'].iloc[0] if solar_row is not None and not solar_row.empty else 0
                                net_val = load_val - (wind_val + solar_val)
                                
                                load_color = get_color_for_value(load_val, pjm_min_load, pjm_max_load)
                                wind_color = get_color_for_value(wind_val, pjm_wind_min, pjm_wind_max, reverse=True)
                                solar_color = '#333333' if solar_val == 0 else get_color_for_value(solar_val, pjm_solar_min, pjm_solar_max, reverse=True)
                                solar_text_color = '#666666' if solar_val == 0 else '#000'
                                net_color = get_color_for_value(net_val, pjm_popup_net_min, pjm_popup_net_max)
                                
                                rows_html += f"""
                                    <div style='display: grid; grid-template-columns: 40px 1fr 1fr 1fr 1fr; gap: 4px; margin-bottom: 2px;'>
                                        <div style='text-align: center; font-size: 11px; font-weight: bold; padding: 4px 0;'>{he}</div>
                                        <div style='background: {load_color}; color: #000; padding: 4px; border-radius: 4px; text-align: center; font-size: 12px; font-weight: bold;'>{load_val:,.0f}</div>
                                        <div style='background: {wind_color}; color: #000; padding: 4px; border-radius: 4px; text-align: center; font-size: 12px; font-weight: bold;'>{wind_val:,.0f}</div>
                                        <div style='background: {solar_color}; color: {solar_text_color}; padding: 4px; border-radius: 4px; text-align: center; font-size: 12px; font-weight: bold;'>{solar_val:,.0f}</div>
                                        <div style='background: {net_color}; color: #000; padding: 4px; border-radius: 4px; text-align: center; font-size: 12px; font-weight: bold;'>{net_val:,.0f}</div>
                                    </div>
                                """
                            st.markdown(rows_html, unsafe_allow_html=True)
                            
                            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                            if st.button(f"View Chart for {date_obj.strftime('%m/%d/%Y')}", key=f"pjm_chart_{popup_date}"):
                                chart_col1, chart_col2 = st.columns(2)
                                with chart_col1:
                                    fig_load = go.Figure()
                                    load_hours = hours_data['HE'].tolist()
                                    load_values = hours_data['value'].tolist()
                                    fig_load.add_trace(go.Scatter(
                                        x=load_hours,
                                        y=load_values,
                                        mode='lines+markers',
                                        line=dict(color='#ff6b6b', width=3),
                                        marker=dict(size=6),
                                        hovertemplate='HE%{x}<br>%{y:,.0f}<extra></extra>'
                                    ))
                                    fig_load.update_layout(
                                        title=f"PJM Load Forecast - {date_obj.strftime('%m/%d/%Y')}",
                                        xaxis=dict(title='Hour Ending', dtick=2),
                                        yaxis=dict(title='Load (MW)', range=[0, max(load_values) * 1.1]),
                                        template='plotly_dark',
                                        height=400,
                                        showlegend=False
                                    )
                                    st.plotly_chart(fig_load, use_container_width=True)

                                with chart_col2:
                                    fig_ren = go.Figure()
                                    if wind_date_data is not None and not wind_date_data.empty:
                                        wind_hours = wind_date_data['HE'].tolist()
                                        wind_values = wind_date_data['value'].tolist()
                                        fig_ren.add_trace(go.Scatter(
                                            x=wind_hours,
                                            y=wind_values,
                                            mode='lines+markers',
                                            name='Wind',
                                            line=dict(color='#51cf66', width=3),
                                            marker=dict(size=6),
                                            hovertemplate='HE%{x}<br>%{y:,.0f}<extra></extra>'
                                        ))
                                    if solar_date_data is not None and not solar_date_data.empty:
                                        solar_hours = solar_date_data['HE'].tolist()
                                        solar_values = solar_date_data['value'].tolist()
                                        fig_ren.add_trace(go.Scatter(
                                            x=solar_hours,
                                            y=solar_values,
                                            mode='lines+markers',
                                            name='Solar',
                                            line=dict(color='#ffd43b', width=3),
                                            marker=dict(size=6),
                                            hovertemplate='HE%{x}<br>%{y:,.0f}<extra></extra>'
                                        ))
                                    max_y = 30000
                                    if wind_date_data is not None and not wind_date_data.empty and solar_date_data is not None and not solar_date_data.empty:
                                        max_y = max(wind_values + solar_values) * 1.1
                                    fig_ren.update_layout(
                                        title=f"PJM Renewable Forecast - {date_obj.strftime('%m/%d/%Y')}",
                                        xaxis=dict(title='Hour Ending', dtick=2),
                                        yaxis=dict(title='Generation (MW)', range=[0, max_y]),
                                        template='plotly_dark',
                                        height=400,
                                        showlegend=True,
                                        legend=dict(x=0.02, y=0.98),
                                        hovermode='x unified'
                                    )
                                    st.plotly_chart(fig_ren, use_container_width=True)
                            if st.button("Close", key="close_pjm", type="primary", use_container_width=True):
                                if 'pjm_dialog_active' in st.session_state:
                                    del st.session_state['pjm_dialog_active']
                                del st.session_state['pjm_popup_date']
                                st.rerun()
                    show_pjm_dialog()
                    # Clear the active flag after showing dialog so clicking outside will clean up
                    if 'pjm_dialog_active' in st.session_state:
                        del st.session_state['pjm_dialog_active']

                st.markdown("---")
            else:
                st.warning("No PJM Meteologica load data available")


        # Tab 3 - Gas
        with tab3:
            st.header("NG Futures")

            chart_type = st.radio("Select Timeframe:", ["Daily", "Weekly"], horizontal=True)

            try:
                ticker = "NG=F"

                if chart_type == "Daily":
                    data = yf.download(ticker, period="6mo", interval="1d", progress=False)
                else:
                    data = yf.download(ticker, period="2y", interval="1wk", progress=False)

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                ohlc_cols = ["Open", "High", "Low", "Close"]
                data = data.dropna(subset=ohlc_cols)
                data[ohlc_cols] = data[ohlc_cols].astype(float)
                data.index = pd.to_datetime(data.index).tz_localize(None)

                if not data.empty:
                    # Get current price (most recent close)
                    current_price = data['Close'].iloc[-1]
                    current_date = data.index[-1].strftime('%Y-%m-%d')

                    # Display current price above chart
                    st.markdown(f"""
                        <div style='text-align: center; padding: 5px; margin-bottom: 3px;'>
                            <div style='font-size: 18px; color: #888888;'>Current Price (as of {current_date})</div>
                            <div style='font-size: 36px; font-weight: bold; color: #4CAF50;'>${current_price:.3f}</div>
                            <div style='font-size: 14px; color: #888888;'>USD per MMBtu</div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Create custom style without grid
                    mc = mpf.make_marketcolors(up='g', down='r', edge='inherit', wick='inherit', volume='in')
                    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='', y_on_right=True, facecolor='#0e1117', edgecolor='#ffffff')

                    fig, axes = mpf.plot(
                        data,
                        type='candle',
                        style=s,
                        ylabel='Price',
                        ylabel_lower='Volume',
                        volume=False,
                        figsize=(14, 7),
                        returnfig=True,
                        datetime_format='%Y-%m-%d',
                        xrotation=45
                    )

                    # Configure main price axis (right side)
                    axes[0].yaxis.set_label_position('right')
                    axes[0].yaxis.tick_right()
                    axes[0].tick_params(axis='both', colors='white', labelsize=10)
                    axes[0].spines['bottom'].set_color('white')
                    axes[0].spines['top'].set_color('white')
                    axes[0].spines['right'].set_color('white')
                    axes[0].spines['left'].set_color('white')
                    axes[0].xaxis.label.set_color('white')
                    axes[0].yaxis.label.set_color('white')

                    # Remove grid but keep axes visible
                    axes[0].grid(False)

                    # Configure volume axis if present
                    if len(axes) > 1:
                        axes[1].grid(False)
                        axes[1].tick_params(axis='both', colors='white', labelsize=10)
                        axes[1].spines['bottom'].set_color('white')
                        axes[1].spines['top'].set_color('white')
                        axes[1].spines['right'].set_color('white')
                        axes[1].spines['left'].set_color('white')
                        axes[1].xaxis.label.set_color('white')
                        axes[1].yaxis.label.set_color('white')

                    buf = BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#0e1117', dpi=100)
                    buf.seek(0)
                    st.image(buf, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("No data available for Natural Gas futures")

            except Exception as e:
                st.error(f"Error fetching Natural Gas data: {str(e)}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.exception(e)

if __name__ == "__main__":
    main()
