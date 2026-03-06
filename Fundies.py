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
import xml.etree.ElementTree as ET
from html import unescape
import re
from gridstatus import Ercot as GridStatusErcot
from gridstatus.ercot import ERCOTSevenDayLoadForecastReport
from zoneinfo import ZoneInfo
import numpy as np


st.set_page_config(page_title="Fundies", layout="wide", initial_sidebar_state="collapsed")


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

# Auto-refresh once per hour at 16:30 past each hour CT
def get_refresh_info():
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo('America/Chicago'))
    target_minute = 16
    target_second = 30
    
    
    current_seconds_into_hour = now.minute * 60 + now.second
    target_seconds_into_hour = target_minute * 60 + target_second  # 990 seconds = 16:30
    
    if current_seconds_into_hour < target_seconds_into_hour:
        
        seconds_until = target_seconds_into_hour - current_seconds_into_hour
        next_refresh = now.replace(minute=target_minute, second=target_second, microsecond=0)
    else:
        
        seconds_remaining_this_hour = 3600 - current_seconds_into_hour
        seconds_until = seconds_remaining_this_hour + target_seconds_into_hour
        next_refresh = (now + timedelta(hours=1)).replace(minute=target_minute, second=target_second, microsecond=0)
    
    return seconds_until, next_refresh.strftime('%I:%M:%S %p CT')

refresh_seconds, next_refresh_time = get_refresh_info()
st.markdown(f'<meta http-equiv="refresh" content="{refresh_seconds}">', unsafe_allow_html=True)

# Load from Streamlit secrets 
baseurl = "https://api-markets.meteologica.com/api/v1/"


CACHE_FILE = Path("/tmp/historical_cache.json")

def make_get_request(endpoint, query_params):
    url = baseurl + endpoint
    return requests.get(url, params=query_params, timeout=60)

def make_post_request(endpoint, json_body):
    url = baseurl + endpoint
    return requests.post(url, json=json_body, timeout=60)

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
    try:
        auth_response = requests.post(AUTH_URL, timeout=60)
        if auth_response.ok:
            access_token = auth_response.json().get("access_token")
            headers = {"Authorization": "Bearer " + access_token, "Ocp-Apim-Subscription-Key": SUBSCRIPTION}
            return headers
        st.error(f"Error in Authentication: {auth_response.text}")
    except requests.exceptions.Timeout:
        st.error("ERCOT authentication timed out")
    except Exception as e:
        st.error(f"ERCOT authentication error: {str(e)}")
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
    try:
        resp = requests.post(AUTH_URL, data=data, timeout=60)
        if resp.ok:
            access_token = resp.json().get("access_token")
            headers = {
                "Authorization": "Bearer " + access_token,
                "Ocp-Apim-Subscription-Key": SUBSCRIPTION
            }
            return headers
        st.error(f"Error in Outages Authentication: {resp.text}")
    except requests.exceptions.Timeout:
        st.error("ERCOT outages authentication timed out")
    except Exception as e:
        st.error(f"ERCOT outages authentication error: {str(e)}")
    return None

def fetch_outage_data_robust(url, headers, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=(60, 120))
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
            result = requests.get(url, headers=pjm_headers, timeout=(60, 120))
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
    if not headers:
        return None, cache_time
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

@st.cache_data(ttl=3600)  # Cache for 1 hour - Gist only updates at HE17 and HE01
def load_historical_cache():
    """Load snapshots from GitHub Gist"""
    try:
        gist_url = st.secrets.get("gist", {}).get("snapshot_url")
        if gist_url:
            # Add cache-busting parameter to avoid GitHub CDN caching issues
            cache_buster = f"?cb={int(time.time() // 3600)}"  # Changes every hour
            response = requests.get(gist_url + cache_buster, timeout=30)
            if response.ok:
                return response.json()
    except Exception as e:
        pass  # Silently fail, return None
    
    # Fallback to local file (for local development)
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return None

def save_historical_cache(cache_data):
    
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
        return True
    except Exception as e:
        return False

def upload_to_gist(cache_data):
    """Upload snapshot data to GitHub Gist"""
    try:
        gist_token = st.secrets.get("gist", {}).get("token")
        gist_id = st.secrets.get("gist", {}).get("id")
        
        if not gist_token or not gist_id:
            return False
        
        url = f"https://api.github.com/gists/{gist_id}"
        headers = {
            "Authorization": f"token {gist_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        payload = {
            "files": {
                "fundies_snapshots.json": {
                    "content": json.dumps(cache_data, indent=2, default=str)
                }
            }
        }
        response = requests.patch(url, headers=headers, json=payload, timeout=60)
        return response.ok
    except Exception as e:
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
    """Load snapshots from Gist, capture new ones if in window, and format for display"""
    cache = load_historical_cache()
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo('America/Chicago'))
    today = now.date()
    current_hour = now.hour

    if cache is None:
        cache = {
            'HE17_snapshot': {'captured_date': None, 'data': {}},
            'HE01_snapshot': {'captured_date': None, 'data': {}}
        }

    # Check snapshot
    snapshot_captured = False
    
    # HE17 window: 4-6 PM CT (hour 16-17)
    if 16 <= current_hour <= 17:
        he17_captured_date = cache.get('HE17_snapshot', {}).get('captured_date')
        session_key = f"he17_captured_{today}"
        if he17_captured_date != str(today) and session_key not in st.session_state:
            # Capture HE17 snapshot
            snapshot = create_snapshot_data(met_load_df, met_wind_df, met_solar_df, df, outage_df,
                               pjm_met_load_df, pjm_met_wind_df, pjm_met_solar_df, pjm_load_df, pjm_outage_df)
            cache['HE17_snapshot'] = {
                'captured_date': str(today),
                'capture_time': now.isoformat(),
                'data': snapshot
            }
            snapshot_captured = True
            st.session_state[session_key] = True
            st.success(f"📸 Captured HE17 snapshot at {now.strftime('%I:%M %p')} CT")
    
    # HE01 
    if current_hour == 0:
        he01_captured_date = cache.get('HE01_snapshot', {}).get('captured_date')
        session_key = f"he01_captured_{today}"
        if he01_captured_date != str(today) and session_key not in st.session_state:
            # Capture HE01 snapshot
            snapshot = create_snapshot_data(met_load_df, met_wind_df, met_solar_df, df, outage_df,
                               pjm_met_load_df, pjm_met_wind_df, pjm_met_solar_df, pjm_load_df, pjm_outage_df)
            cache['HE01_snapshot'] = {
                'captured_date': str(today),
                'capture_time': now.isoformat(),
                'data': snapshot
            }
            snapshot_captured = True
            st.session_state[session_key] = True
            st.success(f"📸 Captured HE01 snapshot at {now.strftime('%I:%M %p')} CT")
    
    # Upload to Gist 
    if snapshot_captured:
        if upload_to_gist(cache):
            st.success(" Snapshot uploaded")
            # Clear the Gist cache so next load gets fresh data
            load_historical_cache.clear()
        else:
            st.warning(" Failed to upload to Gist")
        # Also save locally as backup
        save_historical_cache(cache)

    # Handle both HE1_snapshot or HE01_snapshot
    he01_data = cache.get('HE01_snapshot') or cache.get('HE1_snapshot') or {'captured_date': None, 'data': {}}
    he17_data = cache.get('HE17_snapshot') or {'captured_date': None, 'data': {}}

    display_cache = {}
    yesterday = today - timedelta(days=1)

    # HE17 snapshot 
    he17_captured = he17_data.get('captured_date')
    if he17_captured:
        display_cache['yesterday_HE17'] = {
            'date': he17_captured,
            'data': he17_data.get('data', {})
        }
    else:
        display_cache['yesterday_HE17'] = {
            'date': str(yesterday),
            'data': {}
        }

    # HE01 snapshot 
    he01_captured = he01_data.get('captured_date')
    if he01_captured:
        display_cache['today_HE1'] = {
            'date': he01_captured,
            'data': he01_data.get('data', {})
        }
    else:
        display_cache['today_HE1'] = {
            'date': str(today),
            'data': {}
        }

    display_cache['fetch_time'] = now.strftime('%Y-%m-%d %H:%M:%S CT')

    return display_cache

def display_cache_status(cache):
    with st.sidebar:
        st.markdown("### Snapshot Status")
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo('America/Chicago'))
        
        
        he17_data = cache.get('HE17_snapshot', {})
        he01_data = cache.get('HE01_snapshot') or cache.get('HE1_snapshot', {})
        
        he17_captured = he17_data.get('captured_date')
        he17_updated = he17_data.get('capture_time') or he17_data.get('last_updated')
        if he17_captured:
            st.success(f"HE17: {he17_captured}")
            if he17_updated:
                try:
                    update_time = datetime.fromisoformat(he17_updated.replace('Z', '+00:00')).strftime('%I:%M %p')
                    st.caption(f"Captured: {update_time}")
                except:
                    pass
        else:
            st.warning("HE17: No snapshot")
            
        he01_captured = he01_data.get('captured_date')
        he01_updated = he01_data.get('capture_time') or he01_data.get('last_updated')
        if he01_captured:
            st.success(f"HE01: {he01_captured}")
            if he01_updated:
                try:
                    update_time = datetime.fromisoformat(he01_updated.replace('Z', '+00:00')).strftime('%I:%M %p')
                    st.caption(f"Captured: {update_time}")
                except:
                    pass
        else:
            st.warning("HE01: No snapshot")
            
        st.caption(f"Current: {now.strftime('%I:%M %p')} CT")

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

def check_password():
    
    
    if st.query_params.get("auth") == "1":
        return
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("Goat Farmers Only")
        with st.form("login_form"):
            password = st.text_input("Password:", type="password", key="password_input")
            submitted = st.form_submit_button("Login", type="primary")
            if submitted:
                if password == st.secrets.get("app", {}).get("password", ""):
                    st.session_state.authenticated = True
                    st.query_params["auth"] = "1"
                    st.rerun()
                else:
                    st.error("Wrong password")
        st.stop()

# ── News RSS Functions ──
NEWS_CATEGORIES = {
    "ERCOT": {
        "color": "#22c55e",
        "bg": "rgba(34,197,94,0.12)",
        "queries": [
            "ERCOT grid operations when:3d",
            "ERCOT load forecast record demand when:3d",
            "ERCOT generation outage when:3d",
            "ERCOT interconnection queue MW when:3d",
            "ERCOT plant retirement deactivation when:3d",
            "ERCOT datacenter large load when:3d",
            "PUCT rulemaking ERCOT reliability when:3d",
            "Texas grid reserve margin when:3d",
            "ERCOT conservation appeal emergency when:3d",
            "ERCOT transmission congestion when:3d",
        ],
    },
    "PJM": {
        "color": "#3b82f6",
        "bg": "rgba(59,130,246,0.12)",
        "queries": [
            "PJM grid operations reliability when:3d",
            "PJM capacity auction results when:3d",
            "PJM interconnection queue MW when:3d",
            "PJM plant retirement deactivation when:3d",
            "PJM datacenter large load when:3d",
            "PJM generation outage when:3d",
            "PJM transmission congestion when:3d",
            "PJM reserve margin when:3d",
        ],
    },
    "Gas": {
        "color": "#f97316",
        "bg": "rgba(249,115,22,0.12)",
        "queries": [
            "EIA natural gas storage injection withdrawal when:3d",
            "natural gas production Permian Haynesville when:3d",
            "LNG export feed gas terminal when:3d",
            "Henry Hub natural gas spot when:3d",
            "natural gas power burn demand when:3d",
            "natural gas storage report when:3d",
        ],
    },
    "Pipeline": {
        "color": "#a855f7",
        "bg": "rgba(168,85,247,0.12)",
        "queries": [
            "pipeline maintenance outage natural gas when:3d",
            "operational flow order OFO gas when:3d",
            "pipeline force majeure natural gas when:3d",
            "natural gas freeze off curtailment when:3d",
            "gas pipeline compressor outage when:3d",
            "Transco pipeline capacity when:3d",
            "gas pipeline FERC certificate when:3d",
        ],
    },
    "Load": {
        "color": "#eab308",
        "bg": "rgba(234,179,8,0.12)",
        "queries": [
            "datacenter power interconnection ERCOT PJM when:3d",
            "datacenter electricity demand gigawatt when:3d",
            "large load industrial power interconnection when:3d",
            "behind the meter generation datacenter when:3d",
        ],
    },
    "Regulatory": {
        "color": "#ef4444",
        "bg": "rgba(239,68,68,0.12)",
        "queries": [
            "FERC order rulemaking transmission when:3d",
            "NERC reliability standard grid when:3d",
            "power plant retirement closure MW when:3d",
            "new generation capacity commercial operation when:3d",
            "PUCT rulemaking reliability Texas when:3d",
            "EPA power plant emission rule when:3d",
        ],
    },
}

# ── X / Twitter Feeds via Nitter RSS ──
# These accounts post real-time operational power/gas market updates
X_FEEDS = {
    "ERCOT_ISO": {
        "category": "ERCOT",
        "display_name": "ERCOT",
        "description": "Official ERCOT grid operations",
    },
    "pjminterconnect": {
        "category": "PJM",
        "display_name": "PJM",
        "description": "Official PJM operations",
    },
    "grid_status": {
        "category": "ERCOT",
        "display_name": "Grid Status",
        "description": "Real-time grid data & analysis",
    },
    "douglewinenergy": {
        "category": "ERCOT",
        "display_name": "Doug Lewin",
        "description": "Stoic Energy - TX grid commentary",
    },
    "EIAgov": {
        "category": "Gas",
        "display_name": "EIA",
        "description": "US Energy Information Administration",
    },
    "NatGasWeather": {
        "category": "Gas",
        "display_name": "NatGasWeather",
        "description": "Gas weather & demand forecasts",
    },
}

# Nitter instances to try (public, may rotate)
NITTER_INSTANCES = [
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.woodland.cafe",
    "https://nitter.perennialte.ch",
]

# ── Relevance Filtering ──
# Whitelist: article MUST contain at least one of these to pass.
RELEVANCE_KEYWORDS = {
    # ISOs / RTOs
    "ercot", "pjm", "caiso", "miso", "spp", "nyiso", "isone",
    # Regulatory bodies
    "ferc", "nerc", "puct", "epa",
    # Grid / Power operations
    "grid", "load", "generation", "megawatt", " mw", "gigawatt", " gw",
    "outage", "curtailment", "reliability", "reserve margin",
    "interconnection", "queue", "capacity", "retirement", "deactivation",
    "congestion", "constraint", "transmission", "blackout", "brownout",
    "conservation", "emergency", "scarcity", "demand response",
    "ancillary", "frequency response", "spinning reserve",
    "power plant", "coal plant", "gas plant", "nuclear plant",
    "wind farm", "solar farm", "battery storage", "peaker",
    # Gas physical/operational
    "natural gas", "henry hub", "nat gas", "natgas",
    "pipeline", "compressor", "force majeure",
    "storage injection", "storage withdrawal", "storage report",
    "eia storage", "eia gas", "working gas",
    "freeze off", "freeze-off", "wellhead",
    "ofo", "operational flow order",
    "lng", "liquefaction", "feed gas", "export terminal",
    "permian", "haynesville", "marcellus", "utica", "appalachia",
    "waha", "transco", "sonat", "tetco", "centerpoint",
    "power burn", "gas burn", "gas demand",
    "bcf", "mcf", "mmbtu", "dekatherm",
    # Datacenter / large load
    "datacenter", "data center", "hyperscale", "large load",
    "behind the meter", "colocation", "colo ",
    # Renewables relevant to trading
    "wind generation", "solar generation", "renewable curtailment",
    "wind forecast", "solar forecast",
    # Specific to trading ops
    "day-ahead", "real-time", "dart spread", "basis",
    "wholesale electricity", "power price", "spark spread",
    "heat rate", "capacity auction", "capacity market",
}

# Sources that are financial noise
EXCLUDE_SOURCES = {
    "the motley fool", "motley fool",
    "seeking alpha", "seekingalpha",
    "investopedia", "investor's business daily",
    "benzinga", "zacks", "zacks investment",
    "tipranks", "marketbeat", "stockanalysis",
    "24/7 wall st", "247wallst",
    "insidermonkey", "insider monkey",
    "simply wall st",
    "barron's", "kiplinger",
    "gobankingrates", "bankrate",
    "fortune", "forbes",
}

@st.cache_data(ttl=1800)
def fetch_google_rss_news(query, category, _cache_time):
    """Fetch news from Google News RSS for a given query."""
    articles = []
    try:
        rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(rss_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if resp.ok:
            root = ET.fromstring(resp.content)
            channel = root.find("channel")
            if channel is not None:
                for item in channel.findall("item")[:5]:
                    title_el = item.find("title")
                    link_el = item.find("link")
                    pub_el = item.find("pubDate")
                    source_el = item.find("source")
                    title = unescape(title_el.text) if title_el is not None and title_el.text else ""
                    link = link_el.text if link_el is not None and link_el.text else ""
                    pub_date = pub_el.text if pub_el is not None and pub_el.text else ""
                    source = source_el.text if source_el is not None and source_el.text else ""
                    if title:
                        articles.append({
                            "category": category,
                            "headline": title,
                            "link": link,
                            "source": source,
                            "pubDate": pub_date,
                            "is_x_post": False,
                        })
    except Exception:
        pass
    return articles

@st.cache_data(ttl=1800)
def fetch_x_feed(username, feed_config, _cache_time):
    """Fetch recent posts from an X/Twitter account via Nitter RSS or Google search fallback."""
    articles = []
    category = feed_config["category"]
    display_name = feed_config["display_name"]

    # Method 1: Try Nitter RSS instances
    for nitter_base in NITTER_INSTANCES:
        try:
            rss_url = f"{nitter_base}/{username}/rss"
            resp = requests.get(rss_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if resp.ok and '<?xml' in resp.text[:100]:
                root = ET.fromstring(resp.content)
                channel = root.find("channel")
                if channel is not None:
                    for item in channel.findall("item")[:10]:
                        title_el = item.find("title")
                        link_el = item.find("link")
                        pub_el = item.find("pubDate")
                        title = unescape(title_el.text) if title_el is not None and title_el.text else ""
                        link = link_el.text if link_el is not None and link_el.text else ""
                        pub_date = pub_el.text if pub_el is not None and pub_el.text else ""
                        # Clean up nitter links to point to x.com
                        if link and nitter_base in link:
                            link = link.replace(nitter_base, "https://x.com")
                        # Truncate long tweets for headline display
                        if len(title) > 200:
                            title = title[:197] + "..."
                        if title and title.strip() != "":
                            articles.append({
                                "category": category,
                                "headline": title,
                                "link": link if link else f"https://x.com/{username}",
                                "source": f"@{username}",
                                "pubDate": pub_date,
                                "is_x_post": True,
                            })
                    if articles:
                        return articles  # Got data, stop trying instances
        except Exception:
            continue

    # Method 2: Fallback - Google News search for the account's posts
    if not articles:
        try:
            query = f"from:{username} site:x.com when:3d"
            rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
            resp = requests.get(rss_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if resp.ok:
                root = ET.fromstring(resp.content)
                channel = root.find("channel")
                if channel is not None:
                    for item in channel.findall("item")[:5]:
                        title_el = item.find("title")
                        link_el = item.find("link")
                        pub_el = item.find("pubDate")
                        title = unescape(title_el.text) if title_el is not None and title_el.text else ""
                        link = link_el.text if link_el is not None and link_el.text else ""
                        pub_date = pub_el.text if pub_el is not None and pub_el.text else ""
                        if title:
                            articles.append({
                                "category": category,
                                "headline": title,
                                "link": link,
                                "source": f"@{username}",
                                "pubDate": pub_date,
                                "is_x_post": True,
                            })
        except Exception:
            pass

    return articles

def parse_rss_date(date_str):
    """Parse RSS pubDate string to datetime."""
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(date_str)
    except Exception:
        return datetime.min.replace(tzinfo=None)

def format_relative_time(dt):
    """Format a datetime as relative time string."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo('UTC'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo('UTC'))
        diff = now - dt
        seconds = int(diff.total_seconds())
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            m = seconds // 60
            return f"{m}m ago"
        elif seconds < 86400:
            h = seconds // 3600
            return f"{h}h ago"
        else:
            d = seconds // 86400
            return f"{d}d ago"
    except Exception:
        return ""

@st.cache_data(ttl=1800)
def fetch_all_news(cache_time):
    """Fetch all news across all categories plus X feeds."""
    all_articles = []
    seen_headlines = set()

    # 1. Google News RSS articles
    for category, config in NEWS_CATEGORIES.items():
        for query in config["queries"]:
            articles = fetch_google_rss_news(query, category, cache_time)
            for article in articles:
                clean = re.sub(r'\s+', ' ', article["headline"].strip().lower())
                if clean not in seen_headlines:
                    seen_headlines.add(clean)
                    all_articles.append(article)

    # 2. X / Twitter feeds
    x_articles = []
    for username, feed_config in X_FEEDS.items():
        posts = fetch_x_feed(username, feed_config, cache_time)
        for post in posts:
            clean = re.sub(r'\s+', ' ', post["headline"].strip().lower())
            if clean not in seen_headlines:
                seen_headlines.add(clean)
                x_articles.append(post)
    all_articles.extend(x_articles)

    # Hard filter: drop articles older than 3 days
    from zoneinfo import ZoneInfo
    cutoff = datetime.now(ZoneInfo('UTC')) - timedelta(days=3)
    all_articles = [
        a for a in all_articles
        if parse_rss_date(a.get("pubDate", "")) >= cutoff
    ]

    # Filter: whitelist relevance + source exclusion
    # X posts from trusted accounts skip the whitelist (they're already curated)
    def is_relevant(article):
        # Trusted X accounts always pass
        if article.get("is_x_post", False):
            return True
        text = (article.get("headline", "") + " " + article.get("source", "")).lower()
        # Hard exclude: financial noise sources
        source_lower = article.get("source", "").lower().strip()
        for exc_source in EXCLUDE_SOURCES:
            if exc_source in source_lower:
                return False
        # Whitelist: headline MUST contain at least one relevant keyword
        for kw in RELEVANCE_KEYWORDS:
            if kw in text:
                return True
        return False

    all_articles = [a for a in all_articles if is_relevant(a)]

    all_articles.sort(key=lambda x: parse_rss_date(x.get("pubDate", "")), reverse=True)
    return all_articles

# ── ERCOT Reserve Margin Functions ──
ALL_HOURS = [f'HE{h:02}' for h in range(1, 25)]

def prep_interval(df_in, col='Interval Start'):
    df_out = df_in.copy()
    if col in df_out.columns:
        df_out[col] = pd.to_datetime(df_out[col]).dt.tz_localize(None)
        df_out['Hour Ending'] = (df_out[col].dt.hour + 1).apply(lambda x: f'HE{x:02}')
    return df_out

def filt_date(df_in, d, col='Interval Start'):
    if df_in.empty or col not in df_in.columns:
        return pd.DataFrame()
    return df_in[pd.to_datetime(df_in[col]).dt.date == d].copy()

def safe_int(v, default=0):
    try:
        if pd.isna(v): return default
        return int(v)
    except (ValueError, TypeError):
        return default

def safe_float(v, default=0.0):
    try:
        if pd.isna(v): return default
        return float(v)
    except (ValueError, TypeError):
        return default

def add_now_line(fig, current_he):
    he_str = f"HE{current_he:02}"
    fig.add_shape(type="line", x0=he_str, x1=he_str, y0=0, y1=1,
                  yref="paper", line=dict(color="white", width=1.5, dash="dot"))
    fig.add_annotation(x=he_str, y=1.04, yref="paper", text="Now",
                       showarrow=False, font=dict(color="white", size=11))

def add_marker(fig, he, y, text, color, yshift=30):
    fig.add_annotation(
        x=he, y=y, text=text,
        showarrow=True, arrowhead=2, arrowsize=1, arrowcolor=color,
        ax=0, ay=-yshift,
        font=dict(color=color, size=10, family="monospace"),
        bgcolor="rgba(0,0,0,0.7)", bordercolor=color, borderwidth=1, borderpad=3
    )

def style_reserve_xaxis(fig):
    fig.update_layout(
        xaxis=dict(
            title='Hour Ending',
            tickvals=ALL_HOURS,
            tickangle=0,
            range=[-0.5, 23.5],
            categoryorder='array',
            categoryarray=ALL_HOURS
        )
    )

CPT = ZoneInfo('America/Chicago')

def add_now_line_ts(fig, now_dt):
    fig.add_shape(
        type="line", x0=now_dt, x1=now_dt, y0=0, y1=1,
        yref="paper", line=dict(color="white", width=1.5, dash="dot")
    )
    fig.add_annotation(
        x=now_dt, y=1.04, yref="paper", text="Now",
        showarrow=False, font=dict(color="white", size=11)
    )

def _get_ercot_json(url):
    """Fetch JSON from ERCOT, bypassing Cloudflare."""
    errors = []

    try:
        import cloudscraper
        s = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows", "mobile": False})
        r = s.get(url, timeout=20)
        r.raise_for_status()
        return r.json()
    except ImportError:
        errors.append("cloudscraper not installed")
    except Exception as e:
        errors.append(f"cloudscraper: {e}")

    try:
        from curl_cffi import requests as curl_req
        r = curl_req.get(url, impersonate="chrome", timeout=20,
                         headers={"Accept": "application/json",
                                  "Referer": "https://www.ercot.com/gridmktinfo/dashboards/supplyanddemand"})
        r.raise_for_status()
        return r.json()
    except ImportError:
        errors.append("curl_cffi not installed")
    except Exception as e:
        errors.append(f"curl_cffi: {e}")

    try:
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.ercot.com/gridmktinfo/dashboards/supplyanddemand",
        })
        s.get("https://www.ercot.com/gridmktinfo/dashboards/supplyanddemand", timeout=10)
        r = s.get(url, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        errors.append(f"requests: {e}")

    if errors:
        st.warning(f"6-Day fetch errors: {'; '.join(errors)}")
    return None

@st.cache_data(ttl=3600)
def fetch_6day_supply_demand():
    BASE = "https://www.ercot.com/api/1/services/read/dashboards"
    raw = _get_ercot_json(f"{BASE}/supply-demand.json")
    if raw is None:
        return pd.DataFrame()

    items = raw.get("forecast", raw.get("data", []))
    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)

    if "epoch" not in df.columns:
        return pd.DataFrame()

    df["timestamp"] = (
        pd.to_datetime(df["epoch"], unit="ms")
        .dt.tz_localize("UTC")
        .dt.tz_convert(CPT)
        .dt.tz_localize(None)
    )

    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "availcapgen":
            col_map[c] = "Available Capacity"
        elif cl == "forecasteddemand":
            col_map[c] = "Load Forecast"
    df.rename(columns=col_map, inplace=True)

    for c in ["Available Capacity", "Load Forecast"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Available Capacity" in df.columns and "Load Forecast" in df.columns:
        df["Reserve MW"] = df["Available Capacity"] - df["Load Forecast"]
        mask = df["Load Forecast"] > 0
        df["Reserve %"] = pd.Series(dtype=float)
        df.loc[mask, "Reserve %"] = (df.loc[mask, "Reserve MW"] / df.loc[mask, "Load Forecast"] * 100).round(2)

    df["date"] = df["timestamp"].dt.date
    df["HE"] = df["timestamp"].dt.hour + 1
    df["Hour Ending"] = df["HE"].apply(lambda x: f"HE{x:02}")

    return df.sort_values("timestamp").reset_index(drop=True)

@st.cache_data(ttl=600)
def fetch_reserve_data(cache_time):
    """Fetch capacity forecast, committed capacity, and actual load from gridstatus"""
    gs_ercot = GridStatusErcot()
    data = {}
    try:
        data['fc'] = gs_ercot.get_capacity_forecast(date='latest')
    except:
        data['fc'] = pd.DataFrame()
    try:
        data['cc'] = gs_ercot.get_capacity_committed(date='latest')
    except:
        data['cc'] = pd.DataFrame()
    try:
        data['alz'] = gs_ercot.get_load_by_forecast_zone(date='today')
    except:
        data['alz'] = pd.DataFrame()
    if data['alz'].empty:
        try:
            yesterday = datetime.today().date() - timedelta(days=1)
            data['alz'] = gs_ercot.get_load_by_forecast_zone(date=yesterday)
        except:
            pass
    return data

ERCOT_ONPEAK = (7, 22)
PJM_ONPEAK   = (8, 23)
PJM_WESTERN_HUB_ID = '51217'

@st.cache_data(ttl=3600)
def _balday_ercot_token():
    uid = st.secrets["ercot"]["username"]
    pwd = st.secrets["ercot"]["password"]
    SUBSCRIPTION = st.secrets["ercot"]["subscription"]
    AUTH_URL = (
        "https://ercotb2c.b2clogin.com/"
        "ercotb2c.onmicrosoft.com/"
        "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
    )
    data = {
        'username': uid, 'password': pwd,
        'scope': 'openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access',
        'client_id': 'fec253ea-0d06-4272-a5e6-b478baeecd70',
        'response_type': 'id_token', 'grant_type': 'password',
    }
    resp = requests.post(AUTH_URL, data=data, timeout=30)
    if resp.ok:
        return {
            "Authorization": "Bearer " + resp.json().get("access_token"),
            "Ocp-Apim-Subscription-Key": SUBSCRIPTION,
        }
    return None

def _balday_pjm_headers():
    return {'Ocp-Apim-Subscription-Key': st.secrets["pjm"]["subscription_key"]}

def _balday_pjm_api_call(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = requests.get(url, headers=_balday_pjm_headers(), timeout=(30, 60))
            if result.ok:
                return result.json()
            if result.status_code == 400:
                return None
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(3)
    return None

@st.cache_data(ttl=86400)
def _fetch_ercot_da(today_str):
    """Fetch today's ERCOT DA SPP for HB_NORTH."""
    headers = _balday_ercot_token()
    if not headers:
        return pd.DataFrame()
    url = "https://api.ercot.com/api/public-reports/np4-190-cd/dam_stlmnt_pnt_prices"
    params = {
        'deliveryDateFrom': today_str, 'deliveryDateTo': today_str,
        'settlementPoint': 'HB_NORTH', 'DSTFlag': 'false', 'size': 200000,
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code != 200:
            return pd.DataFrame()
        rows = resp.json().get('data', [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["deliveryDate","hourEnding","settlementPoint","settlementPointPrice","DSTFlag"])
        df['HE'] = df['hourEnding'].str[:2].astype(int)
        df['DA Price'] = pd.to_numeric(df['settlementPointPrice'], errors='coerce')
        return df[['HE', 'DA Price']]
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900)
def _fetch_ercot_rt(today_str, cache_buster):
    """Fetch today's ERCOT RT 15-min SPP for HB_NORTH, averaged to hourly."""
    headers = _balday_ercot_token()
    if not headers:
        return pd.DataFrame()
    url = "https://api.ercot.com/api/public-reports/np6-905-cd/spp_node_zone_hub"
    params = {
        'deliveryDateFrom': today_str, 'deliveryDateTo': today_str,
        'settlementPoint': 'HB_NORTH', 'DSTFlag': 'false', 'size': 200000,
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code != 200:
            return pd.DataFrame()
        data_json = resp.json()
        rows = data_json.get('data', [])
        fields = data_json.get('fields', [])
        if not rows or not fields:
            return pd.DataFrame()
        col_names = [f.get('name') or f.get('label') for f in fields]
        df = pd.DataFrame(rows, columns=col_names)
        df['HE'] = pd.to_numeric(df['deliveryHour'], errors='coerce').astype(int)
        df['settlementPointPrice'] = pd.to_numeric(df['settlementPointPrice'], errors='coerce')
        hourly = df.groupby('HE')['settlementPointPrice'].mean().reset_index()
        hourly.columns = ['HE', 'RT Price']
        return hourly
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def _fetch_pjm_da(today_str):
    """Fetch today's PJM DA LMP for WESTERN HUB."""
    from urllib.parse import quote
    start_dt = f"{today_str} 00:00"
    end_dt = f"{today_str} 23:59"
    date_range = f"{start_dt} to {end_dt}"
    params = {
        'datetime_beginning_ept': date_range,
        'pnode_id': PJM_WESTERN_HUB_ID,
        'rowCount': 50000, 'startRow': 1,
    }
    param_str = '&'.join([f"{k}={quote(str(v))}" for k, v in params.items()])
    url = f"https://api.pjm.com/api/v1/da_hrl_lmps?{param_str}"
    data_json = _balday_pjm_api_call(url)
    if not data_json:
        return pd.DataFrame()
    items = data_json.get('items', [])
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    df['datetime'] = pd.to_datetime(df['datetime_beginning_ept'])
    df['HE'] = df['datetime'].dt.hour + 1
    df['DA Price'] = pd.to_numeric(df['total_lmp_da'], errors='coerce')
    return df[['HE', 'DA Price']]

@st.cache_data(ttl=900)
def _fetch_pjm_rt(today_str, cache_buster):
    """Fetch today's PJM RT LMP for WESTERN HUB via YES Energy. Returns (df, debug_list) tuple."""
    from io import StringIO
    debug = []
    YES_AUTH = ('Leeward_YesAPI1', 'LresYsEnergy202%!')
    YES_BASE = 'https://services.yesenergy.com/PS/rest'

    try:
        url = f"{YES_BASE}/timeseries/RTLMP/WESTERN HUB?agglevel=HOUR&startdate={today_str}&enddate={today_str}"
        resp = requests.get(url, auth=YES_AUTH, timeout=30)
        if resp.ok and len(resp.text) > 100:
            df = pd.read_html(StringIO(resp.text))[0]
            if 'error' not in df.columns and len(df.columns) > 1:
                df['HE'] = pd.to_numeric(df['HOURENDING'], errors='coerce').astype(int)
                df['RT Price'] = pd.to_numeric(df['AVGVALUE'], errors='coerce')
                hourly = df.groupby('HE')['RT Price'].mean().reset_index()
                if not hourly.empty:
                    debug.append(f"YES RTLMP: {len(hourly)} HEs through HE{hourly['HE'].max()}")
                    return hourly, debug
        debug.append(f"YES RTLMP: status={resp.status_code}, bytes={len(resp.text)}")
    except Exception as e:
        debug.append(f"YES error: {e}")

    return pd.DataFrame(), debug

# ─── Gist-based persistent DART cache ───────────────────────────────────────

DART_GIST_FILENAME_ERCOT = "ercot_dart_cache.json"
DART_GIST_FILENAME_PJM = "pjm_dart_cache.json"


def _load_dart_cache_from_gist(filename):
    """Load cached DART history from GitHub Gist."""
    try:
        gist_token = st.secrets.get("gist", {}).get("token")
        gist_id = st.secrets.get("gist", {}).get("dart_gist_id") or st.secrets.get("gist", {}).get("id")
        if not gist_token or not gist_id:
            return None, None
        url = f"https://api.github.com/gists/{gist_id}"
        headers = {
            "Authorization": f"token {gist_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.ok:
            gist_data = resp.json()
            file_info = gist_data.get("files", {}).get(filename)
            if file_info and file_info.get("content"):
                cache = json.loads(file_info["content"])
                df = pd.DataFrame(cache["data"])
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                last_date = cache.get("last_date")
                return df, last_date
    except Exception as e:
        pass
    return None, None


def _save_dart_cache_to_gist(df, filename):
    """Upload DART cache to GitHub Gist."""
    try:
        gist_token = st.secrets.get("gist", {}).get("token")
        gist_id = st.secrets.get("gist", {}).get("dart_gist_id") or st.secrets.get("gist", {}).get("id")
        if not gist_token or not gist_id:
            return False
        cutoff = datetime.now() - timedelta(days=400)
        df_save = df[df['date'] >= cutoff].copy()
        cache_payload = {
            "last_date": df_save['date'].max().strftime('%Y-%m-%d'),
            "row_count": len(df_save),
            "updated_at": datetime.now().isoformat(),
            "data": df_save.to_dict(orient='records'),
        }
        for row in cache_payload["data"]:
            if isinstance(row.get('date'), (pd.Timestamp, datetime)):
                row['date'] = row['date'].strftime('%Y-%m-%d')
        url = f"https://api.github.com/gists/{gist_id}"
        headers = {
            "Authorization": f"token {gist_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        payload = {
            "files": {
                filename: {
                    "content": json.dumps(cache_payload, default=str)
                }
            }
        }
        resp = requests.patch(url, headers=headers, json=payload, timeout=60)
        return resp.ok
    except Exception as e:
        return False


# ─── Chunked ERCOT historical fetchers ──────────────────────────────────────

def _hist_ercot_api_headers():
    """Get ERCOT API auth headers for historical pulls."""
    uid = st.secrets["ercot"]["username"]
    pwd = st.secrets["ercot"]["password"]
    SUBSCRIPTION = st.secrets["ercot"]["subscription"]
    AUTH_URL = (
        "https://ercotb2c.b2clogin.com/"
        "ercotb2c.onmicrosoft.com/"
        "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
    )
    data = {
        'username': uid, 'password': pwd,
        'scope': 'openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access',
        'client_id': 'fec253ea-0d06-4272-a5e6-b478baeecd70',
        'response_type': 'id_token', 'grant_type': 'password',
    }
    resp = requests.post(AUTH_URL, data=data, timeout=30)
    if resp.ok:
        return {
            "Authorization": "Bearer " + resp.json().get("access_token"),
            "Ocp-Apim-Subscription-Key": SUBSCRIPTION,
        }
    return None


def _fetch_ercot_da_chunk(headers, start_date, end_date, max_retries=3):
    """Fetch ERCOT DA prices for HB_NORTH in a date range (max ~30 days)."""
    url = "https://api.ercot.com/api/public-reports/np4-190-cd/dam_stlmnt_pnt_prices"
    params = {
        'deliveryDateFrom': start_date,
        'deliveryDateTo': end_date,
        'settlementPoint': 'HB_NORTH',
        'DSTFlag': 'false',
        'size': 50000,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=60)
            if resp.status_code == 200:
                rows = resp.json().get('data', [])
                if rows:
                    df = pd.DataFrame(rows, columns=[
                        "deliveryDate", "hourEnding", "settlementPoint",
                        "settlementPointPrice", "DSTFlag"
                    ])
                    df['date'] = df['deliveryDate']
                    df['HE'] = df['hourEnding'].str[:2].astype(int)
                    df['DA'] = pd.to_numeric(df['settlementPointPrice'], errors='coerce')
                    return df[['date', 'HE', 'DA']].dropna()
            elif resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
        except requests.exceptions.Timeout:
            time.sleep(3 * (attempt + 1))
        except Exception:
            break
    return pd.DataFrame()


def _fetch_ercot_rt_chunk(headers, start_date, end_date, max_retries=3):
    """Fetch ERCOT RT 15-min prices for HB_NORTH, averaged to hourly."""
    url = "https://api.ercot.com/api/public-reports/np6-905-cd/spp_node_zone_hub"
    params = {
        'deliveryDateFrom': start_date,
        'deliveryDateTo': end_date,
        'settlementPoint': 'HB_NORTH',
        'DSTFlag': 'false',
        'size': 200000,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=90)
            if resp.status_code == 200:
                data_json = resp.json()
                rows = data_json.get('data', [])
                fields = data_json.get('fields', [])
                if rows and fields:
                    col_names = [f.get('name') or f.get('label') for f in fields]
                    df = pd.DataFrame(rows, columns=col_names)
                    df['date'] = df['deliveryDate'].astype(str)
                    df['HE'] = pd.to_numeric(df['deliveryHour'], errors='coerce').astype(int)
                    df['RT'] = pd.to_numeric(df['settlementPointPrice'], errors='coerce')
                    return df.groupby(['date', 'HE'])['RT'].mean().reset_index()
            elif resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
        except requests.exceptions.Timeout:
            time.sleep(3 * (attempt + 1))
        except Exception:
            break
    return pd.DataFrame()


def _fetch_ercot_dart_chunked(start_date_str, end_date_str, progress_bar=None):
    """Fetch ERCOT DART in 30-day chunks between start_date and end_date."""
    headers = _hist_ercot_api_headers()
    if not headers:
        return pd.DataFrame()

    start = datetime.strptime(start_date_str, '%Y-%m-%d')
    end = datetime.strptime(end_date_str, '%Y-%m-%d')

    all_da = []
    all_rt = []
    current = start
    chunk_num = 0
    total_chunks = ((end - start).days // 30) + 1

    while current.date() <= end.date():
        chunk_end = min(current + timedelta(days=29), end)
        cs = current.strftime('%Y-%m-%d')
        ce = chunk_end.strftime('%Y-%m-%d')

        if progress_bar:
            progress_bar.progress(
                min(chunk_num / total_chunks, 1.0),
                text=f"Fetching {cs} to {ce}..."
            )

        da_chunk = _fetch_ercot_da_chunk(headers, cs, ce)
        if not da_chunk.empty:
            all_da.append(da_chunk)

        rt_chunk = _fetch_ercot_rt_chunk(headers, cs, ce)
        if not rt_chunk.empty:
            all_rt.append(rt_chunk)

        time.sleep(0.5)
        current = chunk_end + timedelta(days=1)
        chunk_num += 1

    if not all_da or not all_rt:
        return pd.DataFrame()

    da_df = pd.concat(all_da, ignore_index=True)
    rt_df = pd.concat(all_rt, ignore_index=True)

    merged = da_df.merge(rt_df, on=['date', 'HE'], how='inner')
    merged['DART'] = merged['DA'] - merged['RT']
    merged['date'] = pd.to_datetime(merged['date'])
    merged['doy'] = merged['date'].dt.dayofyear
    return merged[['date', 'doy', 'HE', 'DA', 'RT', 'DART']]


@st.cache_data(ttl=86400)
def _fetch_ercot_historical_dart(cache_date_str):
    """Fetch 365 days of ERCOT DA + RT hourly for HB_NORTH, return per-hour DART.
    Uses Gist cache — only fetches missing days incrementally."""
    today = datetime.now().date()
    target_start = (today - timedelta(days=365)).strftime('%Y-%m-%d')

    cached_df, last_cached_date = _load_dart_cache_from_gist(DART_GIST_FILENAME_ERCOT)

    if cached_df is not None and not cached_df.empty and last_cached_date:
        last_date = datetime.strptime(last_cached_date, '%Y-%m-%d').date()
        days_missing = (today - last_date).days

        if days_missing <= 1:
            return cached_df

        if days_missing <= 60:
            fetch_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            fetch_end = today.strftime('%Y-%m-%d')

            new_data = _fetch_ercot_dart_chunked(fetch_start, fetch_end)

            if not new_data.empty:
                combined = pd.concat([cached_df, new_data], ignore_index=True)
                combined = combined.drop_duplicates(subset=['date', 'HE'], keep='last')
                cutoff = pd.Timestamp(today - timedelta(days=400))
                combined = combined[combined['date'] >= cutoff]
                combined = combined.sort_values(['date', 'HE']).reset_index(drop=True)
                _save_dart_cache_to_gist(combined, DART_GIST_FILENAME_ERCOT)
                return combined
            else:
                return cached_df

    progress = st.progress(0, text="Building ERCOT DART cache (one-time, ~2 min)...")
    full_data = _fetch_ercot_dart_chunked(target_start, today.strftime('%Y-%m-%d'), progress)
    progress.empty()

    if not full_data.empty:
        _save_dart_cache_to_gist(full_data, DART_GIST_FILENAME_ERCOT)

    return full_data


# ─── Chunked PJM historical fetchers ────────────────────────────────────────

def _hist_pjm_api_call(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = requests.get(url, headers=_balday_pjm_headers(), timeout=(30, 90))
            if result.ok:
                return result.json()
            if result.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if result.status_code == 400:
                return None
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
    return None


def _fetch_pjm_dart_chunked(start_date_str, end_date_str, progress_bar=None):
    """Fetch PJM DART in 30-day chunks."""
    from urllib.parse import quote

    start = datetime.strptime(start_date_str, '%Y-%m-%d')
    end = datetime.strptime(end_date_str, '%Y-%m-%d')

    all_da = []
    all_rt = []
    current = start
    chunk_num = 0
    total_chunks = ((end - start).days // 30) + 1

    while current.date() <= end.date():
        chunk_end = min(current + timedelta(days=29), end)
        date_range = f"{current.strftime('%Y-%m-%d')} 00:00 to {chunk_end.strftime('%Y-%m-%d')} 23:59"

        if progress_bar:
            progress_bar.progress(
                min(chunk_num / total_chunks, 1.0),
                text=f"Fetching {current.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}..."
            )

        params = {
            'datetime_beginning_ept': date_range,
            'pnode_id': PJM_WESTERN_HUB_ID,
            'rowCount': 50000, 'startRow': 1,
        }
        param_str = '&'.join([f"{k}={quote(str(v))}" for k, v in params.items()])

        da_json = _hist_pjm_api_call(f"https://api.pjm.com/api/v1/da_hrl_lmps?{param_str}")
        if da_json and da_json.get('items'):
            da_df = pd.DataFrame(da_json['items'])
            da_df['datetime'] = pd.to_datetime(da_df['datetime_beginning_ept'])
            da_df['date'] = da_df['datetime'].dt.strftime('%Y-%m-%d')
            da_df['HE'] = da_df['datetime'].dt.hour + 1
            da_df['DA'] = pd.to_numeric(da_df['total_lmp_da'], errors='coerce')
            all_da.append(da_df[['date', 'HE', 'DA']].dropna())

        time.sleep(0.5)

        rt_json = _hist_pjm_api_call(f"https://api.pjm.com/api/v1/rt_hrl_lmps?{param_str}")
        if rt_json and rt_json.get('items'):
            rt_df = pd.DataFrame(rt_json['items'])
            rt_df['datetime'] = pd.to_datetime(rt_df['datetime_beginning_ept'])
            rt_df['date'] = rt_df['datetime'].dt.strftime('%Y-%m-%d')
            rt_df['HE'] = rt_df['datetime'].dt.hour + 1
            rt_df['RT'] = pd.to_numeric(rt_df['total_lmp_rt'], errors='coerce')
            all_rt.append(rt_df[['date', 'HE', 'RT']].dropna())

        time.sleep(0.5)
        current = chunk_end + timedelta(days=1)
        chunk_num += 1

    if not all_da or not all_rt:
        return pd.DataFrame()

    da_all = pd.concat(all_da, ignore_index=True)
    rt_all = pd.concat(all_rt, ignore_index=True)

    merged = da_all.merge(rt_all, on=['date', 'HE'], how='inner')
    merged['DART'] = merged['DA'] - merged['RT']
    merged['date'] = pd.to_datetime(merged['date'])
    merged['doy'] = merged['date'].dt.dayofyear
    return merged[['date', 'doy', 'HE', 'DA', 'RT', 'DART']]


@st.cache_data(ttl=86400)
def _fetch_pjm_historical_dart(cache_date_str):
    """Fetch 365 days of PJM DA + RT hourly for WESTERN HUB, return per-hour DART.
    Uses Gist cache — only fetches missing days incrementally."""
    today = datetime.now().date()
    target_start = (today - timedelta(days=365)).strftime('%Y-%m-%d')

    cached_df, last_cached_date = _load_dart_cache_from_gist(DART_GIST_FILENAME_PJM)

    if cached_df is not None and not cached_df.empty and last_cached_date:
        last_date = datetime.strptime(last_cached_date, '%Y-%m-%d').date()
        days_missing = (today - last_date).days

        if days_missing <= 1:
            return cached_df

        if days_missing <= 60:
            fetch_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            fetch_end = today.strftime('%Y-%m-%d')

            new_data = _fetch_pjm_dart_chunked(fetch_start, fetch_end)

            if not new_data.empty:
                combined = pd.concat([cached_df, new_data], ignore_index=True)
                combined = combined.drop_duplicates(subset=['date', 'HE'], keep='last')
                cutoff = pd.Timestamp(today - timedelta(days=400))
                combined = combined[combined['date'] >= cutoff]
                combined = combined.sort_values(['date', 'HE']).reset_index(drop=True)
                _save_dart_cache_to_gist(combined, DART_GIST_FILENAME_PJM)
                return combined
            else:
                return cached_df

    progress = st.progress(0, text="Building PJM DART cache (one-time, ~3 min)...")
    full_data = _fetch_pjm_dart_chunked(target_start, today.strftime('%Y-%m-%d'), progress)
    progress.empty()

    if not full_data.empty:
        _save_dart_cache_to_gist(full_data, DART_GIST_FILENAME_PJM)

    return full_data


def _seasonal_weights(hist_doy, today_doy, halflife=30):
    """Gaussian kernel weights — days near today's calendar date get more weight."""
    diff = np.abs(hist_doy - today_doy)
    diff = np.minimum(diff, 365 - diff)
    return np.exp(-0.5 * (diff / halflife) ** 2)


def _regime_weights(hist_df, printed_hes_dart, printed_hes_list):
    """
    Compute per-date regime weights based on how similar each historical day's
    printed-hour DART pattern is to today's.
    """
    if printed_hes_dart.empty or not printed_hes_list:
        return None

    today_avg_dart = printed_hes_dart['DART'].mean()

    hist_printed = hist_df[hist_df['HE'].isin(printed_hes_list)]
    if hist_printed.empty:
        return None

    hist_daily_dart = hist_printed.groupby('date')['DART'].mean().reset_index()
    hist_daily_dart.columns = ['date', 'hist_printed_dart']

    dart_diff = np.abs(hist_daily_dart['hist_printed_dart'].values - today_avg_dart)
    dart_halflife = max(5.0, np.std(hist_daily_dart['hist_printed_dart'].values) * 0.75)
    regime_wt = np.exp(-0.5 * (dart_diff / dart_halflife) ** 2)

    hist_daily_dart['regime_weight'] = regime_wt
    return hist_daily_dart[['date', 'regime_weight']]


def _run_monte_carlo(hist_df, remaining_hes, rt_printed_sum, n_printed,
                     onpk_hours, today_doy, printed_hes_dart, printed_hes_list,
                     today_da_by_he, n_sims=10000):
    """
    Regime-conditional Monte Carlo simulation.
    Samples historical DART spreads (DA - RT) per hour, NOT absolute RT prices.
    Then computes simulated RT = today's DA - sampled DART for each remaining hour.
    """
    if hist_df.empty or not remaining_hes:
        return np.array([])

    date_info = hist_df.groupby('date')['doy'].first().reset_index()
    date_info['seasonal_wt'] = _seasonal_weights(date_info['doy'].values, today_doy, halflife=30)

    regime_df = _regime_weights(hist_df, printed_hes_dart, printed_hes_list)
    if regime_df is not None:
        date_info = date_info.merge(regime_df, on='date', how='left')
        date_info['regime_weight'] = date_info['regime_weight'].fillna(0.1)
    else:
        date_info['regime_weight'] = 1.0

    date_info['combined_wt'] = (
        date_info['seasonal_wt'] * 0.3 +
        date_info['regime_weight'] * 0.7
    )
    date_info['combined_wt'] = date_info['combined_wt'].clip(lower=0.01)

    date_weight_map = dict(zip(date_info['date'], date_info['combined_wt']))

    rng = np.random.default_rng()
    simulated_remaining_rt_sum = np.zeros(n_sims)

    for he in remaining_hes:
        he_data = hist_df[hist_df['HE'] == he].copy()
        if he_data.empty:
            continue

        today_da_he = today_da_by_he.get(he, 0)
        if today_da_he == 0:
            continue

        he_weights = np.array([date_weight_map.get(d, 0.01) for d in he_data['date'].values])
        he_weights = he_weights / he_weights.sum()

        dart_samples = rng.choice(he_data['DART'].values, size=n_sims, p=he_weights)
        simulated_rt_he = today_da_he - dart_samples
        simulated_remaining_rt_sum += simulated_rt_he

    simulated_rt_avg = (rt_printed_sum + simulated_remaining_rt_sum) / onpk_hours
    return simulated_rt_avg


def _render_probability(iso_choice, da_avg, rt_printed_sum, rt_printed_avg,
                        n_printed, n_remaining, onpk_hours, onpk_start, onpk_end,
                        active_he, is_long, hours, now_ct, printed_with_da):
    """Render the regime-conditional Monte Carlo probability section."""

    cache_key = now_ct.strftime('%Y-%m-%d')
    with st.spinner(f"Loading {iso_choice} historical data (365 days)..."):
        if iso_choice == "ERCOT":
            hist = _fetch_ercot_historical_dart(cache_key)
        else:
            hist = _fetch_pjm_historical_dart(cache_key)

    if hist.empty:
        st.warning("Could not load historical data for probability analysis.")
        return

    today_doy = now_ct.timetuple().tm_yday

    remaining_hes = hours[~hours['Printed']]['HE'].tolist()
    printed_hes_list = hours[hours['Printed']]['HE'].tolist()

    if not remaining_hes:
        return

    today_da_by_he = dict(zip(hours['HE'], hours['DA Price']))

    sim_rt_avg = _run_monte_carlo(
        hist, remaining_hes, rt_printed_sum, n_printed,
        onpk_hours, today_doy, printed_with_da, printed_hes_list,
        today_da_by_he, n_sims=10000,
    )

    if len(sim_rt_avg) == 0:
        st.warning("Insufficient historical data for simulation.")
        return

    sim_dart = da_avg - sim_rt_avg
    prob_da_over_rt = (sim_dart > 0).mean() * 100
    prob_rt_over_da = (sim_dart < 0).mean() * 100

    if is_long:
        your_prob = prob_rt_over_da
        your_label = "RT outperforms DA"
    else:
        your_prob = prob_da_over_rt
        your_label = "DA outperforms RT"

    p10 = np.percentile(sim_rt_avg, 10)
    p50 = np.percentile(sim_rt_avg, 50)
    p90 = np.percentile(sim_rt_avg, 90)

    if your_prob >= 65:
        prob_color = "#4CAF50"
    elif your_prob >= 45:
        prob_color = "#FFD54F"
    else:
        prob_color = "#FF5252"

    p1, p2, p3 = st.columns(3)
    p1.markdown(
        f'<div style="text-align:center;padding:10px;">'
        f'<div style="color:#888;font-size:12px;margin-bottom:4px;">Your Position ({("Long" if is_long else "Short")})</div>'
        f'<div style="color:{prob_color};font-size:36px;font-weight:700;">{your_prob:.0f}%</div>'
        f'<div style="color:#aaa;font-size:12px;">{your_label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    p2.markdown(
        f'<div style="text-align:center;padding:10px;">'
        f'<div style="color:#888;font-size:12px;margin-bottom:4px;">Median RT Settle</div>'
        f'<div style="color:#fff;font-size:28px;font-weight:700;">${p50:,.2f}</div>'
        f'<div style="color:#aaa;font-size:12px;">DA: ${da_avg:,.2f}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    p3.markdown(
        f'<div style="text-align:center;padding:10px;">'
        f'<div style="color:#888;font-size:12px;margin-bottom:4px;">80% Range</div>'
        f'<div style="color:#fff;font-size:22px;font-weight:600;">${p10:,.2f} — ${p90:,.2f}</div>'
        f'<div style="color:#aaa;font-size:12px;">10th to 90th percentile</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    fig_dist = go.Figure()

    fig_dist.add_trace(go.Histogram(
        x=sim_rt_avg,
        nbinsx=80,
        marker_color='rgba(66,165,245,0.5)',
        marker_line=dict(color='#42A5F5', width=0.5),
        name='Simulated RT Settle',
    ))

    fig_dist.update_layout(
        title="Monte Carlo — Simulated Final RT OnPk Avg (10,000 scenarios)",
        xaxis=dict(title='Simulated RT OnPk Avg ($/MWh)'),
        yaxis=dict(title='Frequency'),
        height=350,
        showlegend=False,
        bargap=0.02,
    )
    st.plotly_chart(fig_dist, use_container_width=True)


def _render_balday(now_ct, current_he, cache_time):
    """Render the Bal-Day P&L calculator."""
    from zoneinfo import ZoneInfo as ZI

    mins_past_quarter = now_ct.minute % 15
    secs_to_next_quarter = (15 - mins_past_quarter) * 60 - now_ct.second + 30
    if secs_to_next_quarter < 60:
        secs_to_next_quarter += 900
    st.markdown(f'<meta http-equiv="refresh" content="{secs_to_next_quarter}">', unsafe_allow_html=True)

    next_refresh = now_ct + timedelta(seconds=secs_to_next_quarter)
    st.header("Bal-Day Calculator")
    st.caption(f"Next refresh: {next_refresh.strftime('%I:%M %p CT')} · Updated: {now_ct.strftime('%I:%M:%S %p CT')}")

    col_iso, col_pos, _ = st.columns([2, 2, 3])
    with col_iso:
        iso_choice = st.selectbox("Market", ["ERCOT", "PJM"], index=0)
    with col_pos:
        position = st.selectbox("Position", ["Long", "Short"])

    is_long = position == "Long"

    if iso_choice == "ERCOT":
        node_label = "HB_NORTH"
        onpk_start, onpk_end = ERCOT_ONPEAK
        active_he = current_he
    else:
        node_label = "WESTERN HUB"
        onpk_start, onpk_end = PJM_ONPEAK
        now_ept = datetime.now(ZI('America/New_York'))
        active_he = now_ept.hour + 1

    onpk_hours = onpk_end - onpk_start + 1

    if iso_choice == "ERCOT":
        today_str = now_ct.strftime('%Y-%m-%d')
    else:
        today_str = datetime.now(ZI('America/New_York')).strftime('%Y-%m-%d')

    rtcache_buster = now_ct.strftime('%Y-%m-%d %H:') + str(now_ct.minute // 15)

    with st.spinner(f"Fetching {iso_choice} DA & RT for {node_label}..."):
        if iso_choice == "ERCOT":
            da_df = _fetch_ercot_da(today_str)
            rt_df = _fetch_ercot_rt(today_str, rtcache_buster)
            pjm_debug = []
        else:
            da_df = _fetch_pjm_da(today_str)
            rt_result = _fetch_pjm_rt(today_str, rtcache_buster)
            rt_df = rt_result[0]
            pjm_debug = rt_result[1]

    if iso_choice == "PJM":
        if not rt_df.empty:
            latest_he = rt_df['HE'].max()
            if latest_he < active_he - 1:
                st.caption(f"PJM RT through HE{latest_he}")
        if rt_df.empty and pjm_debug:
            with st.expander("PJM RT Debug — NO DATA"):
                st.caption(f"today_str (EPT): {today_str} | active_he: {active_he}")
                for d in pjm_debug:
                    st.text(d)

    if da_df.empty:
        st.warning(f"{iso_choice} DA prices not available yet for today.")
        return

    da_onpk = da_df[(da_df['HE'] >= onpk_start) & (da_df['HE'] <= onpk_end)].copy()
    da_by_he = da_onpk.groupby('HE')['DA Price'].mean().reset_index()
    da_avg = da_by_he['DA Price'].mean()

    rt_by_he = pd.DataFrame()
    if not rt_df.empty:
        rt_onpk = rt_df[(rt_df['HE'] >= onpk_start) & (rt_df['HE'] <= onpk_end)].copy()
        if not rt_onpk.empty:
            rt_by_he = rt_onpk.groupby('HE')['RT Price'].mean().reset_index()

    hours = pd.DataFrame({'HE': range(onpk_start, onpk_end + 1)})
    hours['Hour Ending'] = hours['HE'].apply(lambda x: f'HE{x:02}')
    hours = hours.merge(da_by_he, on='HE', how='left')
    if not rt_by_he.empty:
        hours = hours.merge(rt_by_he, on='HE', how='left')
    else:
        hours['RT Price'] = pd.Series(dtype=float)

    hours['Printed'] = hours['RT Price'].notna() & (hours['HE'] < active_he)
    if not rt_by_he.empty and active_he in rt_by_he['HE'].values:
        hours.loc[hours['HE'] == active_he, 'Printed'] = True

    printed = hours[hours['Printed']].copy()
    remaining = hours[~hours['Printed']].copy()
    n_printed = len(printed)
    n_remaining = len(remaining)

    rt_printed_sum = printed['RT Price'].sum() if n_printed > 0 else 0
    rt_printed_avg = printed['RT Price'].mean() if n_printed > 0 else 0

    if n_remaining > 0:
        breakeven_remaining = (da_avg * onpk_hours - rt_printed_sum) / n_remaining
    else:
        breakeven_remaining = 0

    current_dart = da_avg - rt_printed_avg if n_printed > 0 else 0

    k1, k2, k3 = st.columns(3)
    k1.metric("DA OnPk Avg", f"${da_avg:,.2f}")

    if n_printed > 0:
        if is_long:
            dart_color = "#4CAF50" if current_dart < 0 else "#FF5252" if current_dart > 0 else "#888"
        else:
            dart_color = "#4CAF50" if current_dart > 0 else "#FF5252" if current_dart < 0 else "#888"
        dart_sign = "+" if current_dart >= 0 else ""
        k2.markdown(
            f'<div style="font-size:12px;color:#888;margin-bottom:2px;">RT OnPk Avg</div>'
            f'<div style="font-size:26px;font-weight:700;color:#fafafa;">${rt_printed_avg:,.2f}</div>'
            f'<span style="background:rgba({int(dart_color[1:3],16)},{int(dart_color[3:5],16)},{int(dart_color[5:7],16)},0.2);'
            f'color:{dart_color};font-size:13px;font-weight:600;padding:3px 10px;border-radius:5px;'
            f'border:1px solid {dart_color};">{dart_sign}${current_dart:,.2f} DART</span>',
            unsafe_allow_html=True,
        )
    else:
        k2.metric("RT OnPk Avg", "—")

    k3.metric("Hours Printed / Left", f"{n_printed} / {n_remaining}")

    st.markdown("---")

    if n_printed > 0 and n_remaining > 0:
        rt_leading = rt_printed_avg > da_avg
        if rt_leading:
            leader = "RT outperforming DA"
        else:
            leader = "DA outperforming RT"

        if is_long:
            box_color = "#4CAF50" if rt_leading else "#FF5252"
        else:
            box_color = "#4CAF50" if not rt_leading else "#FF5252"

        r, g, b = int(box_color[1:3],16), int(box_color[3:5],16), int(box_color[5:7],16)
        if is_long:
            dart_goal = "to close with a negative DART"
        else:
            dart_goal = "to close with a positive DART"
        st.markdown(
            f'<div style="background:rgba({r},{g},{b},0.12);'
            f'border:1px solid {box_color};border-radius:8px;padding:14px 22px;margin-bottom:16px;">'
            f'<span style="color:{box_color};font-weight:700;font-size:15px;">{leader}</span><br>'
            f'<span style="color:#ccc;font-size:17px;font-weight:600;margin-top:6px;display:inline-block;">'
            f'Remaining {n_remaining} hours need to avg '
            f'{"above " if is_long else "below "}'
            f'<span style="color:#fff;font-size:20px;font-weight:700;">${breakeven_remaining:,.2f}</span>'
            f' {dart_goal}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif n_remaining == 0 and n_printed > 0:
        final_dart = rt_printed_avg - da_avg
        leader = "RT outperformed DA" if final_dart > 0 else "DA outperformed RT"
        if is_long:
            box_color = "#4CAF50" if final_dart > 0 else "#FF5252"
        else:
            box_color = "#4CAF50" if final_dart < 0 else "#FF5252"
        r, g, b = int(box_color[1:3],16), int(box_color[3:5],16), int(box_color[5:7],16)
        st.markdown(
            f'<div style="background:rgba({r},{g},{b},0.12);'
            f'border:1px solid {box_color};border-radius:8px;padding:12px 20px;margin-bottom:16px;">'
            f'<span style="color:{box_color};font-weight:700;font-size:15px;">ON-PEAK COMPLETE — {leader}</span><br>'
            f'<span style="color:#aaa;font-size:13px;">Final DART: <b style="color:#fff;">${final_dart:+,.2f}</b>/MWh</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=hours['Hour Ending'], y=hours['DA Price'],
        name='DA', marker_color='#AB47BC',
        opacity=0.85, width=0.35, offset=-0.18,
    ))

    if n_printed > 0:
        rt_display = hours['RT Price'].copy()
        rt_display[~hours['Printed']] = None
        fig.add_trace(go.Bar(
            x=hours['Hour Ending'], y=rt_display,
            name='RT', marker_color='#42A5F5',
            opacity=0.85, width=0.35, offset=0.18,
        ))

    fig.add_hline(y=da_avg, line_dash="dash", line_color="#AB47BC", line_width=2,
                  annotation_text=f"DA Avg ${da_avg:,.2f}", annotation_position="top right",
                  annotation_font_color="#AB47BC", annotation_font_size=11)

    if n_printed > 0:
        fig.add_hline(y=rt_printed_avg, line_dash="dot", line_color="#42A5F5", line_width=2,
                      annotation_text=f"RT Avg ${rt_printed_avg:,.2f}", annotation_position="bottom right",
                      annotation_font_color="#42A5F5", annotation_font_size=11)

    if n_remaining > 0 and n_printed > 0:
        fig.add_hline(y=breakeven_remaining, line_dash="dash", line_color="#FFD54F", line_width=1.5,
                      annotation_text=f"Breakeven ${breakeven_remaining:,.2f}", annotation_position="top left",
                      annotation_font_color="#FFD54F", annotation_font_size=11)

    if onpk_start <= active_he <= onpk_end:
        add_now_line(fig, active_he)

    onpk_he_labels = [f'HE{h:02}' for h in range(onpk_start, onpk_end + 1)]
    fig.update_layout(
        title=f"Bal-Day — {iso_choice} {node_label} — {now_ct.date()}",
        yaxis=dict(title='$/MWh'),
        xaxis=dict(title='Hour Ending', tickvals=onpk_he_labels, tickangle=0,
                   categoryorder='array', categoryarray=onpk_he_labels),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        barmode='group', height=500, hovermode='x',
        bargap=0.15, bargroupgap=0.02,
    )
    fig.update_traces(hovertemplate='%{x}: $%{y:,.2f}<extra>%{fullData.name}</extra>')
    st.plotly_chart(fig, use_container_width=True)

    if n_printed > 0 and n_remaining > 0:
        st.markdown("#### RT Settle Probability")

        printed_with_da = printed[['HE', 'DA Price', 'RT Price']].copy()
        printed_with_da['DART'] = printed_with_da['DA Price'] - printed_with_da['RT Price']

        _render_probability(
            iso_choice, da_avg, rt_printed_sum, rt_printed_avg,
            n_printed, n_remaining, onpk_hours, onpk_start, onpk_end,
            active_he, is_long, hours, now_ct, printed_with_da,
        )

    if n_printed > 0 and n_remaining > 0:
        st.markdown("---")
        st.markdown("#### Target Settle Scenario")
        col_input, col_spacer = st.columns([2, 4])
        with col_input:
            target_rt = st.text_input(
                "Target Final RT OnPk Avg ($/MWh)",
                value="",
                placeholder=f"e.g. {da_avg:.2f}",
                key="target_rt_input",
            )

        if target_rt.strip():
            try:
                target_rt_val = float(target_rt.strip().replace('$', '').replace(',', ''))
            except ValueError:
                st.warning("Enter a valid number.")
                target_rt_val = None

            if target_rt_val is not None:
                needed_avg = (target_rt_val * onpk_hours - rt_printed_sum) / n_remaining
                target_dart = da_avg - target_rt_val

                if is_long:
                    pnl_per_mwh = target_rt_val - da_avg
                    outcome = "profit" if pnl_per_mwh > 0 else "loss"
                else:
                    pnl_per_mwh = da_avg - target_rt_val
                    outcome = "profit" if pnl_per_mwh > 0 else "loss"

                pnl_color = "#4CAF50" if outcome == "profit" else "#FF5252"
                rv, gv, bv = int(pnl_color[1:3],16), int(pnl_color[3:5],16), int(pnl_color[5:7],16)

                st.markdown(
                    f'<div style="background:rgba({rv},{gv},{bv},0.10);'
                    f'border:1px solid rgba({rv},{gv},{bv},0.4);border-radius:8px;padding:14px 22px;margin:8px 0 16px 0;">'
                    f'<span style="color:#ccc;font-size:17px;font-weight:600;">'
                    f'Remaining {n_remaining} hours need to avg '
                    f'<span style="color:#fff;font-size:22px;font-weight:700;">${needed_avg:,.2f}</span>'
                    f'</span><br>'
                    f'<span style="color:#aaa;font-size:13px;margin-top:4px;display:inline-block;">'
                    f'Target RT: ${target_rt_val:,.2f} · DA: ${da_avg:,.2f} · '
                    f'DART: {("+" if target_dart >= 0 else "")}${target_dart:,.2f} · '
                    f'<span style="color:{pnl_color};font-weight:700;">'
                    f'{"+" if pnl_per_mwh >= 0 else ""}${pnl_per_mwh:,.2f}/MWh {outcome}</span>'
                    f'</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                fig_proj = go.Figure()
                fig_proj.add_trace(go.Bar(
                    x=hours['Hour Ending'], y=hours['DA Price'],
                    name='DA', marker_color='#AB47BC',
                    opacity=0.85, width=0.35, offset=-0.18,
                ))

                proj_rt = hours[['Hour Ending', 'HE', 'RT Price', 'Printed']].copy()
                proj_rt.loc[~proj_rt['Printed'], 'RT Price'] = needed_avg

                printed_proj = proj_rt[proj_rt['Printed']]
                if not printed_proj.empty:
                    fig_proj.add_trace(go.Bar(
                        x=printed_proj['Hour Ending'], y=printed_proj['RT Price'],
                        name='RT (Actual)', marker_color='#42A5F5',
                        opacity=0.85, width=0.35, offset=0.18,
                    ))

                remaining_proj = proj_rt[~proj_rt['Printed']]
                if not remaining_proj.empty:
                    fig_proj.add_trace(go.Bar(
                        x=remaining_proj['Hour Ending'], y=remaining_proj['RT Price'],
                        name=f'RT (Projected ${needed_avg:,.2f})',
                        marker_color='rgba(66,165,245,0.35)',
                        marker_line=dict(color='#42A5F5', width=1.5),
                        marker_pattern_shape='/',
                        width=0.35, offset=0.18,
                    ))

                fig_proj.add_hline(y=da_avg, line_dash="dash", line_color="#AB47BC", line_width=2,
                                   annotation_text=f"DA Avg ${da_avg:,.2f}", annotation_position="top right",
                                   annotation_font_color="#AB47BC", annotation_font_size=11)
                fig_proj.add_hline(y=target_rt_val, line_dash="dot",
                                   line_color="#4CAF50" if outcome == "profit" else "#FF5252", line_width=2,
                                   annotation_text=f"Target RT ${target_rt_val:,.2f}",
                                   annotation_position="bottom right",
                                   annotation_font_color="#4CAF50" if outcome == "profit" else "#FF5252",
                                   annotation_font_size=11)
                fig_proj.add_hline(y=needed_avg, line_dash="dash", line_color="#FFD54F", line_width=1.5,
                                   annotation_text=f"Needed Avg ${needed_avg:,.2f}", annotation_position="top left",
                                   annotation_font_color="#FFD54F", annotation_font_size=11)

                if onpk_start <= active_he <= onpk_end:
                    add_now_line(fig_proj, active_he)

                onpk_he_labels = [f'HE{h:02}' for h in range(onpk_start, onpk_end + 1)]
                fig_proj.update_layout(
                    title=f"Projected Scenario — RT settles at ${target_rt_val:,.2f}",
                    yaxis=dict(title='$/MWh'),
                    xaxis=dict(title='Hour Ending', tickvals=onpk_he_labels, tickangle=0,
                               categoryorder='array', categoryarray=onpk_he_labels),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    barmode='group', height=450, hovermode='x',
                    bargap=0.15, bargroupgap=0.02,
                )
                fig_proj.update_traces(hovertemplate='%{x}: $%{y:,.2f}<extra>%{fullData.name}</extra>')
                st.plotly_chart(fig_proj, use_container_width=True)


def main():
    check_password()
    _fetch_ercot_historical_dart.clear()
    _fetch_pjm_historical_dart.clear()
    st.title("Fundies")
    try:
        tab1, tab2, tab5, tab6, tab3, tab4 = st.tabs(["ERCOT Weekly", "PJM Weekly", "ERCOT Reserves", "Bal-Day Calc", "Gas", "News"])
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
        
        
        
        if 'ercot_popup_date' in st.session_state and 'ercot_dialog_active' not in st.session_state:
            del st.session_state['ercot_popup_date']
        if 'pjm_popup_date' in st.session_state and 'pjm_dialog_active' not in st.session_state:
            del st.session_state['pjm_popup_date']
        if 'wind_region_popup_date' in st.session_state and 'wind_region_dialog_active' not in st.session_state:
            del st.session_state['wind_region_popup_date']
        
        with tab1:
            from zoneinfo import ZoneInfo
            current_time = datetime.now(ZoneInfo('America/Chicago')).strftime('%H:%M:%S CT')
            st.caption(f"Data last fetched: {current_time} | Next refresh: {next_refresh_time}")
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
                
                # Set up regional wind mapping for popup
                if wind_regional_df is not None and not wind_regional_df.empty:
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
                    available_regions = {k: v for k, v in region_mapping.items() if v in wind_regional_df.columns}
                    st.session_state['wind_region_mapping'] = available_regions
                
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
                            # Add date button for all days - only first 7 open regional popup
                            if st.button(f" {date_obj.strftime('%m/%d')}", key=f"wind_region_{date}", use_container_width=True):
                                if wind_regional_df is not None and not wind_regional_df.empty and idx < 7:
                                    st.session_state['wind_region_popup_date'] = date
                                    st.session_state['wind_region_dialog_active'] = True
                            st.markdown(f"""
                                <div style='text-align: center; padding: 10px 3px; background-color: {peak_color}; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
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
                        
                       
                        region_names = list(available_regions.keys())
                        num_regions = len(region_names)
                        header_cols = "40px " + " ".join(["1fr"] * num_regions)
                        
                        header_html = f"<div style='display: grid; grid-template-columns: {header_cols}; gap: 4px; margin-bottom: 4px; font-size: 10px; font-weight: bold; color: #888;'>"
                        header_html += "<div style='text-align: center;'>HE</div>"
                        for region in region_names:
                            header_html += f"<div style='text-align: center;'>{region}</div>"
                        header_html += "</div>"
                        st.markdown(header_html, unsafe_allow_html=True)
                        
                        
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

                # Popup Dialog for ERCOT date
                if 'ercot_popup_date' in st.session_state:
                    @st.dialog("Load Details", width="large")
                    def show_ercot_dialog():
                        popup_date = st.session_state['ercot_popup_date']
                        date_obj = pd.to_datetime(popup_date)
                        st.markdown(f"#### {date_obj.strftime('%A, %B %d, %Y')}")
                        hours_data = met_load_df[met_load_df['deliveryDate'] == popup_date]
                        wind_date_data = met_wind_df[met_wind_df['deliveryDate'] == popup_date] if met_wind_df is not None else None
                        solar_date_data = met_solar_df[met_solar_df['deliveryDate'] == popup_date] if met_solar_df is not None else None
                        
                        
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
                            
                            st.markdown("""
                                <div style='display: grid; grid-template-columns: 40px 1fr 1fr 1fr 1fr; gap: 4px; margin-bottom: 4px; font-size: 11px; font-weight: bold; color: #888;'>
                                    <div style='text-align: center;'>HE</div>
                                    <div style='text-align: center;'>Load</div>
                                    <div style='text-align: center;'>Wind</div>
                                    <div style='text-align: center;'>Solar</div>
                                    <div style='text-align: center;'>Net</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            
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
                    
                    if 'ercot_dialog_active' in st.session_state:
                        del st.session_state['ercot_dialog_active']

                st.markdown("---")
            else:
                st.warning("No Meteologica load data available")

        # Tab 2 - PJM Weekly
        with tab2:
            from zoneinfo import ZoneInfo
            current_time = datetime.now(ZoneInfo('America/Chicago')).strftime('%H:%M:%S CT')
            st.caption(f"Data last fetched: {current_time} | Next refresh: {next_refresh_time}")
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


                # Zone dropdown 
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

                # Popup  for PJM date
                if 'pjm_popup_date' in st.session_state:
                    @st.dialog("PJM Load Details", width="large")
                    def show_pjm_dialog():
                        popup_date = st.session_state['pjm_popup_date']
                        date_obj = pd.to_datetime(popup_date)
                        st.markdown(f"#### {date_obj.strftime('%A, %B %d, %Y')}")
                        hours_data = pjm_met_load_df[pjm_met_load_df['deliveryDate'] == popup_date]
                        wind_date_data = pjm_met_wind_df[pjm_met_wind_df['deliveryDate'] == popup_date] if pjm_met_wind_df is not None else None
                        solar_date_data = pjm_met_solar_df[pjm_met_solar_df['deliveryDate'] == popup_date] if pjm_met_solar_df is not None else None
                        
                        
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
                            
                            st.markdown("""
                                <div style='display: grid; grid-template-columns: 40px 1fr 1fr 1fr 1fr; gap: 4px; margin-bottom: 4px; font-size: 11px; font-weight: bold; color: #888;'>
                                    <div style='text-align: center;'>HE</div>
                                    <div style='text-align: center;'>Load</div>
                                    <div style='text-align: center;'>Wind</div>
                                    <div style='text-align: center;'>Solar</div>
                                    <div style='text-align: center;'>Net</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            
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
                    
                    current_price = data['Close'].iloc[-1]
                    current_date = data.index[-1].strftime('%Y-%m-%d')

                    
                    st.markdown(f"""
                        <div style='text-align: center; padding: 5px; margin-bottom: 3px;'>
                            <div style='font-size: 18px; color: #888888;'>Current Price (as of {current_date})</div>
                            <div style='font-size: 36px; font-weight: bold; color: #4CAF50;'>${current_price:.3f}</div>
                            <div style='font-size: 14px; color: #888888;'>USD per MMBtu</div>
                        </div>
                    """, unsafe_allow_html=True)

                    
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

                    # e main price axis 
                    axes[0].yaxis.set_label_position('right')
                    axes[0].yaxis.tick_right()
                    axes[0].tick_params(axis='both', colors='white', labelsize=10)
                    axes[0].spines['bottom'].set_color('white')
                    axes[0].spines['top'].set_color('white')
                    axes[0].spines['right'].set_color('white')
                    axes[0].spines['left'].set_color('white')
                    axes[0].xaxis.label.set_color('white')
                    axes[0].yaxis.label.set_color('white')

                    
                    axes[0].grid(False)

                    # volume axis 
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

        # Tab 4 - News
        with tab4:
            st.header("Energy Market News")

            # Fetch news
            with st.spinner("Loading news feeds..."):
                all_articles = fetch_all_news(cache_time)

            # Filter chips - added X Feed option
            cat_options = ["All", "ERCOT", "PJM", "Gas", "Pipeline", "Load", "Regulatory", "X Feed"]
            selected_cat = st.radio(
                "Filter by market:",
                cat_options,
                horizontal=True,
                key="news_filter",
                label_visibility="collapsed",
            )

            if selected_cat == "X Feed":
                filtered_articles = [a for a in all_articles if a.get("is_x_post", False)]
            elif selected_cat != "All":
                filtered_articles = [a for a in all_articles if a["category"] == selected_cat]
            else:
                filtered_articles = all_articles

            if not filtered_articles:
                st.info("No news articles found. Try a different filter or check back later.")
            else:
                # Header row
                hdr1, hdr2, hdr3, hdr4 = st.columns([1, 6, 2, 1])
                hdr1.markdown("**MARKET**")
                hdr2.markdown("**HEADLINE**")
                hdr3.markdown("**SOURCE**")
                hdr4.markdown("**TIME**")
                st.divider()

                for i, article in enumerate(filtered_articles):
                    cat = article["category"]
                    cat_color = NEWS_CATEGORIES.get(cat, {"color": "#888", "bg": "rgba(136,136,136,0.12)"})["color"]
                    cat_bg = NEWS_CATEGORIES.get(cat, {"color": "#888", "bg": "rgba(136,136,136,0.12)"})["bg"]
                    headline = article["headline"]
                    link = article.get("link", "#")
                    source = article.get("source", "")
                    pub_date = article.get("pubDate", "")
                    dt = parse_rss_date(pub_date)
                    rel_time = format_relative_time(dt)
                    is_x = article.get("is_x_post", False)

                    col1, col2, col3, col4 = st.columns([1, 6, 2, 1])
                    with col1:
                        # Show X icon for tweets, category badge for news
                        if is_x:
                            st.markdown(
                                f'<span style="background:{cat_bg};color:{cat_color};'
                                f'font-size:11px;font-weight:700;padding:3px 10px;border-radius:4px;'
                                f'font-family:monospace;">𝕏 {cat}</span>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<span style="background:{cat_bg};color:{cat_color};'
                                f'font-size:11px;font-weight:700;padding:3px 10px;border-radius:4px;'
                                f'font-family:monospace;">{cat}</span>',
                                unsafe_allow_html=True,
                            )
                    with col2:
                        st.markdown(
                            f'<a href="{link}" target="_blank" style="color:#e2e8f0;text-decoration:none;'
                            f'font-size:13.5px;font-weight:500;font-family:monospace;">{headline}</a>',
                            unsafe_allow_html=True,
                        )
                    with col3:
                        st.caption(source)
                    with col4:
                        st.caption(rel_time)

                st.markdown("---")
                x_count = sum(1 for a in filtered_articles if a.get("is_x_post", False))
                news_count = len(filtered_articles) - x_count
                st.caption(f"Showing {len(filtered_articles)} items ({news_count} articles, {x_count} posts) | Refreshes hourly")

        # Tab 5 - ERCOT Reserves
        with tab5:
            from zoneinfo import ZoneInfo
            now_ct = datetime.now(ZoneInfo('America/Chicago'))
            current_he = now_ct.hour + 1
            target_date = now_ct.date()
            is_today = True

            if "reserve_view" not in st.session_state:
                st.session_state.reserve_view = "current"

            c1, c2, _ = st.columns([1, 1, 4])
            with c1:
                if st.button("Current Day", use_container_width=True,
                             type="primary" if st.session_state.reserve_view == "current" else "secondary"):
                    st.session_state.reserve_view = "current"
                    st.rerun()
            with c2:
                if st.button("6-Day Forecast", use_container_width=True,
                             type="primary" if st.session_state.reserve_view == "6day" else "secondary"):
                    st.session_state.reserve_view = "6day"
                    st.rerun()

            st.markdown("")

            if st.session_state.reserve_view == "current":
                st.header("ERCOT Reserve Margin")

                with st.spinner("Loading reserve data..."):
                    reserve_data = fetch_reserve_data(cache_time)

                fc = reserve_data['fc']
                cc = reserve_data['cc']
                alz = reserve_data['alz']

                if not fc.empty:
                    fc = prep_interval(fc)
                if not cc.empty and 'Interval Start' in cc.columns:
                    cc = prep_interval(cc)
                if not alz.empty:
                    alz = prep_interval(alz)
                    alz.columns = [c.capitalize() if c in ['NORTH', 'SOUTH', 'WEST', 'HOUSTON', 'TOTAL'] else c for c in alz.columns]

                ercot_load_for_reserves = None
                if df is not None and not df.empty:
                    today_load = df[df['deliveryDate'] == target_date].copy()
                    if not today_load.empty:
                        ercot_load_for_reserves = today_load.groupby('HE')['systemTotal'].mean().reset_index()
                        ercot_load_for_reserves.columns = ['HN', 'Load Forecast']
                        ercot_load_for_reserves['Hour Ending'] = ercot_load_for_reserves['HN'].apply(lambda x: f'HE{x:02}')

                fc_day = filt_date(fc, target_date) if not fc.empty else pd.DataFrame()

                if fc_day.empty or ercot_load_for_reserves is None or 'Committed Capacity' not in fc_day.columns:
                    st.warning("Reserve data not yet available for today. Try the 6-Day view.")
                else:
                    master = pd.DataFrame({'Hour Ending': ALL_HOURS})
                    master['HN'] = master['Hour Ending'].str.replace('HE', '').astype(int)

                    cap_hr = fc_day.groupby('Hour Ending')['Committed Capacity'].mean().reset_index()
                    master = master.merge(cap_hr, on='Hour Ending', how='left')

                    if 'Available Capacity' in fc_day.columns:
                        avail_hr = fc_day.groupby('Hour Ending')['Available Capacity'].mean().reset_index()
                        master = master.merge(avail_hr, on='Hour Ending', how='left')

                    master = master.merge(ercot_load_for_reserves[['Hour Ending', 'Load Forecast']], on='Hour Ending', how='left')

                    master['Fcst Reserve MW'] = master['Committed Capacity'] - master['Load Forecast']
                    mask_fcst = master['Load Forecast'] > 0
                    master['Fcst Reserve %'] = pd.Series(dtype=float)
                    master.loc[mask_fcst, 'Fcst Reserve %'] = master.loc[mask_fcst, 'Fcst Reserve MW'] / master.loc[mask_fcst, 'Load Forecast'] * 100

                    has_actual_cap = False
                    has_actual_load = False

                    if not cc.empty and 'Capacity' in cc.columns:
                        cc_day = filt_date(cc, target_date)
                        if not cc_day.empty:
                            cc_hr = cc_day.groupby('Hour Ending')['Capacity'].mean().reset_index()
                            cc_hr = cc_hr.rename(columns={'Capacity': 'Committed Actual'})
                            master = master.merge(cc_hr, on='Hour Ending', how='left')
                            has_actual_cap = True

                    if not alz.empty:
                        alz_day = filt_date(alz, target_date)
                        total_col = next((c for c in ['Total', 'TOTAL'] if c in alz_day.columns), None)
                        if not alz_day.empty and total_col:
                            alz_hr = alz_day.groupby('Hour Ending')[total_col].mean().reset_index()
                            alz_hr = alz_hr.rename(columns={total_col: 'Actual Load'})
                            master = master.merge(alz_hr, on='Hour Ending', how='left')
                            has_actual_load = True

                    if 'Committed Actual' in master.columns:
                        master['Best Cap'] = master['Committed Actual'].fillna(master['Committed Capacity'])
                    else:
                        master['Best Cap'] = master['Committed Capacity']

                    if 'Actual Load' in master.columns:
                        master['Best Load'] = master['Actual Load'].fillna(master['Load Forecast'])
                    else:
                        master['Best Load'] = master['Load Forecast']

                    if has_actual_cap or has_actual_load:
                        has_any = pd.Series(False, index=master.index)
                        if has_actual_cap:
                            has_any = has_any | master['Committed Actual'].notna()
                        if has_actual_load:
                            has_any = has_any | master['Actual Load'].notna()
                        master['Actual Reserve MW'] = pd.Series(dtype=float)
                        master['Actual Reserve %'] = pd.Series(dtype=float)
                        valid = has_any & (master['Best Load'] > 0)
                        master.loc[valid, 'Actual Reserve MW'] = master.loc[valid, 'Best Cap'] - master.loc[valid, 'Best Load']
                        master.loc[valid, 'Actual Reserve %'] = master.loc[valid, 'Actual Reserve MW'] / master.loc[valid, 'Best Load'] * 100

                    master = master.sort_values('HN').reset_index(drop=True)

                    fwd = master[master['HN'] >= current_he].dropna(subset=['Fcst Reserve %'])
                    if not fwd.empty:
                        tight_idx = fwd['Fcst Reserve %'].idxmin()
                        tight_he = master.loc[tight_idx, 'Hour Ending']
                        tight_pct = master.loc[tight_idx, 'Fcst Reserve %']
                    else:
                        tight_he, tight_pct = 'N/A', 0

                    fwd_load = master[master['HN'] >= current_he].dropna(subset=['Load Forecast'])
                    if not fwd_load.empty:
                        peak_idx = fwd_load['Load Forecast'].idxmax()
                        res_peak_he = master.loc[peak_idx, 'Hour Ending']
                        res_peak_load = master.loc[peak_idx, 'Load Forecast']
                    else:
                        res_peak_he, res_peak_load = 'N/A', 0

                    if not fwd.empty:
                        min_mw_idx = fwd['Fcst Reserve MW'].idxmin()
                        min_mw_he = master.loc[min_mw_idx, 'Hour Ending']
                        min_mw = master.loc[min_mw_idx, 'Fcst Reserve MW']
                    else:
                        min_mw_he, min_mw = 'N/A', 0

                    k1, k2, k3 = st.columns(3)
                    k1.metric("Tightest Reserve", f"{safe_float(tight_pct):.1f}%  ({tight_he})")
                    k2.metric("Min Reserve MW", f"{safe_int(min_mw):,} MW  ({min_mw_he})")
                    k3.metric("Peak Load", f"{safe_int(res_peak_load):,} MW  ({res_peak_he})")

                    st.markdown("---")

                    fig_cap = go.Figure()

                    if 'Available Capacity' in master.columns:
                        fig_cap.add_trace(go.Scatter(
                            x=master['Hour Ending'], y=master['Available Capacity'],
                            mode='lines', name='Available Capacity',
                            line=dict(color='#FF9800', width=2, dash='dash'), opacity=0.7
                        ))

                    fig_cap.add_trace(go.Scatter(
                        x=master['Hour Ending'], y=master['Committed Capacity'],
                        mode='lines+markers', name='Committed (Forecast)',
                        line=dict(color='#4CAF50', width=2, dash='dash'), marker=dict(size=4)
                    ))

                    if has_actual_cap:
                        act_cap = master.dropna(subset=['Committed Actual'])
                        fig_cap.add_trace(go.Scatter(
                            x=act_cap['Hour Ending'], y=act_cap['Committed Actual'],
                            mode='lines+markers', name='Committed (Actual)',
                            line=dict(color='#00E676', width=2.5), marker=dict(size=5)
                        ))

                    fig_cap.add_trace(go.Scatter(
                        x=master['Hour Ending'], y=master['Load Forecast'],
                        mode='lines+markers', name='Load Forecast (ERCOT)',
                        line=dict(color='#FF5252', width=2, dash='dash'), marker=dict(size=4)
                    ))

                    if has_actual_load:
                        act_load = master.dropna(subset=['Actual Load'])
                        fig_cap.add_trace(go.Scatter(
                            x=act_load['Hour Ending'], y=act_load['Actual Load'],
                            mode='lines+markers', name='Actual Load',
                            line=dict(color='#FF1744', width=2.5), marker=dict(size=5)
                        ))

                    if res_peak_he != 'N/A':
                        add_marker(fig_cap, res_peak_he, res_peak_load,
                                   f"Peak Load<br>{safe_int(res_peak_load):,} MW", '#FF1744', yshift=35)
                    if is_today:
                        add_now_line(fig_cap, current_he)
                    fig_cap.update_traces(hovertemplate='%{x}: %{y:,.0f} MW<extra>%{fullData.name}</extra>')
                    fig_cap.update_layout(
                        title=f"Capacity vs Load — {target_date}",
                        yaxis=dict(title='MW', tickformat=','),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                        height=500, hovermode='x'
                    )
                    style_reserve_xaxis(fig_cap)
                    st.plotly_chart(fig_cap, use_container_width=True)

                    fig_pct = go.Figure()

                    fcst_rm = master.dropna(subset=['Fcst Reserve %'])
                    fig_pct.add_trace(go.Scatter(
                        x=fcst_rm['Hour Ending'], y=fcst_rm['Fcst Reserve %'],
                        mode='lines+markers', name='Reserve % (Forecast)',
                        line=dict(color='#42A5F5', width=2), marker=dict(size=4)
                    ))

                    if 'Actual Reserve %' in master.columns:
                        act_rm = master.dropna(subset=['Actual Reserve %'])
                        if not act_rm.empty:
                            fig_pct.add_trace(go.Scatter(
                                x=act_rm['Hour Ending'], y=act_rm['Actual Reserve %'],
                                mode='lines+markers', name='Reserve % (Actual)',
                                line=dict(color='#29B6F6', width=2.5), marker=dict(size=5)
                            ))

                    fig_pct.add_hline(y=10, line_dash="dash", line_color="red", line_width=1,
                                      annotation_text="10% Area to Watch", annotation_position="top left",
                                      annotation_font_color="red", annotation_font_size=10)

                    if tight_he != 'N/A':
                        add_marker(fig_pct, tight_he, tight_pct,
                                   f"Min {safe_float(tight_pct):.1f}%", '#FF5252', yshift=30)

                    if is_today:
                        add_now_line(fig_pct, current_he)
                    fig_pct.update_traces(hovertemplate='%{x}: %{y:.1f}%<extra>%{fullData.name}</extra>')
                    fig_pct.update_layout(
                        title=f"Reserve Margin % — {target_date}",
                        yaxis=dict(title='Reserve %'),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                        height=450, hovermode='x'
                    )
                    style_reserve_xaxis(fig_pct)
                    st.plotly_chart(fig_pct, use_container_width=True)

                    fig_rmw = go.Figure()

                    fcst_mw = master.dropna(subset=['Fcst Reserve MW'])
                    fig_rmw.add_trace(go.Scatter(
                        x=fcst_mw['Hour Ending'], y=fcst_mw['Fcst Reserve MW'],
                        mode='lines', name='Reserve MW (Forecast)',
                        line=dict(color='#42A5F5', width=2),
                        fill='tozeroy', fillcolor='rgba(66,165,245,0.15)'
                    ))

                    if 'Actual Reserve MW' in master.columns:
                        act_mw = master.dropna(subset=['Actual Reserve MW'])
                        if not act_mw.empty:
                            fig_rmw.add_trace(go.Scatter(
                                x=act_mw['Hour Ending'], y=act_mw['Actual Reserve MW'],
                                mode='lines+markers', name='Reserve MW (Actual)',
                                line=dict(color='#29B6F6', width=2.5), marker=dict(size=5)
                            ))

                    if min_mw_he != 'N/A':
                        add_marker(fig_rmw, min_mw_he, min_mw,
                                   f"Min {safe_int(min_mw):,} MW", '#FF5252', yshift=30)

                    if is_today:
                        add_now_line(fig_rmw, current_he)
                    fig_rmw.update_traces(hovertemplate='%{x}: %{y:,.0f} MW<extra>%{fullData.name}</extra>')
                    fig_rmw.update_layout(
                        title=f"Reserve MW — {target_date}",
                        yaxis=dict(title='MW', tickformat=','),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                        height=400, hovermode='x'
                    )
                    style_reserve_xaxis(fig_rmw)
                    st.plotly_chart(fig_rmw, use_container_width=True)

                    with st.expander("Hourly Data Table"):
                        show_cols = ['Hour Ending']
                        if 'Available Capacity' in master.columns:
                            show_cols.append('Available Capacity')
                        show_cols.append('Committed Capacity')
                        if 'Committed Actual' in master.columns:
                            show_cols.append('Committed Actual')
                        show_cols.append('Load Forecast')
                        if 'Actual Load' in master.columns:
                            show_cols.append('Actual Load')
                        show_cols += ['Fcst Reserve MW', 'Fcst Reserve %']
                        if 'Actual Reserve MW' in master.columns:
                            show_cols.append('Actual Reserve MW')
                        if 'Actual Reserve %' in master.columns:
                            show_cols.append('Actual Reserve %')

                        disp = master[[c for c in show_cols if c in master.columns]].copy()
                        for c in disp.columns:
                            if c == 'Hour Ending':
                                continue
                            elif '%' in c:
                                disp[c] = disp[c].apply(lambda x: f'{float(x):.1f}%' if pd.notna(x) else '')
                            else:
                                disp[c] = disp[c].apply(lambda x: f'{int(float(x)):,}' if pd.notna(x) else '')
                        st.dataframe(disp, use_container_width=True, hide_index=True)

            else:
                st.header("ERCOT 6-Day Supply & Demand Forecast")

                with st.spinner("Fetching 6-day forecast from ERCOT..."):
                    df_6d = fetch_6day_supply_demand()

                if df_6d.empty:
                    st.error("Could not fetch 6-day data. Make sure cloudscraper is installed: pip install cloudscraper")
                else:
                    now_naive = now_ct.replace(tzinfo=None)

                    valid_6d = df_6d.dropna(subset=['Reserve %'])
                    tight_pct_6d, tight_ts_6d = 0, None
                    tight_label_6d = ''
                    min_res_mw_6d, min_res_ts_6d = 0, None
                    min_res_label_6d = ''

                    if not valid_6d.empty:
                        tight_idx_6d = valid_6d['Reserve %'].idxmin()
                        tight_pct_6d = valid_6d.loc[tight_idx_6d, 'Reserve %']
                        tight_ts_6d = valid_6d.loc[tight_idx_6d, 'timestamp']
                        tight_label_6d = tight_ts_6d.strftime('%a %m/%d HE%H') if pd.notna(tight_ts_6d) else ''

                        min_res_mw_idx_6d = valid_6d['Reserve MW'].idxmin()
                        min_res_mw_6d = valid_6d.loc[min_res_mw_idx_6d, 'Reserve MW']
                        min_res_ts_6d = valid_6d.loc[min_res_mw_idx_6d, 'timestamp']
                        min_res_label_6d = min_res_ts_6d.strftime('%a %m/%d HE%H') if pd.notna(min_res_ts_6d) else ''

                        k1, k2 = st.columns(2)
                        k1.metric("Tightest Reserve", f"{safe_float(tight_pct_6d):.1f}%  ({tight_label_6d})")
                        k2.metric("Min Reserve MW", f"{safe_int(min_res_mw_6d):,} MW  ({min_res_label_6d})")

                    st.markdown("---")

                    fig_6d = go.Figure()
                    if 'Available Capacity' in df_6d.columns:
                        fig_6d.add_trace(go.Scatter(x=df_6d['timestamp'], y=df_6d['Available Capacity'],
                            mode='lines', name='Available Capacity', line=dict(color='#AB47BC', width=2), opacity=0.9))
                    fig_6d.add_trace(go.Scatter(x=df_6d['timestamp'], y=df_6d['Load Forecast'],
                        mode='lines', name='Load Forecast', line=dict(color='#42A5F5', width=2),
                        fill='tozeroy', fillcolor='rgba(66,165,245,0.12)'))
                    add_now_line_ts(fig_6d, now_naive)
                    fig_6d.update_traces(hovertemplate='%{x|%a %m/%d HE%H}: %{y:,.0f} MW<extra>%{fullData.name}</extra>')
                    fig_6d.update_layout(
                        title="6-Day — Available Capacity vs Load Forecast",
                        yaxis=dict(title='MW', tickformat=','),
                        xaxis=dict(title='', tickformat='%a %m/%d'),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                        height=500, hovermode='x unified'
                    )
                    st.plotly_chart(fig_6d, use_container_width=True)

        # Tab 6 - Bal-Day Calc
        with tab6:
            from zoneinfo import ZoneInfo
            now_ct_balday = datetime.now(ZoneInfo('America/Chicago'))
            current_he_balday = now_ct_balday.hour + 1
            _render_balday(now_ct_balday, current_he_balday, cache_time)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.exception(e)

if __name__ == "__main__":
    main()

