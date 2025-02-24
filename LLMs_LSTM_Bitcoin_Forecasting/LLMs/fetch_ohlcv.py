import os 
from dotenv import load_dotenv
import pandas as pd
import requests

from datetime import datetime, timedelta, timezone

# load .env
load_dotenv()

# Setting API Key 
bit_api_key = os.getenv("BITSTAMP_API_KEY")
bit_secret = os.getenv("BITSTAMP_SECRET")
bit_user_id = os.getenv("BITSTAMP_USER_ID")
serpapi_key = os.getenv("SERP_API_KEY")
reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
reddit_user_agent = os.getenv("REDDIT_USER_AGENT")

# Setting Bitstamp API 
BASE_URL = "https://www.bitstamp.net/api/v2"
CURRENCY_PAIR = 'btcusd'  # setting currency btc with dollars


#Directory
file_suffix = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
ohlcv_directory = r"D:\DCU\practicum\data\ohlcv"
news_directory = r"D:\DCU\practicum\data\news"
reddit_directory = r"D:\DCU\practicum\data\reddit"
result_directory = r"D:\DCU\practicum\data\result"

class OHLCV(object):
    """
    Here’s a concise summary of the OHLCV class functions:
    1. fetch_chart_data(step, limit, save=False)
        * Retrieves real-time OHLCV data from Bitstamp.
        * step: Interval (e.g., 3600 for 1-hour candles, 86400 for daily candles).
        * limit: Number of data points (max 1000 per request).
        * If save=True, the data is saved (storage path is internally defined).
    2. current_price(save=False)
        *Fetches the current price.
        *If save=True, the price is saved.
    """
    def __init__(self, currenccy_pair = CURRENCY_PAIR, base_url = BASE_URL):
        self.currency_pair = currenccy_pair
        self.base_url = base_url

    # 1.The function for fetching ohlcv data
    def fetch_chart_data(self, step, limit, save = False):
        """get OHLC data from  (time interval: step, amounts: limit)"""
        params = {'step': step, 'limit': limit}
        response = requests.get(f'{self.base_url}/ohlc/{self.currency_pair}', params=params)
        data = response.json()
        json_ohlcv = data['data']['ohlc']
        if save == True:
            self.save_ohlcv_to_csv(json_ohlcv, step, directory=ohlcv_directory)

        return json_ohlcv


    #2. The function for saving data
    def save_ohlcv_to_csv(self, json_data, step, directory=ohlcv_directory):
        """Save the OHLCV data to a CSV file with a timestamp and interval-based prefix"""
        df = pd.DataFrame(json_data)
        df['time'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('time')
        df.drop(columns=['timestamp'], inplace=True)
        # If directory not exists, creating the directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Setting file name based on step(interval)
        if step == 180:
            prefix = "three_minute_ohlcv"
        elif step == 3600:
            prefix = "hourly_ohlcv"
        elif step == 86400:
            prefix = "daily_ohlcv"
        else: prefix = "ohlcv" 

        # Using current time in the file name
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"{directory}/{prefix}_{current_time}.csv"

        # Save as csv file
        df.to_csv(file_name, index=True)
        print(f"Data saved to {file_name}")

    # 3.Get current price of Bitcoin
    def current_price(self, save = False):
        response = requests.get(f'{self.base_url}/ticker/{self.currency_pair}')
        data = response.json()
        current_price_json = {
            "timestamp": data['timestamp'],
            "current_price": data['last']
        }

        if save == True:
        #     self.save_current_price(current_price_json)
            df = pd.DataFrame([current_price_json])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit = 's')
            df.set_index('timestamp', inplace=True)
            file_name = f"{result_directory}\current_price.csv"
            if os.path.exists(file_name):
                df.to_csv(file_name, index=True, mode='a', header=False)
                print(f"Data appended to {file_name}")
            else:
                df.to_csv(file_name, index=True, mode='w', header=True)
                print(f"New file created and data saved to {file_name}")

        return current_price_json


ohlcv = OHLCV()
#if you want to save, then setting the parameter save = True
# print(ohlcv.fetch_chart_data(step=3600, limit=30, save=True))
# print(ohlcv.current_price(save=True))



