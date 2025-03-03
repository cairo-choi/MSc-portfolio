import os 
from dotenv import load_dotenv
import pandas as pd
import requests

from datetime import datetime, timedelta, timezone
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

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
db_id = os.getenv("MONGODB_ID")
db_pass = os.getenv("MONGODB_PASSWORD")

# Setting Bitstamp API 
BASE_URL = "https://www.bitstamp.net/api/v2"
CURRENCY_PAIR = 'btcusd'  # setting currency btc with dollars


class OHLCV(object):
    """
   The OHLCV class retrieves Bitcoin's OHLCV (Open, High, Low, Close, Volume) 
   data and current price, returning the data through respective functions and 
   storing it in MongoDB. The JSON-formatted data returned by the function will 
   later be used as input tokens for LLMs.

    Main Features:
    1. fetch_chart_data: Retrieves OHLCV data based on a specified time interval (step) and number of data points (limit), returns the data, and stores it in MongoDB with a human-readable datetime.
    2. current_price: Fetches Bitcoin's current price, returns the data, and stores it in MongoDB with a human-readable datetime.
    3. save_json_to_mongo: Saves JSON-formatted data to MongoDB.

    Usage Example:
    - Retrieve OHLCV data: `fetch_chart_data(step=3600, limit=100)`
    - Get the current price: `current_price()`

    MongoDB:
    - Data is stored in the `btcPredict` database.
    - Collection names:
        - For `fetch_chart_data`: `{step}_ohlcv_collection`
        - For `current_price`: `"currentPrice"`
    """

    def __init__(self, currenccy_pair = CURRENCY_PAIR, base_url = BASE_URL):
        self.currency_pair = currenccy_pair
        self.base_url = base_url

    # 1.Function for saving json data to MongoAtlas
    def save_json_to_mongo(self, collection_name, json_data):
        # You should put your <db_id> and <db_pass> and your Database name such as btcPredict
        uri = f"mongodb+srv://{db_id}:{db_pass}@btcpredict.fhtnwzs.mongodb.net/?retryWrites=true&w=majority&appName=btcPredict"
        client = MongoClient(uri, server_api=ServerApi('1'))
        db = client["btcPredict"]  # Choose database
        collection = db[collection_name]
        if isinstance(json_data, dict):
            collection.insert_one(json_data)
        elif isinstance(json_data, list):
            collection.insert_many(json_data)
        else: 
            print('The size of json_data should not be less than 1')

        print(f"Data saved successfully, on MongoDB btcPredict/{collection_name}")

    # 2.Function for fetching ohlcv data
    def fetch_chart_data(self, step, limit):
        """get OHLC data from  (time interval: step, amounts: limit)"""
        params = {'step': step, 'limit': limit}
        response = requests.get(f'{self.base_url}/ohlc/{self.currency_pair}', params=params)
        data = response.json()
        json_ohlcv = data['data']['ohlc']
        
        mgdb_json_ohlcv=[
            {
                 "datetime": pd.to_datetime(int(entry['timestamp']), unit='s', utc=True).strftime('%Y-%m-%d %H:%M:%S UTC'),
                **entry
            } 
            for entry in json_ohlcv
        ]
        # Sava data to MogoAtlas
        self.save_json_to_mongo(collection_name=f"{str(step)}_ohlcv_collection", json_data=mgdb_json_ohlcv)

        return json_ohlcv

    #3.Get current price of Bitcoin
    # Return current_price_json without human_readble_time for OpenAI
    # Save mgdb_current_price_json with human_readble_time on MongoDB
    def current_price(self):
        response = requests.get(f'{self.base_url}/ticker/{self.currency_pair}')
        data = response.json()
        current_price_json = {
            "timestamp": data['timestamp'],
            "current_price": data['last']
        }
        human_readable_time = pd.to_datetime(int(data['timestamp']), unit='s', utc=True).strftime('%Y-%m-%d %H:%M:%S UTC')
        mgdb_current_price_json = {
            "datetime": human_readable_time,
            **current_price_json
        }
        # Sava data to MogoAtlas
        self.save_json_to_mongo(collection_name="currentPrice", json_data=mgdb_current_price_json)


        return current_price_json



ohlcv = OHLCV()
# print(ohlcv.current_price())
# print(ohlcv.fetch_chart_data(3600, 30))

    



