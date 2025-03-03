import os 
from dotenv import load_dotenv
import pandas as pd
import requests
import copy
from datetime import datetime, timedelta, timezone
import pytz 
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


class NEWS(object):
    """
    The NEWS class retrieves Bitcoin-related news headlines from Google News using the SerpAPI
    and stores the relevant data in MongoDB.

    Main Features:
    1. fetch_news_titles: Fetches Bitcoin-related news headlines published within the last 'n' hours.
       - Uses SerpAPI to query Google News.
       - Filters news articles based on the provided time range.
       - Returns the news headlines along with their publication timestamps in JSON format.
       - Stores the filtered news in MongoDB.

    2. save_json_to_mongo: Saves JSON-formatted news data to MongoDB.
       - Determines if the input data is a dictionary or a list.
       - Inserts data into the "btcPredict" database under the "newsTitle" collection.

    Usage Example:
    - Fetch news from the last 6 hours: `fetch_news_titles(from_n_hour_ago=6)`

    MongoDB:
    - Data is stored in the `btcPredict` database.
    - The collection name for news headlines is `"newsTitle"`.
    """
      
    def __init__(self, serpapi_key):
        self.serpapi_key = serpapi_key


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

    # fetch latest n hour news from google news 
    def fetch_news_titles(self, from_n_hour_ago):
        """
        Fetches Bitcoin-related news headlines from Google News for the past n hours.
        This method returns the news titles and publication times in JSON format.
        """

        url = "https://serpapi.com/search.json"

        params = {
            'engine': 'google_news',  
            'q': 'bitcoin',  
            'api_key': self.serpapi_key  
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            news_results = data.get("news_results", []) #  'date'format:'02/18/2025, 09:58 PM, +0000 UTC'
            title_results = []

            for item in news_results:
                # Case 1:if stories exists in json structure, get title and date from "stories"
                if "stories" in item:
                    for story in item["stories"]:
                        title = story.get("title", "")
                        date = story.get("date", "")
                        if title and date:  # Only add if both title and date exist
                            title_results.append({"title": title, "date": date})
                # otherwise, fetch title and date from root
                else:
                # Case 2: 'stories' key does not exist
                    title = item.get("title", "")
                    date = item.get("date", "")
                    if title and date:  # Only add if both title and date exist
                        title_results.append({"title": title, "date": date})

            filtered_headline = []
            # current time and n hour ago
            current_time_utc = datetime.now(pytz.UTC)
            # currnt time format: 2025-02-24 13:49:12.906411+00:00
            n_hour_ago_utc = current_time_utc - timedelta(hours=from_n_hour_ago)

            for item in title_results:
                date_str = item.get("date", "")
                #print(date_str)
                try:
                    published_time = datetime.strptime(date_str, "%m/%d/%Y, %I:%M %p, +0000 UTC")
                    # print(published_time)
                    published_time = published_time.replace(tzinfo=pytz.UTC) 
                    # print(published_time)

                    #filtering
                    if n_hour_ago_utc <= published_time <= current_time_utc:
                        filtered_headline.append(item)
                except ValueError as e:
                    print(f"Error parsing date: {date_str}. Error: {e}")
                    continue
            
            if filtered_headline:
                mgdb_headline = copy.deepcopy(filtered_headline)
                self.save_json_to_mongo(collection_name="newsTitle", json_data=mgdb_headline)
                return filtered_headline
                
            else:
                print(f"No news articles published in the last {from_n_hour_ago} hour.")
                return []
    
        except requests.RequestException as e:
            print(f"Error fetching news: {e}")
            return []
            

#news = NEWS(serpapi_key)
#print(news.fetch_news_titles(1.5))


            


