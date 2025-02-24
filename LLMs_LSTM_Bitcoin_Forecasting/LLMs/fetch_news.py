import os 
from dotenv import load_dotenv
import pandas as pd
import requests

from datetime import datetime, timedelta, timezone
import pytz 

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


#Directory
file_suffix = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
ohlcv_directory = r"D:\DCU\practicum\data\ohlcv"
news_directory = r"D:\DCU\practicum\data\news"
reddit_directory = r"D:\DCU\practicum\data\reddit"
result_directory = r"D:\DCU\practicum\data\result"

class NEWS(object):
    """This class is based on the Serp API and retrieves 
    Bitcoin-related news headlines from Google News within 
    the timeframe from n hours ago to the current time.
    """
    def __init__(self, serpapi_key, news_directory):
        self.serpapi_key = serpapi_key
        self.news_directory = news_directory

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
                # if stories exists in json structure, get title and date from "stories"
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
                return filtered_headline
            else:
                print(f"No news articles published in the last {from_n_hour_ago} hour.")
                return []
            
        except requests.RequestException as e:
            print(f"Error fetching news: {e}")
            return []
            


    def save_news_title(self, from_n_hour_ago):
        """
        This method calls the fetch_news_titles method to retrieve the news, 
        then saves it to a CSV file and returns the data in JSON format.

        """
        data = self.fetch_news_titles(from_n_hour_ago)
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        # Using current time in the file name
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"{news_directory}/{current_time}.csv"

        # Save as csv file
        df.to_csv(file_name, index=True)
        print(f"News data saved to {file_name}")


        return data






# news = NEWS(serpapi_key, news_directory)
# news.save_news_title(2)


            


