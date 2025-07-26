import os 
from dotenv import load_dotenv
import pandas as pd
import numpy as np

from openai import OpenAI
from datetime import datetime, timedelta, timezone
import praw
import re
import time
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
base_url = "https://www.bitstamp.net/api/v2"
CURRENCY_PAIR = 'btcusd'  # setting currency btc with dollars



class REDDIT_COMMENTS(object):
    """
    The class to fetch, process, and store Reddit comments related to Bitcoin.
    This class connects to the Reddit API, retrieves comments from the Bitcoin subreddit, 
    filters relevant content, and saves the cleaned data to a MongoDB database.
    """
    def __init__(self, reddit_client_id, reddit_client_secret, reddit_user_agent):
        self.client_id = reddit_client_id
        self.client_secret = reddit_client_secret
        self.user_agent = reddit_user_agent

    # Save latest Comments to MongoDB 
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
    
 
    def clean_text(self, text): 
        """
        Clean and preprocess the Reddit comment text by:
        - Converting to lowercase
        - Removing unnecessary elements like usernames, hashtags, URLs, and special characters
        - Keeping only alphabets, numbers, apostrophes (for contractions), and periods (for decimal numbers)
        - Replacing consecutive spaces with a single space

        :param text: Raw text from the Reddit comment
        :return: Cleaned and formatted text
        """

    # Convert to lowercase
        text = text.lower()
    
    # Remove unnecessary content
        text = re.sub(r"@[A-Za-z0-9_]+", "", text)  # Remove usernames
        text = re.sub(r"#[A-Za-z0-9_]+", "", text)  # Remove hashtags
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"www\.\S+", "", text)  # Remove URLs starting with www.
        text = re.sub(r"[^a-z0-9'.]", " ", text)  # Only keep alphabet, numeric characters, "'" and "." e.g.can't or 3.5 
        text = re.sub(r"\s+", " ", text).strip() # Convert consecutive spaces into a single space.

        return text

    def fetch_reddit_comments(self, n_minutes_ago):
        # 300 is maximum comments per request using praw
        max_comments = 300
        comment_count = 0
        filtered_data = []
        # Authorize Reddit API 
        reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
            )
        current_time = datetime.now(timezone.utc).timestamp()
        # Start from n minutes ago 
        start_time = current_time - 60 * n_minutes_ago
        subreddit = reddit.subreddit("Bitcoin")

        # Fetch latest comments from reddit
        for comment in subreddit.comments(limit=None):  # limit=None: fetch as many as possible
            # comment created time
            comment_time = comment.created_utc
            if comment_time < start_time:
                break  

            if start_time <= comment_time <= current_time:
                # if comment created time is between (start_time, current_time)
                # fetch related submission
                submission = comment.submission
                 # Checking if the title of submission is btc related contents
                if re.search(r"\b(btc|BTC|bitcoin|Bitcoin|BITCOIN)\b", submission.title, re.IGNORECASE):
                    filtered_data.append({
                        "submission_title": submission.title,
                        "comment_created_utc": comment.created_utc,
                        "comment": self.clean_text(comment.body)
                        })
                    comment_count += 1
                    if comment_count >= max_comments:
                        break  # make sure maximum number of comments per request is 300

        mgdb_json_comments=[
            {
                 "datetime": pd.to_datetime(int(entry['comment_created_utc']), unit='s', utc=True).strftime('%Y-%m-%d %H:%M:%S UTC'),
                **entry
            } 
            for entry in filtered_data
        ]
         # Sava data to MogoAtlas
        self.save_json_to_mongo(collection_name="redditComments", json_data=mgdb_json_comments)
        # print(mgdb_json_comments)
        return filtered_data


rc = REDDIT_COMMENTS(reddit_client_id, reddit_client_secret, reddit_user_agent)
print(rc.fetch_reddit_comments(30))
