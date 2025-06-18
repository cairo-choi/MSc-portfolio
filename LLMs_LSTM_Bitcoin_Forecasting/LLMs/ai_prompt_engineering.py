from fetch_news import NEWS
#from fetch_ohlcv import OHLCV
from fetch_reddit import REDDIT_COMMENTS
from openai import OpenAI
import os
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import json
from datetime import datetime, timedelta, timezone
import pandas as pd


load_dotenv() 

serpapi_key = os.getenv("SERP_API_KEY")
reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
reddit_user_agent = os.getenv("REDDIT_USER_AGENT")
openai_key = os.getenv('OPENAI_API_KEY')
deepseek_key = os.getenv('DEEPSEEK_API_KEY')
db_id = os.getenv("MONGODB_ID")
db_pass = os.getenv("MONGODB_PASSWORD")

class AI_DECISION():
    '''
    This class is designed to make short-term trading decisions for the Bitcoin market.
    It primarily performs sentiment analysis based on Reddit comments and news headlines
    to predict whether the price of Bitcoin will increase or decrease.

    Key Features:
    1. Collects data from Reddit submissions and news headlines.
    2. Assigns sentiment scores to each data source through sentiment analysis.
    3. Combines sentiment scores from Reddit and news to make a final decision.
    4. Uses either DeepSeek or GPT models to generate analysis results.
    5. Saves the analysis results to MongoDB.

    Usage Examples:
    - deepseek_decision(): Performs sentiment analysis and decision-making using the DeepSeek model.
    - gpt_decision(): Performs sentiment analysis and decision-making using the GPT-4 model.
    - save_json_to_mongo(): Saves analysis results to MongoDB.

    Notes:
    - Reddit and news, OpenAI and DeepSeek API keys are required.
    - MongoDB connection details (db_id, db_pass) are required.
    '''
    #ohlcv = OHLCV()
    news = NEWS(serpapi_key)
    reddit = REDDIT_COMMENTS(reddit_client_id, reddit_client_secret, reddit_user_agent)
    #daily_ohlcv = ohlcv.fetch_chart_data(step=86400, limit=30) 
    #hourly_ohlcv = ohlcv.fetch_chart_data(step=3600, limit=24)
    #current_price = ohlcv.current_price()
    news_headline = news.fetch_news_titles(1)
    reddit_comments = reddit.fetch_reddit_comments(10)

    def __init__(self, ai_key):
        self.api_key = ai_key
        # self.ai_type = ai_type

    def save_json_to_mongo(self, collection_name, json_data):
        # You should put your <db_id> and <db_pass> and your Database name such as btcPredict
        uri = f"mongodb+srv://{db_id}:{db_pass}@btcpredict.m1mya.mongodb.net/?retryWrites=true&w=majority&appName=btcPredict"
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

    def deepseek_decision(self):
        client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/v1")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system", 
                    "content": 
                    [
                        {
                            "type": "text",
                            "text": """
                            You are the world's best short-term trading strategist for the Bitcoin market, specializing in sentiment analysis.I am a novice investor seeking to leverage your expertise to determine whether Bitcoin's price will increase or decrease in the next hour based purely on sentiment analysis.

                            Analyze the following data sources:
                            1. Reddit Sentiment - Comments from r/Bitcoin in the last hour.
                            2. News Sentiment - News headlines published about Bitcoin in the last hour.

                            Your task:
                            1. Perform sentiment analysis on both Reddit comments and news headlines.
                            2. Assign a sentiment score from -5(very negative) to +5(very positive) for each source, representing the likelihood that the sentiment indicates a price change.
                            3. Combine both scores to produce an overall sentiment score for the final decision. You should combine both scores based on this "Overall Score=(0.6*Reddit Sentiment Score)+(0.4*News Sentiment Score)".
                            4. Output the result in JSON format with a clear decision and reasoning.
                        
                            Provide your reasoning in JSON format.
                            Example Response:
                            {
                            "decision": "increase",  
                            "reason": "some technical reason", 
                            "sentiment_scores": {"reddit": 4,"news": 3,"overall": 3.6}
                            }
                            {
                            "decision": "decrease",
                            "reason": "Reddit sentiment is highly negative, and recent news articles indicate bearish sentiment.",
                            "sentiment_scores": {"reddit": -2, "news": -1, "overall": -1.6}
                            }
                            """
                        }
                    ]
                },
                {
                    "role": "user", 
                    "content": 
                    [
                        # {"type": "text", "text": f"7-day hourly OHLCV data: {hourly_data.to_json()}\n"},
                        # {"type": "text", "text": f"3-minute OHLCV data: {minute_data.to_json()}\n"},
                        {"type": "text", "text": f"Comments from the past 1 hour on r/Bitcoin Reddit: {self.reddit_comments}\n"},
                        {"type": "text", "text": f"News articles published in the last hour: {self.news_headline}\n"}
                        #{"type": "text", "text": f"Current Bitcoin Price: {current_price}"}              
                    ]
                },
            ],
        response_format={"type": "json_object"},
        stream=False,
        max_completion_tokens=2048,
        temperature=0
    )
        result = json.loads(response.choices[0].message.content)
        print("point1")
        mgdb_result=[
            {   
                "run_time": pd.to_datetime(datetime.now(timezone.utc).timestamp(), unit='s', utc=True).strftime('%Y-%m-%d %H:%M:%S'),
                "ai_type": "DeepSeekR1",
                **result
            } 
        ]
        # Sava data to MogoAtlas
        self.save_json_to_mongo(collection_name="aiDecision", json_data=mgdb_result)
        print(mgdb_result)


    def gpt_decision(self):
        client = OpenAI()
        client.api_key = openai_key
        response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
                {
                    "role": "system", 
                    "content": 
                    [
                        {
                            "type": "text",
                            "text": """
                            You are the world's best short-term trading strategist for the Bitcoin market, specializing in sentiment analysis.I am a novice investor seeking to leverage your expertise to determine whether Bitcoin's price will increase or decrease in the next hour based purely on sentiment analysis.

                            Analyze the following data sources:
                            1. Reddit Sentiment - Comments from r/Bitcoin in the last hour.
                            2. News Sentiment - News headlines published about Bitcoin in the last hour.

                            Your task:
                            1. Perform sentiment analysis on both Reddit comments and news headlines.
                            2. Assign a sentiment score from -5(very negative) to +5(very positive) for each source, representing the likelihood that the sentiment indicates a price change.
                            3. Combine both scores to produce an overall sentiment score for the final decision. You should combine both scores based on this "Overall Score=(0.6*Reddit Sentiment Score)+(0.4*News Sentiment Score)".
                            4. Output the result in JSON format with a clear decision and reasoning.
                        
                            Provide your reasoning in JSON format.
                            Example Response:
                            {
                            "decision": "increase",  
                            "reason": "some technical reason", 
                            "sentiment_scores": {"reddit": 4,"news": 3,"overall": 3.6}
                            }
                            {
                            "decision": "decrease",
                            "reason": "Reddit sentiment is highly negative, and recent news articles indicate bearish sentiment.",
                            "sentiment_scores": {"reddit": -2, "news": -1, "overall": -1.6}
                            }
                            """
                        }
                    ]
                },
                {
                    "role": "user", 
                    "content": 
                    [
                        # {"type": "text", "text": f"7-day hourly OHLCV data: {hourly_data.to_json()}\n"},
                        # {"type": "text", "text": f"3-minute OHLCV data: {minute_data.to_json()}\n"},
                        {"type": "text", "text": f"Comments from the past 1 hour on r/Bitcoin Reddit: {self.reddit_comments}\n"},
                        {"type": "text", "text": f"News articles published in the last hour: {self.news_headline}\n"}
                        #{"type": "text", "text": f"Current Bitcoin Price: {current_price}"}              
                    ]
                },
            ],
        response_format={"type": "json_object"},
        max_completion_tokens=2048,
        temperature=0
    )
        result = response.choices[0].message.content
        mgdb_result=[
            {   
                "run_time": pd.to_datetime(datetime.now(timezone.utc).timestamp(), unit='s', utc=True).strftime('%Y-%m-%d %H:%M:%S'),
                "ai_type": "Gpt4o",
                **json.loads(result)
            } 
        ]
        # Sava data to MogoAtlas
        self.save_json_to_mongo(collection_name="aiDecision", json_data=mgdb_result)
        print(mgdb_result)
    










