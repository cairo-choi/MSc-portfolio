import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timezone
import praw
import re
from psaw import PushshiftAPI

# Load environment variables
load_dotenv('api.env')

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

print(f"REDDIT_CLIENT_ID: {REDDIT_CLIENT_ID}")
print(f"REDDIT_CLIENT_SECRET: {REDDIT_CLIENT_SECRET}")
print(f"REDDIT_USER_AGENT: {REDDIT_USER_AGENT}")

class REDDIT_BITCOIN_SCRAPER(object):
    """
    Class to fetch, clean, and store Reddit post titles (headlines)
    from multiple Bitcoin and crypto-related subreddits with date range filtering.
    """
    def __init__(self, reddit_client_id, reddit_client_secret, reddit_user_agent):
        self.client_id = reddit_client_id
        self.client_secret = reddit_client_secret
        self.user_agent = reddit_user_agent

    def clean_title(self, text): 
        """
        Light NLP cleaning of Reddit post titles:
        - Lowercase
        - Remove usernames, hashtags, URLs
        - Keep alphabets, numbers, apostrophes, periods
        - Remove extra whitespace
        """
        text = text.lower()
        text = re.sub(r"@[A-Za-z0-9_]+", "", text)
        text = re.sub(r"#[A-Za-z0-9_]+", "", text)
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"[^a-z0-9'.]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def save_to_csv(self, data, filename=None):
        """
        Save data to CSV file locally
        """
        if not data:
            print("No data to save.")
            return
            
        df = pd.DataFrame(data)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_bitcoin_posts_{timestamp}.csv"
        
        # Create data folder if it doesn't exist
        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", filename)
        
        df.to_csv(filepath, index=False)
        print(f"Saved {len(data)} headlines to: {filepath}")
        return filepath

    def fetch_reddit_posts(self, start_date=None, end_date=None, filename=None):
        """
        Fetch post titles from multiple subreddits within the specified date range using Pushshift + PRAW.
        """
        reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )
        
        psaw_api = PushshiftAPI(reddit)
        
        subreddits = [
            "BitcoinBeginners", "BitcoinMarkets", "btc", "BitcoinMining",
            "BitcoinCA", "BitcoinUK", "CryptoCurrency", "CryptoMarkets", "CryptoTechnology",
            "cryptotrading", "Crypto_General", "CryptoCurrencyMeta", "DeFi", "Altcoin"
        ]
        
        # Date handling
        if start_date:
            start_dt = pd.to_datetime(start_date).tz_localize("UTC")
            start_epoch = int(start_dt.timestamp())
        else:
            start_epoch = 0
        
        if end_date:
            end_dt = pd.to_datetime(end_date).tz_localize("UTC")
            end_epoch = int(end_dt.timestamp())
        else:
            end_epoch = int(datetime.now(timezone.utc).timestamp())
        
        print(f"Scraping posts from {start_dt.date()} to {end_dt.date()}")
        
        headlines = []
        total_processed = 0
        
        for sub in subreddits:
            print(f"Scraping r/{sub} via Pushshift...")
            
            try:
                submissions = psaw_api.search_submissions(
                    after=start_epoch,
                    before=end_epoch,
                    subreddit=sub,
                    filter=['id'],
                    limit=10000   # you can adjust as needed
                )
                
                sub_count = 0
                
                for submission in submissions:
                    total_processed += 1
                    
                    praw_submission = reddit.submission(id=submission.id)
                    
                    # Filter stickies
                    if praw_submission.stickied:
                        continue
                    
                    cleaned = self.clean_title(praw_submission.title.strip())
                    
                    if cleaned:
                        post_datetime = datetime.fromtimestamp(praw_submission.created_utc, timezone.utc)
                        
                        # get top comments
                        praw_submission.comments.replace_more(limit=0)
                        top_comments = []
                        for comment in praw_submission.comments.list():
                            if len(top_comments) >= 10:
                                break
                            if comment.body and comment.body.strip() not in ["[deleted]", "[removed]"]:
                                top_comments.append(comment.body.strip())
                        top_comments_str = " || ".join(top_comments)
                        
                        # collect
                        headlines.append({
                            "subreddit": sub,
                            "title": cleaned,
                            "original_title": praw_submission.title.strip(),
                            "created_utc": praw_submission.created_utc,
                            "datetime": post_datetime.strftime('%Y-%m-%d %H:%M:%S UTC'),
                            "date": post_datetime.strftime('%Y-%m-%d'),
                            "score": praw_submission.score,
                            "num_comments": praw_submission.num_comments,
                            "url": praw_submission.url,
                            "post_id": praw_submission.id,
                            "top_comments": top_comments_str
                        })
                        sub_count += 1
                        
                print(f"   Found {sub_count} posts in date range for r/{sub}")
                        
            except Exception as e:
                print(f"Error scraping r/{sub}: {str(e)}")
                continue
                
        print(f"\nTotal posts processed: {total_processed}")
        print(f"Posts matching date range: {len(headlines)}")
        
        # Sort
        headlines.sort(key=lambda x: x['created_utc'], reverse=True)
        
        # Save
        if headlines:
            saved_file = self.save_to_csv(headlines, filename)
            return headlines, saved_file
        else:
            print("No posts found in the specified date range.")
            return [], None

    def get_recent_posts(self, days_ago=1, filename=None):
        """
        Convenience method to get posts from the last N days
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - pd.Timedelta(days=days_ago)
        
        return self.fetch_reddit_posts(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            filename=filename
        )


# Example usage
if __name__ == "__main__":
    # Initialize scraper
    scraper = REDDIT_BITCOIN_SCRAPER(REDDIT_CLIENT_ID,
    
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT)
    
    start_date = "2017-12-18"#
    end_date = "2018-12-31"  # Adjust as needed
    # Example 1: Scrape posts from a specific date range
    print("=== Date Range Scraping ===")
    headlines, file_path = scraper.fetch_reddit_posts(
        start_date=start_date,  # Start date
        end_date=end_date,    # End date
        filename=f"bitcoin_posts_jan_{start_date}_{end_date}.csv"
    )
    
    if headlines:
        print(f"\nSample post: {headlines[0]['title']}")
        print(f"From subreddit: r/{headlines[0]['subreddit']}")
        print(f"Date: {headlines[0]['datetime']}")
    
    # Example 2: Get posts from last 7 days
    # print("\n=== Example 2: Recent Posts ===")
    # recent_headlines, recent_file = scraper.get_recent_posts(days_ago=7, filename="recent_bitcoin_posts.csv")
    
    # if recent_headlines:
    #     print(f"Found {len(recent_headlines)} recent posts")
    #     print(f"Saved to: {recent_file}")