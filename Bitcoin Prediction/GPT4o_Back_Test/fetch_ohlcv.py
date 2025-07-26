import pandas as pd
import os
import yfinance as yf
from datetime import datetime, timedelta

# Path to save the final CSV
save_path = r"C:/Users/comer/OneDrive - Dublin City University/Bitcoin Thesis Files/GPT_Code/data/ohlcv_ma.csv"

def add_ma_columns(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["ma30"] = df["close"].rolling(window=30).mean()
    df["ma60"] = df["close"].rolling(window=60).mean()
    df["ma230"] = df["close"].rolling(window=230).mean()

    # Future MA30 as prediction target
    df["future_ma30"] = df["ma30"].shift(-30)
    df["ma30_change"] = (df["future_ma30"] > df["ma30"]).map({True: "increase", False: "decrease"})

    return df

def main():
    print("Fetching Bitcoin OHLCV data from Yahoo Finance...")
    
    # Get Bitcoin data for the last 10 years
    btc = yf.Ticker("BTC-USD")
    
    # Calculate date range (10 years ago to today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 10)
    
    # Fetch the data
    df = btc.history(start=start_date, end=end_date, interval="1d")
    
    if df.empty:
        print("ERROR: No data received from Yahoo Finance")
        return
    
    print(f"Received {len(df)} records")
    
    # Reset index to make Date a column
    df = df.reset_index()
    
    # Rename columns to match your expected format (optional)
    df = df.rename(columns={
        'Date': 'timestamp',
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    print("Calculating moving averages and targets...")
    df = add_ma_columns(df)

    print(f"Saving to: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print("Done! OHLCV + MA dataset saved.")
    
    # Show some basic info
    print(f"\nDataset info:")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()