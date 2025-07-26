from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf
import requests


# ----- 1. Abstract Base -----
class BaseDataFetcher(ABC):
    @abstractmethod
    def fetch(self):
        pass

# ----- 2. Yahoo Finance Fetcher -----
class YahooFinanceFetcher(BaseDataFetcher):
# Fetches historical data for a given asset name and start, end time point from Yahoo Finance.
# Args:
#     asset_name (str): The ticker symbol of the asset.
#     e.g. BTC ticker: 'BTC-USD',  S&P 500: '^GSPC', ETH ticker: "ETH-USD", XRP ticker: "XRP-USD"
#     Returns:
#     pandas.DataFrame: A DataFrame containing the historical data.
 

    def __init__(self, asset_name, start, end):
        self.asset_name = asset_name
        self.start = start
        self.end = end

    def fetch(self):
        df = yf.download(self.asset_name, start=self.start, end=self.end)
        if isinstance(df.columns, pd.MultiIndex) and 'Ticker' in df.columns.names:
            df = df.droplevel('Ticker', axis=1)
        df.columns = [f"{self.asset_name}_{col}" for col in df.columns]
        return df

# ----- 3. Blockchain HashRate Fetcher -----
class BlockchainHashRateFetcher(BaseDataFetcher):
    def __init__(self, start=None, end=None, sampled=False, timespan='all'):
        self.start = start
        self.end = end
        self.sampled = sampled
        self.timespan = timespan

    def fetch(self):
        url = "https://api.blockchain.info/charts/hash-rate"
        params = {'timespan': self.timespan, 'format': 'json', 'sampled': str(self.sampled).lower()}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['values'])
        df['Date'] = pd.to_datetime(df['x'], unit='s')
        df['HashRate'] = df['y']
        df = df.drop(['x', 'y'], axis=1)
        df.set_index('Date', inplace=True)
        if self.start:
            df = df[df.index >= pd.to_datetime(self.start)]
        if self.end:
            df = df[df.index < pd.to_datetime(self.end)]
        return df


# ----- 4.load data -----
if __name__ == "__main__":
    btc = YahooFinanceFetcher('BTC-USD', start='2017-01-01', end='2025-06-15')
    eth = YahooFinanceFetcher('ETH-USD', start='2017-11-09', end='2025-06-15')
    xrp = YahooFinanceFetcher('XRP-USD', start='2017-11-09', end='2025-06-15')
    sp500 = YahooFinanceFetcher('^GSPC', start='2017-11-09', end='2025-06-15')
    hash = BlockchainHashRateFetcher(start='2017-11-09', end='2025-06-15')

    print(btc.fetch())   # BTC-USD_Close   BTC-USD_High    BTC-USD_Low   BTC-USD_Open  BTC-USD_Volume
    print(eth.fetch())   # ETH-USD_Close  ETH-USD_High  ETH-USD_Low  ETH-USD_Open  ETH-USD_Volume
    print(xrp.fetch())   # XRP-USD_Close  XRP-USD_High  XRP-USD_Low  XRP-USD_Open  XRP-USD_Volume
    print(sp500.fetch()) # ^GSPC_Close   ^GSPC_High    ^GSPC_Low   ^GSPC_Open  ^GSPC_Volume
    print(hash.fetch())  #  HashRate
    print(f"BTC shape:{btc.fetch().shape} \n",         #(3087, 5)
          f"ETH shape:{eth.fetch().shape} \n",         #(2775, 5)
          f"XRP shape:{xrp.fetch().shape} \n",         #(2775, 5)
          f"SP500 shape:{sp500.fetch().shape} \n",     #(1908, 5)
          f"Hash Rate shape:{hash.fetch().shape}")     #(2775, 1)

