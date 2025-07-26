from data_collection import BaseDataFetcher, YahooFinanceFetcher, BlockchainHashRateFetcher
import pandas as pd

bitcoin_start_date = '2017-01-01'
other_timeseries_start_date = '2017-11-09'
end_date = '2025-06-15'
btc_ticker = 'BTC-USD'
eth_ticker = 'ETH-USD'
xrp_ticker = 'XRP-USD'
sp500_ticker = '^GSPC'

btc = YahooFinanceFetcher(btc_ticker, start=bitcoin_start_date, end=end_date)
eth = YahooFinanceFetcher(eth_ticker, start=other_timeseries_start_date, end=end_date)
xrp = YahooFinanceFetcher(xrp_ticker, start=other_timeseries_start_date, end=end_date)
sp500 = YahooFinanceFetcher(sp500_ticker, start=other_timeseries_start_date, end=end_date)
hash = BlockchainHashRateFetcher(start=other_timeseries_start_date, end=end_date)


btc_data = btc.fetch()
eth_data = eth.fetch()
xrp_data = xrp.fetch()
sp500_data = sp500.fetch()
hash_data = hash.fetch()

# Imputing stock market missing data for closing market, using forward fill method
def forward_fill(df):
    # Create full date range 
    full_date_range = pd.date_range(start=other_timeseries_start_date, end=end_date, freq='D')

    # Reindex to include all dates, this will create NaN for missing dates
    df_full_range = df.reindex(full_date_range)

    # Forward fill for NaN value
    df_imputed = df_full_range.ffill()

    return df_imputed

sp500_imputed = forward_fill(sp500_data)
# print(sp500_imputed.head(14))

# df_list for each timeseries 
df_list = [btc_data, eth_data, xrp_data, sp500_imputed, hash_data]

# Merge multiple time series datasets into a single DataFrame
def merge_timeseries(df_list, how='left'):
    """
    Merge multiple time series DataFrames based on index.
    
    :param df_list: List of DataFrames
    :param how: Type of join (default: 'left')
    :return: Merged DataFrame
    """
    if not df_list:
        raise ValueError("The list of DataFrames is empty.")
    
    merged_df = df_list[0]
    for df in df_list[1:]:
        merged_df = merged_df.join(df, how=how)
    
    return merged_df

merged_df = merge_timeseries(df_list=df_list)
print(merged_df.columns)

merged_df.columns = ['btc_close', 'btc_high', 'btc_low', 'btc_open', 'btc_volume', 
                     'eth_close', 'eth_high', 'eth_low', 'eth_open', 'eth_volume',
                     'xrp_close', 'xrp_high', 'xrp_low', 'xrp_open', 'xrp_volume',
                     'sp_close', 'sp_high', 'sp_low', 'sp_open', 'sp_volume',
                     'hash_rate']
merged_df.to_csv("merged_timeseries.csv", index=True)
# print(merged_df.shape)    # 3087, 21
# print(merged_df.isnull().sum())   # 312 NaN for ETH, XRP, SP500, HashRate 



