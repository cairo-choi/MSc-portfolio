import pandas as pd 
import numpy as np
from fracdiff.sklearn import Fracdiff
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def add_ma(df, column_name, window=30, new_column_name=None):
    """
    Add a moving average column to a DataFrame.

    :param df: pandas DataFrame
    :param column_name: Name of the column to calculate moving average
    :param window: Rolling window size (default: 30)
    :param new_column_name: Custom name for the new column (optional)
    :return: DataFrame with new moving average column
    """
    try:
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' does not exist in DataFrame.")

        if new_column_name is None:
            new_column_name = f'ma{window}_{column_name}'

        df[new_column_name] = df[column_name].rolling(window=window).mean()
        return df

    except KeyError as e:
        print(f"Error: {e}")
        return df

def add_pct_change(df, column_name, new_column_name=None):
    """
    Add a percentage change column to a DataFrame.

    :param df: pandas DataFrame
    :param column_name: Name of the column to calculate percentage change
    :param new_column_name: Custom name for the new column (optional)
    :return: DataFrame with new percentage change column
    """
    try:
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' does not exist in DataFrame.")

        if new_column_name is None:
            new_column_name = f'{column_name}_pct_change'

        df[new_column_name] = df[column_name].pct_change() * 100
        return df

    except KeyError as e:
        print(f"Error: {e}")
        return df
    
def add_log_return(df, column_name, new_column_name=None):
    """
    Calculate daily log returns for a given column and add it to the DataFrame.

    :param df: pandas DataFrame
    :param column_name: Name of the price column (e.g., 'btc_close')
    :param new_column_name: Custom name for the new column (optional)
    :return: DataFrame with new log return column
    """
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame.")

    if new_column_name is None:
        new_column_name = f'{column_name}_log_return'

    df[new_column_name] = np.log(df[column_name] / df[column_name].shift(1))

    return df



def frac_diff(df, column_name, d=0.48, window=130, mode='valid', return_adf=True, return_corr=True, new_column_name=None):
    """
    Apply fractional differencing to a column in a DataFrame and optionally run ADF test and correlation.

    :param df: pandas DataFrame
    :param column_name: Name of the column to apply fractional differencing
    :param d: Fractional differencing order (default: 0.48)
    :param window: Window size for rolling estimation
    :param mode: 'valid' or 'full' (default: 'valid')
    :param return_adf: If True, return p-value from ADF test
    :param return_corr: If True, return correlation between original and fracdiff series
    :param new_column_name: Name for the new column (optional)
    :return: (DataFrame with new column, adf_p_value (if requested), corr (if requested))
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    
    if new_column_name is None:
        new_column_name = f"fracdiff_{column_name}"

    # Convert to numpy for processing
    series = df[column_name].to_numpy().reshape(-1, 1)
    f = Fracdiff(d, mode=mode, window=window)
    frac_diff_series = f.fit_transform(series).flatten()

    # Align with original DataFrame
    df[new_column_name] = np.nan
    df.loc[len(df) - len(frac_diff_series):, new_column_name] = frac_diff_series

    adf_p_value = None
    if return_adf:
        adf_p_value = adfuller(frac_diff_series)[1]

    corr_value = None
    if return_corr:
        original_series = series[-len(frac_diff_series):].flatten()
        corr_value = np.corrcoef(original_series, frac_diff_series)[0, 1]

    # Return tuple dynamically
    results = [df]
    if return_adf:
        results.append(adf_p_value)
    if return_corr:
        results.append(corr_value)

    return tuple(results)

def visualization(df, column1, column2):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[column1], label=column1, alpha=0.5)
    plt.plot(df.index, df[column2], label=column2, color='red')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'{column1} VS {column2}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('merged_timeseries.csv')
    df = add_ma(df, column_name='btc_close', window=30)  # ma30_btc_close
    df = add_pct_change(df, column_name='btc_close') # btc_close_pct_change
    df = add_ma(df, column_name='btc_close_pct_change') # ma30_btc_close_pct_change
    df = add_log_return(df, column_name='btc_close')
    df = add_ma(df, 'btc_close_log_return')
    df, p_val, corr = frac_diff(df, column_name='btc_close', return_adf=True, return_corr=True)
    df = add_ma(df, column_name='fracdiff_btc_close') # ma30_fracdiff_btc_close
    df = add_ma(df, column_name='btc_volume')
    df = add_pct_change(df, column_name='btc_volume')
    df = add_ma(df, column_name='btc_volume_pct_change')
    df = add_ma(df, column_name='hash_rate')
    df = add_pct_change(df, column_name='hash_rate')
    df = add_ma(df, column_name = 'hash_rate_pct_change')
    df = add_ma(df, column_name='eth_close')
    df = add_pct_change(df, column_name='eth_close')
    df = add_ma(df, column_name = 'eth_close_pct_change')
    df = add_ma(df, column_name='xrp_close')
    df = add_pct_change(df, column_name='xrp_close')
    df = add_ma(df, column_name = 'xrp_close_pct_change')
    df = add_ma(df, column_name='sp_close')
    df = add_pct_change(df, column_name='sp_close')
    df = add_ma(df, column_name = 'sp_close_pct_change')
    df['y1_30_lagged_ma30_pct_change'] = df['ma30_btc_close_pct_change'].shift(-30)
    df['y2_1_lagged_pct_change'] = df['btc_close_pct_change'].shift(-1)
    df = df.dropna()
    print(df.columns)    
    visualization(df, column1='btc_close', column2='ma30_btc_close')
    print(df)     #shape: (2715, 46)  46 including 2 y-target candidates
    df.to_csv("feature_engineered.csv", index=True)
    print(p_val)  # 0.0416
    print(corr)   # 0.78164
