import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import sys 
from matplotlib.dates import MonthLocator, DateFormatter

COLS = ['Date', 'vwap', 'transactions']

def parse_input(df : pd.DataFrame, keep_cols = COLS):
    """ 
    Parse inputs
        - Handle data types
        - Datetime 
        - Drop unnecessary cols
    """
    df                  = df.copy()
    df                  = df[keep_cols]
    df                  = df.set_index('Date')
    df.index            = pd.to_datetime(df.index, format='mixed', utc=True)
    df                  = df.astype(float)

    # Daily sample
    vwap                  = df.resample('D')['vwap'].last()
    trans                 = df.resample('D')['transactions'].sum()

    merged                = pd.merge(vwap, trans, left_index=True, right_index=True)

    return merged
    


def process_ts(merged):
    """
    Check null values
        - Linear interpolate
    """
    # Check how many is missing
    missing_data          = 100 * merged.isnull().sum() / len(merged)
    print('Missing Data\n', missing_data)
    # Interpolate
    merged['vwap']       = merged['vwap'].dropna()

    return merged

def summary_stats(processed, rets = 'vwap'):
    """ 
    Returns:
        - Mean
        - Skew
        - StD
        - Kurt
        - Median
    """
    # Calc log rets
    if not isinstance(processed.index, pd.DatetimeIndex):
        processed.index = pd.to_datetime(processed.index, format='mixed', utc=True)
    
    rets_df         = processed[rets].pct_change(fill_method=None).dropna()


    rets_summary    = {'Mean' : rets_df.mean(),
                       'StD'  : rets_df.std() * np.sqrt(252),
                       'Skew' : rets_df.skew(),
                       'Kurt' : rets_df.kurt()}
    rets_summary = pd.DataFrame(rets_summary.values(), rets_summary.keys())

    # Calc annualised rate of return
    total_ret      = 100 * (processed[rets][-1] - processed[rets][0]) / processed[rets][0]
    length_of_time = (processed.index[-1] - processed.index[0]).days / 365 
    cagr           = total_ret / length_of_time 
    # Plots

    fig, axs = plt.subplots(2, 1, figsize=(12,10), sharex=False)
    sns.histplot(
        data = rets_df,
        bins = 30,
        kde = True,
        ax = axs[0]
    )
    axs[0].annotate(f'{rets_summary.round(2)}', xy = (0.15, 10))
    axs[0].set_title('Returns Distribution')
    axs[1].set_title(f'Price Action\nC.A. Growth Rate : {cagr:.2f}%')
    axs[0].set_xlabel('Returns')

    # Time Series Plot
    axs[1].plot(
        processed[rets].dropna(),
        label       = 'AAPL stock price',
        lw          = 2.5,
        c           = 'blue'
    )
    
    axs[0].set_xlabel('Date')
    axs[1].set_ylabel('Dollars')
    plt.grid()
    plt.tight_layout()
    plt.show()

    
    return rets_summary








if __name__ == "__main__":
    # aapl_df        = pd.read_csv('Datasets/AAPL_5minute_2023-09_minute_level_data.csv')
    # processed_aapl = process_ts(aapl_df)
    # processed_sum  = summary_stats(processed_aapl)

    data    = []
    headers = input()
    headers = headers.split(',')

    for i in sys.stdin:
        data.append(i.split(','))

    df              = pd.DataFrame(data)
    df.columns      = headers

    ans       = parse_input(df)
    processed = process_ts(ans)
    df        = summary_stats(processed)


