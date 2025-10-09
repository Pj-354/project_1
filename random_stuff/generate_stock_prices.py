import pandas as pd
import numpy as np
import datetime as dt
import sys
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import PercentFormatter
from matplotlib.dates import MonthLocator, DateFormatter
import seaborn as sns
from scipy.stats import probplot

def calc_summary_stats(df : pd.DataFrame
)-> pd.DataFrame:
  """ 
  Calculates in returns space
    1. mean
    2. std
    3. skew
    4. kurt
    5. hit rate
  Price space
    1. highest and lowest
    2. cagr
    3. Max drawdown 
  pnl metrics
    1. sharpe ratio
    
  """
  df                 = df.dropna()
  df.index           = pd.to_datetime(df.index)
  df['Log Rets']     = np.log(df['Rets'] + 1)
  # Calculate statistical moments of log returns
  rets_mean          = df['Rets'].mean() * 100
  rets_std           = df['Rets'].std(ddof=1) * np.sqrt(252) * 100
  rets_skew          = df['Rets'].skew()
  rets_kurt          = df['Rets'].kurt()

  # Calculate Sharpe Ratio
  date_range             = (df.index[-1] - df.index[0]).days * 1/365
  cagr                   = (df['close'].iloc[-1] / df['open'].iloc[0])**(1/date_range) - 1
  sharpe                 = (rets_mean / rets_kurt) * np.sqrt(252)

  # Yearly Returns
  df_year                = (1+df).resample('Y')['Rets'].prod() - 1
  # Ranges
  max_                   = np.max([df['close'].max(), df['open'].max()])
  min_                   = np.min([df['close'].min(), df['open'].min()])
  summary_stats = pd.Series({'Daily Rets Mean %' : rets_mean,
                             'Yearly Return %'   : df_year.mean() * 100,
                             'RVol %' : rets_std,
                             'Skew'   : rets_skew,
                             'Kurt'   : rets_kurt,
                             'CAGR %' : cagr * 100,
                             'Sharpe ': sharpe,
                             'Max Price': max_,
                             'Min Price': min_
                             }).round(3)
  # Sharpe Ratio Rolling
  df['rolling std day']       = df['Rets'].rolling(window=60).std(ddof=1)
  df['rolling average']       = df['Rets'].rolling(window=60).mean() 
  df['rolling sharpe']        = df['rolling average'] / df['rolling std day'] * np.sqrt(252)

  print(summary_stats)
  plot_returns_dist(df)

  return df

def plot_returns_dist(df : pd.DataFrame)-> None:
  """ 
  Plot returns histogram
  """
  fig, axs   = plt.subplots(3,1, figsize = (12,16))
  start_date = df.index[0].strftime('%Y-%m-%d')
  end_date   = df.index[-1].strftime('%Y-%m-%d')
  # Plot returns histogram
  sns.histplot(
      df['Log Rets'],
      bins = math.floor(len(df['Log Rets']) / 10), 
      kde = True,
      ax = axs[0]
    )
  axs[0].set_xlabel('Returns %')
  axs[0].set_ylabel('Frequency')
  axs[0].set_title(f'Returns from:\n {start_date} to {end_date}')
  axs[0].xaxis.set_major_formatter(PercentFormatter(1))
  axs[0].grid()
  # QQ Plot
  probplot(df['Log Rets'], dist='norm', plot=axs[1], fit=True, rvalue=True)
  axs[1].set_title('Actual vs Theoretical Returns Distribution')
  axs[1].yaxis.set_major_formatter(PercentFormatter(1))
  axs[1].grid()

  # Price Action + Sharpe
  axs[2].set_title('Stock Price and a. Sharpe Ratio (60 day rolling)')
  axs3 = axs[2].twinx()

  axs[2].plot(
    df['close'].index,
    df['close'].ffill(),
    label= 'Stock Price',
    color = 'tab:blue',
  )
  axs3.plot(
    df['rolling sharpe'].index,
    df['rolling sharpe'],
    label = 'Sharpe Ratio',
    color = 'tab:orange',
    ls = '--'
  )
  plt.gca().xaxis.set_major_locator(MonthLocator(interval=2))
  plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%b'))
  plt.tight_layout(pad=2)
  plt.show()


if __name__ == "__main__":
  df          = pd.read_csv(r'/Users/phillip/Desktop/Moon2/Datasets/AVGO_5minute_2023-09_minute_level_data.csv', parse_dates=True, index_col =[0])
  df.index    = pd.to_datetime(df.index, utc=True).tz_convert('US/Eastern')
  df          = df[['open']].copy()
  open_mask   = df.index[df.index.time == dt.time(9,30)] 
  close_mask  = df.index[df.index.time == dt.time(16,00)]

  open_prices         = df.loc[open_mask].copy()
  close_prices        = df.loc[close_mask].copy()
  open_prices.index   = open_prices.index.date 
  close_prices.index  = close_prices.index.date 

  nq_rets             = pd.concat([open_prices, close_prices], axis=1)
  nq_rets.columns     = ['open', 'close']
  nq_rets['Rets']     = nq_rets['close'].pct_change()
  nq_rets.dropna(inplace=True)
  
  nq_rets_summary = calc_summary_stats(nq_rets)
 




