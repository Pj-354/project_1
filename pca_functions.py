import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from stock_data_functions import TickerData, calc_log_rets
import stock_data_functions
from glob import glob
import os
import datetime as dt
import seaborn as sns 
import matplotlib.pyplot as plt
import datetime as dt


palette = sns.color_palette("tab10")

SECTOR_MAP = {
    'Biotech'             : ['GILD', 'ABBV', 'BSX', 'AMGN', 'ISRG', 'MDT', 'ABT', 'TMO', 'LLY', 'DHR', 'SYK',
                             'DHR'],
    'Consumer'            : ['HD', 'LOW', 'PG', 'PEP', 'AMZN', 'MRK', 'JNJ', 'DASH', 'KO', 'APP', 'PM', 'BKNG',
                             'TJX','COST', 'MCD'],
    'Healthcare'          : ['AMGN','MRK', 'UNH'],
    'Business Software'   : ['CSCO', 'NOW', 'ORCL', 'PANW', 'CRM', 'AMZN', 'CRWD', 'ADP', 'ADBE', 'IBM', 'MSFT'],
    'Telecoms'            : ['T', 'VZ', 'APH', 'TMUS', 'INTU'],
    'Mega-Cap Tech'       : ['AAPL', 'UBER', 'META', 'CSCO', 'ORCL', 'NFLX', 'AMZN', 'AVGO', 'PLTR', 'DASH','APP',
                             'GOOGL', 'IBM', 'MSFT', 'ANET'],
    'Financials'          : ['AXP', 'V', 'MA', 'BAC', 'SPGI', 'BLK', 'SCHW', 'BX', 'PGR', 'C', 'CB', 'MS', 'ACN'
                             ,'WFC', 'KKR', 'GS', 'JPM'] ,
    'Industrials'         : ['LIN', 'HON', 'PG', 'GE', 'BA', 'CAT', 'RTX'],
    'Semi'                : ['AMAT', 'TXN', 'QCOM', 'AMD', 'AVGO', 'MU', 'KLAC', 'ADI', 'ANET', 'LRCX'],
    'Power'               : ['ETN', 'HON', 'GE', 'NEE', 'DE'],
    'Oil & Gas'           : ['CVX', 'COP'],
    'Other'               : ['CMCSA', 'DIS', 'GE', 'BA', 'PGR', 'WMT', 'UNP','WELL', 'BKNG', 'RTX']
}

C_Definitions = {
    "PC1": "Market / Broad Risk",
    "PC2": "Defensive vs Risk-On",
    "PC3": "Clean Energy / Transition / Batteries",
    "PC4": "Digital Infrastructure / Backbone",
    "PC5": "Cyclical Value & Industrials vs Growth Tech",
    "PC6": "Quality-Defensive Tech vs Speculative Risk",
    "PC7": "Utilities & Real Assets (Defensive Income)",
    "PC8": "Defensive-Growth (Cyber/Cloud) + Energy/Green vs Cyclicals/Semis/Value",
}


def missing_tickers(collected_tickers, regular_df, sector_map = SECTOR_MAP):
    # Check if all the collected tickers are in the list
    present         = set().union(*(set(v) for v in sector_map.values()))
    missing         = set(collected_tickers).difference(present)

    # Map into df space
    sector_dfs = {}
    for i,v in sector_map.items():
        sector_dfs[i] = regular_df[v]

def get_all_returns_data():

    pattern             = os.path.join('Datasets', "*_5minute_2023-09_minute_level_data.csv")
    csv_files           = glob(pattern)
    collected_tickers   = [os.path.basename(f).split("_")[0] for f in csv_files]
    dfs                 = []
    try_again           = []

    # See what we have
    for i in collected_tickers:
        try:

            df              = stock_data_functions \
                            .get_minute_level_data(i, n =5, period='minute', start_date='2023-09-26')
            x               = df[['vwap']].rename(columns={'vwap':i})
            dfs.append(x)
        # List the actual exception 
        except Exception as e:
            print(e, i)
            try_again.append(i)
    
    
    return dfs, collected_tickers, try_again


def filter_session(dfs     : pd.DataFrame,
                   session : str = 'Regular'
    )-> pd.DataFrame:
    """ 
    Tag time series data based on if it is pre market, regular session or post market
    Can filter depending on what session we want
        - If session = False then returns everything from all sessions
    """
    
    df = pd.concat(dfs, axis=1)

    t = df.index.time  # ndarray of datetime.time

    df['session'] = np.select(
            [
                (t >= dt.time(4, 0))  & (t < dt.time(9, 30)),   # Pre-Market
                (t >= dt.time(9, 30)) & (t < dt.time(16, 0)),   # Regular
                (t >= dt.time(16, 0)) & (t < dt.time(20, 0)),   # Post-Market
            ],
            ['Pre-Market', 'Regular', 'Post-Market'],
            default='closed'
        )
    if session == False:
        return df.iloc[:, :-1].apply(calc_log_rets)
    else:
        regular_df = df[df['session'] == session]
        regular_df = regular_df.iloc[:, :-1].apply(calc_log_rets)

        return regular_df



def pca_analysis(tag : str, dfs, start_date, end_date):
    """
    Perform PCA analysis on either a sector or the entire dataset of returns 
        dfs = returns (not standardised)
        start_date and end_date
    """
    if tag == False:
        df_i = dfs
    else:
        df_i              = dfs[tag] 
    
    df_i = df_i.loc[start_date : end_date].copy()

    print(df_i.index[0], df_i.index[-1])
    # Standardise
    standardised_rets = pd.DataFrame(StandardScaler().fit_transform(df_i),
                                     columns = df_i.columns,
                                     index   = df_i.index)
    pca      = PCA().fit(standardised_rets)
    
    loadings                = pd.DataFrame(pca.components_.T,
                                           index   = standardised_rets.columns,
                                           columns = [f'PC{i+1}' for i in range(len(standardised_rets.columns))])
    explained_var_by_PC     = pd.DataFrame(pca.explained_variance_ratio_,
                                           index = loadings.columns,
                                           columns = ['PC'])
    explained_var_by_PC['cumsum']   = explained_var_by_PC['PC'].cumsum()

    eigenvalues             = pd.DataFrame(pca.explained_variance_,
                                           index = loadings.columns,
                                           columns = ['PC']
                                           )
    return loadings, explained_var_by_PC, eigenvalues, 


def top_stocks_per_pc(loadings, n = 20, top=10):
    """  
    Given a dataframe of principle components and loadings
    """
    
    group       = {}
    for i in range(n):
        loadings_pc_i               = loadings.sort_values(by=f'PC{i+1}', ascending=False)
        group[f'PC{i+1}']           = loadings_pc_i.index[:top] 
        group[f'PC{i+1}']           = loadings_pc_i.iloc[:top, i]

    
    df = pd.concat(group, axis=0).to_frame('Value')
    df.reset_index(inplace=True)
    df.rename(columns={'level_0':'PC', 'level_1':'Ticker'}, inplace=True)
    df.set_index(['PC', 'Ticker'], inplace=True)
    
    return df    

def sector_centroids(loadings, sector_map = SECTOR_MAP):
    rows = []
    for sector, names in sector_map.items():
        names = [n for n in names if n in loadings.index]
        if not names: 
            continue
        rows.append(pd.DataFrame(loadings.loc[names].mean().rename(sector)))
    return pd.concat(rows, axis=1).T.sort_index()


def pc_weights_from_loadings(loadings: pd.DataFrame) -> pd.DataFrame:
    """
    Convert loading matrix (tickers × PCs) into weights that sum to 1 per PC.
    loadings: DataFrame indexed by ticker, columns = PCs
    Returns weights: same shape, each column sums to 1.
    """
    # If you want *signed* weights (for long/short)
    w = loadings.div(loadings.sum(axis=0), axis=1)
    return w

def make_pc_portfolios(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PC portfolio returns given stock returns and weights.
    returns: DataFrame indexed by date, columns = tickers
    weights: DataFrame indexed by ticker, columns = PCs (weights sum = 1 per PC)
    Returns: DataFrame indexed by date, columns = PCs (portfolio returns)
    """
    # Align tickers
    common = returns.columns.intersection(weights.index)
    R = returns[common]
    W = weights.loc[common]
    # For each PC, do R × w
    # returns is (T × N); weights is (N × K) → result is (T × K)
    port = R.dot(W)
    return port

def get_pc_weights_from_loadings(loadings,
                                 pc : int):
    
    x               = loadings[f'PC{pc}'].copy()
    sum_x           = x.sum()

    x               = x / sum_x
    return x


def get_weighted_returns(col,rets_df):

    ticker                  = col.name
    weight                  = col.values

    ticker_rets             = rets_df[ticker].copy().to_frame(ticker)
    ticker_rets['Cum Rets'] = ticker_rets[ticker].cumsum()
    ticker_rets             = np.exp(ticker_rets)

    return ticker_rets['Cum Rets'] * (weight / ticker_rets[ticker].std())


def plot_pca_vs_benchmark_three_timeframes(n : int,
                          daily_rets,
                          hourly_rets,
                          minute_rets,
                          good_dates,
                          end_date,
                          benchmark_series,
                          text,
                          colors=palette,
                          output = False):
    
    minute_loadings, minute_var_ratios, minute_eigenvalues   = pca_analysis(False, minute_rets, start_date=good_dates, end_date=end_date)
    hourly_loadings, hourly_var_ratios, hourly_eigenvalues   = pca_analysis(False, hourly_rets, start_date=good_dates, end_date=end_date)
    daily_loadings, daily_var_ratios, daily_eigenvalues      = pca_analysis(False, daily_rets, start_date=good_dates, end_date=end_date)    
        
    minutely_weights           = get_pc_weights_from_loadings(minute_loadings, n)
    hourly_weights             = get_pc_weights_from_loadings(hourly_loadings, n)
    daily_weights              = get_pc_weights_from_loadings(daily_loadings, n)

    weights_minute_df          = pd.DataFrame(minutely_weights).T
    weights_hour_df            = pd.DataFrame(hourly_weights).T
    weights_daily_df           = pd.DataFrame(daily_weights).T

    weighted_rets_daily        = weights_minute_df.apply(get_weighted_returns, axis=0, args=(daily_rets,)).sum(axis=1).to_frame(f'PC{n} rets')
    weighted_rets_daily        = weighted_rets_daily / weighted_rets_daily.iloc[0, 0]

    weighted_rets_hourly       = weights_hour_df.apply(get_weighted_returns, axis=0, args=(hourly_rets,)).sum(axis=1).to_frame(f'PC{n} rets')
    weighted_rets_hourly       = weighted_rets_hourly / weighted_rets_hourly.iloc[0, 0]

    weighted_rets_minute       = weights_daily_df.apply(get_weighted_returns, axis=0, args=(minute_rets,)).sum(axis=1).to_frame(f'PC{n} rets')
    weighted_rets_minute       = weighted_rets_minute / weighted_rets_minute.iloc[0, 0]

    weighted_rets_test_daily   = weighted_rets_daily[daily_rets.index.date > end_date.date()].copy()
    weighted_rets_train_daily  = weighted_rets_daily[daily_rets.index.date < end_date.date()].copy()

    weighted_rets_test_hourly   = weighted_rets_hourly[hourly_rets.index.date > end_date.date()].copy()
    weighted_rets_train_hourly  = weighted_rets_hourly[hourly_rets.index.date < end_date.date()].copy()

    weighted_rets_test_minute   = weighted_rets_minute[minute_rets.index.date > end_date.date()].copy()
    weighted_rets_train_minute  = weighted_rets_minute[minute_rets.index.date < end_date.date()].copy()

    benchmark_rets              = benchmark_series.apply(calc_log_rets)
    benchmark_rets['Cum Rets']  = benchmark_rets.cumsum()
    benchmark_rets['Benchmark Gains'] = np.exp(benchmark_rets['Cum Rets']) 

    fig, axs = plt.subplots(1, figsize=(12,6))
    axs.plot(benchmark_rets['Benchmark Gains'], label=f'{text}', color=colors[0], lw=1.5)

    axs.plot(weighted_rets_test_daily[f'PC{n} rets'], label=f'PC {n} Daily', color=colors[1], ls ='--')
    axs.plot(weighted_rets_train_daily[f'PC{n} rets'], color=colors[1])

    axs.plot(weighted_rets_test_hourly[f'PC{n} rets'], label=f'PC {n} Hourly', color=colors[2], ls ='--')
    axs.plot(weighted_rets_train_hourly[f'PC{n} rets'], color=colors[2])

    axs.plot(weighted_rets_test_minute[f'PC{n} rets'], label=f'PC {n} Minute', color=colors[5], ls ='--')
    axs.plot(weighted_rets_train_minute[f'PC{n} rets'], color=colors[5])

    axs.legend()
    plt.title(f'PC {n} Test')

    if output:
        print('Rets goes daily, hourly, minute in the tuple returned')
        return {'Minute' : [minute_loadings, minute_var_ratios, minute_eigenvalues],
                'Hour'   : [hourly_loadings, hourly_var_ratios, hourly_eigenvalues],
                'Daily'  : [daily_loadings, daily_var_ratios, daily_eigenvalues],
                'Rets'   : [weighted_rets_test_daily, weighted_rets_train_daily,
                            weighted_rets_train_hourly, weighted_rets_train_hourly,
                            weighted_rets_test_minute, weighted_rets_train_minute]}
    


def plot_pca_vs_benchmark(n_pc : int,
                          tag : str,
                          rets : pd.DataFrame,
                          start_date : dt.datetime,
                          split_date : dt.datetime,
                          benchmark_series : pd.Series,
                          benchmark_label : str,
                          colors=palette,
                          output = False):
    
    loadings, var_ratios, eigen_values   = pca_analysis(tag,
                                                        rets,
                                                        start_date=start_date,
                                                        end_date=split_date)
        
    weights              = get_pc_weights_from_loadings(loadings, n_pc)


    weights__df          = pd.DataFrame(weights).T

    weighted_rets_df     = weights__df.apply(get_weighted_returns, axis=0, args=(rets,)).sum(axis=1).to_frame(f'PC{n_pc} rets')
    weighted_rets_df     = weighted_rets_df / weighted_rets_df.iloc[0, 0]

    weighted_rets_df_test   = weighted_rets_df[weighted_rets_df.index.date > split_date.date()].copy()
    weighted_rets_df_train  = weighted_rets_df[weighted_rets_df.index.date < split_date.date()].copy()

    benchmark_rets              = calc_log_rets(benchmark_series)
    benchmark_rets['Cum Rets']  = benchmark_rets.cumsum()
    benchmark_rets['Benchmark Gains'] = np.exp(benchmark_rets['Cum Rets']) 

    fig, axs = plt.subplots(1, figsize=(12,6))
    axs.plot(benchmark_rets['Benchmark Gains'], label=f'{benchmark_label}', color=colors[0], lw=1.5)

    axs.plot(weighted_rets_df_test[f'PC{n_pc} rets'], label=f'PC {n_pc} Daily', color=colors[1], ls ='--')
    axs.plot(weighted_rets_df_train[f'PC{n_pc} rets'], color=colors[2])

    
    axs.legend()
    plt.title(f'PC {n_pc} Test')

    if output:
        print('Returning PCA Loadings, Var Ratios, Eigen values, ' \
              'Rets goestraining then testing')
        
        to_out =  {'PCA Loadings' : [loadings, var_ratios, eigen_values],
                         'Rets'   : [weighted_rets_df_train, weighted_rets_df_test]}
        return output