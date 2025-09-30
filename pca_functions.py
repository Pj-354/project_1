import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from stock_data_functions import TickerData, calc_log_rets
import stock_data_functions
from glob import glob
import os
import datetime as dt

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



def missing_tickers(collected_tickers, regular_df, sector_map = SECTOR_MAP):
    # Check if all the collected tickers are in the list
    present         = set().union(*(set(v) for v in sector_map.values()))
    missing         = set(collected_tickers).difference(present)
    missing

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
        except Exception as e:
            print(e, i)
            try_again.append(i)
    
    
    return dfs, collected_tickers, try_again


def filter_session(dfs : pd.DataFrame,
                   session = 'Regular'):
    
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
    for sector, names in sector_maps.items():
        names = [n for n in names if n in loadings.index]
        if not names: 
            continue
        rows.append(pd.DataFrame(loadings.loc[names].mean().rename(sector)))
    return pd.concat(rows, axis=1).T.sort_index()