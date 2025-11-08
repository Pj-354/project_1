# Read and updated returns from datasets
from os import read
import pandas as pd 
import numpy as np 
import requests
from io import StringIO
import warnings
import datetime as dt
from datetime import datetime
import time 
from .stock_data_functions import TickerData
from pathlib import Path
from glob import glob 

# Mute the specific warnings you showed
warnings.filterwarnings("ignore", category=FutureWarning, module=r"^pandas\.io\.html")
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

filing_date = dt.datetime.today().date() - pd.Timedelta(days=365*2 -1)
sector_tickers = {'Semiconductor'       : ['NVDA', 'AVGO', 'AMAT', 'AMD', 'MU', 'QCOM', 'TXN','INTC', 'ADI', 'MRVL', 'NXPI', 'MPWR', 'MCHP', 'ALAB', 'CRDO', 'STM', 'ASX', 'ON', 'GFS', 'UMC', 'SMTC', 'LRCX', 'KLAC', 'TSM', 'TER', 'ENTG', 'ARM', 'RMBS', 'MTSI'],
                  'Biotech'             : ['GILD', 'ABBV', 'BSX', 'AMGN', 'ISRG', 'MDT', 'ABT', 'TMO', 'LLY', 'DHR', 'SYK','DHR'],
                  'Consumer'            : ['HD', 'LOW', 'PG', 'PEP', 'AMZN', 'MRK', 'JNJ', 'DASH', 'KO', 'APP', 'PM', 'BKNG', 'TJX','COST', 'MCD'],
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
    'Other'               : ['CMCSA', 'DIS', 'GE', 'BA', 'PGR', 'WMT', 'UNP','WELL', 'BKNG', 'RTX']}


def read_sector_tickers(*sectors : str
    ) -> pd.DataFrame:
    """ 
    Read in sector returns
    Valid sectors:
        'Technology'
        'Healthcare'
        'Consumer Disc'
        'Consumer Staples'
        'Financials'
        'Industrials'
        'Energy'
        'Utilities'
        'Real Estate'
        'Materials'
        'Communication Services'
    Read in sector symbols and organises them descending by market cap

    Call the isolate_sector_tickers function to remove sector tickers from a set of symbols
    """
    sector_tickers = {'Technology' : 'technology',
                      'Healthcare' : 'healthcare',
                      'Consumer Disc' : 'consumer-discretionary',
                      'Consumer Staples' : 'consumer-staples', 
                      'Financials' : 'financials',
                      'Industrials' : 'industrials',
                      'Energy' : 'energy',
                      'Utilities'  : 'utilities',
                      'Real Estate' : 'real-estate',
                      'Materials' : 'materials',
                      'Communication Services' : 'communication-services'}
    
    if not sectors:
        sectors = list(sector_tickers.keys())

    frames = {}

    for sector in sectors:

        sector_url = sector_tickers[sector]
        if sector not in sector_tickers.keys():
            raise ValueError(f"Sector '{sector}' not recognized. Valid sectors are: {list(sector_tickers.keys())}")
        
        url            = f'https://stockanalysis.com/stocks/sector/{sector_url}/'
        r              = requests.get(url)
        tables         = pd.read_html(StringIO(r.text))
        symbols        = tables[0][['Symbol', 'Market Cap']]

        symbols['Market Cap'] = symbols['Market Cap'].str.replace('M', '').str.replace('B', '000').str.replace('T','000000').str.replace('.', '')
        

        symbols['Market Cap'] = pd.to_numeric(symbols['Market Cap'], errors='coerce')
        frames[sector]        = symbols.set_index('Symbol').sort_values(by='Market Cap', ascending=False)

    return frames

def get_sub_sectors_names():
    """ 
    Just gets the names of all the sub sectors in an industry
    """
    url    = 'https://stockanalysis.com/stocks/industry/'
    r      = requests.get(url)
    tables = pd.read_html(StringIO(r.text))
    dfs    = [i[['Industry Name', 'Stocks', 'Market Cap']] for i in tables]

    # Handle the market cap stuff
    for i in dfs:

        clean = i['Market Cap'].str.replace(',', '', regex=False).str.upper()
        parts = clean.str.extract(r'(?P<value>\d*\.?\d+)\s*(?P<unit>[MB])')
        
        i['Market Cap'] = (
                pd.to_numeric(parts['value'], errors='coerce') *
                parts['unit'].map({'M': 1.0, 'B': 1000.0})
            )


    return dfs

def isolate_sector_tickers(df_symbols       : pd.DataFrame,
                           sector_to_remove : str,
                           sector_tickers   : dict = sector_tickers,
                           n_components     : int = 50
    )-> pd.DataFrame:
    """ 
    Isolate sector tickers from a set of symbols : removes sector tickers from request call 
    """
    symbols             = set(df_symbols.index) 
    symbols_to_remove   = set(sector_tickers[sector_to_remove])
    
    isolated_tickers    = symbols - symbols_to_remove
    isolated_tickers    = list(isolated_tickers)

    df = df_symbols.loc[isolated_tickers]
    df = df.sort_values(by='Market Cap', ascending=False)

    return df.iloc[:n_components, :]

def read_etf_returns(*categories : str)-> pd.DataFrame:
    """ 
    Returns ETF returns 
        1. Sector
        2. Style
        3. Theme
    """

    df_registry = {
        'SECTOR' : 'Daily/sector_ETF_rets.csv',
        'STYLE'  : 'Daily/style_ETF_rets.csv',
        'THEME'  : 'Daily/theme_ETF_rets.csv'
                }
    
    if not categories:
        categories = tuple(df_registry.keys())
    else:
        categories = [i.upper() for i in categories]
        unknown    = set(categories) - set(df_registry.keys())
        print('Invalid keys : ', unknown)
        categories = set(categories) - set(unknown)
    
    frames = []
    for name in categories:
        print(name)
        file_name = df_registry[name]
        df        = pd.read_csv(file_name, index_col = [0], parse_dates=True)
        frames.append(df)

    return pd.concat(frames, axis=1)

def get_rets_cumsum(rets     : pd.DataFrame,
                    constant : int = 1):
    """ 
    Must be log returns -> turns series of log returns into the price
    Can be normalized so starts at value of 1 -> use constant
    """
    if isinstance(rets.index, pd.core.indexes.datetimes.DatetimeIndex):
        rets_cumsum     = rets.cumsum()
        rets_cumsum     = np.exp(rets_cumsum)
        start_date      = rets_cumsum.index[0]
        rets_cumsum.loc[start_date, :] = constant
        rets_cumsum.sort_index(inplace=True)
        return rets_cumsum
    else:
        print('Must fix index') 

def get_existing_subsector_stock_prices(sector, filing_date, n_stocks_per):

    daily_base_path     = Path('Daily')
    pattern             = f"Daily_{sector}_*_prices.csv"
    file_path           = next(daily_base_path.glob(pattern), None)

    return file_path

def check_columns(col, df_old, n = 20):
    """ 
    Checks column by column
    """
    sub_sector_name   = col.name[0]
    unordered_old     = set(df_old.xs(sub_sector_name, axis=1).iloc[:n, 0])
    unordered_col     = set(col.iloc[:n])
    condition         = (unordered_col == unordered_old)
    if condition == False:
        print(sub_sector_name, 'False')
        changes = {o: n for o, n in zip(set.difference(unordered_old,unordered_col), set.difference(unordered_col,unordered_old))}
        return changes
    else:
        return None

def map_sub_sector_differences(df_old: pd.DataFrame, df_new: pd.DataFrame, n: int):
    """
    Compares for a dataframe of stocks and market caps across two different days
        - Calculates the market cap difference in replacement
        - And as % of total market cap of what we're evaluating to see how important it is
    Can specify n = 20 components etc...
    """
    # Basic sanity checks
    df_old_subsectors = df_old.loc[slice(None), (slice(None), 'Symbol')]
    df_new_subsectors = df_new.loc[slice(None), (slice(None), 'Symbol')]

    # Check if sectors are the same 
    if df_old_subsectors.shape != df_new_subsectors.shape:
        raise ValueError(f"Shape mismatch: {df_old.shape} vs {df_new.shape}")
    check_sectors = set(df_old_subsectors.columns.get_level_values(0)) == set(df_new_subsectors.columns.get_level_values(0))
    if check_sectors ==  False:
        raise ValueError("Column order/names differ between the two DataFrames.")
    
    n = min(n, len(df_old))

    # Compare top N; treat NaN as a distinct value
    x = df_new_subsectors.apply(check_columns, df_old = df_old_subsectors, n = n).dropna()
    x = x.droplevel(1)

    old_entries_dict = {}
    for i in range(len(x)):
        sub_sector_name =  x.index[i]
        replaced        =  x.iloc[i]
        df_old_slice    = df_old.xs(sub_sector_name, axis=1)
        df_new_slice    = df_new.xs(sub_sector_name, axis=1)

        old_entries      = []
        new_entries      = []
        change_marketcap = []
        
        for old, new in replaced.items():
            

            if isinstance(old, str) and isinstance(new, str):

                old_entries_i     = df_old_slice[df_old_slice['Symbol'] == old]
                new_entries_i     = df_new_slice[df_new_slice['Symbol'] == new]

                old_entries.append(old_entries_i['Symbol'].iloc[0])
                new_entries.append(new_entries_i['Symbol'].iloc[0])
                change_in_marketcap_i = (new_entries_i['Market Cap'].values - old_entries_i['Market Cap'].values)[0]
                change_marketcap.append(change_in_marketcap_i)
        
        new = { 'Old' : old_entries,
                'New' : new_entries,
                'Change in Market Cap' : change_marketcap
                }
        
        old_entries_dict[sub_sector_name]   = pd.DataFrame(new.values(), new.keys())
    
    total_change_in_market_cap_per_sector   = pd.concat(old_entries_dict)\
                                                .loc[(slice(None), 'Change in Market Cap'), :]\
                                                .sum(axis=1).dropna().sort_values(ascending=True)\
                                                .to_frame('Total Change')\
                                                .droplevel(1)\
                                                .dropna()
        
    return pd.concat(old_entries_dict), total_change_in_market_cap_per_sector


####################################################################################

sub_sector_names      = get_sub_sectors_names()
    
SUB_SECTOR_DICT       = { 
        'Healthcare'             : sub_sector_names[0],
        'Financials'             : sub_sector_names[1],
        'Technology'             : sub_sector_names[2],
        'Energy'                 : sub_sector_names[3],
        'Industrials'            : sub_sector_names[4],
        'Communication Services' : sub_sector_names[5],
        'Consumer Staples'       : sub_sector_names[6],
        'Materials'              : sub_sector_names[7],
        'Consumer Discretionary' : sub_sector_names[8],
        'Real Estate'            : sub_sector_names[9],
        'Utilities'              : sub_sector_names[10]
    }


def get_sub_sectors_tickers(sector : str,
                            sub_sector_dict : dict = SUB_SECTOR_DICT,
                            n : int = 50
    )-> pd.DataFrame:
    """ 
    Returns a dataframe of the stocks and market cap of each 
    subsectors in the sector

    Input :
        - sub sector : the dataframe of all the sub sectors in a sector
        - n          : number of stocks to keep
    """
    sub_sector   = sub_sector_dict[sector].copy()
    url_suffixes = sub_sector['Industry Name'].str.replace(' - ', '-')\
                                              .str.replace('&', 'and')\
                                              .str.replace(' ', '-')\
                                              .str.lower()
    frames = {}
    for i in url_suffixes:
        try:
            url     = f'https://stockanalysis.com/stocks/industry/{i}'
            r       = requests.get(url)
        
            tables  = pd.read_html(StringIO(r.text))
            symbols = tables[0][['Symbol', 'Market Cap']]
                    
            clean   = symbols['Market Cap'].str.replace(',', '', regex=False).str.upper()
            parts   = clean.str.extract(r'(?P<value>\d*\.?\d+)\s*(?P<unit>[MBT])')
            
            symbols['Market Cap'] = (
                pd.to_numeric(parts['value'], errors='coerce') *
                parts['unit'].map({'M': 1.0, 'B': 1000.0, 'T' : 1000000.0})
            )
            frames[i]             = symbols.set_index('Symbol').sort_values(by='Market Cap', ascending=False)
        except ValueError as e:
            print(f"Error processing {i}: {e}")
            continue

    
    frames = {k:v.reset_index()[:n] for k,v in frames.items()}
    return pd.concat(frames, axis=1)


def get_subsector_tickers_stock_prices(sub_sector_dict : dict = SUB_SECTOR_DICT,
                                       sector          : str = 'Technology',
                                       filing_date     : dt.date = filing_date,
                                       n_stocks_per    : int = 10
)-> tuple[pd.DataFrame, list[str]]:
    """
    Pass in the sub_sector_dictionaries which should be automatically initiated
    Enter in the string for the sub sectors we want
    Returns the dataframe of returns for 2 years and the tickers that didn't work
    """
    # Get the names for the sub sector we want e.g., tech
    df_subsector_names = sub_sector_dict[sector]

    # Check if we already have the data

    file_path          = get_existing_subsector_stock_prices(sector, filing_date, n_stocks_per)

    # Get the sub sector names e.g., semi conductors, software and add the stocks 
    sub_sectors        = df_subsector_names.columns.get_level_values(0).unique()
    ticker_groups      = {}
    for i in sub_sectors:
        ticker_groups[i]  = df_subsector_names.loc[:, (i,'Symbol')][:n_stocks_per]

    # Call ticker obj
    failed = []
    frames = {}

    for key in ticker_groups.keys():
    
        stocks_per_sub = {}

        for count,i in enumerate(ticker_groups[key]):
            print(i, count)
            if count % 4 == 0 and count > 0:
                time.sleep(60)
            else:
                try:
                    ticker_obj = TickerData(i, filing_date)
                    ticker_obj.get_historical_prices(limit = 500000)
                    stocks_per_sub[i] = ticker_obj.historical_prices.drop(columns=['timestamp', 'otc'])
                
                except Exception as e:
                    failed.append(f'Ticker {i} failed due to : {e}')
                    print(f'failed at {i}')
                    time.sleep(60)

    frames[key] = pd.concat(stocks_per_sub, axis = 1)

    return frames, failed





if __name__ == '__main__':

    # df = read_sector_tickers('Technology')
    # non_semi = isolate_sector_tickers(df['Technology'], 'Semiconductor')

    pathfile     = get_existing_subsector_stock_prices('Technology', '2025-09-01', 10)
    df           = pd.read_csv(pathfile, index_col = [0], parse_dates = True, header = [0,1,2])
    last_date    = pathfile.stem.split('_')[2]
    tickers_list = df.columns.get_level_values(2).unique()

    tickers_sub = get_sub_sectors_tickers('Technology')