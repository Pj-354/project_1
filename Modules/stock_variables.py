import pandas as pd
from stock_data_functions import TickerComparison, calc_log_rets
import datetime as dt

def get_avg_daily_volume(tc, last_n_days_quantiles = 5, lookback=200):
    from scipy.stats import percentileofscore
    vol = tc.tickers_stocks_prices.loc[:, (slice(None), 'volume')]
    vol = vol.droplevel(1, axis=1)

    # Volume per month and year
    vol['month_year'] = vol.index.strftime('%Y-%m')
    avg_daily_vol = vol.groupby(['month_year']).agg(['mean', 'median'])

    # Volume by day of week
    vol['day_of_week'] = vol.index.day_name()
    avg_daily_vol_dow = vol.groupby(['month_year', 'day_of_week']).agg(['mean', 'median', 'std'])
    
    # Volume Quantiles per last 200 days
    def get_last_days_volume_quantiles(s, last_n_days_quantiles = last_n_days_quantiles, lookback=lookback):
        last_200_days = s.dropna().tail(lookback)
        last_day      = last_200_days.iloc[-last_n_days_quantiles:]
        quantiles     = percentileofscore(last_200_days, last_day)
        return quantiles
    vol.drop(columns=['month_year', 'day_of_week'], inplace=True)
    quantiles_vol = vol.apply(get_last_days_volume_quantiles)

    return avg_daily_vol, avg_daily_vol_dow, quantiles_vol

def get_avg_volatility(tc, kind='close'):
    from scipy.stats import percentileofscore
    prices = tc.tickers_stocks_prices.loc[:, (slice(None), kind)]
    rets = prices.apply(calc_log_rets) 
    rets = rets.droplevel(1, axis=1)
    yearly_vol = rets.groupby(rets.index.year).std() * 1600
    monthly_vol = rets.groupby(rets.index.to_period("M")).std() * 1600
    last_3weeks_vol = rets.rolling(window=15).std() * (252**0.5)  # Annualized volatility
    def vol_quantile_lookback(s, window, lookback=200):
        last_200_days = s.dropna().tail(lookback)
        last_day      = last_200_days.iloc[-window:]
        quantile      = pd.Series(percentileofscore(last_200_days, last_day), index=last_day.index)
        return quantile
    vol_quantile = rets.apply(vol_quantile_lookback, window=21)
    return yearly_vol, monthly_vol,last_3weeks_vol, vol_quantile

# params
def filter_out_missing_volume_data(tc, date = None):
    if date is None:
        date = dt.date(2023,11,15)
    volume = tc.tickers_stocks_prices.loc[:, (slice(None), 'volume')].copy()
    volume = volume.droplevel(1, axis=1)

    remove = []
    for i in volume.columns:
        first = volume[i].first_valid_index().date()
        if first > date:
            remove.append(i)
        else:
            pass

    volume = volume.drop(columns=remove)
    volume = volume.fillna(0)

    return volume

def identify_tod_volume_outliers(tc : TickerComparison, X: int =None, THRESH: float=5.0, eps = 1e-12):
    """ 
    Identify time-of-day volume outliers based on historical volume patterns. 
    Divide the volume at each time-of-day by the total daily volume (median or avg over X days)
    Returns 4 dictionaries : flags, ratios of median and average
    eps is epsilon to avoid division by zero
    """
    from Modules.factor_analysis_functions import (get_session_and_trade_date,
                                                   _select_ext_hours_index)
    
    if isinstance(tc, pd.DataFrame):
        volume = tc.fillna(0.0)
    else:
        volume = filter_out_missing_volume_data(tc)

    if X is None:
        X = len(set(volume.index.date))
    # Results
    matrices          = {}
    stocks            = volume.columns.to_list()
    # Volume Dataframe
    vol = volume.copy()
    valid_tod         = _select_ext_hours_index(vol.index)
    vol               = get_session_and_trade_date(vol)
    vol               = vol.reindex(valid_tod)
    vol['tod']        = vol.index.strftime('%H:%M')
    vol['trade_date'] = pd.to_datetime(vol['trade_date']).dt.date
    # Start and end date
    last_date   = vol['trade_date'].max()
    start_date = (pd.to_datetime(last_date) - pd.Timedelta(days=X-1)).date()

    # Selection window
    sel = vol[(vol['trade_date'] >= start_date) & (vol['trade_date'] <= last_date)].copy()
    grouped = sel.groupby(['trade_date', 'tod'])
    sel['trade_date'] = pd.to_datetime(sel['trade_date'])
    weekdays = sel['trade_date'].dt.dayofweek < 5

    sel = sel.loc[weekdays]
    trade_dates = sorted(sel['trade_date'].unique())
    tod         = sorted(sel['tod'].unique())
    for t in stocks:
        temp = grouped[t].sum().unstack().reindex(index=trade_dates, columns=tod).fillna(0.0)
        matrices[t] = temp


    baseline_med  = {}
    baseline_mean = {}
    for t, mat in matrices.items():
        baseline_med[t]  = mat.shift(1).rolling(window=X, min_periods=1).median().round()
        baseline_mean[t] = mat.shift(1).rolling(window=X, min_periods=1).mean().round()

    day_pct = {}
    baseline_med_pct = {}
    baseline_mean_pct = {}
    for t in stocks:
        # Build out historical volume curves
        mat=matrices[t]; day_sum=mat.sum(axis=1).replace(0, eps); day_pct[t]=mat.div(day_sum, axis=0)
        bmed=baseline_med[t]; bmed_sum=bmed.sum(axis=1).replace(0, eps); baseline_med_pct[t]=bmed.div(bmed_sum, axis=0)
        bmean=baseline_mean[t]; bmean_sum=bmean.sum(axis=1).replace(0, eps); baseline_mean_pct[t]=bmean.div(bmean_sum, axis=0)

    ratio_med = {t: day_pct[t].divide(baseline_med_pct[t].replace(0, eps)) for t in stocks}
    flag_med  = {t: ratio_med[t] > THRESH for t in stocks}

    ratio_mean = {t: day_pct[t].divide(baseline_mean_pct[t].replace(0, eps)) for t in stocks}
    flag_mean  = {t: ratio_mean[t] > THRESH for t in stocks}
        
    return ratio_med, flag_med, ratio_mean, flag_mean, baseline_med_pct, baseline_mean_pct


    missing_tally = {}
    for i in missing_names:
        ser_ = close_prices[i]
        start_date = ser_.first_valid_index()
        ser_start  = ser_[start_date:]
        total_len = len(ser_[start_date:])
        missing_pct = ser_start.isnull().sum() / total_len
        missing_tally[i] = pd.Series([missing_pct, start_date, total_len], index=['missing_pct_after_start', 'start_date', 'total_len'])

    missing_check = pd.DataFrame(missing_tally.values(), missing_tally.keys(), columns=['missing_pct_after_start', 'start_date', 'total_len'])
    must_drop     = missing_check[missing_check['missing_pct_after_start'] > 0.1].index
    stock_universe = pd.read_csv('/Users/phillip/Desktop/Moon2/data/stock_tickers_universe.csv', index_col='ticker')
    bad_df = missing_check.loc[must_drop]
    bad_df['industry'] = bad_df.index.map(stock_universe['industry'])
    bad_df['sector']   = bad_df.index.map(stock_universe['sector'])
    missing_by_sector = bad_df.groupby('sector').size()
    missing_by_industry = bad_df.groupby('industry').size()
    missing_by_industry.sort_values(ascending=False)

    return missing_check, must_drop, missing_by_sector.sort_values(ascending=False), missing_by_industry.sort_values(ascending=False)

# utils - pass ticker comparison to see what returns are missing and why
def analyse_missing_tickers(tC, col : str = 'close', start_date : str = None, end_date : str = None, stock_universe : pd.DataFrame = None, skip_industry = False):
    """ 
    Pass the tC object. Will show you how many tickers are missing, and also groups by industry and sector.
    When grouping, it starts at the first valid index because e.g., IPO in last 2 years.
    
    """
    print('Looking at the {} prices'.format(col))
    print(f'Skip Industry : {skip_industry}')
    close_prices = tC.tickers_stocks_prices.loc[:, (slice(None), 'close')].droplevel(axis=1, level=1)
    if start_date is not None:
        close_prices = close_prices[close_prices.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        close_prices = close_prices[close_prices.index <= pd.to_datetime(end_date)]

    def missing_tickers_sector_industry(close_prices, missing_names, stock_universe, skip_industry):
        missing_tally = {}
        for i in missing_names:
            ser_ = close_prices[i]
            start_date = ser_.first_valid_index()
            ser_start  = ser_[start_date:]
            total_len = len(ser_[start_date:])
            missing_pct = ser_start.isnull().sum() / total_len
            missing_tally[i] = pd.Series([missing_pct, start_date, total_len], index=['missing_pct_after_start', 'start_date', 'total_len'])

        missing_check = pd.DataFrame(missing_tally.values(), missing_tally.keys(), columns=['missing_pct_after_start', 'start_date', 'total_len'])
        must_drop     = missing_check[missing_check['missing_pct_after_start'] > 0.1].index
       
        if skip_industry == True:
            return missing_check, must_drop, close_prices, None
       
        bad_df = missing_check.loc[must_drop]
        bad_df['industry'] = bad_df.index.map(stock_universe['industry'])
        bad_df['sector']   = bad_df.index.map(stock_universe['sector'])
        missing_by_sector = bad_df.groupby('sector').size()
        missing_by_industry = bad_df.groupby('industry').size()
        missing_by_industry.sort_values(ascending=False)

        return missing_check, must_drop, missing_by_sector.sort_values(ascending=False), missing_by_industry.sort_values(ascending=False)

    def check_null(df, thresh : int = 50):
        null_df      = df.isnull().sum()
        good_tickers = list(null_df[null_df < thresh].index)
        missing_names = list(null_df[null_df >= thresh].index)
        print('Number of Tickers Discarded : ', len(df.columns) - len(df[good_tickers].columns))
        return df[good_tickers], missing_names, df

    prices_df, missing_names, close_prices = check_null(close_prices)

    missing_check, must_drop, missing_by_sector, missing_by_industry = missing_tickers_sector_industry(close_prices, missing_names, stock_universe, skip_industry=skip_industry)

    return prices_df, missing_names, close_prices, missing_check, must_drop, missing_by_sector, missing_by_industry

if __name__ == "__main__":
    from stock_data_functions import TickerComparison, calc_log_rets

    tc = TickerComparison(['NVDA', 'AMD'], '2023-11-08', date_updated=True)
    prices = tc.tickers_stocks_prices.loc[:, (slice(None), 'close')]
    rets = prices.apply(calc_log_rets)
