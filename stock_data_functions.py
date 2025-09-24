import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from polygon import RESTClient
import datetime as dt
from pandas.tseries.offsets import BDay
import requests
import time
import json
from matplotlib.dates import MonthLocator, AutoDateFormatter, AutoDateFormatter
import seaborn as sns
from matplotlib.ticker import PercentFormatter



API_key = 'tt2gOLH0fHAmPX70a4QURLFy59PRCZr3'
client = RESTClient(API_key, trace=True)
base_url = "https://api.polygon.io/vX/reference/financials"
headers = {"Authorization": f"Bearer {API_key}"}

def rate_limited_request(url, headers, params):
    
    response = requests.get(url, headers=headers, params=params)

    return response

def calc_log_rets(df : pd.DataFrame):
    rets = df.pct_change().dropna()
    rets = np.log(1 + rets) 
    
    return rets

def get_top_N_US_stocks(N):

    URL = "https://www.slickcharts.com/sp500"
    headers = {"User-Agent": "Mozilla/5.0"}

    resp = requests.get(URL, headers=headers, timeout=30)
    resp.raise_for_status()  # will raise if still forbidden

    tables = pd.read_html(resp.text)
    spx_df = tables[0]  # first table = current constituents
    tickers = spx_df["Symbol"].astype(str).str.upper().str.replace(r"[^\w\.-]", "", regex=True).tolist()

    return tickers[:N+1]

def handle_missing_fundamental_data(df):
    """
    Calculates by backing out the share count and eps if it is missing
        - Handles yearly vs quarterly cases differently
        - Yearly average share count is rough approx 
    """
    # When share counts are missing (quarter-only) -> backout by dividing quarterly earnings and basic_eps (quarterly)
    df['basic_avg_shares'] = np.where(df['basic_avg_shares'].isnull() & df['quarterly'] == True,
                                                    df['earnings_quarterly'] / df['basic_eps'],
                                                    df['basic_avg_shares'])

    df['diluted_avg_shares'] = np.where(df['diluted_avg_shares'].isnull() & df['quarterly'] == True,
                                                    df['earnings_quarterly'] / df['diluted_eps'],
                                                    df['diluted_avg_shares'])

    # When share counts are missing (yearly) -> divide yearly earnings by yearly EPS to get rough approx for quarterly share count
    df['basic_avg_shares'] = np.where(df['basic_avg_shares'].isnull() & df['quarterly'] == False,
                                                    df['earnings'] / df['basic_eps_'],
                                                    df['basic_avg_shares'])

    df['diluted_avg_shares'] = np.where(df['diluted_avg_shares'].isnull() & df['quarterly'] == False,
                                                    df['earnings'] / df['diluted_eps_'],
                                                    df['diluted_avg_shares'])
    # When basic eps data is missing for yearly -> rough approx by dividing quarterly earnings and share count calculated above
    df['basic_eps']          = np.where(df['basic_eps'].isna() & df['quarterly'] == False,
                                                        df['earnings_quarterly'] / df['basic_avg_shares'],
                                                        df['basic_eps'])

    df['diluted_eps']        = np.where(df['basic_eps'].isna() & df['quarterly'] == False,
                                                        df['earnings_quarterly'] / df['diluted_avg_shares'],
                                                        df['diluted_eps'])
    
    return df


class TickerData:

    def __init__(self, ticker, filing_date_gte):
        self.ticker             = ticker
        self.filing_date_gte    = filing_date_gte
        self.forecast_date      = (pd.to_datetime(filing_date_gte) - dt.timedelta(days =365*2)).strftime('%Y-%m-%d')
        date_today              = dt.datetime.now().date() - BDay(1)
        self.date_today         = date_today.strftime('%Y-%m-%d')

    def get_historical_prices(self, period='day', limit=5000):

        raw = []
        forecast_date = self.forecast_date
        date_today = self.date_today
        
        # Query API to get OHLC and volume data
        for a in client.list_aggs(
            self.ticker,
            1,
            period,
            forecast_date,
            date_today,
            limit = limit):
            
            raw.append(a)

        # Format
        df                      = pd.DataFrame(raw)
        df['Date']              = pd.to_datetime(df['timestamp'], unit='ms').dt.strftime('%Y-%m-%d')
        df.set_index('Date', inplace=True)

        df.index                = pd.to_datetime(df.index)
        self.historical_prices  = df




    def get_fundamental_data(self, test : bool = False):

        fundamentals = {}

        # Build your API calls manually

        params = {
            "ticker": self.ticker,
            "filing_date.gte": self.forecast_date,
            "limit": 50  # Adjust as needed
                 }

        raw_df = rate_limited_request(base_url, headers, params)
        json_raw_df = raw_df.json()
        # Get important data to calculate metrics

        number_of_periods = len(json_raw_df.get('results'))


        for i in range(0, number_of_periods):
            raw_df_period                   = json_raw_df.get('results', np.nan)[i]
            income_statement                = raw_df_period.get('financials').get('income_statement', np.nan)
            end_date                        = raw_df_period.get('end_date')

            if raw_df_period.get('timeframe') != 'quarterly':
                quarterly        = False
            else:
                quarterly        = True 
            
            
            try:
                basic_avg_shares = income_statement.get('basic_average_shares', {})
                basic_avg_shares = basic_avg_shares['value']
            except Exception:
                print(f"{self.ticker} period {end_date}: basic_avg_shares missing")
                basic_avg_shares = np.nan

            try:
                diluted_avg_shares = income_statement.get('diluted_average_shares', {})
                diluted_avg_shares = diluted_avg_shares['value']
            except Exception:
                print(f"{self.ticker} period {end_date}: diluted_avg_shares missing")
                diluted_avg_shares = np.nan

            try:
                diluted_eps = income_statement.get('diluted_earnings_per_share', {})
                diluted_eps = diluted_eps['value']
            except Exception:
                print(f"{self.ticker} period {end_date}: diluted_eps missing")
                diluted_eps = np.nan

            try:
                basic_eps = income_statement.get('basic_earnings_per_share', {})
                basic_eps = basic_eps['value']
            except Exception:
                print(f"{self.ticker} period {end_date}: basic_eps missing")
                basic_eps = np.nan

            try:
                earnings = income_statement.get('net_income_loss_available_to_common_stockholders_basic', {})
                earnings = earnings['value']
            except Exception:
                print(f"{self.ticker} period {end_date}: earnings missing")
                earnings = np.nan

            try:
                revenue = income_statement.get('revenues', {})
                revenue = revenue['value']
            except Exception:
                print(f"{self.ticker} period {end_date}: revenue missing")
                revenue = np.nan

            try:
                r_d = income_statement.get('research_and_development', {})
                r_d = r_d['value']
            except Exception:
                r_d = np.nan

            file_date = raw_df_period.get('filing_date', np.nan)
            if file_date != np.nan:
                file_date = pd.to_datetime(file_date).date()
                file_date = (file_date - BDay(1)).date()


            fundamentals[end_date] = pd.Series([
                                    basic_avg_shares,
                                    basic_eps, 
                                    diluted_avg_shares,
                                    diluted_eps,
                                    earnings,
                                    r_d,
                                    revenue,
                                    file_date,
                                    quarterly
                                    ]
                                )   
        fundamentals_df = pd.DataFrame(fundamentals).T
        fundamentals_df.rename({
                0 : 'basic_avg_shares',
                1 : 'basic_eps', 
                2 : 'diluted_avg_shares',
                3 : 'diluted_eps',
                4 : 'earnings',
                5 : 'r&d',
                6 : 'revenue',
                7 : 'earnings tag',
                8 : 'quarterly'

            }, axis = 1, inplace = True)

        fundamentals_df.index           = pd.to_datetime(fundamentals_df.index).strftime('%Y-%m-%d')
        fundamentals_df.index           = pd.to_datetime(fundamentals_df.index)
        fundamentals_df.sort_index(ascending=True,inplace=True)
            
        # Test : returns json file
        if test:
            self.json_raw_df                = json_raw_df


        def get_quarterly_fundamentals(df):
            """ 
            Some stocks list their 4th quarter (or whatever quarter) earnings 
            as sum of the total year, so must back out what that quarter earnigns are

            used to calculate TTM EPS and TTM P/E
            
            """
            # ensure sorted by date
            df = df.sort_index()

            # sum of prior 3 quarters' earnings
            prior3          = df['earnings'].shift(1).rolling(3, min_periods=3).sum()
            prior3_rev      = df['revenue'].shift(1).rolling(3, min_periods=3).sum()
            prior3_rd       = df['r&d'].shift(1).rolling(3, min_periods=3).sum()

            # flag annual rows per your rule
            is_annual = df['quarterly'] == False

            # Placeholder for data processing
            df['basic_eps_']               = df['basic_eps'].copy()
            df['diluted_eps_']             = df['diluted_eps'].copy()
            # adjusted quarterly earnings:
            df['earnings_quarterly']       = np.where(is_annual, df['earnings'] - prior3, df['earnings'])
            df['basic_eps']                = np.where(is_annual, df['earnings_quarterly'] / df['basic_avg_shares'],
                                                                 df['basic_eps']) \
                                                                 if ~(df['basic_avg_shares'].isna() | df['basic_avg_shares'] == 0).any() else np.nan
            
            df['diluted_eps']              = np.where(is_annual, df['earnings_quarterly'] / df['diluted_avg_shares'],
                                                                 df['diluted_eps']) \
                                                                 if ~(df['diluted_avg_shares'].isna() | df['diluted_avg_shares'] == 0).any() else np.nan
            
            df['revenue']                  = np.where(is_annual, df['revenue'] - prior3_rev, df['revenue'])
            df['r&d']                      = np.where(is_annual, df['r&d'] - prior3_rd, df['r&d'])


            # Backfill the eps that we do have but set to np.nan because of zero division error
            df['basic_eps']                = np.where(df['basic_eps'] == np.nan,
                                                      df['basic_eps_'],
                                                      df['basic_eps'])

            df['diluted_eps']              = np.where(df['diluted_eps'] == np.nan,
                                                      df['diluted_eps_'],
                                                      df['diluted_eps'])
            return df
    
    
        # Process for missing values (share count and eps)
        df = get_quarterly_fundamentals(fundamentals_df)

        df = handle_missing_fundamental_data(df)
        self.fundamentals = df

    
    def get_ratios(self):
        """  
        Calculate P/E Ratios and TTM Earnings
        """

        if not hasattr(self, 'fundamentals'):
            self.get_fundamental_data()

        if not hasattr(self, 'historical_prices'):
            self.get_historical_prices()

        quarterly_fundamentals                 = self.fundamentals
        historical_prices                      = self.historical_prices

        quarterly_fundamentals['TTM earnings'] = quarterly_fundamentals['earnings_quarterly'].rolling(window=4, 
                                                                                                      min_periods=4) \
                                                                                                      .sum()
        quarterly_fundamentals['TTM eps']      = quarterly_fundamentals['TTM earnings'] \
                                               / quarterly_fundamentals['diluted_avg_shares'] \
                                               if ~(quarterly_fundamentals['diluted_avg_shares'].isna() \
                                               | quarterly_fundamentals['diluted_avg_shares'] == 0) \
                                               .any() else np.nan

        merged = pd.merge_asof(left                 = historical_prices,
                                right               = quarterly_fundamentals,
                                left_index          = True,
                                right_index         = True,
                                allow_exact_matches = True,
                                direction           = 'backward',
                                tolerance           = None
                                )

        merged['TTM P/E']                   = merged['vwap'] / merged['TTM eps'] 
        merged['quarterly P/E diluted']     = merged['vwap'] / merged['diluted_eps']    * 0.25 
        merged['quarterly P/E basic']       = merged['vwap'] / merged['basic_eps']      * 0.25 
        
        keep = ['open', 'high', 'low', 'close', 'vwap', 'quarterly P/E diluted', 'quarterly P/E basic', 'TTM P/E', 'TTM eps', 'TTM earnings']

        self.merged_test = merged
        # negative p/e ratio makes no sense
        merged['TTM P/E']                  = np.where(merged['TTM P/E'] < 0, 0, merged['TTM P/E'])
        merged['quarterly P/E diluted']    = np.where(merged['quarterly P/E diluted'] < 0, 0, merged['quarterly P/E diluted'])
        merged['quarterly P/E basic']      = np.where(merged['quarterly P/E basic'] < 0, 0, merged['quarterly P/E basic'])
        merged['earnings tag']             = pd.to_datetime(merged['earnings tag']).dt.strftime('%Y-%m-%d')

        self.summary_data                  = merged[keep]            
        self.earning_dates                 = pd.Series(merged['earnings tag'].unique())




    def map_earnings_to_prices(self,
                               method: str = "backward",
                               tolerance = dt.timedelta(5)) -> pd.DataFrame:
        """
        Maps the share price on the day closest to the earnings date so you can see on the time series plots
            - make sure it has :
                1. Historical prices
                2. Earning dates
        """
        
        prices              = self.historical_prices
        earnings_dates      = self.earning_dates

        events              = pd.DataFrame({"earnings_date": pd.to_datetime(earnings_dates.dropna().unique())})
        events              = events.sort_values("earnings_date")

        merged = pd.merge_asof(
            left=events,
            right=prices,
            left_on="earnings_date",
            right_index=True,
            direction=method,                
            tolerance=pd.Timedelta(tolerance) if tolerance else None
        )

        merged.set_index('earnings_date', inplace= True)
        merged.index = pd.to_datetime(merged.index)
        self.earning_dates_with_prices = merged


    def plot_single_timeseries(self, plot=True):
        """ 
        Plots the time series with earning dates highlighted. Dependencies
            - Needs map_earnings_to_prices to get the share price on earnigns day
            - Needs historical price chart
            - Needs Ratios (it is where the earnings logic is done)
        """
        if not hasattr(self, 'historical_prices'):
            self.get_historical_prices()

        if not hasattr(self, 'earning_dates'):
            self.get_fundamental_data()
            self.get_ratios()
        
        if not hasattr(self, 'earning_dates_with_prices'):
            self.map_earnings_to_prices()

        df = self.summary_data
        earnings_dates = self.earning_dates_with_prices


        if plot == True:                    
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12,8))

            axs[0].scatter(earnings_dates.index, earnings_dates['close'],marker='x', color='tab:red', label='Earnings Date')
            axs[0].plot(df['close'], label='Close Price')
            axs[0].legend()
            axs[0].set_title(f'Share Price of {self.ticker}')
            axs[0].set_ylabel('$ Dollars')
            axs[0].grid()
            axs[1].plot(df['quarterly P/E diluted'], label='Quarterly P/E', alpha = 0.5, linestyle='--')
            axs[1].plot(df['TTM P/E'], label='TTM P/E', color='tab:blue')

            df['negative TTM P/E'] = np.where(df['TTM P/E'] < 1, df['TTM P/E'], np.nan)
            df['negative Q P/E'] = np.where(df['quarterly P/E diluted'] < 1, df['quarterly P/E diluted'], np.nan)


            axs[1].plot(df['negative TTM P/E'], color = 'lightgray', linewidth=3)
            axs[1].plot(df['negative Q P/E'], color = 'lightgray',linewidth=3)
            axs[1].grid()
            axs[1].legend()
            axs[1].set_title('P/E Ratios')


    def calc_log_rets(self, returns = 'vwap'):
        rets = self.summary_data[returns].pct_change().dropna()
        rets = np.log(1 + rets).to_frame('returns')
        self.returns = rets

    def plot_returns_distribution(self,
                                  title: str = "Returns Distribution",
                                  xlabel: str = "Returns",
                                  bins: int | int = 25,
                                  kde: bool = True,
                                  percent_axis: bool = True,
                                  show_stats: bool = True,
                                  ax: plt.Axes | None = None,
                                  ):
        """
        Plot a histogram + KDE for returns.
        """
        # sanitize
        if not hasattr(self, 'returns'):
            self.calc_log_rets()
        
        r = self.returns 

        if r.empty:
            raise ValueError("No non-NaN returns to plot.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5.5))

        # Use density so the KDE is on the same scale as the histogram
        sns.histplot(data = r, x = 'returns', bins=bins, stat="density", kde=kde, ax=ax)

        # Labels / title
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")

        # Optional percent axis (for decimal returns like 0.01 = 1%)
        if percent_axis:
            ax.xaxis.set_major_formatter(PercentFormatter(1.0))

        # Optional mean/median lines
        if show_stats:
            mu, med             = r['returns'].mean() , r['returns'].median() 
            stD, mad            = (r['returns'].std() * np.sqrt(252) * 100).round(2), (np.mean(np.abs(r['returns'] - mu)) * np.sqrt(252) * 100).round(2)
            stD_30day           = ((r['returns'].iloc[-31:-1]).std() * np.sqrt(252) * 100).round(2)

            ax.axvline(mu, linestyle="--", linewidth=1.2, label=f"Mean: {mu:.4f}")
            ax.axvline(med, linestyle=":", linewidth=1.2, label=f"Median: {med:.4f}")

            ax.annotate(
                f'Realized Vol : {stD}%\nMAD : {mad}%\n30-day Realized Vol: {stD_30day}%', # Text for Annotation
                xy = (0.025, 10), # Location of the annotation
                xytext =(r['returns'].max() - 1.5*r['returns'].std(), 15), # Location of the text
            )
            
            ax.legend()

        plt.tight_layout()
        return ax

