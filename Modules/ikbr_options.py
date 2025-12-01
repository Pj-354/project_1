from ib_insync import IB, Index, Option, util
import pandas as pd
from typing import Tuple
import time
import pandas as pd
from ib_async import Stock  # ib_insync/ib_async
from typing import List, Union, Tuple
from ib_async import IB, Stock
import pandas as pd
from math import ceil


def snapshot_option_greeks(
    ib: IB,
    symbol: str = "SPX",
    index_exchange: str = "CBOE",
    expiries: int = 10,
    strike_window: float = 500,
    step: float = 5,
    market_data_type: int = 4,  # 1=live, 2=freeze, 3=delayed, 4=delayed-frozen
) -> Tuple[pd.DataFrame, float]:
    
    """Return DataFrame of snapshot prices + model greeks near-the-money."""
    idx = Index(symbol, index_exchange)
    ib.qualifyContracts(idx)

    ib.reqMarketDataType(market_data_type)
    [t_under] = ib.reqTickers(idx)
    spot = t_under.marketPrice()

    chains = ib.reqSecDefOptParams(idx.symbol, "", idx.secType, idx.conId)
    # pick SPX chain via SMART or the listed exchange if present
    chain = next(c for c in chains if c.tradingClass == symbol and c.exchange in ("SMART", index_exchange))

    strikes = [s for s in chain.strikes if s % step == 0 and (spot - strike_window) < s < (spot + strike_window)]
    expirations = sorted(chain.expirations)[:expiries]
    rights = ("C", "P")

    contracts = [
        Option(symbol, exp, strike, right, "SMART", tradingClass=symbol)
        for exp in expirations for strike in strikes for right in rights
    ]
    contracts = ib.qualifyContracts(*contracts)
    tickers = ib.reqTickers(*contracts)

    rows = []
    for c, t in zip(contracts, tickers):
        g = t.modelGreeks or t.bidGreeks or t.askGreeks or t.lastGreeks
        rows.append({
            "expiry": c.lastTradeDateOrContractMonth,
            "strike": c.strike,
            "right": c.right,
            "bid": t.bid,
            "ask": t.ask,
            "last": t.last,
            "iv_model": getattr(g, "impliedVol", None) if g else None,
            "delta": getattr(g, "delta", None) if g else None,
            "gamma": getattr(g, "gamma", None) if g else None,
            "vega": getattr(g, "vega", None) if g else None,
            "theta": getattr(g, "theta", None) if g else None,
            "undPrice": getattr(g, "undPrice", None) if g else spot,
        })
    df = pd.DataFrame(rows).sort_values(["expiry", "strike", "right"]).reset_index(drop=True)
    return df, spot, chain




def fetch_nbbo_1min_bars(
    ib,
    tickers: Union[str, List[str]],
    start_date: str,                 # 'YYYY-MM-DD'
    end_date: str,                   # 'YYYY-MM-DD'
    tws_timezone: str = "America/New_York",    # TWS login timezone
    output_timezone: str = "America/New_York", # index timezone you want
    use_rth: bool = True,
    pace: bool = True                # True = sleep ~21s per request (BID_ASK counts double)
) -> pd.DataFrame:
    """
    Returns a DataFrame of 1-minute NBBO bars for all tickers over [start_date, end_date],
    indexed by time in output_timezone, with columns: ['symbol','open','high','low','close','volume','wap','count'].
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    start = pd.Timestamp(start_date).tz_localize(tws_timezone).normalize()
    end   = pd.Timestamp(end_date).tz_localize(tws_timezone).normalize()
    days  = pd.date_range(start, end, freq="D", tz=tws_timezone)
    per_req_sleep = 21.0 if pace else 0.0   # BID_ASK requests count double toward 60/10min

    dict_df = {}
    for sym in tickers:
        c = Stock(sym, "SMART", "USD"); ib.qualifyContracts(c)
        for day in days:
            end_dt = day + pd.Timedelta(hours=23, minutes=59, seconds=59)
            end_str = end_dt.strftime("%Y%m%d %H:%M:%S")  # interpreted in TWS timezone
            bars = ib.reqHistoricalData(
                c,
                endDateTime=end_str,
                durationStr="1 D",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=use_rth,
                formatDate=1,
                keepUpToDate=False
            )
            if bars:
                df = pd.DataFrame([{
                    "time": pd.to_datetime(b.date),  # in TWS tz per IB docs
                    "twap_bid": b.open, "max_ask": b.high, "max_bid": b.low, "twap_ask": b.close,
                } for b in bars])
                # localize to TWS tz then convert to requested output tz
                df["symbol"] = sym
                df = df.set_index("time").sort_index()
            if per_req_sleep:
                time.sleep(per_req_sleep)
        dict_df[sym] = df

    return dict_df
