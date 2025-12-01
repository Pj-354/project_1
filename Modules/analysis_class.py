import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class TTMRatiosAnalyzer:
    api_key: str = 
    tickers: List[str] = field(default_factory=list)
    fundamentals_dir: Optional[str] = None
    ratios_dir: str = "./ttm_ratios"
    local_price_dir: Optional[str] = None
    min_end_date: Optional[str] = None
    max_end_date: Optional[str] = None
    verbose: bool = False
    base_url: str = "https://api.polygon.io"
    _alias_map: Dict[str, List[str]] = field(default_factory=lambda: {
        "is_interest_expense_operating": [
            "interest_expense_operating", "interest_expense_operating_loss",
            "interest_expense_operating_activities"
        ],
        "is_research_and_development": [
            "research_and_development", "rnd_expense", "r&d_expense"
        ],
        # Add further aliases here
    })

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("API key required")
        if self.fundamentals_dir:
            Path(self.fundamentals_dir).mkdir(parents=True, exist_ok=True)
        Path(self.ratios_dir).mkdir(parents=True, exist_ok=True)

    def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _canonical_field(self, raw_key: str) -> str:
        raw = raw_key.lower()
        for canon, synonyms in self._alias_map.items():
            if raw == canon.lower() or any(raw == s.lower() for s in synonyms):
                return canon
            for s in synonyms:
                if s.lower() in raw:
                    return canon
        return raw_key

    def _flatten_section(self, sec: Optional[Dict[str, Any]], prefix: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not isinstance(sec, dict):
            return out
        for k, v in sec.items():
            val = v.get("value") if isinstance(v, dict) else v
            if isinstance(val, (int, float)):
                canon = self._canonical_field(k)
                out[f"{prefix}{canon}"] = float(val)
        return out

    def _num(self, s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    def _load_latest_local_price(self, ticker: str) -> Optional[float]:
        """Load most recent close from local parquet if available."""
        if not self.local_price_dir:
            return None
        pattern = f"{ticker.upper()}_*.parquet"
        files = list(Path(self.local_price_dir).glob(pattern))
        if not files:
            return None
        latest = max(files, key=lambda p: p.stat().st_mtime)
        df = pd.read_parquet(latest)
        if "close" in df.columns:
            return float(df["close"].iloc[-1])
        try:
            return float(df.iloc[-1].iloc[0])
        except Exception:
            return None

    def _ensure_q4_as_annual(self, fins: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """If Q4 quarterly record missing filing date, fetch annual instead."""
        q4_mask = (fins["fiscal_period"] == "Q4") | (fins["end_date"].dt.month == 12)
        missing_filing = fins["filing_date"].isna()
        if not fins[q4_mask & missing_filing].empty:
            if self.verbose:
                print(f"{ticker}: Q4 row missing filing date → fetching annual")
            url = f"{self.base_url}/vX/reference/financials"
            params = {"ticker": ticker.upper(), "timeframe": "annual", "apiKey": self.api_key}
            resp = self._get_json(url, params=params)
            items = resp.get("results", [])
            if items:
                item = items[0]
                fin = item.get("financials", {}) or {}
                row = {
                    "ticker": ticker.upper(),
                    "fiscal_year": item.get("fiscal_year"),
                    "fiscal_period": item.get("fiscal_period"),
                    "timeframe": "annual",
                    "filing_date": item.get("filing_date"),
                    "start_date": item.get("start_date"),
                    "end_date": item.get("end_date"),
                }
                row.update(self._flatten_section(fin.get("income_statement", {}), "is_"))
                row.update(self._flatten_section(fin.get("balance_sheet", {}), "bs_"))
                row.update(self._flatten_section(fin.get("cash_flow_statement", {}), "cf_"))
                row.update(self._flatten_section(fin.get("comprehensive_income", {}), "ci_"))
                df_ann = pd.DataFrame([row])
                df_ann["end_date"] = pd.to_datetime(df_ann["end_date"], errors="coerce")
                fins = pd.concat([fins, df_ann], ignore_index=True)
                fins = fins.drop_duplicates(subset=["end_date","fiscal_period"], keep="last")\
                           .sort_values("end_date").reset_index(drop=True)
        return fins

    def fetch_quarterly_financials(self, ticker: str) -> pd.DataFrame:
        """Fetch quarterly financials from API for the ticker."""
        url = f"{self.base_url}/vX/reference/financials"
        params = {
            "ticker": ticker.upper(),
            "timeframe": "quarterly",
            "sort": "period_of_report_date",
            "order": "asc",
            "limit": 100,
            "apiKey": self.api_key
        }
        rows: List[Dict[str, Any]] = []
        first = True
        while url:
            data = self._get_json(url, params=params if first else None)
            first = False
            for item in data.get("results", []):
                fin = item.get("financials", {}) or {}
                row = {
                    "ticker": (item.get("tickers") or [ticker.upper()])[0],
                    "fiscal_year": item.get("fiscal_year"),
                    "fiscal_period": item.get("fiscal_period"),
                    "timeframe": item.get("timeframe"),
                    "filing_date": item.get("filing_date"),
                    "start_date": item.get("start_date"),
                    "end_date": item.get("end_date"),
                }
                row.update(self._flatten_section(fin.get("income_statement", {}), "is_"))
                row.update(self._flatten_section(fin.get("balance_sheet", {}), "bs_"))
                row.update(self._flatten_section(fin.get("cash_flow_statement", {}), "cf_"))
                row.update(self._flatten_section(fin.get("comprehensive_income", {}), "ci_"))
                rows.append(row)
            next_url = data.get("next_url")
            url = next_url
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        for col in ["filing_date","start_date","end_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        stmt_cols = [c for c in df.columns if c.startswith(("is_","bs_","cf_","ci_"))]
        df[stmt_cols] = df[stmt_cols].apply(pd.to_numeric, errors="coerce")
        df = df.sort_values("end_date").reset_index(drop=True)
        return df

    def _compute_ttm_metrics(self, fins: pd.DataFrame) -> pd.DataFrame:
        """Compute TTM flows and metrics — edit here when you change metric definitions."""
        df = fins.copy().set_index("end_date").sort_index()
        for col in ["is_revenues","is_net_income_loss","is_operating_income_loss",
                    "is_depreciation_and_amortization","is_basic_earnings_per_share",
                    "is_diluted_earnings_per_share"]:
            if col not in df.columns:
                df[col] = pd.NA
        df["ebitda_q"] = self._num(df["is_operating_income_loss"]) + self._num(df["is_depreciation_and_amortization"])
        roll4 = lambda c: self._num(df[c]).rolling(window=4, min_periods=4).sum()
        df["revenues_ttm"] = roll4("is_revenues")
        df["net_income_ttm"] = roll4("is_net_income_loss")
        df["ebitda_ttm"] = roll4("ebitda_q")
        df["eps_basic_ttm"] = roll4("is_basic_earnings_per_share")
        df["eps_diluted_ttm"] = roll4("is_diluted_earnings_per_share")
        df["shares_diluted"] = self._num(df.get("is_diluted_average_shares", pd.Series()))
        df["equity"] = self._num(df.get("bs_equity_attributable_to_parent", df.get("bs_equity", pd.Series())))
        df["cash"] = self._num(df.get("bs_cash", pd.Series()))
        df["total_debt"] = self._num(df.get("bs_long_term_debt", pd.Series()))
        df["assets"] = self._num(df.get("bs_assets", pd.Series()))
        df["current_assets"] = self._num(df.get("bs_current_assets", pd.Series()))
        df["current_liabilities"] = self._num(df.get("bs_current_liabilities", pd.Series()))
        df["inventory"] = self._num(df.get("bs_inventory", pd.Series()))
        df["nwc"] = df["current_assets"] - df["current_liabilities"]
        df["nwc_avg_4q"] = df["nwc"].rolling(window=4, min_periods=2).mean()
        df["capital_employed"] = df["assets"] - df["current_liabilities"]
        df["capital_employed_avg_4q"] = df["capital_employed"].rolling(window=4, min_periods=2).mean()
        df["assets_avg_4q"] = df["assets"].rolling(window=4, min_periods=2).mean()
        df["inventory_avg_4q"] = df["inventory"].rolling(window=4, min_periods=2).mean()
        df["cf_investing_q"] = self._num(df.get("cf_net_cash_flow_from_investing_activities", pd.Series()))
        df["cf_investing_ttm"] = df["cf_investing_q"].rolling(window=4, min_periods=4).sum()
        df["capex_ttm"] = (-df["cf_investing_ttm"]).where(df["cf_investing_ttm"] < 0)
        df["interest_expense_q"] = self._num(df.get("is_interest_expense_operating", pd.Series()))
        df["interest_expense_ttm"] = df["interest_expense_q"].rolling(window=4, min_periods=4).sum().abs()

        # valuation & ratio calculations
        price = df["price"]
        market_cap = price * df["shares_diluted"]
        ev = market_cap + df["total_debt"] - df["cash"]
        df["market_cap"] = market_cap
        df["enterprise_value"] = ev
        safe_div = lambda n, d: n / d.replace(0, pd.NA)
        df["pe_ttm"] = safe_div(price, df["eps_basic_ttm"])
        df["ps_ttm"] = safe_div(market_cap, df["revenues_ttm"])
        df["pb_ttm"] = safe_div(market_cap, df["equity"])
        df["ev_ebitda_ttm"] = safe_div(ev, df["ebitda_ttm"])
        df["net_margin_ttm"] = safe_div(df["net_income_ttm"], df["revenues_ttm"])
        df["ebitda_margin_ttm"] = safe_div(df["ebitda_ttm"], df["revenues_ttm"])
        # income‐% of revenue metrics
        df["sga_pct_rev_ttm"] = safe_div(df.get("is_selling_general_and_administrative_expenses", pd.Series()).rolling(window=4, min_periods=4).sum(), df["revenues_ttm"])
        df["rnd_pct_rev_ttm"] = safe_div(df.get("is_research_and_development", pd.Series()).rolling(window=4, min_periods=4).sum(), df["revenues_ttm"])
        df["capex_to_da_ttm"] = safe_div(df["capex_ttm"], df["is_depreciation_and_amortization"].rolling(window=4, min_periods=4).sum())
        df["roce_ttm"] = safe_div(df["is_operating_income_loss"].rolling(window=4, min_periods=4).sum(), df["capital_employed_avg_4q"])
        df["roa_ttm"] = safe_div(df["net_income_ttm"], df["assets_avg_4q"])
        df["inventory_turnover_ttm"] = safe_div(df["cogs_ttm"], df["inventory_avg_4q"])
        df["interest_coverage_ttm"] = safe_div(df["is_operating_income_loss"].rolling(window=4, min_periods=4).sum(), df["interest_expense_ttm"])

        return df.reset_index().rename(columns={"end_date": "end_date"})

    def build_ttm_ratios_from_financials(self, ticker: str) -> pd.DataFrame:
        """Build historical TTM ratios for the ticker."""
        t = ticker.strip().upper()
        if self.verbose:
            print(f"Starting {t}")
        # fetch fundamentals
        if self.fundamentals_dir:
            fpath = Path(self.fundamentals_dir) / f"{t}_fundamentals.parquet"
            fins = pd.read_parquet(fpath) if fpath.exists() else self.fetch_quarterly_financials(t)
        else:
            fins = self.fetch_quarterly_financials(t)
        if fins.empty:
            raise ValueError(f"No fundamentals for {t}")
        fins["end_date"] = pd.to_datetime(fins["end_date"], errors="coerce")
        fins = fins[fins["timeframe"] == "quarterly"]
        if self.min_end_date:
            fins = fins[fins["end_date"] >= pd.to_datetime(self.min_end_date)]
        if self.max_end_date:
            fins = fins[fins["end_date"] <= pd.to_datetime(self.max_end_date)]
        fins = fins.sort_values("end_date").reset_index(drop=True)
        fins = self._ensure_q4_as_annual(fins, t)

        # price alignment
        price_override = self._load_latest_local_price(t)
        if price_override is not None:
            fins["price"] = price_override
        else:
            from pandas.tseries.offsets import BDay
            price_start = (fins["end_date"].min() - BDay(1)).normalize()
            price_end = (fins["end_date"].max() + BDay(1)).normalize()
            prices = self.fetch_daily_prices(t, price_start, price_end)
            if prices.empty:
                raise ValueError(f"No price data for {t}")
            series = prices.set_index("date")["close"].sort_index()
            full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D").tz_localize(None)
            filled = series.reindex(full_idx).ffill()
            aligned = filled.reindex(fins["end_date"], method="ffill")
            fins["price"] = aligned.astype(float)

        df_metrics = self._compute_ttm_metrics(fins)
        keep = df_metrics.columns.tolist()
        # save fundamentals (raw) and metrics
        if self.fundamentals_dir:
            Path(self.fundamentals_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.fundamentals_dir)/f"{t}_fundamentals.parquet").to_parquet(fins.reset_index(drop=True))
        Path(self.ratios_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.ratios_dir)/f"{t}_ratios.parquet").to_parquet(df_metrics[keep])
        if self.verbose:
            print(f"Finished {t}, {len(df_metrics)} rows")
        return df_metrics

    def build_and_save_ttm_ratios(self, filename_template: str = "{ticker}_ratios.parquet") -> (pd.DataFrame, List[str]):
        """Batch build & save TTM ratios for all tickers."""
        all_frames: List[pd.DataFrame] = []
        failed: List[str] = []
        for t0 in self.tickers:
            t = t0.strip().upper()
            if self.verbose:
                print(f"Processing {t}")
            try:
                df_ttm = self.build_ttm_ratios_from_financials(t)
            except Exception as e:
                print(f"{t} failed: {e}")
                failed.append(t)
                continue
            if df_ttm.empty:
                failed.append(t)
                continue
            all_frames.append(df_ttm.assign(ticker=t))
        if not all_frames:
            return pd.DataFrame(), failed
        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.set_index(["ticker", "end_date"])
        return combined, failed

    def fetch_daily_prices(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp, adjusted: bool = True) -> pd.DataFrame:
        """Fetch daily closing prices via API."""
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        params = {"adjusted": "true" if adjusted else "false", "sort": "asc", "limit": 50000, "apiKey": self.api_key}
        data = self._get_json(url, params=params)
        results = data.get("results") or []
        if not results:
            return pd.DataFrame(columns=["date","close"])
        recs = [{"date": pd.to_datetime(r["t"], unit="ms", utc=True).tz_convert("US/Eastern").normalize(),
                 "close": float(r["c"])} for r in results]
        df = pd.DataFrame(recs).groupby("date", as_index=False)["close"].last().sort_values("date").reset_index(drop=True)
        return df
