# # --- Core & Factors (broad beta + factor/ESG + active thematic rotation) ---
# core_and_factors = {
#     "SPY":  "SPDR S&P 500 (US large-cap core)",
#     "QQQ":  "Invesco QQQ (Nasdaq-100 growth)",
#     "ESGU": "iShares ESG Aware MSCI USA (ESG-screened core)",
#     "IMTM": "iShares MSCI Intl Momentum Factor (developed ex-US momentum)",
#     "DYNF": "BlackRock U.S. Equity Factor Rotation (active multi-factor)",
#     "THRO": "iShares U.S. Thematic Rotation Active (active multi-theme)",
#     'IB01': "iShares 1 year Treasury Bond ETF",
#     'IBTA': "iShares 3+ Year Treasury Bond ETF",
#     'LQDA': 'iShares Corporate Bonds',
#     'CEMG': 'iShares Emerging Markets Consumer Growth',
# }

# # --- Semiconductors (note: XSD is equal-weighted; SOXX/SMH are cap-weighted) ---
# semiconductors = {
#     "XSD": "SPDR S&P Semiconductor (equal-weight)",
#     "SOXX": "iShares Semiconductor (PHLX Semiconductor Index)",
#     "SMH": "VanEck Semiconductor (MVIS US Listed Semiconductor 25)",
# }

# # --- Software, Cloud & AI (software/application + cloud infra + AI strategies) ---
# software_cloud_ai = {
#     "IGV":  "iShares Expanded Tech–Software",
#     "SKYY": "First Trust Cloud Computing",
#     "CLOU": "Global X Cloud Computing",
#     "IGPT": "Invesco AI & Next Gen Software UCITS",
#     'LOCK': 'iShares Digital Security UCITS',
    
# }

# # --- Innovation: Robotics / Exponential Tech / China Tech / Biotech ---
# innovation_and_china = {
#     "XT":   "iShares Exponential Technologies (broad innovation)",
#     "ROBO": "ROBO Global Robotics & Automation",
#     "KTEC": "KraneShares Hang Seng TECH (China internet/tech leaders)",
#     "TCHI": "iShares China Multisector Tech UCITS",
#     "IBB":  "iShares Biotechnology (biotech industry)",
#     'CTCE': 'iShares MSCI China Tech UCITS',
#     'HEAL': 'iShares Healthcare Innovation UCITS',
#     'AIAA': 'iShares AI and Adopters UCITS',

# }

# # --- Clean Energy, EV & Storage Materials ---
# clean_energy_ev_storage = {
#     "ICLN": "iShares Global Clean Energy",
#     "IBAT": "iShares Energy Storage & Materials UCITS",
#     "ETEC": "iShares Breakthrough Environmental Solutions UCITS",
#     "IDRV": "iShares Self-Driving EV & Tech",
#     'WENS': 'iShares World Energy Sector',
#     'URA':  'VanEck Uranium',
#     'INRG': 'iShares Global Clean Energy UCITS',
#     'ECAR': 'iShares Electric Vehicles & Driving Technology UCITS',

# }

# # --- Infrastructure, Manufacturing & Defense (onshoring/reshoring themes) ---
# infrastructure_mfg_defense = {
#     "IFRA": "iShares U.S. Infrastructure",
#     "EMIF": "iShares Emerging Markets Infrastructure",
#     "MADE": "iShares U.S. Manufacturing Renaissance (active)",
#     "ITA":  "iShares U.S. Aerospace & Defense",
#     "IETC": "iShares U.S. Tech Independence Focused",
#     'AINF': 'iShares US AI Infra',
# }

# # --- Emerging Markets Focus ---
# emerging_markets = {
#     "EMGF": "iShares MSCI Emerging Markets Multifactor",
#     "EMXC": "iShares MSCI Emerging Markets ex-China",
#     "EWY":  "iShares MSCI South Korea",
# }

# # --- Communications & Market Structure (telecoms, comm services, brokers/exchanges) ---
# communications_market_structure = {
#     "XLC": "Communication Services Select Sector SPDR",
#     "IYZ": "iShares U.S. Telecommunications",
#     "IAI": "iShares U.S. Broker-Dealers & Securities Exchanges",
# }

# # Convenience containers
# all_groups = {
#     "core_and_factors": core_and_factors,
#     "semiconductors": semiconductors,
#     "software_cloud_ai": software_cloud_ai,
#     "innovation_and_china": innovation_and_china,
#     "clean_energy_ev_storage": clean_energy_ev_storage,
#     "infrastructure_mfg_defense": infrastructure_mfg_defense,
#     "emerging_markets": emerging_markets,
#     "communications_market_structure": communications_market_structure,
# }

# # Optional: reverse map from ticker -> group name
# ETF_TICKERS = {
#     tkr: group for group, bucket in all_groups.items() for tkr in bucket.keys()
# }

GENERIC_TICKS = {
    # Options
    100: {"name": "Option Volume", "fields": ["callVolume", "putVolume"]},
    101: {"name": "Option Open Interest", "fields": ["callOpenInterest", "putOpenInterest"]},
    104: {"name": "Option Historical Volatility", "fields": ["histVolatility"]},
    105: {"name": "Average Option Volume", "fields": ["avOptionVolume"]},
    106: {"name": "Option Implied Volatility", "fields": ["impliedVolatility"]},

    # Index/futures and misc stats
    162: {"name": "Index Future Premium", "fields": ["indexFuturePremium"]},
    165: {"name": "13/26/52w L/H + AvgVol", "fields": [
        "low13week","high13week","low26week","high26week","low52week","high52week","avVolume"
    ]},

    # Mark price family
    232: {"name": "Mark Price (PL Price)", "fields": ["markPrice"]},
    221: {"name": "Creditman Mark Price (deprecated/not available)", "fields": []},
    619: {"name": "Creditman Slow Mark Price", "fields": []},

    # Auctions
    225: {"name": "Auction Vol/Price/Imbalance", "fields": ["auctionVolume","auctionPrice","auctionImbalance"]},

    # Time & sales / volume stats
    233: {"name": "RTVolume (Time & Sales)", "fields": ["last","lastSize","rtVolume","rtTime","vwap"]},
    375: {"name": "RT Trade Volume (excl. unreportable)", "fields": ["rtTradeVolume"]},
    293: {"name": "Trade Count", "fields": ["tradeCount"]},
    294: {"name": "Trade Rate per min", "fields": ["tradeRate"]},
    295: {"name": "Volume Rate per min", "fields": ["volumeRate"]},
    595: {"name": "Short-Term Volume 3/5/10 min", "fields": []},

    # Shortability
    236: {"name": "Shortable / Shortable Shares", "fields": ["shortableShares"]},

    # Dividends and fundamentals
    456: {"name": "IB Dividends", "fields": ["dividends"]},
    258: {"name": "Fundamental Ratios", "fields": ["fundamentalRatios"]},

    # Vol surface realtime
    411: {"name": "RT Historical Volatility (30d)", "fields": ["rtHistVolatility"]},

    # Futures OI
    588: {"name": "Futures Open Interest", "fields": ["futuresOpenInterest"]},

    # IPO indications
    586: {"name": "IPO Midpoint / Final Price", "fields": []},

    # ETF NAVs
    578: {"name": "ETF NAV Close", "fields": []},
    576: {"name": "ETF NAV Bid/Ask", "fields": []},
    577: {"name": "ETF NAV Last", "fields": []},
    614: {"name": "ETF NAV High/Low", "fields": []},
    623: {"name": "ETF NAV Frozen Last", "fields": []},

    # News
    292: {"name": "News", "fields": []},

    # RTH last
    318: {"name": "Last RTH Trade", "fields": []},
}

FUNDAMENTAL_DEF = {
  # 1) Core earnings and income quality (TTM unless noted)
  "TTMREV": "Total revenue, trailing 12 months",
  "TTMEBITD": "EBITDA, TTM",
  "TTMEBT": "Earnings before taxes, TTM",
  "TTMNIAC": "Net income available to common, TTM",
  "TTMEPSXCLX": "EPS excluding extraordinary items, TTM",
  "TTMEPSCHG": "EPS change %, TTM vs prior TTM",

  # 2) Margins and returns
  "TTMGROSMGN": "Gross margin %, TTM",
  "TTMOPMGN": "Operating margin %, TTM",
  "TTMNPMGN": "Net profit margin %, TTM",
  "TTMPTMGN": "Pretax margin %, TTM",
  "TTMROEPCT": "Return on average equity %, TTM",
  "TTMROAPCT": "Return on average assets %, TTM",
  "TTMROIPCT": "Return on investment %, TTM",
  "APTMGNPCT": "Pretax margin %, fiscal year",
  "AROAPCT": "Return on average assets %, fiscal year",

  # 3) Leverage and coverage
  "TTMINTCOV": "Interest coverage (EBIT/interest), TTM",
  "QLTD2EQ": "Long-term debt to equity, MRQ",
  "QTOTD2EQ": "Total debt to equity, MRQ",
  "NetDebt_I": "Net debt (interim): debt + preferred + minority − cash & ST investments",

  # 4) Liquidity and cash flow
  "QCURRATIO": "Current ratio, MRQ",
  "QQUICKRATI": "Quick ratio, MRQ",
  "QBVPS": "Book value per share (common equity), MRQ",
  "QTANBVPS": "Tangible book value per share, MRQ",
  "QCSHPS": "Cash per share, MRQ",
  "TTMCFSHR": "Cash flow per share, TTM",
  "TTMFCF": "Free cash flow, TTM",
  "TTMFCFSHR": "Free cash flow per share, TTM",

  # 5) Efficiency and per-employee
  "TTMINVTURN": "Inventory turnover, TTM",
  "TTMRECTURN": "Receivables turnover, TTM",
  "TTMREVPERE": "Revenue per employee, TTM",
  "TTMREVPS": "Revenue per share, TTM",

  # 6) Growth
  "REVCHNGYR": "Revenue change %, last interim vs same quarter a year ago",
  "TTMREVCHG": "Revenue change %, TTM vs prior TTM",
  "REVTRENDGR": "Revenue CAGR, 5-year",
  "EPSCHNGYR": "EPS change %, last interim vs year-ago interim",
  "EPSTRENDGR": "EPS CAGR, 5-year (ex-extraordinary items)",
  "DIVGRPCT": "Dividend per share growth rate %",
  "ADIV5YAVG": "Dividend per share 5-year average",
  "YLD5YAVG": "Dividend yield 5-year average",

  # 7) Valuation
  "PEEXCLXOR": "P/E excluding extraordinary items (TTM EPS ex-extra)",
  "APENORM": "P/E on normalized EPS (latest annual)",
  "PRICE2BK": "Price to book, MRQ",
  "PR2TANBK": "Price to tangible book",
  "TTMPR2REV": "Price to sales, TTM",
  "TTMPRCFPS": "Price to cash flow per share, TTM",
  "TTMPRFCFPS": "Price to free cash flow per share, TTM",
  "MKTCAP": "Market capitalization",

  # 8) Price level and momentum
  "NPRICE": "Last closing price",
  "NHIG": "52-week high price",
  "NLOW": "52-week low price",
  "PR1WKPCT": "1-week price change %",
  "PR4WKPCT": "4-week price change %",
  "PR13WKPCT": "13-week price change %",
  "PR52WKPCT": "52-week price change %",
  "PRYTDPCTR": "YTD price change % relative to S&P 500",
  "BETA": "Reuters 5-year beta vs local index",

  # 9) Meta, forecasts, normalizations
  "CURRENCY": "Reporting currency code",
  "LATESTADATE": "Last available date for these ratios",
  "AFEEPSNTM": "Analyst forecast EPS, next 12 months",
  "AEBTNORM": "Earnings before taxes, normalized (annual)",
  "ANIACNORM": "Net income available to common, normalized (annual)",
  "AEPSNORM": "EPS, normalized (annual)",
}

