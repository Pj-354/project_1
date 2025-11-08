# --- Core & Factors (broad beta + factor/ESG + active thematic rotation) ---
core_and_factors = {
    "SPY":  "SPDR S&P 500 (US large-cap core)",
    "QQQ":  "Invesco QQQ (Nasdaq-100 growth)",
    "ESGU": "iShares ESG Aware MSCI USA (ESG-screened core)",
    "IMTM": "iShares MSCI Intl Momentum Factor (developed ex-US momentum)",
    "DYNF": "BlackRock U.S. Equity Factor Rotation (active multi-factor)",
    "THRO": "iShares U.S. Thematic Rotation Active (active multi-theme)",
    'IB01': "iShares 1 year Treasury Bond ETF",
    'IBTA': "iShares 3+ Year Treasury Bond ETF",
    'LQDA': 'iShares Corporate Bonds',
    'CEMG': 'iShares Emerging Markets Consumer Growth',
}

# --- Semiconductors (note: XSD is equal-weighted; SOXX/SMH are cap-weighted) ---
semiconductors = {
    "XSD": "SPDR S&P Semiconductor (equal-weight)",
    "SOXX": "iShares Semiconductor (PHLX Semiconductor Index)",
    "SMH": "VanEck Semiconductor (MVIS US Listed Semiconductor 25)",
}

# --- Software, Cloud & AI (software/application + cloud infra + AI strategies) ---
software_cloud_ai = {
    "IGV":  "iShares Expanded Tech–Software",
    "SKYY": "First Trust Cloud Computing",
    "CLOU": "Global X Cloud Computing",
    "IGPT": "Invesco AI & Next Gen Software UCITS",
    'LOCK': 'iShares Digital Security UCITS',
    
}

# --- Innovation: Robotics / Exponential Tech / China Tech / Biotech ---
innovation_and_china = {
    "XT":   "iShares Exponential Technologies (broad innovation)",
    "ROBO": "ROBO Global Robotics & Automation",
    "KTEC": "KraneShares Hang Seng TECH (China internet/tech leaders)",
    "TCHI": "iShares China Multisector Tech UCITS",
    "IBB":  "iShares Biotechnology (biotech industry)",
    'CTCE': 'iShares MSCI China Tech UCITS',
    'HEAL': 'iShares Healthcare Innovation UCITS',
    'AIAA': 'iShares AI and Adopters UCITS',

}

# --- Clean Energy, EV & Storage Materials ---
clean_energy_ev_storage = {
    "ICLN": "iShares Global Clean Energy",
    "IBAT": "iShares Energy Storage & Materials UCITS",
    "ETEC": "iShares Breakthrough Environmental Solutions UCITS",
    "IDRV": "iShares Self-Driving EV & Tech",
    'WENS': 'iShares World Energy Sector',
    'URA':  'VanEck Uranium',
    'INRG': 'iShares Global Clean Energy UCITS',
    'ECAR': 'iShares Electric Vehicles & Driving Technology UCITS',

}

# --- Infrastructure, Manufacturing & Defense (onshoring/reshoring themes) ---
infrastructure_mfg_defense = {
    "IFRA": "iShares U.S. Infrastructure",
    "EMIF": "iShares Emerging Markets Infrastructure",
    "MADE": "iShares U.S. Manufacturing Renaissance (active)",
    "ITA":  "iShares U.S. Aerospace & Defense",
    "IETC": "iShares U.S. Tech Independence Focused",
    'AINF': 'iShares US AI Infra',
}

# --- Emerging Markets Focus ---
emerging_markets = {
    "EMGF": "iShares MSCI Emerging Markets Multifactor",
    "EMXC": "iShares MSCI Emerging Markets ex-China",
    "EWY":  "iShares MSCI South Korea",
}

# --- Communications & Market Structure (telecoms, comm services, brokers/exchanges) ---
communications_market_structure = {
    "XLC": "Communication Services Select Sector SPDR",
    "IYZ": "iShares U.S. Telecommunications",
    "IAI": "iShares U.S. Broker-Dealers & Securities Exchanges",
}

# Convenience containers
all_groups = {
    "core_and_factors": core_and_factors,
    "semiconductors": semiconductors,
    "software_cloud_ai": software_cloud_ai,
    "innovation_and_china": innovation_and_china,
    "clean_energy_ev_storage": clean_energy_ev_storage,
    "infrastructure_mfg_defense": infrastructure_mfg_defense,
    "emerging_markets": emerging_markets,
    "communications_market_structure": communications_market_structure,
}

# Optional: reverse map from ticker -> group name
ETF_TICKERS = {
    tkr: group for group, bucket in all_groups.items() for tkr in bucket.keys()
}

####################### Getting Tech Sector Tickers ###########################


import pandas as pd
from .read_in_data_functions import get_sub_sectors_tickers, SUB_SECTOR_DICT

cut_off_cap           = 10_000 
tech_stocks_marketcap = get_sub_sectors_tickers(sector='Technology', n = 60) 
tech_sectors          = list(tech_stocks_marketcap.columns.get_level_values(0).unique())

mask = (tech_stocks_marketcap[tech_stocks_marketcap.loc[:,(slice(None), 'Market Cap')] > cut_off_cap] > 0)
tech_stocks_marketcap = tech_stocks_marketcap.loc[mask.any(axis=1), :]

long = (
    tech_stocks_marketcap
    .stack(level=0, future_stack=True)                       # index: (row, sector); columns: ['Symbol','Market Cap']
    .dropna(subset=['Market Cap'])        # keep rows with a market cap
)

result = (
    long
    .groupby(level=1)                     # group by sector (the stacked level)
    .agg(avg_market_cap=('Market Cap', lambda x : x.mean().round(2)),  # average market cap per sector
        n_stocks=('Symbol', 'count'))    # counts non-null symbols
    .sort_index()
)

semi_tickers                             = tech_stocks_marketcap['semiconductors']['Symbol']
semi_equipment_tickers                   = tech_stocks_marketcap['semiconductor-equipment-and-materials']['Symbol']
software_tickers                         = tech_stocks_marketcap['software-application']['Symbol']
computer_hardware_tickers                = tech_stocks_marketcap['computer-hardware']['Symbol']
comm_equipment_tickers                   = tech_stocks_marketcap['communication-equipment']['Symbol']
software_infrastructure_tickers          = tech_stocks_marketcap['software-infrastructure']['Symbol']
electronic_components_tickers            = tech_stocks_marketcap['electronic-components']['Symbol']
scientific_technical_instruments_tickers = tech_stocks_marketcap['scientific-and-technical-instruments']['Symbol']

stock_tickers = {
            'semi' : semi_tickers,
            'semi_equipment' : semi_equipment_tickers,
            'software' : software_tickers, 'computer_hardware' : computer_hardware_tickers,
            'comm_equipment' : comm_equipment_tickers,
            'software_infrastructure' : software_infrastructure_tickers,
            'electronic_components' : electronic_components_tickers,
            'scientific_technical_instruments' : scientific_technical_instruments_tickers
        }

STOCK_TICKERS =[]
for key, val in stock_tickers.items():
    for i in val:
        STOCK_TICKERS.append(i)

temp = pd.Series(STOCK_TICKERS, index=None)
STOCK_TICKERS = temp.dropna().tolist()