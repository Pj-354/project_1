import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter, FuncFormatter
import numpy as np, math, textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


#### Data Handling Functions
def combine_fx_positions(positions : pd.DataFrame, fx : pd.DataFrame
    )-> pd.DataFrame:
    """ 
    Combine FX and positions csv data and adds:
        1. Net Exposure in USD
        2. Gross Exposure (absolute position size)
        3. Beta adjusted net and gross exposures
        4. Unrealized Pnl in dollar terms
        5. Portfolio weight of each position
    """
    # Map currency conversions to Positions
    positions['currency_conversion'] = positions['currency'].map(fx['to_USD'])

    # Convert cost basis and market price to USD
    positions['cost_basis_usd']   = positions['cost_basis_local'] * positions['currency_conversion']
    positions['market_price_usd'] = positions['market_price_local'] * positions['currency_conversion']

    # Get Position value : logic to handle for short and long positions
    positions['unrealized_pnl_per_share_usd'] = np.where(positions['side'] == 'SHORT',
                                                        positions['cost_basis_usd'] - positions['market_price_usd'],
                                                        positions['market_price_usd'] - positions['cost_basis_usd'])

    # Get total unrealized gains or losses per position
    positions['unrealized_pnl_usd'] = positions['unrealized_pnl_per_share_usd'] * positions['posn_shares'].abs()
    positions['unrealized_pnl_pct']  = positions['unrealized_pnl_per_share_usd'] / positions['cost_basis_usd'] * 100

    # Get total exposure per position
    positions['net_exposure_usd']                 = positions['market_price_usd'].abs() * positions['posn_shares']
    positions['gross_exposure_usd']               = positions['market_price_usd'].abs() * positions['posn_shares'].abs()
    positions['beta_weighted_exposure_usd']       = positions['beta'] * positions['net_exposure_usd']
    positions['gross_beta_weighted_exposure_usd'] = positions['beta_weighted_exposure_usd'].abs()
    
    # Calculate portfolio weight
    positions['portfolio_weight']                 = positions['gross_exposure_usd'] / positions['gross_exposure_usd'].sum()

    #Drop columns that are not needed for analysis
    cols_to_hide = ['name', 'currency_conversion', 'cost_basis_local', 'unrealized_pnl_pct','market_price_local', 'cost_basis_usd', 'unrealized_pnl_per_share_usd']

    #Keep certain columns only
    concise_positions = positions.loc[:, ~positions.columns.isin(cols_to_hide)].copy(deep=True)

    return concise_positions

def shorten_labels(sectors):
    """ 
    Shorten sectors
    """
    repl = {
        'Information Technology'    : 'IT',
        'Consumer Discretionary'    : 'Cons Disc',
        'Consumer Staples'          : 'Cons Stap',
        'Telecommunication Services': 'Tele Comm',
        'Industrials'               : 'Industrials',
        'Financials'                : 'Financials'
    }

    return [repl.get(item, item) for item in sectors]

#### Summary Functions
def build_overall_exposure_summary(
        concise_positions : pd.DataFrame
    )-> pd.DataFrame:
    """ 
    Provide a summary of the portfolio:
        1. Gross Market Value (GMV)
        2. Net Exposure 
        3. Net Exposure Ratio (as % of GMV)
        4. Portfolio Beta (weighted-avg)
        5. Gross Beta-Weighted Exposure 
        6. Net Beta-Weighted Exposure
        7. Beta Tilt (6. / 5.)
        8. Longs Market Value (LMV)
        9. Shorts Market Value (SMV)
        10. Value Weighted Average Days to Liquidation assuming 10% ADV participation
        11. Gross Exposure Unwindable in 1 Day (% of GMV)
    """
    concise_positions = days_to_liquidate(concise_positions, 0.10)
    concise_positions = bucket_by_liquidity(concise_positions, 0.10)
    overall_exposures = concise_positions[['beta_weighted_exposure_usd', 'gross_beta_weighted_exposure_usd','net_exposure_usd', 'gross_exposure_usd']].sum().to_frame('Overall Book Summary') 

    # Overall Exposures
    gross_exposure                                               = concise_positions['gross_exposure_usd'].sum()
    dollar_beta                                                  = concise_positions['beta_weighted_exposure_usd'].sum()
    gross_beta_weighted                                          = concise_positions['beta_weighted_exposure_usd'].abs().sum()
    overall_exposures.loc['Gross Beta-Weighted Exposure ($Mn)']  = gross_beta_weighted
    # Long Short Breakdown
    long_positions                                               = concise_positions.groupby('side')['gross_exposure_usd'].sum()['LONG']
    overall_exposures.loc['Longs Market Value ($Mn)', :]         = long_positions
    short_positions                                              = concise_positions.groupby('side')['gross_exposure_usd'].sum()['SHORT']
    overall_exposures.loc['Shorts Market Value ($Mn)', :]        = short_positions

    # Calculate Value Weighted Avg Position Size/ADV
    gross_exposure_unwindable_1d, value_weighted_dtl = calc_liquidity_port_summary(concise_positions, p=0.10)
    
    # Turn into $mn
    overall_exposures                                                  = overall_exposures / 1e6

    # Assign back to output df
    overall_exposures.loc['Value Weighted Avg Position Size/ADV']      = (concise_positions['portfolio_weight'] * concise_positions['position_pct_of_adv']).sum()
    overall_exposures.loc['Net Exposure Ratio (%)', :]                 = (long_positions - short_positions) / gross_exposure * 100
    overall_exposures.loc['Portfolio Beta', :]                         = dollar_beta / gross_exposure
    overall_exposures.loc['Beta Tilt (%)', :]                          = dollar_beta / gross_beta_weighted * 100
    overall_exposures.loc['(Value-Weighted) Avg Days to Liquidate', :] = value_weighted_dtl
    overall_exposures.loc['Gross Exposure Unwindable 1D (%)']          = gross_exposure_unwindable_1d / gross_exposure * 100

    overall_exposures.rename(index={'beta_weighted_exposure_usd'     : 'Net Beta-Weighted Exposure ($Mn)',
                                    'net_exposure_usd'               : 'Net Exposure ($Mn)',
                                    'gross_exposure_usd'             : 'Gross Market Value ($Mn)',
                                },
                            inplace=True
                            )

    display_order = ['Net Exposure ($Mn)',
                    'Gross Market Value ($Mn)',
                    'Net Beta-Weighted Exposure ($Mn)',
                    'Gross Beta-Weighted Exposure ($Mn)',
                    '(Value-Weighted) Avg Days to Liquidate',
                    'Longs Market Value ($Mn)',
                    'Shorts Market Value ($Mn)',
                    'Net Exposure Ratio (%)',
                    'Beta Tilt (%)',
                    'Portfolio Beta',
                    'Gross Exposure Unwindable 1D (%)',
                    ]

    return overall_exposures.loc[display_order]

def plot_top_longs(
    df: pd.DataFrame,
    value_col: str = "net_exposure_usd",
    tilt_col: str = "net_tilt_pct",
    unit: str = "M",             
    title: str = "Top 3 Longs \nNet Exposure & Net Tilt",
    tilt_is_pct: bool = True,
    color = 'green',
    xlabel = 'Net Exposure',
    figsize = (6,3)
):
    """
    Plot top-N positive exposures (longs) as a horizontal bar chart.
    X-axis shows dollars (with $ formatting). Each bar is annotated with net tilt %.
    """
    # Keep required columns and ensure numeric
    work = df[[value_col, tilt_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work[tilt_col]  = pd.to_numeric(work[tilt_col],  errors="coerce")

    # Scale for display
    if unit.upper() == "M":
        plot_vals = work[value_col] / 1e6
        xfmt = FuncFormatter(lambda x, pos: f"${x:,.1f}M")
        xlabel = f"{xlabel} ($Mn)"
    elif unit.upper() == "B":
        plot_vals = work[value_col] / 1e9
        xfmt = FuncFormatter(lambda x, pos: f"${x:,.1f}B")
        xlabel = f"{xlabel} ($Bn)"
    else:
        plot_vals = work[value_col]
        xfmt = FuncFormatter(lambda x, pos: f"${x:,.0f}")
        xlabel = f"{xlabel} ($)"

    labels = work.index.astype(str)
    labels = shorten_labels(labels)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, plot_vals.values, color=color)
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.axvline(0, linewidth=1)
    ax.set_xlim(left=min(0, plot_vals.min() * 1.1), right=plot_vals.max() * 1.2)
    ax.invert_yaxis() 
    ax.grid()

    # Annotate each bar with net tilt %
    for bar, tilt in zip(bars, work[tilt_col].values):
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        tilt_pct = tilt if tilt_is_pct else (tilt * 100.0)
        ax.text(width, y, f"  {tilt_pct:.2f}%", va="center", ha="left")

    fig.tight_layout()
    return fig, ax

def plot_top_shorts(
    df: pd.DataFrame,
    value_col: str = "net_exposure_usd",
    tilt_col: str = "net_tilt_pct",
    unit: str = "M",             
    title: str = "Top 3 Shorts \n Net Exposure & Net Tilt",
    tilt_is_pct: bool = True,
    color = 'red',
    figsize = None   
):
    """
    Plot top-N short exposures (most negative) as a horizontal bar chart.
    X-axis shows dollars; each bar is annotated with the net tilt %.
    """
    work = df[[value_col, tilt_col]].copy()
    
    if unit.upper() == "M":
        scaler = 1e6
        xfmt = FuncFormatter(lambda x, pos: f"${x:,.1f}M")
        xlabel = "Net Exposure ($ mn)"
    elif unit.upper() == "B":
        scaler = 1e9
        xfmt = FuncFormatter(lambda x, pos: f"${x:,.1f}B")
        xlabel = "Net Exposure ($ bn)"
    else:
        scaler = 1.0
        xfmt = FuncFormatter(lambda x, pos: f"${x:,.0f}")
        xlabel = "Net Exposure ($)"

    plot_vals = (work[value_col]) / scaler

    labels = shorten_labels(work.index.astype(str))  

    fig, ax = plt.subplots(figsize=(6.0, 1.0 + 0.8 * len(work))) if figsize is None else plt.subplots(figsize=figsize)
    bars = ax.barh(labels, plot_vals.values, color=color)
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.axvline(0, linewidth=1)
    ax.invert_yaxis() 
    ax.grid()
    ax.set_xlim(left=min(0, plot_vals.min() * 1.15), right=0)
    # annotate net tilt %
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    pad = 0.01 * span
    for bar, tilt in zip(bars, work[tilt_col].values):
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        tilt_pct = tilt if tilt_is_pct else (tilt * 100.0)
        if width >= 0:
            ax.text(width + pad, y, f"{tilt_pct:.2f}%", va="center", ha="left")
        else:
            ax.text(width - pad, y, f"{tilt_pct:.2f}%", va="center", ha="right")

    return fig, ax

#### Sector Analysis Functions

def calc_sector_allocations(
        concise_positions
    )-> tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Calculate the allocation for sectors :
        - gross market value of a sector / gross market value of portfolio
    
    Calculate the sector tilt :
        - net exposure of a sector / gross market value of portfolio
    """
    # Calculate sector weights
    sector_allocations = concise_positions.groupby('sector')[['gross_exposure_usd']].sum()
    
    sector_allocations['sector_weight'] = (sector_allocations['gross_exposure_usd']
                                         / sector_allocations['gross_exposure_usd']
                                         .sum()) * 100
    
    sector_allocations = sector_allocations.sort_values(by='sector_weight', ascending=False)

    # Calculate net tilt
    sector_exposures = concise_positions.groupby('sector')[['net_exposure_usd', 'gross_exposure_usd']].sum()
    sector_exposures['net_tilt_pct'] =    ( sector_exposures['net_exposure_usd']
                                          / sector_exposures['gross_exposure_usd'].sum()
                                          * 100)

    sector_exposures = sector_exposures.sort_values('net_tilt_pct', ascending=False)

    return sector_allocations, sector_exposures

def calc_long_short_exposure_per_sector(concise_positions):
    """ 
    Break down each sector into long or short exposures, calculates :
    1. Net Exposure within Sector as a % (Tilt)
    """
    l_s_per_sector     =(  concise_positions.groupby(['side', 'sector'])[['gross_exposure_usd']]
                                            .sum()
                         ) 
    l_s_per_sector = l_s_per_sector.unstack()
    l_s_per_sector.loc['GROSS'] = l_s_per_sector.loc['LONG', :] + l_s_per_sector.loc['SHORT', :]
    l_s_per_sector.loc['% Long'] = l_s_per_sector.loc['LONG',:] / l_s_per_sector.loc['GROSS'] * 100
    l_s_per_sector.loc['% Short'] = l_s_per_sector.loc['SHORT',:] / l_s_per_sector.loc['GROSS'] * 100
    l_s_per_sector.loc['directional_sector_tilt_pct'] = l_s_per_sector.loc['% Long'] - l_s_per_sector.loc['% Short']

    net_exposures = l_s_per_sector.loc['directional_sector_tilt_pct'].sort_values(ascending=False)
    net_exposures = net_exposures.droplevel(0)

    return net_exposures, l_s_per_sector

def get_sector_metrics(l_s_by_sector,
                       sector_exposures
)->pd.DataFrame:
        """ 
        Combines long/short exposure by sector with sector exposures to get key metrics per sector
        """
        return (l_s_by_sector.T.reset_index()
                               .merge(sector_exposures.reset_index(),
                                      left_on='sector', right_on='sector'
                              ).set_index('sector')
                               .drop(columns='level_0')
                )

#### Beta/Market Factor Analysis Functions
def calc_beta_weighted_sector_allocations(concise_positions
)-> tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Calculate the beta weighted allocation for sectors :
        - gross beta-weighted exposure of a sector / gross beta-weighted exposure of portfolio
    
    Calculate the sector tilt :
        - net beta-weighted exposure of a sector / gross beta-weighted exposure of portfolio
    """
    concise_positions['gross_beta_exposure_usd'] = concise_positions['beta_weighted_exposure_usd'].abs()

    beta_weighted_sector_allocations = concise_positions.groupby('sector')[['beta_weighted_exposure_usd', 'gross_beta_exposure_usd']].sum()

    beta_weighted_sector_allocations['sector_weight'] = (beta_weighted_sector_allocations['gross_beta_exposure_usd'] / concise_positions['gross_beta_exposure_usd'].sum()) * 100
    beta_weighted_sector_allocations = beta_weighted_sector_allocations.sort_values(by='sector_weight', ascending=False)

    
    beta_weighted_sector_allocations['net_tilt_pct']  =  beta_weighted_sector_allocations['beta_weighted_exposure_usd'] / concise_positions['gross_beta_exposure_usd'].sum() * 100

    beta_weighted_sector_exposures = beta_weighted_sector_allocations.sort_values('net_tilt_pct', ascending=False)

    return beta_weighted_sector_allocations, beta_weighted_sector_exposures

def calc_l_s_beta_weighted_sector_exposures(concise_positions):
    """ 
    Break down each sector into long or short exposures beta weighted, calculates :
    1. L / S breakdown by $Mn and % of sector 
    2. Sector bias
    3. Sector tilt
    """
    beta_l_s_per_sector         = concise_positions.groupby(['side', 'sector'])[['beta_weighted_exposure_usd']].apply(lambda x : x.abs().sum())
    beta_l_s_per_sector         = beta_l_s_per_sector.unstack()

    beta_l_s_per_sector.loc['GROSS']            = beta_l_s_per_sector.loc['LONG', :] + beta_l_s_per_sector.loc['SHORT', :]
    beta_l_s_per_sector.loc['% Long']           = beta_l_s_per_sector.loc['LONG',:] / beta_l_s_per_sector.loc['GROSS'] * 100
    beta_l_s_per_sector.loc['% Short']          = beta_l_s_per_sector.loc['SHORT',:] / beta_l_s_per_sector.loc['GROSS'] * 100
    beta_l_s_per_sector.loc['beta_sector_bias'] = beta_l_s_per_sector.loc['% Long'] - beta_l_s_per_sector.loc['% Short']
    beta_l_s_per_sector.loc['beta_net_tilt_pct'] = (beta_l_s_per_sector.loc['LONG'] - beta_l_s_per_sector.loc['SHORT']) / concise_positions['beta_weighted_exposure_usd'].abs().sum() * 100

    net_beta_weighted_exposures = beta_l_s_per_sector.loc['beta_net_tilt_pct'].sort_values(ascending=False)
    net_beta_weighted_exposures = net_beta_weighted_exposures.droplevel(0)

    return net_beta_weighted_exposures, beta_l_s_per_sector 

def calc_residual_mkt_exposure_by_sector(concise_positions):
    """
    Calculate the residual market exposure by sector in
        - dollar terms (millions)
        - % of gross beta weighted exposure
    """
    gross_market_exposure           = concise_positions['beta_weighted_exposure_usd'].abs().sum()
    residual_mkt_exposure_by_sector = concise_positions.groupby('sector')[['beta_weighted_exposure_usd']].sum() 

    # Express as % of gross beta weighted market exposure
    residual_mkt_exposure_pct_of_gross_mkt_exposure = (concise_positions.groupby('sector')[
                                                            ['beta_weighted_exposure_usd']
                                                            ].sum()
                                                            / gross_market_exposure 
                                                            * 100
                                                          )

    residual_mkt_exposure_by_sector['beta_tilt_pct'] = residual_mkt_exposure_pct_of_gross_mkt_exposure

    residual_mkt_exposure_by_sector = residual_mkt_exposure_by_sector.sort_values('beta_weighted_exposure_usd', ascending=False)
    return residual_mkt_exposure_by_sector

def breakdown_by_industry(concise_positions, figsize):
    """
    Breakdown of our positions by industry within each sector

    Plots a donut chart for each sector with multiple industries showing
    the allocation to each industry within that sector
    """

    industry_breakdown =  (
        concise_positions.groupby(['sector','industry'])[ # Group by each industry 
                        ['net_exposure_usd',
                        'beta_weighted_exposure_usd',
                        'gross_exposure_usd',]
                        ].sum()
                            .sort_values(['sector', 'industry', 'net_exposure_usd'],
                                        ascending=[True, False, False]
                            )
        ) / 1e6

    # How directional we are in each industry as a % of the absolate value of our positions in that industry
    industry_breakdown['% net exposure']         = industry_breakdown['net_exposure_usd'] / industry_breakdown['gross_exposure_usd'] * 100

    temp_df            = concise_positions.groupby(['sector'])[['gross_exposure_usd']].sum() / 1e6
    industry_breakdown = industry_breakdown.reset_index().merge(temp_df, on='sector', suffixes=('', '_sector_total'), how='left')

    # Other metric : Calculates the contribution of the industry to our sector's net exposure 
    industry_breakdown['Industry allocation % of sector total'] = industry_breakdown['gross_exposure_usd'] / industry_breakdown['gross_exposure_usd_sector_total'] * 100

    industry_breakdown.set_index(['sector','industry'], inplace=True)
    industry_breakdown = industry_breakdown[['net_exposure_usd', 'beta_weighted_exposure_usd', 'gross_exposure_usd', '% net exposure', 'Industry allocation % of sector total']]
    industry_breakdown.sort_values(by=['sector','industry', 'Industry allocation % of sector total'], ascending=[True, False, False], inplace=True)

                                      
    def short_label(s, width=20):
        repl = {
            'Pharmaceuticals & Biotechnology': 'Pharma & Bio',
            'Semiconductors & Semiconductor Equipment': 'Semis & Equip',
            'Food Beverage & Tobacco': 'F&B & Tobacco',
            'Household & Personal Products': 'Household & Personal',
            'Commercial Services & Supplies': 'Commercial Services',
            'Hotels Restaurants & Leisure': 'Hotels & Leisure',
        }
        s = repl.get(s, s)
        return textwrap.shorten(s, width=width, placeholder='…')

    sectors = temp_df.sort_values(by='gross_exposure_usd', ascending=False).index
    n = len(sectors); ncols = 3; nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = np.array(axs).ravel()

    for i, sector in enumerate(sectors):
        df   = industry_breakdown.loc[sector]
        vals = df['Industry allocation % of sector total'].values
        labs = [short_label(x) for x in df.index]
        ax   = axs[i]

        wedges, texts, autotexts = ax.pie(
            vals, startangle=90, counterclock=False, labels=None,
            autopct=lambda p: f'{p:.0f}%' if p >= 2.5 else '',  # remove the labels jic there's many small industries
            pctdistance=0.78,
            wedgeprops=dict(width=0.5, edgecolor='white')      # donut style
        )
        ax.set_title(f'{sector} : ${temp_df.loc[sector, "gross_exposure_usd"].round(2)} Mn', fontsize=12, pad = 5)
        ax.set_aspect('equal')
        ax.legend(wedges, labs, fontsize=8, frameon=False,
                loc='center left', bbox_to_anchor=(1.0, 0.5))
    # hide unused axes
    for j in range(i+1, len(axs)):
        axs[j].axis('off')
    return industry_breakdown

#### Currency and Country Analysis Functions
def calc_currency_positions(concise_positions):
    currency_positions = (concise_positions.groupby('currency')
                                [['net_exposure_usd', 'gross_exposure_usd']]
                                .sum() 
                            / 1e6)

    currency_positions['curr_bias'] = 100 * currency_positions['net_exposure_usd'] / currency_positions['gross_exposure_usd']
    currency_positions['net_tilt_pct'] = 100 * currency_positions['net_exposure_usd'] / currency_positions['gross_exposure_usd'].sum()
    currency_positions['curr_allocation'] = 100 * currency_positions['gross_exposure_usd'] / currency_positions['gross_exposure_usd'].sum()
    # Also calculate how much we have allocated to each currency


    currency_positions = currency_positions.sort_values(by='net_tilt_pct', ascending=False)

    return currency_positions

def build_country_allocation_summary(concise_positions):
    """ 
    Calculate the allocation for countries :
        - gross market value of a country / gross market value of portfolio

    And the beta weighted exposure contribution of countries :
        - gross beta weighted exposure of a country / gross beta weighted market value of portfolio
    """

    country_allocations = concise_positions.groupby('country')[['gross_exposure_usd']].sum()
    country_allocations['gross_country_allocation_pct'] = (country_allocations['gross_exposure_usd']
                                                         / country_allocations['gross_exposure_usd'].sum()) * 100
    
    # Calculate beta weighted exposure

    country_beta_exposure = concise_positions.groupby('country')[['beta_weighted_exposure_usd']].apply(lambda x : x.abs().sum())
    country_beta_exposure['beta_weighted_exposure_pct'] = (country_beta_exposure['beta_weighted_exposure_usd']
                                                         / country_beta_exposure['beta_weighted_exposure_usd'].sum() * 100)
    return country_allocations, country_beta_exposure

def build_country_tilt(
        concise_positions
    )-> tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Calculate:
    - country tilt : net exposure of a country / gross exposure of a country
    - country beta weighted tilt : beta weighted exposure of a country / gross beta weighted exposure of a country
    """ 
    # Country tilt (net/gross)
    country_tilt_pct            = (concise_positions.groupby('country')
                                   [['net_exposure_usd', 'gross_exposure_usd']]
                                    .sum()
                                    .sort_values(by='net_exposure_usd', ascending=False))
    
    country_tilt_pct['country_bias'] = country_tilt_pct['net_exposure_usd'] / country_tilt_pct['gross_exposure_usd'] * 100

    gmv = country_tilt_pct['gross_exposure_usd'].sum()
    beta_gmv = concise_positions['beta_weighted_exposure_usd'].abs().sum()

    country_tilt_pct['country_tilt_pct'] = country_tilt_pct['net_exposure_usd'] / gmv * 100


    # Beta weighted country tilt (beta weighted/gross beta weighted)
    country_gross_beta_weighted_exposure  = (concise_positions.groupby('country')
                                                    [['beta_weighted_exposure_usd']]
                                                    .apply(lambda x : x.abs().sum())
                                                )      
    country_beta_weighted_tilt_pct        = (concise_positions.groupby('country')
                                                    [['beta_weighted_exposure_usd']]   
                                                    .sum()
                                                )
    country_beta_weighted_tilt_pct['country_gross_beta_weighted_exposure'] = country_gross_beta_weighted_exposure

    country_beta_weighted_tilt_pct['country_beta_bias'] =   (country_beta_weighted_tilt_pct['beta_weighted_exposure_usd'] 
                                                            / country_beta_weighted_tilt_pct['country_gross_beta_weighted_exposure']
                                                                    * 100)
    country_beta_weighted_tilt_pct['country_beta_tilt_pct'] = (country_beta_weighted_tilt_pct['beta_weighted_exposure_usd'] / beta_gmv * 100)
    return country_tilt_pct, country_beta_weighted_tilt_pct

def build_long_short_exposure_per_country(concise_positions):
    """ 
    Break down each country into long or short exposures, calculates :
        1. Long / Short position size and % of gross exposure within country as a % (Tilt)
    """
    l_s_per_country     = (concise_positions.groupby(['side', 'country'])[['gross_exposure_usd']]
                                            .sum()
                                ) 
    l_s_per_country     = l_s_per_country.unstack()

    l_s_per_country.loc['GROSS']    = l_s_per_country.loc['LONG', :] + l_s_per_country.loc['SHORT', :]
    l_s_per_country.loc['% Long']   = l_s_per_country.loc['LONG',:] / l_s_per_country.loc['GROSS'] * 100
    l_s_per_country.loc['% Short']  = l_s_per_country.loc['SHORT',:] / l_s_per_country.loc['GROSS'] * 100

    l_s_per_country.loc['country_bias'] = l_s_per_country.loc['% Long'] - l_s_per_country.loc['% Short']

    return l_s_per_country

def build_long_short_beta_exposure_per_country(concise_positions):
    """ 
    Break down each country into long or short exposures, calculates :
        1. Beta Weighted position size and % of gross exposure within country as a % (Beta Weighted Tilt)
    """
    beta_w_l_s_per_country     = concise_positions.groupby(['side', 'country'])[['beta_weighted_exposure_usd']].sum() 
    beta_w_l_s_per_country     = beta_w_l_s_per_country.unstack()

    beta_w_l_s_per_country.loc['GROSS']    = beta_w_l_s_per_country.loc['LONG', :] + beta_w_l_s_per_country.loc['SHORT', :].abs()
    beta_w_l_s_per_country.loc['% Long']   = beta_w_l_s_per_country.loc['LONG',:] / beta_w_l_s_per_country.loc['GROSS'] * 100
    beta_w_l_s_per_country.loc['% Short']  = beta_w_l_s_per_country.loc['SHORT',:] / beta_w_l_s_per_country.loc['GROSS'] * 100

    beta_w_l_s_per_country.loc['beta_weighted_bias'] = beta_w_l_s_per_country.loc['% Long'] + beta_w_l_s_per_country.loc['% Short']
    
    return beta_w_l_s_per_country

def calc_country_tilt(concise_positions):
    """
    Calculate the:
        1. Country tilt : net exposure as % of gross exposure allocated to each country
        2. Allocation to each country : Gross exposure to each country as % of total gross exposure
        3. Total Long vs Short exposure to each country : Long and Short exposure to each country 

    And for foreign market exposures:
        1. Foreign Beta Tilt : Beta weighted net exposure to each country's market as % of beta weighted gross exposure
        2. Allocation Foreign Market Exposures : Beta weighted exposure to each country's market as % of beta weighted gross country exposure
        3. Total long/short market exposures of each country
    """
    # 1) How much we have allocated to each country as a % of gross exposure
    country_tilt_pct, country_beta_weighted_tilt_pct = build_country_tilt(concise_positions)

    # 2) How much of our total gross exposure and beta weighted gross exposure is in each country
    country_allocations, country_beta_exposure = build_country_allocation_summary(concise_positions)

    # 3) Long short breakdown of each country 
    l_s_per_country,beta_w_l_s_per_country = build_long_short_exposure_per_country(concise_positions), build_long_short_beta_exposure_per_country(concise_positions)


    return country_tilt_pct, country_beta_weighted_tilt_pct, country_allocations, country_beta_exposure, l_s_per_country, beta_w_l_s_per_country

def get_country_sector_split(concise_positions: pd.DataFrame):
    """
    For each sector, get the breakdown of gross exposure by country
    and what % of the sector allocation that represents
    """
    sector_allocation = calc_sector_allocations(concise_positions)[0]

    country_sector_split = (concise_positions.groupby(['sector', 'country'])[['gross_exposure_usd']]
                                             .sum().reset_index().merge(
                                                    sector_allocation[['gross_exposure_usd']].reset_index(),
                                                    left_on='sector', right_on='sector',
                                                    suffixes=('', '_sector_total')
                                        ).set_index(['sector', 'country']
                                    ).sort_index()
    )

    country_sector_split['country_pct'] = country_sector_split['gross_exposure_usd'] / country_sector_split['gross_exposure_usd_sector_total'] * 100
    
    return country_sector_split

def plot_country_allocation_by_sector(concise_positions, thresh = 10):
        """ 
        Break down each sector by the countries and show only with exposure of X %
        """
        country_allocations = calc_country_tilt(concise_positions)[2]
        sector_allocation = calc_sector_allocations(concise_positions)[0]
        country_sector_split = get_country_sector_split(concise_positions)
        cmap = plt.get_cmap('tab20')                # pick any palette you like
        palette = [cmap(i % cmap.N) for i in range(len(country_allocations.index))]
        COLOR_OF = dict(zip(country_allocations.index, palette))

        sectors = sector_allocation.index
        n = len(sectors); ncols = 5; nrows = math.ceil(n / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, 4*nrows))
        axs = np.array(axs).ravel()

        for i, sector in enumerate(sectors):
                country_pct = sector_allocation.loc[sector]['sector_weight'].round(2) 
                df   = country_sector_split.loc[sector].sort_values(by='country_pct', ascending=False)
                vals = df['country_pct'].values
                labs = df.index
                ax   = axs[i]

                colors = [COLOR_OF[c] for c in labs]

                wedges, texts, autotexts = ax.pie(
                        vals, startangle=90, counterclock=False, labels=None,
                        autopct=lambda p: f'{p:.0f}%' if p >= thresh else '',  # remove the labels jic there's many small industries
                        pctdistance=0.78, colors = colors,
                        wedgeprops=dict(width=0.5, edgecolor='white')      # donut style
                        )
                
                ax.set_title(f'{sector}\n{country_pct}%', fontsize=10, pad = 10)
                ax.set_aspect('equal')
                # hide unused axes
                for j in range(i+1, len(axs)):
                        axs[j].axis('off')


        plt.suptitle('Country Allocation by Sector\n', fontsize=16, y=0.92)

        handles = [Patch(facecolor=COLOR_OF[c], edgecolor='white', label=c) for c in country_allocations.index]
        fig.legend(handles=handles, title='Country', loc='center left', frameon=False)

#### Sub industry Analysis Functions
def get_sub_industry_breakdown(
        concise_positions,
    )-> tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Compute metrics for each sub-industry
    1) net exposure ratio % (net exposure / gross exposure * 100)
    2) gross market risk of sub-industry % (beta weighted exposure / gross beta
    """
    # Group data by industry and get dollar and beta terms
    gross_exposure, gross_exposure_beta_weighted = concise_positions['gross_exposure_usd'].abs().sum(), concise_positions['beta_weighted_exposure_usd'].abs().sum()
    
    df = concise_positions.copy()
    df['gross_beta_w_exposure_usd'] = df['beta_weighted_exposure_usd'].abs()

    by_sub_ind = df.groupby(['sector','industry', 'sub_industry'])[['net_exposure_usd', 'gross_exposure_usd', 'beta_weighted_exposure_usd', 'gross_beta_w_exposure_usd']].sum()

    # Calculate the sub industry net exposure ratio % 
    by_sub_ind['net_bias_pct']                      = by_sub_ind['net_exposure_usd'] / by_sub_ind['gross_exposure_usd'] * 100
    # Calculate gross market risk of sub-industry %
    by_sub_ind['net_beta_bias_pct']                 = by_sub_ind['beta_weighted_exposure_usd'] / by_sub_ind['gross_beta_w_exposure_usd'] * 100

    # Calculate how much is allocated to that sector as a % of the portfolio
    by_sub_ind['portfolio_weight']                  = by_sub_ind['gross_exposure_usd'] / gross_exposure * 100
    by_sub_ind['portfolio_weighted_beta']           = by_sub_ind['beta_weighted_exposure_usd'] / gross_exposure_beta_weighted * 100

    # Calc net tilts
    by_sub_ind['net_tilt_pct']                      = by_sub_ind['net_exposure_usd'] / gross_exposure.sum() * 100
    by_sub_ind['beta_net_tilt_pct']                 = by_sub_ind['beta_weighted_exposure_usd'] / gross_exposure_beta_weighted.sum() * 100
    # Express in millions
    by_sub_ind[['net_exposure_usd', 'gross_exposure_usd', 'beta_weighted_exposure_usd', 'gross_beta_w_exposure_usd']] = by_sub_ind[['net_exposure_usd', 'gross_exposure_usd', 'beta_weighted_exposure_usd', 'gross_beta_w_exposure_usd']] / 1e6

    beta_df_sub_ind = by_sub_ind[by_sub_ind.columns[by_sub_ind.columns.str.contains('beta')]].copy()
    df_sub_ind = by_sub_ind[list(set(by_sub_ind.columns) - set(beta_df_sub_ind.columns))].copy()
    # Find notable industry exposures (by direction and magnitude)
    return df_sub_ind, beta_df_sub_ind
     
def get_sub_industry_allocation(concise_positions):
    """ 
    Calculate for each sub industry: 
    1) the individual gross exposure as a % of the overall portfolio's gross exposure 
    2) the running sum of the above 
    """
    sub_gross_exposure                  = (concise_positions.groupby('sub_industry')[['gross_exposure_usd']].sum().sort_values(by='gross_exposure_usd', ascending=False) / 1e6)
    sub_gross_exposure['cumsum_GE_usd'] = sub_gross_exposure['gross_exposure_usd'].cumsum()
    sub_gross_exposure['cumsum_GE_pct'] = (sub_gross_exposure['cumsum_GE_usd'] / sub_gross_exposure['gross_exposure_usd'].sum()) * 100

    return sub_gross_exposure

def get_sub_industry_tilts(concise_positions, beta_weighted = False):
    """ 
    Calculate for each sub industry: 
    1) the net exposure ratio % (net exposure / gross exposure * 100)
    2) the net exposure in dollar terms ($Mn)
    """
    select_df = 1 if beta_weighted else 0
    b_df2   = get_sub_industry_breakdown(concise_positions)[select_df].sort_values(by='net_exposure_usd', key=lambda x: x.abs(), ascending=False)

    sub_ind_net                      = (concise_positions.groupby('sub_industry')[['net_exposure_usd', 'gross_exposure_usd']].sum())
    sub_ind_net['industry_tilt_pct'] = sub_ind_net['net_exposure_usd'] / sub_ind_net['gross_exposure_usd'] * 100

    sub_ind_net.sort_values(by='net_exposure_usd',
                            key=lambda x: x.abs(),
                            ascending=False,
                            inplace=True)
    ne_sub_ind_abs = sub_ind_net.loc[b_df2.droplevel(['sector', 'industry']).index]['industry_tilt_pct'].abs()

    return ne_sub_ind_abs

#### liquidity analysis functions
def calc_liquidity_port_summary(
        concise_positions, p=0.10
    ) -> tuple[float, float]:
    """
    Calculate liquidity portfolio summary for positions
        - Average days to liquidity of positions (value weighted)
        - % of Gross Exposure that can be liquidated in 1 day at 10% ADV
    """
    
    # Filter out unnecessary columns
    liq_df = concise_positions.copy()
    # Shares you can execute at p% rate : ADV_shares * 10% * price_of_1_share
    executable_1day_usd         = liq_df['avg_daily_volume'] * p * liq_df['market_price_usd']
    # Days to liquidation at 10% : $notional position size / number of shares you can do in a day
    liq_df.loc[:, 'dtl_p10']    = liq_df.loc[:,'gross_exposure_usd'] / executable_1day_usd

    liq_df['entire_pos_liq'] = np.where(liq_df['gross_exposure_usd'] <= executable_1day_usd,
                                        liq_df['gross_exposure_usd'],
                                        executable_1day_usd)

    value_weighted_dtl = (liq_df['dtl_p10'] * liq_df['portfolio_weight']).sum()

    gross_exposure_unwindable_1d = liq_df['entire_pos_liq'].sum()

    return gross_exposure_unwindable_1d, value_weighted_dtl

def days_to_liquidate(concise_positions, p : float):
    """ 
    Computes how many days to liquidate at different 
    rates of liquidation (parameter p)

    e.g., if p is 10% (participation rate) then it would
    take us 5 days if we hold 50% of the ADV
    """
    
    # How many shares we can liquidate in a day at p rate
    shares_per_day                        = concise_positions['avg_daily_volume'] * p
    concise_positions['posn_shares']      = concise_positions['posn_shares'].copy()
    
    # How much of our position we can liquidate in a day
    concise_positions['position_pct_of_adv'] = concise_positions['posn_shares'].abs() * 100 / concise_positions['avg_daily_volume'] 


    concise_positions[f'days_to_liquidate_{int(p * 100)}_pct'] = (concise_positions['posn_shares'].abs() / shares_per_day).replace(np.inf, np.nan).round(2)

    # If ADV < 2 shares and position < 3 shares, then assume can liquidate in 1 day
    concise_positions[f'days_to_liquidate_{int(p * 100)}_pct'] = np.where((concise_positions['avg_daily_volume'] < 2) & (concise_positions['posn_shares'].abs() < 3),
                                                                         10, concise_positions[f'days_to_liquidate_{int(p * 100)}_pct'])
    return concise_positions

def bucket_by_liquidity(concise_positions, p : float):
    """ 
    Create buckets based on participation rate used
    """
    p = int(p * 100)
    bins = [-np.inf, 0.2, 1, 3, 10, np.inf]
    labels = ['<4 h','~1d','1–3d','3–10d','>10d']

    concise_positions['dtl_bucket_10_pct'] = pd.cut(concise_positions[f'days_to_liquidate_{p}_pct'], bins=bins, labels=labels)

    return concise_positions

def plot_overall_liquidity_profile(concise_positions, p = 0.10):
        """ 
        Show the general liquidity profile of our book in a pie chart
        """
        p = int(100 *p)

        by_liq_bucket = (concise_positions.groupby([f'dtl_bucket_{p}_pct'], observed=False)[['gross_exposure_usd']].sum().unstack() / 1e6)
        by_liq_bucket = by_liq_bucket.droplevel(0)


        plt.pie(by_liq_bucket, labels=[f'{index}\n\n${val}Mn' for val,index in zip(by_liq_bucket.round(-1), by_liq_bucket.index)],
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor': 'white', 'width' : 0.75}
                )
        plt.title(f'Liquidity Breakdown \nby Days to Liquidate\n({p}% Participation Rate)')
        plt.tight_layout()
        plt.show();
        return by_liq_bucket   

def plot_by_liq_bucket(concise_positions,
                       col : str, p = 0.10,
                       exposure_type : str = 'gross_exposure_usd',
                       sharex = False
    ) -> pd.DataFrame:
    """ 
    Two bar charts to visualise how much illiquidity exposure we have in each sector
    1) Normalized by sector : what % of each sector is in each liquidity bucket
    2) Absolute $Mn exposure in each sector by liquidity bucket
    """
    p = int(p * 100)
    dtl_bucket_breakdown = concise_positions.pivot_table(
                                    index = f'dtl_bucket_{p}_pct',
                                    values=exposure_type,
                                    columns = col,
                                    aggfunc = ['sum'],
                                    margins = True
                                )

    dtl_bucket_breakdown.columns = dtl_bucket_breakdown.columns.droplevel(0)
    dtl_bucket_breakdown_norm = dtl_bucket_breakdown / dtl_bucket_breakdown.loc['All'] * 100
    dtl_bucket_breakdown_norm.drop(['All'], axis=0, inplace=True)
    dtl_bucket_breakdown_norm.drop(['All'], axis=1, inplace=True)

    dtl_bucket_breakdown = dtl_bucket_breakdown / 1e6
    dtl_bucket_breakdown.drop(['All'], axis=0, inplace=True)
    dtl_bucket_breakdown.drop(['All'], axis=1, inplace=True)

    fig, axs = plt.subplots(2, 1, figsize = (12,10), sharex = sharex)

    dtl_bucket_breakdown_norm.T.plot(
        kind = 'bar',
        stacked = True,
        ax = axs[0],
    )

    axs[0].set_ylabel('% of Positions')
    axs[0].set_title(f'{exposure_type} by {col} and Liquidation Bucket (Normalized)', pad=20)
    axs[0].yaxis.set_major_formatter(PercentFormatter(100))
    axs[0].grid()
    axs[0].legend(loc='lower right')
    dtl_bucket_breakdown.T.plot(
        kind = 'bar',
        stacked = True,
        ax = axs[1],
    )
    if sharex == False:
        axs[0].set_xticklabels(shorten_labels(dtl_bucket_breakdown_norm.columns), rotation = 0)
        axs[1].set_xticklabels(shorten_labels(dtl_bucket_breakdown.columns), rotation = 0)    
    else:
        axs[1].set_xticklabels(shorten_labels(dtl_bucket_breakdown.columns), rotation = 90)    

    axs[1].set_ylabel('$Mn')
    axs[1].set_title(f'Gross Exposure by {col} and Liquidation Bucket', pad=20)
    axs[1].grid()

    plt.suptitle('Days to Liquidate (at 10% Participation Rate)')
    plt.tight_layout()

    return dtl_bucket_breakdown

def plot_most_illiq_group(concise_positions, col, p = 0.10,
                          illiq_exposure_threshold = 10,
                          pad = 2,
                          figsize=(10, 8),
                          gmv = None):
    """
    Params:
    1) Illiq_exposure_threshold : in $Mn, the minimum gross illiquid exposure to show up on the chart
    Identify industries with most illiquid exposure
    1) Illiquid exposure : takes more than 3 days to liquidate at 10% participation
    2) Liquid exposure   : takes less than 3 days to liquidate at 10% participation
    """
    # Get the gross exposure by industry and liquidation bucket
    p = int(p * 100)
    df2 = concise_positions.pivot_table(
        observed = True,
        values = 'gross_exposure_usd',
        index  = col,
        columns= f'dtl_bucket_{p}_pct',
        aggfunc= 'sum',
    ).fillna(0) /1e6

    # Group the exposures that takes more than 3 days to liquidate
    df2['illiquid_exposure']    = df2[['3–10d', '>10d']].sum(axis=1)
    df2['liquid_exposure']      = df2[['<4 h', '~1d', '1–3d']].sum(axis=1)

    # Find when gross illiquid exposure more than $thresh million 
    cond = (df2['illiquid_exposure'] > illiq_exposure_threshold) 

    # plot 
    plot_illiq_df = df2[cond].copy().sort_values(by='illiquid_exposure', ascending=True)

    plot_illiq_df['total_exposure'] = df2['liquid_exposure'] + df2['illiquid_exposure']

    ax = plot_illiq_df[['illiquid_exposure', 'total_exposure']].plot(
        kind = 'barh',
        alpha = 0.8,
        color = ('tab:red', 'tab:blue'),
        figsize = figsize
    )
    ax.legend(title='Exposures')
    ax.set_ylabel(f'{col}')
    ax.grid()

    if isinstance(col, list):
        ax.set_yticklabels(plot_illiq_df.index.get_level_values(1).str.replace('&', '\n&'))
        col = col[1]
    else:
        ax.set_yticklabels(plot_illiq_df.index.str.replace('&', '\n&'))
    
    illiq_container = ax.containers[0]
    
    # Annotate 
    gmv                                  = concise_positions['gross_exposure_usd'].sum() if gmv == None else gmv
    plot_illiq_df['illiquid_pct_of_gmv'] = plot_illiq_df['illiquid_exposure'] * 1e6 / gmv * 100

    for bar, pct in zip(illiq_container, plot_illiq_df['illiquid_pct_of_gmv']):
        x = bar.get_x() + bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(x + pad, y, f'{pct:.1f}% of Gross Portfolio Value', va='center', ha='left', fontsize=9)

    
    plt.title(f'Top {len(plot_illiq_df)} {col} with Notable Illiquid Exposure\n(DTL > 3 @ {p}% participation rate of ADV)')


    return df2, plot_illiq_df

# Risk Factors : EM vs Developed
def calc_emerging_market_exposure(concise_positions):
    """ 
    EM vs Developed Market Exposure
    """
    em_countries        = ['BRA', 'CHN', 'GRC', 'RUS']
    developed_countries = [c for c in concise_positions['country'].unique() if c not in em_countries]

    concise_positions['market_type'] = np.where(
        concise_positions['country'].isin(em_countries),
        'EM',
        'Developed'
    )

    market_type_exposure = concise_positions.groupby('market_type')[['net_exposure_usd', 'gross_exposure_usd']].sum() / 1e6
    beta_weighted_mkt_exposure = concise_positions.groupby('market_type')[['beta_weighted_exposure_usd', 'gross_beta_exposure_usd']].sum() / 1e6

    market_type_exposure['net_tilt_pct'] = market_type_exposure['net_exposure_usd'] / market_type_exposure['gross_exposure_usd'].sum() * 100
    market_type_exposure['net_to_gross_ratio'] = market_type_exposure['net_exposure_usd'] / market_type_exposure['gross_exposure_usd'] * 100
    beta_weighted_mkt_exposure['net_tilt_pct'] = beta_weighted_mkt_exposure['beta_weighted_exposure_usd'] / beta_weighted_mkt_exposure['gross_beta_exposure_usd'].sum() * 100
    beta_weighted_mkt_exposure['net_to_gross_ratio'] = beta_weighted_mkt_exposure['beta_weighted_exposure_usd'] / beta_weighted_mkt_exposure['gross_beta_exposure_usd'] * 100
    
    return market_type_exposure, beta_weighted_mkt_exposure

def plot_dev_em_dollar_vs_beta(
    market_type_exposure,          # index=['Developed','EM']; cols: gross_exposure_usd, net_exposure_usd, net_tilt_pct
    beta_weighted_mkt_exposure,    # index=['Developed','EM']; cols: gross_beta_exposure_usd, beta_weighted_exposure_usd, net_tilt_pct
    figsize=(12,5)
):

    # order consistent
    order = [i for i in ["Developed","EM"] if i in market_type_exposure.index]
    mkt  = market_type_exposure.loc[order]
    beta = beta_weighted_mkt_exposure.loc[order]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # plot sep
    def panel(ax, df, gross_col, net_col, pct_col, title):
        y = np.arange(len(df)); h = 0.35
        max_abs = float(np.max(np.abs(np.r_[df[gross_col].values, df[net_col].values])))
        xlim = 1.15 * max_abs if max_abs > 0 else 1.0

        ax.barh(y + h/2, df[gross_col].values, height=h, label="Gross")
        ax.barh(y - h/2, df[net_col].values,   height=h, label="Net")

        # dot + label just to the LEFT
        margin = 0.05 * xlim
        for i, (netv, pct) in enumerate(zip(df[net_col].values, df[pct_col].values)):
            ax.text(netv - margin, y[i] - h/2, f" {pct:+.1f}%", va="center", ha="right", fontsize=9)

        ax.set_xlim(-xlim, xlim)
        ax.set_yticks(y)
        ax.set_yticklabels(df.index.tolist())
        ax.axvline(0, linewidth=1)
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid()

    panel(axL, mkt,  "gross_exposure_usd",       "net_exposure_usd",           "net_tilt_pct", "Dollar View")
    panel(axR, beta, "gross_beta_exposure_usd",  "beta_weighted_exposure_usd", "net_tilt_pct", "Market Beta view")

    axL.set_ylabel("Market type")
    axL.set_xlabel("($Mn)")
    axR.set_xlabel("($Mn)")

    plt.tight_layout()
    return fig

def plot_biggest_tilts(concise_positions, col = 'industry',
                       thresh = 0.5, ref_col = 'sector',
                       figsize = (6,4)):
    """ 
    Plot the biggest tilts by industry or sub-industry etc...
        - thresh : threshold in % of portfolio gross market value
        - ref_col : column to use to group colors (legend)
    Returns a plot showing where our most undiversified exposures are 

    Variable names are for industry - sector but can be used for any group
    """
    from matplotlib.patches import Patch

    gmv = concise_positions['gross_exposure_usd'].sum()
    
    by_industry = (concise_positions
                    .groupby([col, ref_col], as_index=False)
                    .agg(net_exposure_usd=('net_exposure_usd','sum'))
                    .sort_values('net_exposure_usd', ascending=False))
    by_industry['net_tilt_pct'] = by_industry['net_exposure_usd'] / gmv * 100

    by_industry = by_industry[by_industry['net_tilt_pct'].abs() > thresh]
    
    sectors = by_industry[ref_col].unique()
    cmap = plt.get_cmap('tab20')
    sector_color = {s: cmap(i / max(1, len(sectors)-1)) for i, s in enumerate(sectors)}

    ax = (by_industry).plot.bar(
        x=col, y='net_tilt_pct',
        color=by_industry[ref_col].map(sector_color),
        figsize=figsize, legend=False
    )
    ax.grid()
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_ylabel('Net Tilt (% of Portfolio Value)')
    ax.set_xlabel(col)
    ax.set_xticklabels(shorten_labels(by_industry[col]))

    # 4) Legend for sectors
    handles = [Patch(facecolor=sector_color[s], label=s) for s in sectors]
    ax.legend(handles=handles, title=f'{ref_col}', loc='lower left', frameon=True)
    ax.set_title(f'Biggest {col} Exposures\n(where net exposure exceeds {thresh}% of portfolio gross market value)', pad=20)
    return by_industry, ax

def get_concentrated_bets(concise_positions, levels : list[str],
                          tilt_thresh : float, bias_thresh : float
                          )-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ 
    Identify concentrated bets at a given level of aggregation (e.g. country x sector)
    A concentrated bet is defined as one where:
    1. The absolute tilt (exposure as % of portfolio gmv) is greater than tilt_thresh
    2. The absolute bias (exposure as % of sector gmv) is greater than bias_thresh
    Returns:
        - DataFrame of all concentrated bets
        - DataFrame of top 3 concentrated longs
        - DataFrame of top 3 concentrated shorts
    """
    df                      = concise_positions.groupby(levels)[['net_exposure_usd', 'gross_exposure_usd']].sum()
    gmv                     = concise_positions['gross_exposure_usd'].sum()
    sector_allocation       = calc_sector_allocations(concise_positions)[0]
    
    # Get the total value of all positions in that sector to calculate local_sector_tilt
    df = df.reset_index().merge(
            sector_allocation.reset_index(),
            left_on='sector', right_on='sector',
            suffixes=('', '_sector_total')
            ).set_index(['sector', 'country']
        ).sort_index()

    # Get how much is allocated to that local market as % of GMV
    df['gmv_allocation_pct'] = df['gross_exposure_usd'] / gmv * 100
    # Get how much is allocated to that country as % of the sector
    df['country_sector_weight_pct'] = df['gross_exposure_usd'] / df['gross_exposure_usd_sector_total'] * 100
    # Calculate tilt (% of portfolio gmv) of each local market
    df['tilt'] = df['net_exposure_usd'] / gmv * 100
    # Calculate local sector bias (per country per sector)
    df['bias'] = df['net_exposure_usd'] / df['gross_exposure_usd'] * 100 

    # Condition for notable exposure
    cond       = (df['tilt'].abs() > tilt_thresh) & (df['bias'].abs() > bias_thresh)

    concentrated_bets = df[cond].copy()
    
    concentrated_bets[['gross_exposure_usd', 'net_exposure_usd',
        'gross_exposure_usd_sector_total']]  = (concentrated_bets[['gross_exposure_usd', 'net_exposure_usd',
        'gross_exposure_usd_sector_total']] / 1e6)

    concentrated_bets = concentrated_bets.sort_values(by='tilt', ascending=False, key = lambda x : x.abs())
    concentrated_bets['side'] = np.where(concentrated_bets['bias'] > 0, 'LONG', 'SHORT')

    most_concentrated_longs = concentrated_bets.groupby('side').get_group('LONG').head(3)
    most_concentrated_shorts = concentrated_bets.groupby('side').get_group('SHORT').sort_values(by='tilt', ascending=True).head(3)

    return concentrated_bets, most_concentrated_longs, most_concentrated_shorts