import math
import warnings
from datetime import timedelta
import numpy as np
import cvxopt as cvx
import pandas as pd
import scipy.stats
from loguru import logger
from prophet import Prophet
from scipy.stats import norm
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

def tipp_dd(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.02, drawdown=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = account_value
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate / 12  # fast way to set all values to a number
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)
        cushion = (account_value - floor_value) / account_value
        risky_w = m * cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1 - risky_w
        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w
        # recompute the new account value at the end of this step
        account_value = risky_alloc * (1 + risky_r.iloc[step]) + safe_alloc * (1 + safe_r.iloc[step])
        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    risky_wealth = start * (1 + risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "risky_r": risky_r,
        "safe_r": safe_r,
        "drawdown": drawdown,
        "peak": peak_history,
        "floor": floorval_history,
    }
    return backtest_result


def tipp_sp(risky_r, safe_r, ci=100, floor_pct=0.80, m=4.5, safe_asset_rate=0.02, gap=1, col_ret=None):
    """
    This function computes computes:
    a. the floor value : F = CI * Floor
    b. the cushion value: C = CI - F
    c. the allocation to the risky asset: E=C x m
    d. the allocation in the safe asset which is the remaining part: B=CI-E
    e. the cumulate growth of the risky and safe allocations

    --- Required Parameters --------------------------------------------------
    risky_r: pd.Series : percantage or log returns of the risky component of prices for index
    safe_r: pd.Series : percantage or log returns of the safe component of prices for index

    col_ret: list of strings : list of column names of returns

    CI: Initial Capital. Default 100.

    floor_pct: minimum percentage value of the portfolio we want to protect.
    Default=80%.

    m: multiplier. The constant we multiply the ci for to obtain the risky
    proportion of the ptf. Default=5

    safe_asset_rate: interest rate of the safe security. Default=2%

    gap: period to consider when we recompute the parameters. Default=1
    --------------------------------------------------------------------------
    """

    # Da call con Investments let's consider a spot rate=2%
    # i.e we assume the safe portion is paying 2% per year
    if safe_r is None:
        safe_assets = pd.DataFrame().reindex_like(risky_r)
    else:
        safe_assets = safe_r

    # Initialize ptf valus
    init_capital = ci  # 100
    F = init_capital * floor_pct  # 80

    # handle non imputed params
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=col_ret)

    if safe_asset_rate is None:
        safe_asset_rate = pd.DataFrame().reindex_like(risky_r)
        safe_assets[:] = safe_asset_rate / 12

    # dfs to store the trackrecords for ci, c, f, e and b
    ci_values = pd.Series().reindex_like(risky_r)
    c_values = pd.Series().reindex_like(risky_r)
    floor_values = pd.Series().reindex_like(risky_r)
    risky_pct = pd.Series().reindex_like(risky_r)
    safe_pct = pd.Series().reindex_like(risky_r)

    # if current floor is below the updated one it will be replaced
    # if the drawdown happen at the first obs we land into the gap probelm
    for i in range(len(risky_r.index)):
        F_updated = init_capital * floor_pct

        if F < F_updated:
            F = F_updated
        # handle exceptions
        if i % gap != 0 and i != 0:
            # updates the ci
            init_capital = risky_alloc_abs * (1 + risky_r.iloc[i].item()) + safe_alloc_abs * (
                    1 + safe_assets.iloc[i].item()
            )
            ci_values.iloc[i] = init_capital
            logger.info("updating the gap")
            continue

        # updates the cushion in relative terms
        cushion_relative = (init_capital - F) / init_capital  # 0,2
        # updates the weights of the risky asset: E=C x m (c)...
        risky_asset_e = m * cushion_relative  # TODO reset max(min(m * cushion_relative, 0.9), 0)  # 0.8
        # logger.info(f"using multiplier{m}, with cushion percentage: {cushion_relative}, hence risky_asset pct is {risky_asset_e}")
        # ...and the weights in the safe asset which is the remaining part: B=CI-E (d)
        riskless_asset_b = 1 - risky_asset_e  # 0.1 Should be minimum 10%!!!!
        # assert riskless_asset_b >= .1, "Riskless asset should be minimum .1"
        # updates the allocations in absolute terms
        risky_alloc_abs = init_capital * risky_asset_e
        safe_alloc_abs = init_capital * riskless_asset_b
        # updates the capital value in the account
        init_capital = risky_alloc_abs * (1 + risky_r.iloc[i].item()) + safe_alloc_abs * (
                1 + safe_assets.iloc[i].item()
        )

        ci_values.iloc[i] = init_capital
        c_values.iloc[i] = cushion_relative
        floor_values.iloc[i] = F
        risky_pct.iloc[i] = risky_asset_e
        safe_pct.iloc[i] = riskless_asset_b

    risky_pl = init_capital * (1 + risky_r).cumprod()

    hist_trackrecords = pd.DataFrame(
        {
            "Ptf P&L": ci_values,
            "Risky Budget": c_values,
            "Risky Allocation (%)": risky_pct,
            "Safe Allocation (%)": safe_pct,
            "floor": floor_values,
        }
    )

    return hist_trackrecords

def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor

    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate / 12  # fast way to set all values to a number
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        cushion = (account_value - floor_value) / account_value
        risky_w = m * cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1 - risky_w
        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w
        # recompute the new account value at the end of this step
        account_value = risky_alloc * (1 + risky_r.iloc[step]) + safe_alloc * (1 + safe_r.iloc[step])
        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
    risky_wealth = start * (1 + risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r": risky_r,
        "safe_r": safe_r,
    }
    return backtest_result


def show_cppi(n_scenarios=50, mu=0.07, sigma=0.15, m=3, floor=0.0, riskfree_rate=0.03, steps_per_year=12, y_max=100):
    """
    Plot the results of a Monte Carlo Simulation of CPPI
    """
    start = 100
    sim_rets = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, steps_per_year=steps_per_year)
    risky_r = pd.DataFrame(sim_rets)
    # run the "back"-test
    btr = run_cppi(risky_r, riskfree_rate=riskfree_rate, m=m, start=start, floor=floor)
    wealth = btr["Wealth"]

    # calculate terminal wealth stats
    y_max = wealth.values.max() * y_max / 100
    terminal_wealth = wealth.iloc[-1]

    tw_mean = terminal_wealth.mean()
    tw_median = terminal_wealth.median()
    failure_mask = np.less(terminal_wealth, start * floor)
    n_failures = failure_mask.sum()
    p_fail = n_failures / n_scenarios

    e_shortfall = np.dot(terminal_wealth - start * floor, failure_mask) / n_failures if n_failures > 0 else 0.0

