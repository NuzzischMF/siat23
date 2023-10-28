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

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 3).mean()
    return exp / sigma_r ** 3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 4).mean()
    return exp / sigma_r ** 4


def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    """
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compounded_growth ** (periods_per_year / n_periods) - 1


def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    """
    return r.std() * (periods_per_year ** 0.5)


def sharpe_ratio_annualized(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol


def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(r)
        return p_value > level


def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
    returns a DataFrame with columns for
    the wealth index,
    the previous peaks, and
    the percentage drawdown
    """
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, "Previous Peak": previous_peaks, "Drawdown": drawdowns})


def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    if isinstance(r, pd.Series):
        is_negative = r < 0
        return r[is_negative].std(ddof=0)
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(semideviation)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level / 100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * (k - 3) / 24 - (2 * z ** 3 - 5 * z) * (s ** 2) / 36
    return -(r.mean() + z * r.std(ddof=0))

def volatility(data, col_M_return):
    """
    Compute the volatility
    """
    df = data.copy()
    trading_months = 12
    vol = df[col_M_return].std() * np.sqrt(trading_months)
    return vol

def sharpe_ratio(data, rf, col_ret_M):
    """
    Compute the Sharpe Ratio
    """
    df = data.copy()
    sharpe = (CAGR(df, col_ret_M) - rf) / volatility(df, col_ret_M)
    return sharpe

def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns a DataFrame with columns for
       the wealth index, 
       the previous peaks, and 
       the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})


def maximum_drawdown(data, col_M_return):
    """
    Compute the Maximum Drawdown
    """
    df = data.copy()
    df["cumulative_returns"] = (1 + df[col_M_return]).cumprod()
    df["cumulative_max"] = df["cumulative_returns"].cummax()
    df["drawdown"] = df["cumulative_max"] - df["cumulative_returns"]
    df["drawdown_pct"] = df["drawdown"] / df["cumulative_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd


def summary_stats(r, riskfree_rate=0.02, t=252):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=t)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=t)
    ann_sr = r.aggregate(sharpe_ratio_annualized, riskfree_rate=riskfree_rate, periods_per_year=t)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame(
        {
            "Annualized Return": ann_r,
            "Annualized Vol": ann_vol,
            "Skewness": skew,
            "Kurtosis": kurt,
            "Cornish-Fisher VaR (5%)": cf_var5,
            "Historic CVaR (5%)": hist_cvar5,
            "Sharpe Ratio": ann_sr,
            "Max Drawdown": dd,
        }
    )


