import math
import warnings
from datetime import timedelta

import cvxopt as cvx
import pandas as pd
import scipy.stats
from loguru import logger
from prophet import Prophet
from scipy.stats import norm

warnings.filterwarnings("ignore")


def forecast_accuracy(forecast, actual):
    """
    Return a dictionary of performance metrics
    """
    forecast = forecast.astype(float)
    actual = actual.astype(float)
    ape = np.abs(forecast - actual) / np.abs(actual)
    ape.replace([np.inf, -np.inf], np.nan, inplace=True)
    mape = np.nanmean(ape)  # MAPE
    me = np.mean(forecast - actual)  # ME
    mae = np.mean(np.abs(forecast - actual))  # MAE
    pe= ((forecast - actual) / actual)
    pe.replace([np.inf, -np.inf], np.nan, inplace=True)
    mpe = np.nanmean(pe)  # MPE
    rmse = np.mean((forecast - actual) ** 2) ** 0.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0, 1]  # corr
    return {"mape": mape, "me": me, "mae": mae, "mpe": mpe, "rmse": rmse, "corr": corr}


def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
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


def univariate_train_test_split(data, n_test):
    """
    Split a univariate dataset into train/test sets
    """
    return data[:-n_test], data[-n_test:]


def prophet_train_test_split(df, forecasted_window):
    """
    Divide in training and test set, using the `forecasted_window` parameter (int)
    returns the training and test dataframes
    Parameters
    ----------
    df : pandas.DataFrame
    forecasted_window: integer
        The number of days separating the training set and the test set
    Returns
    -------
    data_train : pandas.DataFrame
        The training set, formatted for prophet.
    data_test :  pandas.Dataframe
        The test set, formatted for prophet.
    """
    last_date = df["ds"].iloc[-1]
    subtracted_date = pd.to_datetime(last_date) - timedelta(days=forecasted_window)
    subtracted_date = subtracted_date.strftime("%Y-%m-%d")
    data_train = df.sort_values(by="ds").set_index("ds").truncate(after=subtracted_date)
    data_test = df.sort_values(by="ds").set_index("ds").truncate(before=subtracted_date)
    return data_train, data_test


def train_and_forecast_prophet(group: pd.DataFrame, len_data_test: int):
    """
    fit the prophet model on the group data (historical data for a single ticker)
    :return: predicted values for training set and forecast window of length len_data_test
    """
    m = Prophet()
    group.replace([np.inf, -np.inf], np.nan, inplace=True)
    group = group.fillna(method='ffill')
    m.fit(group)
    future = m.make_future_dataframe(periods=len_data_test, freq="B")
    forecast = m.predict(future)
    # now filter the non business days!! https://facebook.github.io/prophet/docs/non-daily_data.html
    forecast["Ticker"] = group["Ticker"].iloc[0]
    return forecast, m


def forecast_with_regressors(ticker: str, df_train: pd.DataFrame, df_val: pd.DataFrame, macro_truncated: pd.DataFrame):
    """
    Make prediction using a set of exogenous variables
    """
    exogs = list(macro_truncated.columns)
    m0 = Prophet(
        mcmc_samples=100,
        holidays_prior_scale=0.25,
        changepoint_prior_scale=0.01,
        seasonality_mode="multiplicative",
        yearly_seasonality=10,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    # Adding Regressors
    for e in exogs:
        m0.add_regressor(e, prior_scale=0.5, mode="multiplicative")

    x = ["Date", ticker] + exogs
    z = x[:-1]

    sample = df_train[x]
    sample = sample.rename(columns={ticker: "y", "Date": "ds"})
    sample["ds"] = sample["ds"].dt.tz_localize(None)
    m0.fit(sample)
    future_y = m0.make_future_dataframe(periods=30, freq="1D")
    ret_future = future_y.merge(df_val[x], right_on="Date", left_on="ds", how="inner")
    ret_forecast = m0.predict(ret_future)
    return ret_forecast


def prophet_forecast_no_regressors(df_prophet, len_data_test, is_return=False):
    """
    Make predictions without any regressor (autoregressive)

    :return: for_loop_forecast appended dataframes (for each ticker) of predictions and model fit specs
    :return: all_models: list of the prophet fitted objects
    :return: all_forecasts list of dataframes with model info for each
    ticker, included the predicted value yhat
    """
    sample = df_prophet[df_prophet.Ticker.str.contains("log_ret|pct_ret") == is_return]
    df_prophet_by_ticker = sample.groupby("Ticker")
    ticker_list = [k for k in df_prophet_by_ticker.groups.keys()]
    for_loop_forecast = pd.DataFrame()
    all_models = []
    all_forecasts = []
    for ticker in ticker_list:
        group = df_prophet_by_ticker.get_group(ticker)
        forecast, m = train_and_forecast_prophet(group, len_data_test)
        for_loop_forecast = pd.concat((for_loop_forecast, forecast))
        all_models.append(m)
        all_forecasts.append(forecast)
    return for_loop_forecast, all_models, all_forecasts


def prophet_with_regressors(df_prophet, exo, len_data_test, is_return=False):
    """
    Call the function to make prediction, create and parse the Data Frame with the inferences
    """
    sample = df_prophet[df_prophet.Ticker.str.contains("log_ret|pct_ret") == is_return]
    df_prophet_by_ticker = sample.groupby("Ticker")
    ticker_list = []
    for k in df_prophet_by_ticker.groups.keys():
        ticker_list.append(k)
    for_loop_forecast = pd.DataFrame()
    all_models = []
    all_forecasts = []
    for ticker in ticker_list:
        group = df_prophet_by_ticker.get_group(ticker)
        forecast, m = forecast_with_regressors(group, exo, len_data_test)
        for_loop_forecast = pd.concat((for_loop_forecast, forecast))
        all_models.append(m)
        all_forecasts.append(forecast)
    return for_loop_forecast, all_models, all_forecasts


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


def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights) ** 0.5


def ENC(df):
    df["w_norm"] = df.apply(lambda x: x * 100 / x.sum())
    df["w_norm_2"] = df["w_norm"].apply(lambda x: (x ** 2))
    enc = 1 / df["w_norm_2"].sum()
    return enc


from scipy.optimize import minimize


def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    return_is_target = {
        "type": "eq",
        "args": (er,),
        "fun": lambda weights, er: target_return - portfolio_return(weights, er),
    }
    weights = minimize(
        portfolio_vol,
        init_guess,
        args=(cov,),
        method="SLSQP",
        options={"disp": False},
        constraints=(weights_sum_to_1, return_is_target),
        bounds=bounds,
    )
    return weights.x


def optimal_weights(n_points, er, cov):
    """
    Optimize weights
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0001, 0.3),) * n  # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate) / vol

    weights = minimize(
        neg_sharpe,
        init_guess,
        args=(riskfree_rate, er, cov),
        method="SLSQP",
        options={"disp": False},
        constraints=(weights_sum_to_1,),
        bounds=bounds,
    )
    return weights.x


def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def plot_ef(
        n_points, er, cov, style=".-", legend=False, show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False
):
    """
    Plots the multi-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
    if show_cml:
        ax.set_xlim(left=0)
        # get MSR
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color="green", marker="o", linestyle="dashed", linewidth=2, markersize=10)

        param = {"weights": [w_msr], "ret": [r_msr]}
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1 / n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        param = {"weights": [w_ew], "ret": [r_ew]}
        # add EW
        ax.plot([vol_ew], [r_ew], color="goldenrod", marker="o", markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # add EW
        ax.plot([vol_gmv], [r_gmv], color="midnightblue", marker="o", markersize=10)
        param = {"weights": [w_gmv], "ret": [r_gmv]}
    return ax, param


def risk_contribution(w, cov):
    """
    Compute the contributions to risk of the constituents of a portfolio, given a set of portfolio
    weights and a covariance matrix
    """
    total_portfolio_var = portfolio_vol(w, cov) ** 2
    # Marginal contribution of each constituent
    marginal_contrib = cov @ w
    risk_contrib = np.multiply(marginal_contrib, w.T) / total_portfolio_var
    return risk_contrib


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


def ptf_trackrecord(r, estimation_window=20, rf=0.02, **kwargs):
    """
    summarize the stats for the ptf return in a given window
    : param r: ptf return
    :param rf: risk free rate: default 2%
    """

    if "Date" in r.columns:
        r = r.set_index("Date")

    r = r.select_dtypes(include="float64")
    n_periods = r.shape[0]

    windows = [(start, start + estimation_window) for start in range(n_periods - estimation_window + 1)]
    stats = [summary_stats(r.iloc[win[0]: win[1]], riskfree_rate=rf, t=252) for win in windows]
    trackrecord = pd.concat(stats)
    # trackrecord['day']=[i for i in range(0,trackrecord.shape[0])]
    trackrecord = trackrecord.set_index(r.iloc[estimation_window - 1:].index)

    return trackrecord


def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1 / steps_per_year
    n_steps = int(n_years * steps_per_year) + 1
    rets_plus_1 = np.random.normal(loc=mu * dt + 1, scale=sigma * np.sqrt(dt), size=(n_steps, n_scenarios))
    # or better ...
    # rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    prices = s_0 * pd.DataFrame(rets_plus_1).cumprod()
    return prices


def ptf_composition(data, numStocks, numRev, col_M_return):
    df = data.copy()
    selected_stocks = []
    avg_monthly_ret = [0]
    for i in range(len(df)):
        if len(selected_stocks) > 0:
            avg_monthly_ret.append(df[selected_stocks].iloc[i, :].mean())
            bad_stocks = df[selected_stocks].iloc[i, :].sort_values(ascending=True)[:numRev].index.values.tolist()
            selected_stocks = [t for t in selected_stocks if t not in bad_stocks]
        fill = numStocks - len(selected_stocks)
        new_picks = df.iloc[i, :].sort_values(ascending=False)[:fill].index.values.tolist()
        selected_stocks = selected_stocks + new_picks
        logger.info(selected_stocks)
    returns_df = pd.DataFrame(np.array(avg_monthly_ret), columns=[col_M_return])
    return returns_df


def CAGR(data, col_M_return):
    """
    Compute the cumulative annual growth rate
    """
    df = data.copy()
    df["cumulative_returns"] = (1 + df[col_M_return]).cumprod()
    trading_months = 12
    n = len(df) / trading_months
    cagr = (df["cumulative_returns"][len(df) - 1]) ** (1 / n) - 1
    return cagr


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


def tipp_pl(risky_r, safe_r, ci=100, floor_pct=0.85, m=5, safe_asset_rate=2, gap=1):
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
    logger.info("Function: tss.tipp_sp")
    # Da call con Investments let's consider a spot rate=2%
    # i.e we assume the safe portion is paying 2% per year
    safe_assets = safe_r

    # Initialize ptf valus
    init_capital = ci
    F = init_capital * floor_pct

    # handle non imputed params
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r)

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
        cushion_relative = (init_capital - F) / init_capital
        # updates the weights of the risky asset: E=C x m (c)...
        risky_asset_e = max(min(m * cushion_relative, 1), 0)
        # ...and the weights in the safe asset which is the remaining part: B=CI-E (d)
        riskless_asset_b = 1 - risky_asset_e
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

    return hist_trackrecords["Ptf P&L"]


def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at
    time t where t is in years and r is the annual interest rate
    """
    return (1 + r) ** (-t)


def pv(l, r):
    """
    Compute the present value of a list of liabilities given by the time (as an index) and amounts
    """
    dates = l.index
    discounts = discount(dates, r)
    return (discounts * l).sum()


def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of a series of liabilities, based on an interest rate and current value of assets
    """
    return assets / pv(liabilities, r)


def inst_to_ann(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.expm1(r)


def ann_to_inst(r):
    """
    Convert an instantaneous interest rate to an annual interest rate
    """
    return np.log1p(r)


def cir(n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None:
        r_0 = b
    r_0 = ann_to_inst(r_0)
    dt = 1 / steps_per_year
    num_steps = int(n_years * steps_per_year) + 1  # because n_years might be a float

    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    # For Price Generation
    h = math.sqrt(a ** 2 + 2 * sigma ** 2)
    prices = np.empty_like(shock)

    ####

    def price(ttm, r):
        _A = ((2 * h * math.exp((h + a) * ttm / 2)) / (2 * h + (h + a) * (math.exp(h * ttm) - 1))) ** (
                2 * a * b / sigma ** 2
        )
        _B = (2 * (math.exp(h * ttm) - 1)) / (2 * h + (h + a) * (math.exp(h * ttm) - 1))
        _P = _A * np.exp(-_B * r)
        return _P

    prices[0] = price(n_years, r_0)
    ####

    for step in range(1, num_steps):
        r_t = rates[step - 1]
        d_r_t = a * (b - r_t) * dt + sigma * np.sqrt(r_t) * shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years - step * dt, rates[step])

    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    # for prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    ###
    return rates, prices


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


import numpy as np


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


def markowitz_opt(ret_vec, covar_mat, max_risk):
    """
    Compute the Markowitz frontieer
    """
    U, V = np.linalg.eig(covar_mat)
    U[U < 0] = 0
    Usqrt = np.sqrt(U)
    A = np.dot(np.diag(Usqrt), V.T)

    # Calculating G and h matrix
    G1temp = np.zeros((A.shape[0] + 1, A.shape[1]))
    G1temp[1:, :] = -A
    h1temp = np.zeros((A.shape[0] + 1, 1))
    h1temp[0] = max_risk

    ret_c = len(ret_vec)
    for i in np.arange(ret_c):
        ei = np.zeros((1, ret_c))
        ei[0, i] = 1
        if i == 0:
            G2temp = [cvx.matrix(-ei)]
            h2temp = [cvx.matrix(np.zeros((1, 1)))]
        else:
            G2temp += [cvx.matrix(-ei)]
            h2temp += [cvx.matrix(np.zeros((1, 1)))]

    # Construct list of matrices
    Ftemp = np.ones((1, ret_c))
    F = cvx.matrix(Ftemp)
    g = cvx.matrix(np.ones((1, 1)))

    G = [cvx.matrix(G1temp)] + G2temp
    H = [cvx.matrix(h1temp)] + h2temp

    # Solce using QCQP
    cvx.solvers.options["show_progress"] = False
    sol = cvx.solvers.socp(-cvx.matrix(ret_vec), Gq=G, hq=H, A=F, b=g)
    xsol = np.array(sol["x"])
    return xsol, sol["status"]
