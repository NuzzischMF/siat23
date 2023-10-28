import math
import warnings
import numpy as np
from datetime import timedelta

import cvxopt as cvx
import pandas as pd
import scipy.stats
from loguru import logger
from prophet import Prophet
from scipy.stats import norm
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


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


