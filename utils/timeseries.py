import pandas as pd
import numpy as np
import math
import scipy.stats

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
