import pandas as pd


def add_time_features(df):
    df = df.copy()
    df["hour"] = df["Datetime"].dt.hour
    df["day_of_week"] = df["Datetime"].dt.dayofweek
    df["month"] = df["Datetime"].dt.month
    return df


def add_lag_features(df, lags=(1, 24, 168)):
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["MW"].shift(lag)
    return df


def add_rolling_features(df, windows=(24, 168)):
    df = df.copy()
    for window in windows:
        df[f"rolling_mean_{window}"] = (
            df["MW"]
            .shift(1)
            .rolling(window)
            .mean()
        )
    return df


def build_features_for_utility(df):
    """
    Feature engineering for a single utility time series.
    """
    df = df.sort_values("Datetime")

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Drop NaNs created by lags and rolling windows
    df = df.dropna()

    return df
