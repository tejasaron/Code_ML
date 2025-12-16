import pandas as pd



def time_based_split(df, target_col, test_size=0.2):
    """
    Splits time-series data into train and validation sets
    based on chronological order and converts time to numeric features.
    """
    df = df.sort_values("Datetime").copy()

    # ---- Time feature engineering ----
    dt = df["Datetime"]
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # Drop original datetime
    df = df.drop(columns=["Datetime"])

    # ---- Time-based split ----
    split_index = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_index]
    val_df = df.iloc[split_index:]

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]

    # ---- Final safety check ----
    assert X_train.select_dtypes(include="datetime").empty
    assert X_val.select_dtypes(include="datetime").empty

    return X_train, X_val, y_train, y_val