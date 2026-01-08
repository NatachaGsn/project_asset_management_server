from __future__ import annotations

import pandas as pd
from typing import Union
from pandas import DataFrame, Series
from statsmodels.tsa.arima.model import ARIMA

def prepare_series_for_arima(
    df_raw: Union[pd.DataFrame, pd.Series],
    interval: str = "1d"
) -> tuple[Series, dict]:
    """
    Prepare a price series for ARIMA by enforcing a regular time index and handling missing data.

    This function does NOT modify the raw dataset. It creates a model-ready series:
    - Accepts either a Series (already a price series) or a DataFrame with a 'price' column
    - Ensures a DateTimeIndex
    - Sorts and de-duplicates timestamps
    - Builds a regular index according to the expected frequency
    - Handles missing values via a simple, reported policy

    Notes (yfinance-friendly)
    -------------------------
    - For equities/indices from yfinance, daily data are typically business days ("B")
    - Weekly data from yfinance aligns better with market week ending on Friday ("W-FRI")
      and should be built via resampling rather than reindexing (avoids anchor mismatch)

    Parameters
    ----------
    df_raw : DataFrame or Series
        Raw data with a DateTimeIndex.
        - If DataFrame: must contain a 'price' column.
        - If Series: interpreted as the price series.
    interval : str, default '1d'
        Data interval (e.g. '1d', '1wk').

    Returns
    -------
    y : Series
        Model-ready price series, regular DateTimeIndex with explicit frequency.
    info : dict
        Preparation details (missing %, fill method, frequency, etc.).
    """
    # --- 1) Normalize input to a clean 1D series y0 ---
    if isinstance(df_raw, pd.Series):
        y0 = df_raw.copy()
        y0.index = pd.to_datetime(y0.index)
        y0 = y0[~y0.index.duplicated(keep="first")].sort_index().astype(float).dropna()
        y0.name = "price"

    elif isinstance(df_raw, pd.DataFrame):
        if "price" not in df_raw.columns:
            raise ValueError("df_raw must contain a 'price' column.")

        df = df_raw.copy()
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="first")].sort_index()

        y0 = df["price"].astype(float).dropna()
        y0.name = "price"

    else:
        raise TypeError("df_raw must be a pandas Series or DataFrame.")

    if y0.empty:
        raise ValueError("No valid price data after dropping missing values.")

    # --- 2) Choose frequency and build regular series (y) ---
    # yfinance equities/indices: prefer business days and week ending Friday
    if interval == "1wk":
        freq = "W-FRI"
        # Use resample to avoid anchor mismatch (W-SUN vs trading dates)
        y = y0.resample(freq).last()
    else:
        # For equities/indices, daily should be business days.
        # (If you later reuse for crypto, you can switch to 'D' when weekends exist.)
        freq = "B"
        full_index = pd.date_range(start=y0.index.min(), end=y0.index.max(), freq=freq)
        y = y0.reindex(full_index)

    # --- 3) Missingness + fill policy ---
    n_missing = int(y.isna().sum())
    pct_missing = (n_missing / len(y) * 100.0) if len(y) > 0 else 0.0

    if n_missing == 0:
        fill_method = "none"
    else:
        if pct_missing < 5:
            y = y.interpolate(method="time")
            fill_method = "time interpolation"
        elif pct_missing < 15:
            y = y.ffill().bfill()
            fill_method = "forward/backward fill"
        else:
            y = y.ffill().bfill()
            fill_method = "forward/backward fill (high missing rate)"

    # --- 4) Ensure explicit freq + final validation ---
    y = y.asfreq(freq).dropna()
    y.name = "price"

    if y.empty:
        raise ValueError(
            "Prepared series is empty after resampling/reindexing. "
            "For yfinance weekly data, 'W-FRI' with resample should work; "
            "check that your downloaded data range is not too short and contains prices."
        )

    # --- 5) Info / reporting ---
    info = {
        "n_original": int(len(y0)),
        "n_final": int(len(y)),
        "n_missing_introduced": n_missing,
        "pct_missing_introduced": float(pct_missing),
        "fill_method": fill_method,
        "start_date": y.index[0],
        "end_date": y.index[-1],
        "frequency": freq,
    }
    return y, info


def temporal_train_test_split(y: Series, train_ratio: float = 0.8) -> tuple[Series, Series]:
    """
    Split a time series into train and test sets while preserving time order (no shuffling).
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1.")
    split_idx = int(len(y) * train_ratio)
    return y.iloc[:split_idx], y.iloc[split_idx:]

def arima_forecast_split(
    df_raw: DataFrame,
    interval: str = "1d",
    order: tuple[int, int, int] = (1, 1, 1),
    horizon: int = 30,
    train_ratio: float = 0.8,
    ci_level: float = 0.95,
) -> dict:
    """
    Train/test evaluation + final future forecast with confidence intervals.

    Steps
    -----
    1) Prepare a model-ready series (regular index + missing handling)
    2) Temporal split into train/test
    3) Fit ARIMA on train and forecast over the test period (+ CI)
    4) Refit ARIMA on the full series and forecast into the future (+ CI)

    Returns a dict containing everything needed for plotting and reporting.
    """
    y, prep_info = prepare_series_for_arima(df_raw, interval=interval)
    y_train, y_test = temporal_train_test_split(y, train_ratio=train_ratio)

    alpha = 1.0 - float(ci_level)
    if not (0.0 < alpha < 1.0):
        raise ValueError("ci_level must be between 0 and 1 (e.g. 0.95).")

    try:
        # --- Fit on train, predict test ---
        model_train = ARIMA(y_train, order=order)
        res_train = model_train.fit()

        pred_test = res_train.get_forecast(steps=len(y_test))
        pred_test_mean = pred_test.predicted_mean
        pred_test_mean.name = "pred_test"
        
        pred_test_ci = pred_test.conf_int(alpha=alpha)
        
        # Debug: afficher le type
        # print(f"Type de pred_test_ci: {type(pred_test_ci)}")
        
        if isinstance(pred_test_ci, pd.Series):
            pred_test_ci = pred_test_ci.to_frame().T
        elif not isinstance(pred_test_ci, pd.DataFrame):
            raise TypeError(f"Unexpected type for conf_int: {type(pred_test_ci)}")
        
        if pred_test_ci.shape[1] == 2:
            pred_test_ci.columns = ["lower", "upper"]
        else:
            raise ValueError(f"Expected 2 columns for CI, got {pred_test_ci.shape[1]}")

        # --- Fit on full data, forecast future ---
        model_full = ARIMA(y, order=order)
        res_full = model_full.fit()

        pred_future = res_full.get_forecast(steps=horizon)
        future_mean = pred_future.predicted_mean
        
        future_ci = pred_future.conf_int(alpha=alpha)
        
        if isinstance(future_ci, pd.Series):
            future_ci = future_ci.to_frame().T
        elif not isinstance(future_ci, pd.DataFrame):
            raise TypeError(f"Unexpected type for conf_int: {type(future_ci)}")
        
        if future_ci.shape[1] == 2:
            future_ci.columns = ["lower", "upper"]
        else:
            raise ValueError(f"Expected 2 columns for CI, got {future_ci.shape[1]}")

        return {
            "y": y,
            "prep_info": prep_info,
            "train": y_train,
            "test": y_test,
            "pred_test": pred_test_mean,
            "pred_test_ci": pred_test_ci,
            "forecast_future": future_mean,
            "forecast_future_ci": future_ci,
            "model_summary": res_full.summary().as_text(),
        }
    
    except Exception as e:
        # Ajouter du contexte à l'erreur
        raise RuntimeError(f"ARIMA forecast failed with order={order}, interval={interval}: {str(e)}") from e
 