from __future__ import annotations

import warnings
from typing import Union
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from statsmodels.tsa.arima.model import ARIMA


def prepare_series_for_arima(
    df_raw: Union[pd.DataFrame, pd.Series],
    interval: str = "1d",
    transform: str = "log",
) -> tuple[Series, dict]:
    """
    Prepare a price series for ARIMA by enforcing a regular time index and handling missing data.

    - Accepts a Series (price series) or a DataFrame with a 'price' column
    - Ensures DateTimeIndex, sorted and de-duplicated
    - Builds a regular index:
        * daily: 'D' if weekends are present (crypto), else 'B' (equities/FX)
        * weekly: 'W-SUN' if weekends are present, else 'W-FRI'
    - Handles missing values with a simple policy (interpolate/ffill+bfill)
    - Optionally applies a transform ('log' or 'none') to stabilize ARIMA fitting

    Returns
    -------
    y : Series
        Model-ready series with explicit frequency.
    info : dict
        Preparation details (frequency, missing %, fill method, etc.).
    """
    # --- 1) Normalize input to a clean 1D series y0 ---
    if isinstance(df_raw, pd.Series):
        y0 = df_raw.copy()
        y0.index = pd.to_datetime(y0.index)
        y0 = (
            y0[~y0.index.duplicated(keep="first")]
            .sort_index()
            .astype(float)
            .dropna()
        )
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

    # Detect if the raw series contains weekends (crypto-like behavior)
    has_weekends = (y0.index.dayofweek >= 5).any()  # 5=Sat, 6=Sun

    # --- 2) Choose frequency and build regular series (y) ---
    if interval == "1wk":
        # Weekly: use W-FRI for markets (equities/FX), W-SUN for 7/7 assets (crypto)
        freq = "W-SUN" if has_weekends else "W-FRI"
        y = y0.resample(freq).last()
    else:
        # Daily: use calendar days for 7/7 assets, business days otherwise
        freq = "D" if has_weekends else "B"
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

    if y.empty:
        raise ValueError(
            "Prepared series is empty after resampling/reindexing. "
            "Check your date range and input data."
        )

        # --- 5) Optional transform ---
    if transform == "log":
        if (y <= 0).any():
            raise ValueError("Log transform requires strictly positive prices.")
        y = pd.Series(np.log(y.values), index=y.index, name="log_price")
    elif transform == "none":
        y.name = "price"
    else:
        raise ValueError("transform must be 'log' or 'none'.")


    # --- 6) Info / reporting ---
    info = {
        "n_original": int(len(y0)),
        "n_final": int(len(y)),
        "n_missing_introduced": n_missing,
        "pct_missing_introduced": float(pct_missing),
        "fill_method": fill_method,
        "start_date": y.index[0],
        "end_date": y.index[-1],
        "frequency": freq,
        "interval": interval,
        "has_weekends": bool(has_weekends),
        "transform": transform,
    }
    return y, info


def temporal_train_test_split(y: Series, train_ratio: float = 0.8) -> tuple[Series, Series]:
    """
    Split a time series into train and test sets while preserving time order (no shuffling).
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1.")
    split_idx = int(len(y) * train_ratio)
    if split_idx < 5 or (len(y) - split_idx) < 2:
        raise ValueError("Not enough data for a meaningful train/test split.")
    return y.iloc[:split_idx], y.iloc[split_idx:]


def arima_forecast_split(
    df_raw: Union[DataFrame, Series],
    interval: str = "1d",
    order: tuple[int, int, int] = (1, 1, 1),
    horizon: int = 30,
    train_ratio: float = 0.8,
    ci_level: float = 0.95,
    transform: str = "log",
    maxiter: int = 200,
) -> dict:
    """
    Train/test evaluation + final future forecast with confidence intervals.

    - Prepares series (regular index + missing handling + optional log transform)
    - Temporal split train/test
    - Fit ARIMA on train -> forecast test (+ CI)
    - Fit ARIMA on full -> forecast future horizon (+ CI)

    Warnings from statsmodels are silenced (no display in terminal or Streamlit).
    """
    y, prep_info = prepare_series_for_arima(df_raw, interval=interval, transform=transform)
    y_train, y_test = temporal_train_test_split(y, train_ratio=train_ratio)

    alpha = 1.0 - float(ci_level)
    if not (0.0 < alpha < 1.0):
        raise ValueError("ci_level must be between 0 and 1 (e.g. 0.95).")

    try:
        with warnings.catch_warnings():
            # Silence all statsmodels warnings (including convergence / start params)
            warnings.simplefilter("ignore")

            # --- Fit on train, predict test ---
            model_train = ARIMA(
                y_train,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res_train = model_train.fit(method_kwargs={"maxiter": int(maxiter)})

            pred_test = res_train.get_forecast(steps=len(y_test))
            pred_test_mean = pred_test.predicted_mean.rename("pred_test")

            pred_test_ci = pd.DataFrame(pred_test.conf_int(alpha=alpha))
            pred_test_ci.columns = ["lower", "upper"]

            # --- Fit on full data, forecast future ---
            model_full = ARIMA(
                y,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res_full = model_full.fit(method_kwargs={"maxiter": int(maxiter)})

            pred_future = res_full.get_forecast(steps=int(horizon))
            future_mean = pred_future.predicted_mean.rename("forecast_future")

            future_ci = pd.DataFrame(pred_future.conf_int(alpha=alpha))
            future_ci.columns = ["lower", "upper"]

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
        raise RuntimeError(
            f"ARIMA forecast failed with order={order}, interval={interval}, transform={transform}: {str(e)}"
        ) from e

def inverse_transform_for_display(
    x: Union[Series, DataFrame, None],
    transform: str,
) -> Union[Series, DataFrame, None]:
    """
    Convert a series or DataFrame from model space back to price space for display purposes.

    If transform == "log", applies exp(.) (inverse of the log transform).
    Otherwise, returns the input unchanged.

    Parameters
    ----------
    x : Series | DataFrame | None
        Data to be converted (e.g. forecasts or confidence interval bounds).
    transform : str
        Transformation applied during model preparation ("log" or "none").

    Returns
    -------
    Series | DataFrame | None
        Data expressed in price space if required, otherwise unchanged.
    """

    if x is None or transform != "log":
        return x

    if isinstance(x, pd.Series):
        return pd.Series(np.exp(x.values), index=x.index, name=x.name)

    if isinstance(x, pd.DataFrame):
        out = x.copy()
        out.iloc[:, :] = np.exp(out.values)
        return out

    raise TypeError(f"Unsupported type for inverse transform: {type(x)}")