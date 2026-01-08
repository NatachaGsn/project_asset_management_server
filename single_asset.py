import yfinance as yf
import numpy as np
import pandas as pd
from pandas import DataFrame, Series


def load_asset(asset: str,
               start: str = None,
               end: str = None,
               period: str = "1y",
               interval: str = "1d") -> DataFrame:
    """
    Load asset price data from Yahoo Finance in a standardized format.
    """

    df = yf.download(
        asset,
        start=start,
        end=end,
        period=None if start else period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError("No data downloaded.")

    df = df[["Adj Close"]].rename(columns={"Adj Close": "price"})
    df.index = pd.to_datetime(df.index)

    return df.dropna()

def canonicalise_price_data(df: DataFrame) -> Series:
    """
    Convert raw loaded asset data into a canonical 1D price series.

    This function guarantees:
    - a pandas Series
    - float dtype
    - datetime index
    - no duplicated timestamps
    - sorted index

    Parameters
    ----------
    df : DataFrame
        DataFrame containing a 'price' column.

    Returns
    -------
    Series
        Clean price series ready for all downstream computations.
    """
    

    price = df["price"]

    # If price is accidentally a DataFrame (1 column), flatten it
    if isinstance(price, pd.DataFrame):
        price = price.iloc[:, 0]

    if not isinstance(price, Series):
        raise TypeError("Price must be a pandas Series after extraction.")

    price = price.astype(float)
    price.index = pd.to_datetime(price.index)

    price = price[~price.index.duplicated(keep="last")]
    price = price.sort_index()
    price = price.dropna()

    return price

# ---------- Signals (strategies output ONLY signals) ----------

def signal_buy_and_hold(price: Series) -> Series:
    """
    Generate trading signals for a Buy & Hold strategy.

    The strategy stays fully invested in the asset at all times.

    Parameters
    ----------
    price : Series
        Asset price series indexed by datetime.

    Returns
    -------
    Series
        Signal series (+1 for long), indexed like price.
    """
    signal = Series(1, index=price.index, name="signal")
    return signal

def signal_moving_average(
    price: Series,
    short_window: int = 20,
    long_window: int = 50
) -> Series:
    """
    Generate trading signals using a moving-average crossover strategy.

    Long when short MA > long MA, otherwise flat.

    Parameters
    ----------
    price : Series
        Asset price series indexed by datetime.
    short_window : int
        Short moving average window.
    long_window : int
        Long moving average window.

    Returns
    -------
    Series
        Signal series (+1 for long, 0 for flat), indexed like price.
    """
    price = price.astype(float)

    short_ma = price.rolling(window=short_window).mean()
    long_ma = price.rolling(window=long_window).mean()

    signal = (short_ma > long_ma).astype(int)
    signal.name = "signal"

    return signal

# ---------- Backtest engine (common for all strategies) ----------

def backtest(price: Series, signal: Series) -> tuple[Series, Series]:
    """
    Backtest a trading strategy from a price series and a signal series.

    The signal is shifted by one period to avoid look-ahead bias.

    Parameters
    ----------
    price : Series
        Asset price series indexed by datetime.
    signal : Series
        Signal series (+1 for long, 0 for flat), indexed like price.

    Returns
    -------
    equity : Series
        Cumulative equity curve (base 1), indexed like price.
    strategy_returns : Series
        Period returns of the strategy, indexed like price.
    """
    # Ensure alignment
    price = price.astype(float)
    signal = signal.reindex(price.index).fillna(0.0).astype(float)

    asset_returns = price.pct_change().fillna(0.0)
    position = signal.shift(1).fillna(0.0)

    strategy_returns = position * asset_returns
    strategy_returns.name = "strategy_returns"

    equity = (1.0 + strategy_returns).cumprod()
    equity.name = "equity"

    return equity, strategy_returns



# ---------- Metrics ----------

def periods_per_year_from_interval(interval: str) -> int:
    """
    Map a data frequency to the corresponding number of periods per year.

    Parameters
    ----------
    interval : str
        Data frequency ('1d' for daily, '1wk' for weekly).

    Returns
    -------
    int
        Number of periods per year used for annualisation.
    """
    mapping = {
        "1d": 252,   # trading days per year
        "1wk": 52    # weeks per year
    }
    return mapping.get(interval, 252)

def sharpe_ratio(returns: Series,
                 interval: str,
                 risk_free_rate: float = 0.0) -> float:
    """
    Compute the annualised Sharpe ratio of a return series.

    The annualisation factor is automatically inferred from the
    data frequency.

    Parameters
    ----------
    returns : Series
        Strategy returns.
    interval : str
        Data frequency ('1d' for daily, '1wk' for weekly).
    risk_free_rate : float
        Risk-free rate per period (default 0).

    Returns
    -------
    float
        Annualised Sharpe ratio.
    """
    # number of periods per year
    periods_per_year = periods_per_year_from_interval(interval)

    # risk free rate per period
    rf_per_period = risk_free_rate / periods_per_year

    r = (returns.astype(float) - rf_per_period).dropna()

    if r.empty:
        return 0.0

    std = r.std()
    if std == 0 or np.isnan(std):
        return 0.0
    
    return float((r.mean() / std) * np.sqrt(periods_per_year))

def max_drawdown(equity: Series) -> float:
    """
    Compute the maximum drawdown of an equity curve.

    Parameters
    ----------
    equity : Series
        Equity curve (base 1), indexed by datetime.

    Returns
    -------
    float
        Maximum drawdown as a negative number (e.g. -0.25 for -25%).
    """
    eq = equity.astype(float).dropna()

    if eq.empty:
        return 0.0

    running_max = eq.cummax()
    drawdown = (eq / running_max) - 1.0

    return float(drawdown.min())

def annualised_return(returns: Series, interval: str) -> float:
    """
    Compute the annualised return from a series of periodic returns.

    The annualisation is adjusted according to the data frequency
    (daily or weekly).

    Parameters
    ----------
    returns : Series
        Periodic strategy returns.
    interval : str
        Data frequency ('1d' or '1wk').

    Returns
    -------
    float
        Annualised return.
    """

    r = returns.astype(float).dropna()
    if r.empty:
        return 0.0

    periods_per_year = periods_per_year_from_interval(interval)
    total_return = (1.0 + r).prod()
    n_years = len(r) / periods_per_year

    if n_years <= 0:
        return 0.0

    return total_return ** (1.0 / n_years) - 1.0


def annualised_volatility(returns: Series, interval: str) -> float:
    """
    Compute the annualised volatility of a return series.

    Volatility is scaled using the square-root-of-time rule,
    adjusted for the data frequency.

    Parameters
    ----------
    returns : Series
        Periodic strategy returns.
    interval : str
        Data frequency ('1d' or '1wk').

    Returns
    -------
    float
        Annualised volatility.
    """
    
    r = returns.astype(float).dropna()
    if r.empty:
        return 0.0

    periods_per_year = periods_per_year_from_interval(interval)
    return r.std() * np.sqrt(periods_per_year)

# -------------------- Main plot ---------------------------
def prepare_price_equity_plot(price: Series, equity: Series) -> DataFrame:
    """
    Prepare normalised price and equity series for joint plotting.

    Both series are normalised to base 100 to allow direct comparison.

    Parameters
    ----------
    price : Series
        Asset price series.
    equity : Series
        Strategy equity curve (base 1).

    Returns
    -------
    DataFrame
        DataFrame containing normalised price and equity (base 100).
    """

    price = price.astype(float).dropna()
    equity = equity.astype(float).dropna()

    # Align on common index
    price, equity = price.align(equity, join="inner")

    df_plot = pd.DataFrame({
        "Price (base 1)": price / price.iloc[0],
        "Strategy equity (base 1)": equity / equity.iloc[0]
    }, index=price.index)

    return df_plot