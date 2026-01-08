import yfinance as yf
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

# ------------------------------------
# Data
# ------------------------------------

def load_asset(asset: str,
               start: str = None,
               end: str = None,
               period: str = "1y",
               interval: str = "1d") -> DataFrame:
    
    """
    Downloads historical price data for a financial asset from Yahoo Finance
    and returns a cleaned DataFrame containing adjusted closing prices.

    Parameters
    ----------
    asset : str
        Ticker symbol of the asset (e.g. 'AAPL', 'BTC-USD').
    start : str, optional
        Start date of the historical data (format 'YYYY-MM-DD').
    end : str, optional
        End date of the historical data (format 'YYYY-MM-DD').
    period : str, optional
        Length of the historical period to download (e.g. '1y', '6mo', '5d').
    interval : str, optional
        Data frequency (e.g. '1d' for daily, '1wk' for weekly).

    Returns
    -------
    DataFrame
        DataFrame indexed by datetime containing a single column 'price'
        corresponding to the adjusted closing price.
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

# ------------------------------------
# Signals (strategies output ONLY signals)
# ------------------------------------

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

# ------------------------------------
# Backtest engine
# ------------------------------------

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


# ------------------------------------
# Metrics
# ------------------------------------

def periods_per_year(interval: str, asset_class: str) -> int:
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
    if interval == "1wk":
        return 52
    if interval == "1d":
        return 365 if asset_class == "crypto" else 252
    raise ValueError(f"Unsupported interval: {interval}")

def sharpe_ratio(returns: Series, periods_per_year: int) -> float:
    """
    Compute the annualised Sharpe ratio of a return series.

    Parameters
    ----------
    returns : Series
        Periodic return series, indexed by datetime.
    periods_per_year : int
        Number of return periods per year (e.g. 252 for daily, 52 for weekly).

    Returns
    -------
    float
        Annualised Sharpe ratio (risk-free rate assumed to be zero).
    """

    r = returns.dropna().astype(float)
    if r.empty:
        return 0.0
    vol = r.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(r.mean() / vol * np.sqrt(periods_per_year))

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

def annualised_return(equity: Series) -> float:
    """
    Compute the annualised return (CAGR) of an equity curve.

    Parameters
    ----------
    equity : Series
        Equity curve (base 1), indexed by datetime.

    Returns
    -------
    float
        Annualised return expressed as a decimal (e.g. 0.12 for 12%).
    """

    eq = equity.dropna().astype(float)
    if len(eq) < 2:
        return 0.0

    T = (eq.index[-1] - eq.index[0]).days / 365.25
    if T <= 0:
        return 0.0

    total = eq.iloc[-1] / eq.iloc[0]
    if total <= 0:
        raise ValueError("Equity must remain positive to compute CAGR.")

    return float(total ** (1.0 / T) - 1.0)


def annualised_volatility(returns: Series, periods_per_year: int) -> float:
    """
    Compute the annualised volatility of a return series.

    Parameters
    ----------
    returns : Series
        Periodic return series, indexed by datetime.
    periods_per_year : int
        Number of return periods per year (e.g. 252 for daily, 52 for weekly).

    Returns
    -------
    float
        Annualised volatility expressed as a decimal (e.g. 0.18 for 18%).
    """

    r = returns.dropna().astype(float)
    if r.empty:
        return 0.0
    return float(r.std(ddof=0) * np.sqrt(periods_per_year))

# ------------------------------------
# Main plot
# ------------------------------------
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