import yfinance as yf
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from typing import List, Tuple

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_TICKERS = ["SPY", "TLT", "GLD", "VNQ"]
TICKER_NAMES = {
    "SPY": "S&P 500 (US Equities)",
    "TLT": "Treasury Bonds 20+ Years",
    "GLD": "Gold Trust",
    "VNQ": "Real Estate (REITs)",
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_multi_asset(
    tickers: List[str] = DEFAULT_TICKERS,
    start: str = None,
    end: str = None,
    period: str = "1y",
    interval: str = "1d"
) -> DataFrame:
    """
    Load price data for multiple assets from Yahoo Finance.

    Parameters
    ----------
    tickers : List[str]
        List of asset tickers to download.
    start : str
        Start date (YYYY-MM-DD). If provided, period is ignored.
    end : str
        End date (YYYY-MM-DD).
    period : str
        Historical period (e.g., '1y', '6mo', '5d').
    interval : str
        Data frequency ('1d' for daily, '1wk' for weekly).

    Returns
    -------
    DataFrame
        DataFrame with adjusted close prices for each asset.
    """
    df = yf.download(
        tickers,
        start=start,
        end=end,
        period=None if start else period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError("No data downloaded.")

    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Adj Close"]
    else:
        df = df[["Adj Close"]]
        df.columns = tickers

    df.index = pd.to_datetime(df.index)
    return df.dropna()


# =============================================================================
# RETURNS CALCULATION
# =============================================================================

def calculate_returns(prices: DataFrame) -> DataFrame:
    """
    Calculate daily returns from price data.

    Parameters
    ----------
    prices : DataFrame
        DataFrame of asset prices.

    Returns
    -------
    DataFrame
        DataFrame of daily returns.
    """
    return prices.pct_change().dropna()


def calculate_cumulative_returns(returns: DataFrame) -> DataFrame:
    """
    Calculate cumulative returns from daily returns.

    Parameters
    ----------
    returns : DataFrame
        DataFrame of daily returns.

    Returns
    -------
    DataFrame
        DataFrame of cumulative returns (base 1).
    """
    return (1 + returns).cumprod()


# =============================================================================
# PORTFOLIO SIMULATION
# =============================================================================

def calculate_weights(
    n_assets: int,
    method: str = "equal",
    custom_weights: List[float] = None
) -> np.ndarray:
    """
    Calculate portfolio weights.

    Parameters
    ----------
    n_assets : int
        Number of assets in the portfolio.
    method : str
        Weighting method ('equal' or 'custom').
    custom_weights : List[float]
        Custom weights (used if method='custom').

    Returns
    -------
    np.ndarray
        Array of normalised weights (sum = 1).
    """
    if method == "equal":
        weights = np.ones(n_assets) / n_assets
    elif method == "custom" and custom_weights is not None:
        weights = np.array(custom_weights)
        weights = weights / weights.sum()
    else:
        weights = np.ones(n_assets) / n_assets

    return weights


def simulate_portfolio(
    prices: DataFrame,
    weights: np.ndarray,
    rebalance_frequency: str = "daily"
) -> Tuple[Series, Series]:
    """
    Simulate portfolio performance with given weights.

    Parameters
    ----------
    prices : DataFrame
        DataFrame of asset prices.
    weights : np.ndarray
        Array of portfolio weights.
    rebalance_frequency : str
        Rebalancing frequency ('daily', 'weekly', 'monthly', 'none').

    Returns
    -------
    equity : Series
        Portfolio equity curve (base 1).
    returns : Series
        Portfolio returns series.
    """
    returns = calculate_returns(prices)

    if rebalance_frequency in ["none", "daily"]:
        portfolio_returns = (returns * weights).sum(axis=1)

    elif rebalance_frequency == "weekly":
        portfolio_returns = _rebalance_portfolio(returns, weights, "W")

    elif rebalance_frequency == "monthly":
        portfolio_returns = _rebalance_portfolio(returns, weights, "ME")

    else:
        portfolio_returns = (returns * weights).sum(axis=1)

    portfolio_returns.name = "portfolio_returns"
    equity = (1 + portfolio_returns).cumprod()
    equity.name = "equity"

    return equity, portfolio_returns


def _rebalance_portfolio(
    returns: DataFrame,
    target_weights: np.ndarray,
    freq: str
) -> Series:
    """
    Internal function to simulate portfolio with periodic rebalancing.
    """
    portfolio_returns = []
    current_weights = target_weights.copy()
    rebalance_dates = returns.resample(freq).last().index

    for date, row in returns.iterrows():
        daily_return = (row * current_weights).sum()
        portfolio_returns.append(daily_return)

        current_weights = current_weights * (1 + row.values)
        current_weights = current_weights / current_weights.sum()

        if date in rebalance_dates:
            current_weights = target_weights.copy()

    return pd.Series(portfolio_returns, index=returns.index)


# =============================================================================
# PORTFOLIO METRICS
# =============================================================================

def correlation_matrix(prices: DataFrame) -> DataFrame:
    """
    Calculate correlation matrix of asset returns.

    Parameters
    ----------
    prices : DataFrame
        DataFrame of asset prices.

    Returns
    -------
    DataFrame
        Correlation matrix.
    """
    returns = calculate_returns(prices)
    return returns.corr()


def covariance_matrix(prices: DataFrame, annualise: bool = True) -> DataFrame:
    """
    Calculate covariance matrix of asset returns.

    Parameters
    ----------
    prices : DataFrame
        DataFrame of asset prices.
    annualise : bool
        If True, annualise the covariance (x252).

    Returns
    -------
    DataFrame
        Covariance matrix.
    """
    returns = calculate_returns(prices)
    cov = returns.cov()

    if annualise:
        cov = cov * 252

    return cov


def portfolio_volatility(
    prices: DataFrame,
    weights: np.ndarray,
    annualise: bool = True
) -> float:
    """
    Calculate portfolio volatility.

    Parameters
    ----------
    prices : DataFrame
        DataFrame of asset prices.
    weights : np.ndarray
        Array of portfolio weights.
    annualise : bool
        If True, annualise the volatility.

    Returns
    -------
    float
        Portfolio volatility.
    """
    cov = covariance_matrix(prices, annualise=False)
    port_var = np.dot(weights.T, np.dot(cov, weights))
    port_vol = np.sqrt(port_var)

    if annualise:
        port_vol = port_vol * np.sqrt(252)

    return float(port_vol)


def portfolio_expected_return(
    prices: DataFrame,
    weights: np.ndarray,
    annualise: bool = True
) -> float:
    """
    Calculate portfolio expected return.

    Parameters
    ----------
    prices : DataFrame
        DataFrame of asset prices.
    weights : np.ndarray
        Array of portfolio weights.
    annualise : bool
        If True, annualise the return.

    Returns
    -------
    float
        Portfolio expected return.
    """
    returns = calculate_returns(prices)
    mean_returns = returns.mean()
    port_return = np.dot(weights, mean_returns)

    if annualise:
        port_return = port_return * 252

    return float(port_return)


def diversification_ratio(prices: DataFrame, weights: np.ndarray) -> float:
    """
    Calculate portfolio diversification ratio.

    DR = weighted average of individual volatilities / portfolio volatility
    A DR > 1 indicates diversification benefit.

    Parameters
    ----------
    prices : DataFrame
        DataFrame of asset prices.
    weights : np.ndarray
        Array of portfolio weights.

    Returns
    -------
    float
        Diversification ratio.
    """
    returns = calculate_returns(prices)
    individual_vols = returns.std() * np.sqrt(252)
    weighted_avg_vol = np.dot(weights, individual_vols)

    port_vol = portfolio_volatility(prices, weights, annualise=True)

    if port_vol == 0:
        return 1.0

    return float(weighted_avg_vol / port_vol)


def max_drawdown(equity: Series) -> float:
    """
    Calculate maximum drawdown from equity curve.

    Parameters
    ----------
    equity : Series
        Equity curve (base 1).

    Returns
    -------
    float
        Maximum drawdown (negative value, e.g., -0.25 for -25%).
    """
    if not isinstance(equity, Series):
        equity = pd.Series(equity.squeeze())

    eq = equity.astype(float).dropna()

    if len(eq) == 0:
        return 0.0

    running_max = eq.cummax()
    dd = (eq - running_max) / running_max
    return float(dd.min())


def sharpe_ratio(
    returns: Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualised Sharpe ratio.

    Parameters
    ----------
    returns : Series
        Portfolio returns series.
    risk_free_rate : float
        Annualised risk-free rate.
    periods_per_year : int
        Number of periods per year (252 for daily).

    Returns
    -------
    float
        Annualised Sharpe ratio.
    """
    if not isinstance(returns, Series):
        returns = pd.Series(returns.squeeze())

    r = returns.astype(float).dropna()

    if r.empty:
        return 0.0

    excess_returns = r - risk_free_rate / periods_per_year
    std = r.std()

    if std == 0 or np.isnan(std):
        return 0.0

    return float(np.sqrt(periods_per_year) * excess_returns.mean() / std)


def individual_metrics(prices: DataFrame) -> DataFrame:
    """
    Calculate metrics for each asset individually.

    Parameters
    ----------
    prices : DataFrame
        DataFrame of asset prices.

    Returns
    -------
    DataFrame
        DataFrame with metrics per asset.
    """
    returns = calculate_returns(prices)
    cumulative = calculate_cumulative_returns(returns)

    metrics = {}
    for ticker in prices.columns:
        metrics[ticker] = {
            "Annual Return": returns[ticker].mean() * 252,
            "Annual Volatility": returns[ticker].std() * np.sqrt(252),
            "Sharpe Ratio": sharpe_ratio(returns[ticker]),
            "Max Drawdown": max_drawdown(cumulative[ticker]),
            "Final Value": cumulative[ticker].iloc[-1],
        }

    return pd.DataFrame(metrics).T


# =============================================================================
# COMPARISON TOOLS
# =============================================================================

def compare_portfolio_vs_assets(
    prices: DataFrame,
    weights: np.ndarray,
    rebalance_frequency: str = "daily"
) -> DataFrame:
    """
    Compare portfolio performance with individual assets.

    Parameters
    ----------
    prices : DataFrame
        DataFrame of asset prices.
    weights : np.ndarray
        Array of portfolio weights.
    rebalance_frequency : str
        Rebalancing frequency.

    Returns
    -------
    DataFrame
        DataFrame with cumulative returns of portfolio and assets.
    """
    returns = calculate_returns(prices)
    cumulative_assets = calculate_cumulative_returns(returns)

    portfolio_equity, _ = simulate_portfolio(prices, weights, rebalance_frequency)

    comparison = cumulative_assets.copy()
    comparison["Portfolio"] = portfolio_equity

    return comparison


def prepare_portfolio_plot(prices: DataFrame, portfolio_equity: Series) -> DataFrame:
    """
    Prepare normalised price and portfolio equity for plotting.

    Both series are normalised to base 1 to allow direct comparison.

    Parameters
    ----------
    prices : DataFrame
        Asset price DataFrame.
    portfolio_equity : Series
        Portfolio equity curve (base 1).

    Returns
    -------
    DataFrame
        DataFrame with normalised prices and portfolio equity.
    """
    returns = calculate_returns(prices)
    cumulative = calculate_cumulative_returns(returns)

    df_plot = cumulative.copy()
    df_plot["Portfolio"] = portfolio_equity

    return df_plot


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PORTFOLIO MODULE TEST")
    print("=" * 60)

    print("\n1. Loading data...")
    prices = load_multi_asset()
    print(f"   Assets: {list(prices.columns)}")
    print(f"   Period: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Days: {len(prices)}")

    print("\n2. Correlation matrix:")
    corr = correlation_matrix(prices)
    print(corr.round(3))

    print("\n3. Equal weight portfolio simulation...")
    weights = calculate_weights(len(prices.columns), method="equal")
    print(f"   Weights: {dict(zip(prices.columns, weights.round(3)))}")

    equity, returns = simulate_portfolio(prices, weights)

    print(f"\n4. Portfolio metrics:")
    print(f"   Annual Return:      {portfolio_expected_return(prices, weights):.2%}")
    print(f"   Annual Volatility:  {portfolio_volatility(prices, weights):.2%}")
    print(f"   Sharpe Ratio:       {sharpe_ratio(returns):.3f}")
    print(f"   Max Drawdown:       {max_drawdown(equity):.2%}")
    print(f"   Diversification:    {diversification_ratio(prices, weights):.3f}")

    print("\n5. Individual asset metrics:")
    individual = individual_metrics(prices)
    print(individual.round(3))

    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
