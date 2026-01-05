import yfinance as yf
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from typing import List, Dict, Tuple

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_TICKERS = ["SPY", "TLT", "GLD", "VNQ"]
TICKER_NAMES = {
    "SPY": "S&P 500 (Actions US)",
    "TLT": "Obligations 20+ ans",
    "GLD": "Or (Gold Trust)",
    "VNQ": "Immobilier (REITs)",
}


# =============================================================================
# DATA FETCHING
# =============================================================================

def get_multi_asset_data(
    tickers: List[str] = DEFAULT_TICKERS,
    period: str = "1y",
    interval: str = "1d"
) -> DataFrame:
    """
    Récupère les données de plusieurs actifs via Yahoo Finance.

    Parameters
    ----------
    tickers : List[str]
        Liste des tickers à récupérer.
    period : str
        Période d'historique (ex: '1y', '6mo', '5d').
    interval : str
        Fréquence des données (ex: '1d', '1h').

    Returns
    -------
    DataFrame
        DataFrame avec les prix de clôture de chaque actif en colonnes.
    """
    df = yf.download(tickers, period=period, interval=interval, progress=False)
    
    # yfinance retourne un MultiIndex si plusieurs tickers
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    else:
        df = df[["Close"]]
        df.columns = tickers
    
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    
    return df


def get_asset_info(ticker: str) -> Dict:
    """
    Récupère les informations d'un actif (nom, secteur, etc.).

    Parameters
    ----------
    ticker : str
        Ticker de l'actif.

    Returns
    -------
    Dict
        Dictionnaire avec les informations de l'actif.
    """
    try:
        asset = yf.Ticker(ticker)
        info = asset.info
        return {
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "currency": info.get("currency", "USD"),
        }
    except Exception:
        return {"name": ticker, "sector": "N/A", "currency": "USD"}


# =============================================================================
# RETURNS CALCULATION
# =============================================================================

def calculate_returns(prices: DataFrame) -> DataFrame:
    """
    Calcule les rendements journaliers à partir des prix.

    Parameters
    ----------
    prices : DataFrame
        DataFrame des prix de clôture.

    Returns
    -------
    DataFrame
        DataFrame des rendements journaliers.
    """
    returns = prices.pct_change().dropna()
    return returns


def calculate_cumulative_returns(returns: DataFrame) -> DataFrame:
    """
    Calcule les rendements cumulés à partir des rendements journaliers.

    Parameters
    ----------
    returns : DataFrame
        DataFrame des rendements journaliers.

    Returns
    -------
    DataFrame
        DataFrame des rendements cumulés (base 1).
    """
    cumulative = (1 + returns).cumprod()
    return cumulative


# =============================================================================
# PORTFOLIO SIMULATION
# =============================================================================

def calculate_portfolio_weights(
    n_assets: int,
    method: str = "equal",
    custom_weights: List[float] = None
) -> np.ndarray:
    """
    Calcule les poids du portfolio selon la méthode choisie.

    Parameters
    ----------
    n_assets : int
        Nombre d'actifs dans le portfolio.
    method : str
        Méthode de pondération ('equal' ou 'custom').
    custom_weights : List[float]
        Poids personnalisés (utilisé si method='custom').

    Returns
    -------
    np.ndarray
        Array des poids normalisés (somme = 1).
    """
    if method == "equal":
        weights = np.ones(n_assets) / n_assets
    elif method == "custom" and custom_weights is not None:
        weights = np.array(custom_weights)
        weights = weights / weights.sum()  # Normalisation
    else:
        weights = np.ones(n_assets) / n_assets
    
    return weights


def simulate_portfolio(
    prices: DataFrame,
    weights: np.ndarray,
    rebalance_frequency: str = "none"
) -> Tuple[Series, Series]:
    """
    Simule la performance d'un portfolio avec les poids donnés.

    Parameters
    ----------
    prices : DataFrame
        DataFrame des prix de clôture.
    weights : np.ndarray
        Array des poids du portfolio.
    rebalance_frequency : str
        Fréquence de rebalancement ('none', 'daily', 'weekly', 'monthly').

    Returns
    -------
    cumulative : Series
        Série du capital cumulé du portfolio (base 1).
    returns : Series
        Série des rendements du portfolio.
    """
    returns = calculate_returns(prices)
    
    if rebalance_frequency == "none" or rebalance_frequency == "daily":
        # Portfolio à poids constants (rebalancé quotidiennement)
        portfolio_returns = (returns * weights).sum(axis=1)
    
    elif rebalance_frequency == "weekly":
        portfolio_returns = _rebalance_portfolio(returns, weights, "W")
    
    elif rebalance_frequency == "monthly":
        portfolio_returns = _rebalance_portfolio(returns, weights, "ME")
    
    else:
        portfolio_returns = (returns * weights).sum(axis=1)
    
    cumulative = (1 + portfolio_returns).cumprod()
    
    return cumulative, portfolio_returns


def _rebalance_portfolio(
    returns: DataFrame,
    target_weights: np.ndarray,
    freq: str
) -> Series:
    """
    Fonction interne pour simuler un portfolio avec rebalancement périodique.

    Parameters
    ----------
    returns : DataFrame
        DataFrame des rendements journaliers.
    target_weights : np.ndarray
        Poids cibles du portfolio.
    freq : str
        Fréquence pandas ('W' pour weekly, 'ME' pour monthly).

    Returns
    -------
    Series
        Série des rendements du portfolio.
    """
    portfolio_returns = []
    current_weights = target_weights.copy()
    
    # Identifier les dates de rebalancement
    rebalance_dates = returns.resample(freq).last().index
    
    for date, row in returns.iterrows():
        # Rendement du jour avec les poids actuels
        daily_return = (row * current_weights).sum()
        portfolio_returns.append(daily_return)
        
        # Mise à jour des poids selon la performance
        current_weights = current_weights * (1 + row.values)
        current_weights = current_weights / current_weights.sum()
        
        # Rebalancement si nécessaire
        if date in rebalance_dates:
            current_weights = target_weights.copy()
    
    return pd.Series(portfolio_returns, index=returns.index)


# =============================================================================
# PORTFOLIO METRICS
# =============================================================================

def calculate_correlation_matrix(prices: DataFrame) -> DataFrame:
    """
    Calcule la matrice de corrélation des rendements.

    Parameters
    ----------
    prices : DataFrame
        DataFrame des prix de clôture.

    Returns
    -------
    DataFrame
        Matrice de corrélation.
    """
    returns = calculate_returns(prices)
    return returns.corr()


def calculate_covariance_matrix(prices: DataFrame, annualize: bool = True) -> DataFrame:
    """
    Calcule la matrice de covariance des rendements.

    Parameters
    ----------
    prices : DataFrame
        DataFrame des prix de clôture.
    annualize : bool
        Si True, annualise la covariance (x252).

    Returns
    -------
    DataFrame
        Matrice de covariance.
    """
    returns = calculate_returns(prices)
    cov_matrix = returns.cov()
    
    if annualize:
        cov_matrix = cov_matrix * 252
    
    return cov_matrix


def calculate_portfolio_volatility(
    prices: DataFrame,
    weights: np.ndarray,
    annualize: bool = True
) -> float:
    """
    Calcule la volatilité du portfolio.

    Parameters
    ----------
    prices : DataFrame
        DataFrame des prix de clôture.
    weights : np.ndarray
        Array des poids du portfolio.
    annualize : bool
        Si True, annualise la volatilité.

    Returns
    -------
    float
        Volatilité du portfolio.
    """
    cov_matrix = calculate_covariance_matrix(prices, annualize=False)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_vol = np.sqrt(portfolio_variance)
    
    if annualize:
        portfolio_vol = portfolio_vol * np.sqrt(252)
    
    return portfolio_vol


def calculate_portfolio_expected_return(
    prices: DataFrame,
    weights: np.ndarray,
    annualize: bool = True
) -> float:
    """
    Calcule le rendement espéré du portfolio.

    Parameters
    ----------
    prices : DataFrame
        DataFrame des prix de clôture.
    weights : np.ndarray
        Array des poids du portfolio.
    annualize : bool
        Si True, annualise le rendement.

    Returns
    -------
    float
        Rendement espéré du portfolio.
    """
    returns = calculate_returns(prices)
    mean_returns = returns.mean()
    portfolio_return = np.dot(weights, mean_returns)
    
    if annualize:
        portfolio_return = portfolio_return * 252
    
    return portfolio_return


def calculate_diversification_ratio(
    prices: DataFrame,
    weights: np.ndarray
) -> float:
    """
    Calcule le ratio de diversification du portfolio.
    
    DR = (somme pondérée des volatilités individuelles) / volatilité du portfolio
    
    Un DR > 1 indique un bénéfice de diversification.

    Parameters
    ----------
    prices : DataFrame
        DataFrame des prix de clôture.
    weights : np.ndarray
        Array des poids du portfolio.

    Returns
    -------
    float
        Ratio de diversification.
    """
    returns = calculate_returns(prices)
    individual_vols = returns.std() * np.sqrt(252)
    weighted_avg_vol = np.dot(weights, individual_vols)
    
    portfolio_vol = calculate_portfolio_volatility(prices, weights, annualize=True)
    
    if portfolio_vol == 0:
        return 1.0
    
    return weighted_avg_vol / portfolio_vol


def max_drawdown(cumulative_series: Series) -> float:
    """
    Calcule le max drawdown à partir d'une série de capital cumulé.

    Parameters
    ----------
    cumulative_series : Series
        Série du capital cumulé (base 1).

    Returns
    -------
    float
        Max drawdown (valeur négative, ex: -0.25 = -25%).
    """
    running_max = cumulative_series.cummax()
    drawdown = cumulative_series / running_max - 1.0
    return drawdown.min()


def sharpe_ratio(returns: Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calcule le Sharpe ratio à partir d'une série de rendements.

    Parameters
    ----------
    returns : Series
        Série de rendements par période.
    risk_free_rate : float
        Taux sans risque annualisé.
    periods_per_year : int
        Nombre de périodes par an (252 pour des données journalières).

    Returns
    -------
    float
        Sharpe ratio annualisé.
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    mean_ret = excess_returns.mean()
    std_ret = returns.std()
    
    if std_ret == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * mean_ret / std_ret


def calculate_individual_metrics(prices: DataFrame) -> DataFrame:
    """
    Calcule les métriques pour chaque actif individuellement.

    Parameters
    ----------
    prices : DataFrame
        DataFrame des prix de clôture.

    Returns
    -------
    DataFrame
        DataFrame avec les métriques par actif.
    """
    returns = calculate_returns(prices)
    cumulative = calculate_cumulative_returns(returns)
    
    metrics = {}
    for ticker in prices.columns:
        metrics[ticker] = {
            "Return (Ann.)": returns[ticker].mean() * 252,
            "Volatility (Ann.)": returns[ticker].std() * np.sqrt(252),
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
    rebalance_frequency: str = "none"
) -> DataFrame:
    """
    Compare la performance du portfolio avec les actifs individuels.

    Parameters
    ----------
    prices : DataFrame
        DataFrame des prix de clôture.
    weights : np.ndarray
        Array des poids du portfolio.
    rebalance_frequency : str
        Fréquence de rebalancement.

    Returns
    -------
    DataFrame
        DataFrame avec les rendements cumulés du portfolio et des actifs.
    """
    # Rendements cumulés des actifs individuels
    returns = calculate_returns(prices)
    cumulative_assets = calculate_cumulative_returns(returns)
    
    # Rendement cumulé du portfolio
    portfolio_cumulative, _ = simulate_portfolio(prices, weights, rebalance_frequency)
    
    # Combiner
    comparison = cumulative_assets.copy()
    comparison["Portfolio"] = portfolio_cumulative
    
    return comparison


# =============================================================================
# MAIN (pour tester le module)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DU MODULE PORTFOLIO")
    print("=" * 60)
    
    # Récupérer les données
    print("\n1. Récupération des données...")
    prices = get_multi_asset_data()
    print(f"   Actifs : {list(prices.columns)}")
    print(f"   Période : {prices.index[0].date()} à {prices.index[-1].date()}")
    print(f"   Nombre de jours : {len(prices)}")
    
    # Matrice de corrélation
    print("\n2. Matrice de corrélation :")
    corr = calculate_correlation_matrix(prices)
    print(corr.round(3))
    
    # Portfolio equal weight
    print("\n3. Simulation Portfolio Equal Weight...")
    weights = calculate_portfolio_weights(len(prices.columns), method="equal")
    print(f"   Poids : {dict(zip(prices.columns, weights.round(3)))}")
    
    cumulative, returns = simulate_portfolio(prices, weights)
    
    print(f"\n4. Métriques du Portfolio :")
    print(f"   Rendement annualisé : {calculate_portfolio_expected_return(prices, weights):.2%}")
    print(f"   Volatilité annualisée : {calculate_portfolio_volatility(prices, weights):.2%}")
    print(f"   Sharpe Ratio : {sharpe_ratio(returns):.3f}")
    print(f"   Max Drawdown : {max_drawdown(cumulative):.2%}")
    print(f"   Ratio de diversification : {calculate_diversification_ratio(prices, weights):.3f}")
    
    # Métriques individuelles
    print("\n5. Métriques par actif :")
    individual = calculate_individual_metrics(prices)
    print(individual.round(3))
    
    print("\n" + "=" * 60)
    print("TEST TERMINÉ AVEC SUCCÈS ✓")
    print("=" * 60)
