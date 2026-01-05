import yfinance as yf
import numpy as np
import pandas as pd
from pandas import DataFrame, Series


def get_apple_data(period:str ="1y", interval:str ="1d") -> DataFrame:
    """
    Récupère les données de l'action Apple (AAPL) via Yahoo Finance.

    Parameters
    ----------
    period : str
        Période d'historique (ex: '1y', '6mo', '5d').
    interval : str
        Fréquence des données (ex: '1d', '1h').

    Returns
    -------
    DataFrame
        DataFrame avec au moins une colonne 'Close' et un index datetime.
    """
    df = yf.download("AAPL", period=period, interval=interval, progress=False)
    df = df[["Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df

def strategy_buy_and_hold(df:DataFrame) -> tuple[Series, Series]:
    """
    Stratégie Buy & Hold : investir au début et conserver l'actif.

    Parameters
    ----------
    df : DataFrame
        DataFrame contenant une colonne 'Close'.

    Returns
    -------
    cumulative : Series
        Série du capital cumulé (base 1).
    strategy_return : Series
        Série des rendements de la stratégie.
    """
    df = df.copy()

    # Rendement journalier de l'actif
    df["return"] = df["Close"].pct_change().fillna(0)

    # Pour Buy & Hold, le rendement de la stratégie = rendement de l'actif
    df["strategy_return"] = df["return"]

    # Capital cumulatif (base 1)
    df["cumulative"] = (1 + df["strategy_return"]).cumprod()

    return df["cumulative"], df["strategy_return"]


def strategy_moving_average(df: DataFrame, short: int = 20, long: int = 50) -> tuple[Series, Series]:
    """
    Stratégie de crossover de moyennes mobiles (short / long).

    Parameters
    ----------
    df : DataFrame
        DataFrame contenant une colonne 'Close'.
    short : int
        Fenêtre de la moyenne mobile courte.
    long : int
        Fenêtre de la moyenne mobile longue.

    Returns
    -------
    cumulative : Series
        Série du capital cumulé (base 1).
    strategy_return : Series
        Série des rendements de la stratégie.
    """

    df = df.copy()

    df["short_ma"] = df["Close"].rolling(window=short).mean()
    df["long_ma"] = df["Close"].rolling(window=long).mean()

    df["signal"] = (df["short_ma"] > df["long_ma"]).astype(int)

    df["return"] = df["Close"].pct_change().fillna(0)

    df["strategy_return"] = df["signal"].shift(1).fillna(0) * df["return"]

    df["cumulative"] = (1 + df["strategy_return"]).cumprod()

    return df["cumulative"], df["strategy_return"]

def max_drawdown(cumulative_series):
    """
    Calcule le max drawdown à partir d'une série de capital cumulé.

    Parameters
    ----------
    cumulative : Series
        Série du capital cumulé (base 1).

    Returns
    -------
    float
        Max drawdown (valeur négative, ex: -0.25 = -25%).
    """
    running_max = cumulative_series.cummax()
    drawdown = cumulative_series / running_max - 1.0
    return drawdown.min()


def sharpe_ratio(returns: Series, periods_per_year: int = 252) -> float:
    """
    Calcule le Sharpe ratio à partir d'une série de rendements.

    Parameters
    ----------
    returns : Series
        Série de rendements par période.
    periods_per_year : int
        Nombre de périodes par an (252 pour des données journalières).

    Returns
    -------
    float
        Sharpe ratio annualisé.
    """
    mean_ret = returns.mean()
    std_ret = returns.std()

    if std_ret == 0:
        return 0.0

    return np.sqrt(periods_per_year) * mean_ret / std_ret


if __name__ == "__main__":
    data = get_apple_data()

    print(data.head())
    print(f"Nombre de lignes : {len(data)}")

    # --- Buy & Hold ---
    cum_bh, ret_bh = strategy_buy_and_hold(data)
    print("\n[Buy & Hold]")
    print("Valeur cumulée (5 premières valeurs) :")
    print(cum_bh.head())

    print("Sharpe Buy & Hold :", sharpe_ratio(ret_bh))
    print("Max Drawdown Buy & Hold :", max_drawdown(cum_bh))

    # --- Moving Average ---
    cum_ma, ret_ma = strategy_moving_average(data, short=20, long=50)
    print("\n[Moving Average 20/50]")
    print("Valeur cumulée (5 premières valeurs) :")
    print(cum_ma.head())

    print("Sharpe Moving Average :", sharpe_ratio(ret_ma))
    print("Max Drawdown Moving Average :", max_drawdown(cum_ma))

