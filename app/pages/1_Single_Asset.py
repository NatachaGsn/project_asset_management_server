import streamlit as st
import os
import sys

# Ajoute la racine du projet pour autoriser "import single_asset"
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from single_asset import (
    get_apple_data,
    strategy_buy_and_hold,
    strategy_moving_average,
    sharpe_ratio,
    max_drawdown,
)

# Titre de la page
st.title("Module Quant A - Analyse Single Asset")

# Récupération des données
data = get_apple_data(period="1y", interval="1d")

# Choix de la stratégie
strategy_choice = st.selectbox(
    "Choisir la stratégie :",
    ("Buy & Hold", "Moving Average")
)

# Paramètres strat Moving Average
if strategy_choice == "Moving Average":
    col1, col2 = st.columns(2)
    with col1:
        short = st.slider("Short Moving Average", 5, 50, 20, step=1)
    with col2:
        long = st.slider("Long Moving Average", 20, 200, 50, step=1)
        if long <= short:
            st.warning("Warning : La long MA doit être > short MA")

# Calcul de la stratégie
if strategy_choice == "Buy & Hold":
    cumulative, returns = strategy_buy_and_hold(data)
else:
    cumulative, returns = strategy_moving_average(data, short=short, long=long)

# Affichage du prix
st.subheader("Prix de l'actif (AAPL)")
st.line_chart(data["Close"])

# Affichage du capital cumulé
st.subheader("Capital Cumulé")
st.line_chart(cumulative)

# Affichage des métriques
st.subheader("Métriques")

st.write(f"**Sharpe Ratio :** {sharpe_ratio(returns):.3f}")
st.write(f"**Max Drawdown :** {max_drawdown(cumulative):.2%}")

