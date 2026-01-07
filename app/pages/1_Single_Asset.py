import streamlit as st
import os
import sys


# Ajoute la racine du projet pour autoriser "import single_asset"
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from single_asset import (
    load_asset,
    signal_buy_and_hold,
    signal_moving_average,
    backtest,
    sharpe_ratio,
    max_drawdown,
    annualised_return,
    annualised_volatility,
    prepare_price_equity_plot
)


# ---------------------------------------------------
# Page title
# ---------------------------------------------------
st.title("Module Quant A - Analyse Single Asset")

# ----------------------------
# Sidebar — Global controls
# ----------------------------
st.sidebar.header("Global Parameters")

# asset selection
st.sidebar.header("Asset selection")

asset_map = {
    "Apple (AAPL)": "AAPL",
    "Gold (GC=F)": "GC=F",
    "EUR/USD (EURUSD=X)": "EURUSD=X",
    "Bitcoin (BTC-USD)": "BTC-USD",
}

asset_label = st.sidebar.selectbox(
    "Choose an asset",
    list(asset_map.keys()),
    index=0
)
asset = asset_map[asset_label]

# Periodicity
periodicity = st.sidebar.selectbox(
    "Data frequency",
    ("Daily (1d)", "Weekly (1wk)"),
    index=0,
    key="periodicity"
)

interval_map = {
    "Daily (1d)": "1d",
    "Weekly (1wk)": "1wk",
}
interval = interval_map[periodicity]

# Data history period
period_choice = st.sidebar.selectbox(
    "Historical data range",
    ("6 months", "1 year", "2 years", "5 years"),
    index=1,
    key="period_choice"
)

period_map = {
    "6 months": "6mo",
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y",
}
period = period_map[period_choice]

# ---------------------------------------------------
# Data loading
# ---------------------------------------------------

data = load_asset(
    asset=asset,
    period=period,
    interval=interval
)

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------

tab_price, tab_backtest, tab_forecast = st.tabs(["📈 Price", "📊 Backtest", "🔮 Forecast"])

# ---------------------------------------------------
# Tab 1 - Price
# ---------------------------------------------------

# Display the price
with tab_price:
    st.subheader(f"Asset price {asset_label}")
    #st.line_chart(data["price"])
    data_plot = data.copy()
    data_plot.index = data_plot.index.strftime("%Y-%m-%d")

    st.line_chart(data_plot["price"])

# ---------------------------------------------------
# Tab 2 - Backtesting 
# ---------------------------------------------------

with tab_backtest:
    st.subheader("Strategy parameters")
    
    # Strategy choice
    strategy_choice = st.selectbox(
        "Strategy choice :",
        ("Buy & Hold", "Moving Average"),
        key="strategy_choice"
    )
    if strategy_choice == "Moving Average":
        col1, col2 = st.columns(2)
        with col1:
            short = st.slider("Short Moving Average", 5, 50, 20, step=1, key="short_ma")
        with col2:
            long = st.slider("Long Moving Average", 20, 200, 50, step=1, key="long_ma")
            if long <= short:
                st.warning("Warning La long MA doit être > short MA")

    # --- Strategy choice ---
    if strategy_choice == "Buy & Hold":
        signal = signal_buy_and_hold(data)
    else:
        signal = signal_moving_average(data, short_window=short, long_window=long)

    equity, strat_returns = backtest(data, signal)

    # --- Display Cumulative Capital ---
    st.subheader("Cumulative Capital")

    equity_plot = equity.copy()
    equity_plot.index = equity_plot.index.strftime("%Y-%m-%d")

    st.line_chart(equity_plot)

    # --- Main backtesting chart ---
    st.subheader("Price vs Strategy Performance")

    plot_df = prepare_price_equity_plot(
        price=data["price"],
        equity=equity
    )
    plot_df.index = plot_df.index.strftime("%Y-%m-%d")

    st.line_chart(plot_df)

    # --- Display metrics ---
    st.subheader("Metrics")
    c1, c2, c3, c4 = st.columns(4)

    # choose annualization factor based on interval (adapt this mapping to your UI)
    c1.metric("Sharpe Ratio", f"{sharpe_ratio(strat_returns, interval=interval):.3f}")
    c2.metric("Max Drawdown", f"{max_drawdown(equity):.2%}")
    c3.metric("Annualised return", f"{annualised_return(strat_returns, interval=interval):.2%}")
    c4.metric("Annualised volatility", f"{annualised_volatility(strat_returns, interval=interval):.2%}")

# ---------------------------------------------------
# Tab 3 — Forecast
# ---------------------------------------------------

with tab_forecast:
    st.subheader("Predictive model")
    st.info(
        "Ici on ajoutera un modèle simple (ex: régression linéaire ou ARIMA) pour prédire "
        "les valeurs futures et visualiser la prédiction avec un intervalle de confiance."
    )