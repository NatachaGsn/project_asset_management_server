import streamlit as st
import os
import sys
import plotly.graph_objects as go

# Ajoute la racine du projet pour autoriser "import single_asset"
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from single_asset import (
    load_asset,
    canonicalise_price_data,
    signal_buy_and_hold,
    signal_moving_average,
    backtest,
    periods_per_year,
    sharpe_ratio,
    max_drawdown,
    annualised_return,
    annualised_volatility,
    prepare_price_equity_plot
)

from single_asset_modelling import (
    arima_forecast_split,
    inverse_transform_for_display)

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
asset_class = "crypto" if asset.endswith("-USD") else "equity"

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

raw_data = load_asset(
    asset=asset,
    period=period,
    interval=interval
)
data = canonicalise_price_data(raw_data)

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
    
    data_plot = data.copy()
    data_plot.index = data_plot.index.strftime("%Y-%m-%d")

    st.line_chart(data_plot)

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
        price=data,
        equity=equity
    )
    plot_df.index = plot_df.index.strftime("%Y-%m-%d")

    st.line_chart(plot_df)

    if strategy_choice == "Buy & Hold":
        st.caption(
            "Note — For the Buy & Hold strategy, the cumulative strategy value "
            "evolves proportionally to the asset price. "
            "Both curves therefore perfectly overlap and appear as a single line."
        )

    # --- Display metrics ---
    st.subheader("Metrics")
    c1, c2, c3, c4 = st.columns(4)

    ppyear = periods_per_year(interval=interval, asset_class=asset_class)

    # choose annualization factor based on interval
    c1.metric("Sharpe Ratio", f"{sharpe_ratio(strat_returns, periods_per_year=ppyear):.3f}")
    c2.metric("Max Drawdown", f"{max_drawdown(equity):.2%}")
    c3.metric("Annualised return (CAGR)", f"{annualised_return(equity):.2%}")
    c4.metric("Annualised volatility", f"{annualised_volatility(strat_returns, periods_per_year=ppyear):.2%}")

# ---------------------------------------------------
# Tab 3 — Forecast
# ---------------------------------------------------

with tab_forecast:
    st.subheader("Forecast (ARIMA)")

    col1, col2, col3 = st.columns(3)
    with col1:
        horizon = st.slider("Forecast horizon (periods)", 5, 60, 30, step=5)
    with col2:
        train_ratio = st.slider("Train ratio", 0.6, 0.9, 0.8, step=0.05)
    with col3:
        ci_level = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1)

    # Run model
    try:
        res = arima_forecast_split(
            df_raw=data,
            interval=interval,
            horizon=horizon,
            train_ratio=train_ratio,
            order=(1, 1, 1),
            ci_level=ci_level
        )
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        st.stop()

    # Show preparation info (useful and professional)
    info = res["prep_info"]
    if info["pct_missing_introduced"] > 0:
        st.warning(
            f"Missing dates introduced: {info['n_missing_introduced']} "
            f"({info['pct_missing_introduced']:.1f}%). Fill method: {info['fill_method']}."
        )
    st.caption(
        f"Frequency used: {info['frequency']} | "
        f"Period: {info['start_date'].date()} → {info['end_date'].date()} | "
        f"Observations: {info['n_final']}"
    )

    # Extract series
    y = res["y"]
    y_train = res["train"]
    y_test = res["test"]

    pred_test = res["pred_test"]
    pred_test_ci = res["pred_test_ci"]

    forecast_future = res["forecast_future"]
    forecast_future_ci = res["forecast_future_ci"]

    # --- Table: future forecast ---
    st.markdown("**Future forecast (with confidence intervals)**")
    future_table = forecast_future_ci.copy()
    future_table.insert(0, "forecast", forecast_future)
    st.dataframe(future_table, width="stretch")

    # ------------ chart --------------
    st.markdown("**Chart: historical, test prediction, and future forecast (interactive)**")

    show_full_history = st.checkbox("Show full history", value=False)
    if show_full_history:
        y_plot = y
    else:
        max_pts = min(3000, len(y))
        default_pts = min(500, len(y))
        min_pts = min(100, max_pts)

        if max_pts <= min_pts:   # covers 0/1 point edge cases too
            history_window = max_pts
            st.caption(f"Only {max_pts} points available — showing full history.")
        else:
            default_pts = max(min_pts, default_pts)
            history_window = st.slider(
                "History window (points)",
                min_value=min_pts,
                max_value=max_pts,
                value=default_pts,
                step=50 if max_pts >= 150 else 1
            )

        y_plot = y.iloc[-history_window:]

    xmin = y_plot.index[0]

    transform = res["prep_info"]["transform"]

    # Restrict test & CI to visible window
    y_test_plot = y_test[y_test.index >= xmin]
    pred_test_plot = pred_test[pred_test.index >= xmin]
    pred_test_ci_plot = pred_test_ci.loc[pred_test_ci.index >= xmin]

    # Convert to price space for display (if needed)
    y_plot_disp = inverse_transform_for_display(y_plot, transform)
    y_test_plot_disp = inverse_transform_for_display(y_test_plot, transform)
    pred_test_plot_disp = inverse_transform_for_display(pred_test_plot, transform)
    pred_test_ci_plot_disp = inverse_transform_for_display(pred_test_ci_plot, transform)

    forecast_future_disp = inverse_transform_for_display(forecast_future, transform)
    forecast_future_ci_disp = inverse_transform_for_display(forecast_future_ci, transform)

    # ------- Trace ---------

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=y_plot_disp.index, y=y_plot_disp.values,
        mode="lines", name="Historical price"
    ))

    # Test actual
    fig.add_trace(go.Scatter(
        x=y_test_plot_disp.index, y=y_test_plot_disp.values,
        mode="lines", name="Actual (test)"
    ))

    # Test prediction
    fig.add_trace(go.Scatter(
        x=pred_test_plot_disp.index, y=pred_test_plot_disp.values,
        mode="lines", name="Prediction (test)"
    ))

    # Test CI band
    fig.add_trace(go.Scatter(
        x=pred_test_ci_plot_disp.index, y=pred_test_ci_plot_disp["upper"],
        mode="lines", line=dict(width=0),
        name=f"Test CI ({int(ci_level*100)}%)",
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=pred_test_ci_plot_disp.index, y=pred_test_ci_plot_disp["lower"],
        mode="lines", line=dict(width=0),
        fill="tonexty",
        name=f"Test CI ({int(ci_level*100)}%)"
    ))

    # Future forecast
    fig.add_trace(go.Scatter(
        x=forecast_future_disp.index, y=forecast_future_disp.values,
        mode="lines", name="Forecast (future)"
    ))

    # Future CI band
    fig.add_trace(go.Scatter(
        x=forecast_future_ci_disp.index, y=forecast_future_ci_disp["upper"],
        mode="lines", line=dict(width=0),
        name=f"Forecast CI ({int(ci_level*100)}%)",
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast_future_ci_disp.index, y=forecast_future_ci_disp["lower"],
        mode="lines", line=dict(width=0),
        fill="tonexty",
        name=f"Forecast CI ({int(ci_level*100)}%)"
    ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=30, b=10),
    )

    st.plotly_chart(fig, width="stretch")