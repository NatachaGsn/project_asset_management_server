import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from portfolio import (
    load_multi_asset,
    calculate_weights,
    simulate_portfolio,
    correlation_matrix,
    portfolio_volatility,
    portfolio_expected_return,
    diversification_ratio,
    individual_metrics,
    compare_portfolio_vs_assets,
    sharpe_ratio,
    max_drawdown,
    calculate_returns,
    calculate_cumulative_returns,
    TICKER_NAMES,
    DEFAULT_TICKERS,
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(page_title="Portfolio Analysis", page_icon="📊", layout="wide")

st.title("📊 Quant B - Multi-Asset Portfolio Analysis")
st.markdown("*Portfolio simulation and diversification analysis*")

# =============================================================================
# SIDEBAR - PARAMETERS
# =============================================================================

st.sidebar.header("⚙️ Portfolio Parameters")

# Asset selection
st.sidebar.subheader("1. Asset Selection")

available_tickers = {
    "SPY": "S&P 500 (US Equities)",
    "TLT": "Treasury Bonds 20+ Years",
    "GLD": "Gold Trust",
    "VNQ": "Real Estate (REITs)",
    "QQQ": "Nasdaq 100",
    "EFA": "International Equities",
    "AGG": "Aggregate Bonds",
    "VWO": "Emerging Markets",
    "USO": "Oil",
    "SLV": "Silver",
}

selected_tickers = st.sidebar.multiselect(
    "Select assets (min. 3):",
    options=list(available_tickers.keys()),
    default=DEFAULT_TICKERS,
    format_func=lambda x: f"{x} - {available_tickers[x]}",
)

if len(selected_tickers) < 3:
    st.warning("⚠️ Please select at least 3 assets for portfolio analysis.")
    st.stop()

# Period
st.sidebar.subheader("2. Analysis Period")
period = st.sidebar.selectbox(
    "Historical period:",
    options=["6mo", "1y", "2y", "5y"],
    index=1,
    format_func=lambda x: {"6mo": "6 months", "1y": "1 year", "2y": "2 years", "5y": "5 years"}[x]
)

# Weighting method
st.sidebar.subheader("3. Weighting Method")
weight_method = st.sidebar.radio(
    "Method:",
    options=["equal", "custom"],
    format_func=lambda x: "Equal Weight" if x == "equal" else "Custom Weights"
)

# Custom weights
custom_weights = []
if weight_method == "custom":
    st.sidebar.markdown("**Set weights (%):**")
    total_weight = 0
    for ticker in selected_tickers:
        weight = st.sidebar.slider(
            f"{ticker}",
            min_value=0,
            max_value=100,
            value=100 // len(selected_tickers),
            step=5,
            key=f"weight_{ticker}"
        )
        custom_weights.append(weight)
        total_weight += weight

    if total_weight != 100:
        st.sidebar.warning(f"⚠️ Total = {total_weight}% (must be 100%)")

    custom_weights = [w / 100 for w in custom_weights]

# Rebalancing frequency
st.sidebar.subheader("4. Rebalancing")
rebalance_freq = st.sidebar.selectbox(
    "Frequency:",
    options=["daily", "weekly", "monthly", "none"],
    index=0,
    format_func=lambda x: {
        "daily": "Daily",
        "weekly": "Weekly",
        "monthly": "Monthly",
        "none": "No rebalancing"
    }[x]
)

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(ttl=300)
def load_data(tickers, period):
    """Load data with caching."""
    return load_multi_asset(list(tickers), period=period)

with st.spinner("Loading data..."):
    try:
        prices = load_data(tuple(selected_tickers), period)
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

# =============================================================================
# CALCULATE PORTFOLIO
# =============================================================================

if weight_method == "equal":
    weights = calculate_weights(len(selected_tickers), method="equal")
else:
    weights = calculate_weights(len(selected_tickers), method="custom", custom_weights=custom_weights)

portfolio_equity, portfolio_returns = simulate_portfolio(prices, weights, rebalance_freq)

# =============================================================================
# MAIN CONTENT
# =============================================================================

# Info row
# Info row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("📅 Start", prices.index[0].strftime('%d/%m/%y'))
with col2:
    st.metric("📅 End", prices.index[-1].strftime('%d/%m/%y'))
with col3:
    st.metric("📈 Assets", len(selected_tickers))
with col4:
    st.metric("📊 Observations", len(prices))
with col5:
    st.metric("🔄 Rebalancing", rebalance_freq.capitalize())

st.divider()
# =============================================================================
# TAB LAYOUT
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance",
    "📊 Correlations",
    "📋 Metrics",
    "⚖️ Allocation",
    "📄 Daily Reports"
])

# -----------------------------------------------------------------------------
# TAB 1: PERFORMANCE
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Performance Comparison: Assets vs Portfolio")

    comparison_data = compare_portfolio_vs_assets(prices, weights, rebalance_freq)

    fig = go.Figure()

    colors = px.colors.qualitative.Set2
    for i, ticker in enumerate(selected_tickers):
        fig.add_trace(go.Scatter(
            x=comparison_data.index,
            y=comparison_data[ticker],
            name=f"{ticker}",
            line=dict(width=1.5, color=colors[i % len(colors)]),
            opacity=0.7
        ))

    fig.add_trace(go.Scatter(
        x=comparison_data.index,
        y=comparison_data["Portfolio"],
        name="📊 PORTFOLIO",
        line=dict(width=3, color="#FF6B6B", dash="solid"),
    ))

    fig.update_layout(
        title="Cumulative Returns (Base 1)",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Asset prices
    st.subheader("Asset Prices")

    normalized_prices = prices / prices.iloc[0] * 100

    fig_prices = go.Figure()
    for i, ticker in enumerate(selected_tickers):
        fig_prices.add_trace(go.Scatter(
            x=normalized_prices.index,
            y=normalized_prices[ticker],
            name=ticker,
            line=dict(width=2, color=colors[i % len(colors)]),
        ))

    fig_prices.update_layout(
        title="Normalised Prices (Base 100)",
        xaxis_title="Date",
        yaxis_title="Price (base 100)",
        hovermode="x unified",
        height=400,
    )

    st.plotly_chart(fig_prices, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: CORRELATIONS
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Correlation Matrix")

    corr_matrix = correlation_matrix(prices)

    fig_corr = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.index,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
    )

    fig_corr.update_layout(
        title="Return Correlations",
        height=500,
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    # Interpretation
    st.markdown("### 💡 Interpretation")

    corr_values = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    min_corr = corr_values.min().min()
    max_corr = corr_values.max().max()

    min_pair = corr_values.stack().idxmin()
    max_pair = corr_values.stack().idxmax()

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Lowest correlation:** {min_pair[0]} / {min_pair[1]} = {min_corr:.3f}")
        st.caption("→ Good diversification between these assets")
    with col2:
        st.warning(f"**Highest correlation:** {max_pair[0]} / {max_pair[1]} = {max_corr:.3f}")
        st.caption("→ These assets often move together")

# -----------------------------------------------------------------------------
# TAB 3: METRICS
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Portfolio Metrics")

    col1, col2, col3, col4 = st.columns(4)

    port_return = portfolio_expected_return(prices, weights)
    port_vol = portfolio_volatility(prices, weights)
    port_sharpe = sharpe_ratio(portfolio_returns)
    port_mdd = max_drawdown(portfolio_equity)
    div_ratio = diversification_ratio(prices, weights)

    with col1:
        st.metric("📈 Annual Return", f"{port_return:.2%}")
    with col2:
        st.metric("📉 Annual Volatility", f"{port_vol:.2%}")
    with col3:
        st.metric("⚡ Sharpe Ratio", f"{port_sharpe:.3f}")
    with col4:
        st.metric("🔻 Max Drawdown", f"{port_mdd:.2%}")

    st.divider()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            "🎯 Diversification Ratio",
            f"{div_ratio:.3f}",
            help="A ratio > 1 indicates diversification benefit"
        )
    with col2:
        if div_ratio > 1:
            st.success(f"✅ Portfolio benefits from diversification! Volatility is reduced by {(div_ratio - 1) * 100:.1f}% compared to a simple average.")
        else:
            st.warning("⚠️ Low diversification effect - assets are too correlated.")

    st.divider()
    st.subheader("Comparison with Individual Assets")

    ind_metrics = individual_metrics(prices)

    portfolio_metrics = pd.DataFrame({
        "📊 PORTFOLIO": {
            "Annual Return": port_return,
            "Annual Volatility": port_vol,
            "Sharpe Ratio": port_sharpe,
            "Max Drawdown": port_mdd,
            "Final Value": portfolio_equity.iloc[-1],
        }
    }).T

    all_metrics = pd.concat([ind_metrics, portfolio_metrics])

    styled_df = all_metrics.style.format({
        "Annual Return": "{:.2%}",
        "Annual Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.3f}",
        "Max Drawdown": "{:.2%}",
        "Final Value": "{:.3f}",
    }).background_gradient(cmap="RdYlGn", subset=["Sharpe Ratio"])

    st.dataframe(styled_df, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: ALLOCATION
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Portfolio Allocation")

    col1, col2 = st.columns(2)

    with col1:
        weight_df = pd.DataFrame({
            "Asset": selected_tickers,
            "Weight": weights,
            "Weight (%)": [f"{w:.1%}" for w in weights]
        })

        fig_pie = px.pie(
            weight_df,
            values="Weight",
            names="Asset",
            title="Portfolio Allocation",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')

        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("### Weight Details")

        weight_detail = pd.DataFrame({
            "Asset": selected_tickers,
            "Description": [available_tickers.get(t, t) for t in selected_tickers],
            "Weight": [f"{w:.2%}" for w in weights],
        })

        st.dataframe(weight_detail, use_container_width=True, hide_index=True)

        st.markdown("### 📋 Summary")
        st.info(f"""
        - **Method**: {"Equal Weight" if weight_method == "equal" else "Custom"}
        - **Number of assets**: {len(selected_tickers)}
        - **Rebalancing**: {rebalance_freq.capitalize()}
        """)

# -----------------------------------------------------------------------------
# TAB 5: DAILY REPORTS
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("📄 Daily Reports")

    reports_dir = os.path.join(ROOT_DIR, "reports")

    if os.path.exists(reports_dir):
        report_files = sorted(
            [f for f in os.listdir(reports_dir) if f.startswith("report_") and f.endswith(".txt")],
            reverse=True
        )

        if report_files:
            st.success(f"✅ {len(report_files)} report(s) available")

            selected_report = st.selectbox(
                "Select a report:",
                options=report_files,
                format_func=lambda x: x.replace("report_", "").replace(".txt", "")
            )

            if selected_report:
                report_path = os.path.join(reports_dir, selected_report)
                with open(report_path, "r") as f:
                    report_content = f.read()

                st.code(report_content, language=None)

                st.download_button(
                    label="📥 Download Report",
                    data=report_content,
                    file_name=selected_report,
                    mime="text/plain"
                )
        else:
            st.warning("No reports available yet. Reports are generated daily at 8pm.")
    else:
        st.warning("Reports directory not found.")

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption("📊 Quant B - Multi-Asset Portfolio | Data: Yahoo Finance | Auto-refresh: 5 min")
