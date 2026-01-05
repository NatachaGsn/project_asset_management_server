import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Ajoute la racine du projet pour autoriser "import portfolio"
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from portfolio import (
    get_multi_asset_data,
    calculate_portfolio_weights,
    simulate_portfolio,
    calculate_correlation_matrix,
    calculate_portfolio_volatility,
    calculate_portfolio_expected_return,
    calculate_diversification_ratio,
    calculate_individual_metrics,
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

st.title("📊 Module Quant B - Portfolio Multi-Asset")
st.markdown("*Analyse et simulation de portefeuille diversifié*")

# =============================================================================
# SIDEBAR - PARAMETERS
# =============================================================================

st.sidebar.header("⚙️ Paramètres du Portfolio")

# Sélection des actifs
st.sidebar.subheader("1. Sélection des actifs")

available_tickers = {
    "SPY": "S&P 500 (Actions US)",
    "TLT": "Obligations 20+ ans",
    "GLD": "Or (Gold Trust)",
    "VNQ": "Immobilier (REITs)",
    "QQQ": "Nasdaq 100",
    "EFA": "Actions Internationales",
    "AGG": "Obligations Aggregate",
    "VWO": "Marchés Émergents",
    "USO": "Pétrole",
    "SLV": "Argent",
}

selected_tickers = st.sidebar.multiselect(
    "Choisir les actifs (min. 3) :",
    options=list(available_tickers.keys()),
    default=DEFAULT_TICKERS,
    format_func=lambda x: f"{x} - {available_tickers[x]}",
)

if len(selected_tickers) < 3:
    st.warning("⚠️ Veuillez sélectionner au moins 3 actifs pour l'analyse portfolio.")
    st.stop()

# Période
st.sidebar.subheader("2. Période d'analyse")
period = st.sidebar.selectbox(
    "Période historique :",
    options=["6mo", "1y", "2y", "5y"],
    index=1,
    format_func=lambda x: {"6mo": "6 mois", "1y": "1 an", "2y": "2 ans", "5y": "5 ans"}[x]
)

# Méthode de pondération
st.sidebar.subheader("3. Pondération")
weight_method = st.sidebar.radio(
    "Méthode :",
    options=["equal", "custom"],
    format_func=lambda x: "Equal Weight (équipondéré)" if x == "equal" else "Custom (personnalisé)"
)

# Poids personnalisés
custom_weights = []
if weight_method == "custom":
    st.sidebar.markdown("**Définir les poids (%) :**")
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
        st.sidebar.warning(f"⚠️ Total = {total_weight}% (doit être 100%)")
    
    custom_weights = [w / 100 for w in custom_weights]

# Fréquence de rebalancement
st.sidebar.subheader("4. Rebalancement")
rebalance_freq = st.sidebar.selectbox(
    "Fréquence :",
    options=["daily", "weekly", "monthly", "none"],
    index=0,
    format_func=lambda x: {
        "daily": "Quotidien",
        "weekly": "Hebdomadaire", 
        "monthly": "Mensuel",
        "none": "Pas de rebalancement"
    }[x]
)

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(ttl=300)  # Cache de 5 minutes
def load_data(tickers, period):
    """Charge les données avec mise en cache."""
    return get_multi_asset_data(tickers, period=period)

with st.spinner("Chargement des données..."):
    try:
        prices = load_data(tuple(selected_tickers), period)
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données : {e}")
        st.stop()

# =============================================================================
# CALCULATE PORTFOLIO
# =============================================================================

# Calculer les poids
if weight_method == "equal":
    weights = calculate_portfolio_weights(len(selected_tickers), method="equal")
else:
    weights = calculate_portfolio_weights(len(selected_tickers), method="custom", custom_weights=custom_weights)

# Simuler le portfolio
portfolio_cumulative, portfolio_returns = simulate_portfolio(prices, weights, rebalance_freq)

# =============================================================================
# MAIN CONTENT
# =============================================================================

# Ligne d'info
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📅 Période", f"{prices.index[0].strftime('%Y-%m-%d')} → {prices.index[-1].strftime('%Y-%m-%d')}")
with col2:
    st.metric("📈 Nb d'actifs", len(selected_tickers))
with col3:
    st.metric("📊 Observations", len(prices))
with col4:
    st.metric("🔄 Rebalancement", rebalance_freq.capitalize())

st.divider()

# =============================================================================
# TAB LAYOUT
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Performance",
    "📊 Corrélations",
    "📋 Métriques",
    "⚖️ Allocation"
])

# -----------------------------------------------------------------------------
# TAB 1: PERFORMANCE
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Performance Comparée : Actifs vs Portfolio")
    
    # Données de comparaison
    comparison_data = compare_portfolio_vs_assets(prices, weights, rebalance_freq)
    
    # Graphique principal avec Plotly
    fig = go.Figure()
    
    # Ajouter les actifs individuels
    colors = px.colors.qualitative.Set2
    for i, ticker in enumerate(selected_tickers):
        fig.add_trace(go.Scatter(
            x=comparison_data.index,
            y=comparison_data[ticker],
            name=f"{ticker}",
            line=dict(width=1.5, color=colors[i % len(colors)]),
            opacity=0.7
        ))
    
    # Ajouter le portfolio (plus épais)
    fig.add_trace(go.Scatter(
        x=comparison_data.index,
        y=comparison_data["Portfolio"],
        name="📊 PORTFOLIO",
        line=dict(width=3, color="#FF6B6B", dash="solid"),
    ))
    
    fig.update_layout(
        title="Valeur Cumulée (Base 1)",
        xaxis_title="Date",
        yaxis_title="Valeur",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Prix des actifs
    st.subheader("Prix des Actifs")
    
    # Normaliser les prix pour comparaison
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
        title="Prix Normalisés (Base 100)",
        xaxis_title="Date",
        yaxis_title="Prix (base 100)",
        hovermode="x unified",
        height=400,
    )
    
    st.plotly_chart(fig_prices, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: CORRELATIONS
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Matrice de Corrélation")
    
    corr_matrix = calculate_correlation_matrix(prices)
    
    # Heatmap avec Plotly
    fig_corr = px.imshow(
        corr_matrix,
        labels=dict(color="Corrélation"),
        x=corr_matrix.columns,
        y=corr_matrix.index,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
    )
    
    fig_corr.update_layout(
        title="Corrélation des Rendements",
        height=500,
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Interprétation
    st.markdown("### 💡 Interprétation")
    
    # Trouver les corrélations extrêmes
    corr_values = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    min_corr = corr_values.min().min()
    max_corr = corr_values.max().max()
    
    min_pair = corr_values.stack().idxmin()
    max_pair = corr_values.stack().idxmax()
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Corrélation la plus faible :** {min_pair[0]} / {min_pair[1]} = {min_corr:.3f}")
        st.caption("→ Bonne diversification entre ces actifs")
    with col2:
        st.warning(f"**Corrélation la plus forte :** {max_pair[0]} / {max_pair[1]} = {max_corr:.3f}")
        st.caption("→ Ces actifs bougent souvent ensemble")

# -----------------------------------------------------------------------------
# TAB 3: METRICS
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Métriques du Portfolio")
    
    # Métriques portfolio
    col1, col2, col3, col4 = st.columns(4)
    
    port_return = calculate_portfolio_expected_return(prices, weights)
    port_vol = calculate_portfolio_volatility(prices, weights)
    port_sharpe = sharpe_ratio(portfolio_returns)
    port_mdd = max_drawdown(portfolio_cumulative)
    div_ratio = calculate_diversification_ratio(prices, weights)
    
    with col1:
        st.metric(
            "📈 Rendement Annualisé",
            f"{port_return:.2%}",
            delta=None
        )
    with col2:
        st.metric(
            "📉 Volatilité Annualisée",
            f"{port_vol:.2%}",
            delta=None
        )
    with col3:
        st.metric(
            "⚡ Sharpe Ratio",
            f"{port_sharpe:.3f}",
            delta=None
        )
    with col4:
        st.metric(
            "🔻 Max Drawdown",
            f"{port_mdd:.2%}",
            delta=None
        )
    
    # Ratio de diversification
    st.divider()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            "🎯 Ratio de Diversification",
            f"{div_ratio:.3f}",
            help="Un ratio > 1 indique un bénéfice de diversification"
        )
    with col2:
        if div_ratio > 1:
            st.success(f"✅ Le portfolio bénéficie de la diversification ! La volatilité est réduite de {(div_ratio - 1) * 100:.1f}% par rapport à une moyenne simple.")
        else:
            st.warning("⚠️ Faible effet de diversification - les actifs sont trop corrélés.")
    
    # Tableau des métriques individuelles
    st.divider()
    st.subheader("Comparaison avec les Actifs Individuels")
    
    individual_metrics = calculate_individual_metrics(prices)
    
    # Ajouter le portfolio au tableau
    portfolio_metrics = pd.DataFrame({
        "📊 PORTFOLIO": {
            "Return (Ann.)": port_return,
            "Volatility (Ann.)": port_vol,
            "Sharpe Ratio": port_sharpe,
            "Max Drawdown": port_mdd,
            "Final Value": portfolio_cumulative.iloc[-1],
        }
    }).T
    
    all_metrics = pd.concat([individual_metrics, portfolio_metrics])
    
    # Formater le tableau
    styled_df = all_metrics.style.format({
        "Return (Ann.)": "{:.2%}",
        "Volatility (Ann.)": "{:.2%}",
        "Sharpe Ratio": "{:.3f}",
        "Max Drawdown": "{:.2%}",
        "Final Value": "{:.3f}",
    }).background_gradient(cmap="RdYlGn", subset=["Sharpe Ratio"])
    
    st.dataframe(styled_df, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: ALLOCATION
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Allocation du Portfolio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart des poids
        weight_df = pd.DataFrame({
            "Actif": selected_tickers,
            "Poids": weights,
            "Poids (%)": [f"{w:.1%}" for w in weights]
        })
        
        fig_pie = px.pie(
            weight_df,
            values="Poids",
            names="Actif",
            title="Répartition du Portfolio",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Tableau des poids
        st.markdown("### Détail des Poids")
        
        weight_detail = pd.DataFrame({
            "Actif": selected_tickers,
            "Description": [available_tickers.get(t, t) for t in selected_tickers],
            "Poids": [f"{w:.2%}" for w in weights],
        })
        
        st.dataframe(weight_detail, use_container_width=True, hide_index=True)
        
        # Résumé
        st.markdown("### 📋 Résumé")
        st.info(f"""
        - **Méthode** : {"Équipondéré" if weight_method == "equal" else "Personnalisé"}
        - **Nombre d'actifs** : {len(selected_tickers)}
        - **Rebalancement** : {rebalance_freq.capitalize()}
        """)

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption("📊 Module Quant B - Portfolio Multi-Asset | Données : Yahoo Finance | Refresh auto : 5 min")
