# Asset Management Quantitative Dashboard

This project was developed in the context of a quantitative finance assignment simulating a professional asset management environment.
We act as members of a quantitative research team supporting fundamental portfolio managers with data-driven tools and interactive analytics.

The objective is to design and deploy a professional-looking online dashboard capable of continuously retrieving financial data, implementing quantitative strategies, and clearly presenting results and key performance metrics.
The platform is built using Python and Streamlit, version-controlled with Git/GitHub, and deployed on a Linux environment to reflect real-world industry workflows.

The project is developed by a two-person team, with each member responsible for a specific quantitative module.
Despite this division of responsibilities, the final deliverable is a single, integrated platform providing a coherent user experience.

**Authors:**  
- Natacha GAUSSIN  
- Margaux GIRONA  

The project is structured around two complementary modules:

---

## Quant A — Single Asset Analysis Module

### Overview
The Single Asset Analysis Module focuses on the quantitative analysis of a single financial asset at a time (equity, FX, commodity, or cryptocurrency).
It provides an interactive environment to explore historical prices, evaluate trading strategies, and analyse their performance using standard quantitative metrics.
An optional forecasting component allows users to experiment with time-series prediction techniques.

---

### Asset Universe
The module supports the analysis of one selected asset among equities, commodities, foreign exchange rates, and cryptocurrencies.
Users can easily switch between assets such as Apple (AAPL), Gold, EUR/USD, or Bitcoin.
Historical price data is retrieved from Yahoo Finance.

---

### Data Handling
Users can configure both the historical time range and the data frequency (daily or weekly).
Raw market data is automatically cleaned, sorted, and deduplicated before analysis.
To ensure efficiency and stability, market data is cached and automatically refreshed every five minutes when new observations are available.

---

### Backtesting Strategies
Two trading strategies are implemented to illustrate different investment approaches.

The **Buy & Hold** strategy remains fully invested in the asset at all times and serves as a natural benchmark.
The **Moving Average Crossover** strategy takes a long position when a short-term moving average exceeds a long-term moving average and remains flat otherwise.
Strategy parameters are fully configurable, and trading signals are shifted to avoid look-ahead bias.

---

### Performance Metrics
For each strategy, the module computes key performance indicators commonly used in quantitative finance.
These include the cumulative equity curve, the annualised Sharpe ratio, maximum drawdown, annualised return (CAGR), and annualised volatility.
Annualisation factors are automatically adapted to the selected data frequency and asset class.

---

### Visualisation
Interactive charts allow users to visualise both market behaviour and strategy performance.
The main chart displays the raw asset price together with the cumulative value of the selected strategy, enabling a direct comparison between passive and active approaches.

---

### Forecasting (Optional Bonus)
An optional ARIMA-based forecasting module is provided.
The model is trained on historical data and produces predictions on both a test set and future horizons.
Forecasts are displayed alongside historical prices and include confidence intervals to illustrate predictive uncertainty.

---

### User Interaction
The interface offers interactive controls for asset selection, data frequency, historical range, strategy parameters, and forecasting settings.
User selections are preserved across application reruns and automatic data refreshes to ensure a smooth experience.

---

## Quant B — Portfolio / Multi-Asset Analysis Module
