# 📊 Quant B - Multi-Asset Portfolio Module

## Overview

This module is part of the **Asset Management Platform** project, developed for the Python, Git & Linux for Finance course. The Quant B module focuses on **multi-asset portfolio analysis**, providing tools for portfolio simulation, diversification analysis, and performance tracking.

**Live Application**: http://13.60.98.189:8501

---

## 🎯 Features

### Core Functionality

| Feature | Description |
|---------|-------------|
| **Multi-Asset Data** | Retrieves real-time data for multiple assets via Yahoo Finance API |
| **Portfolio Simulation** | Simulates portfolio performance with customizable weights |
| **Rebalancing Strategies** | Supports daily, weekly, monthly, or no rebalancing |
| **Correlation Analysis** | Computes and visualizes correlation matrix between assets |
| **Performance Metrics** | Calculates Sharpe ratio, max drawdown, volatility, returns |
| **Diversification Ratio** | Measures portfolio diversification benefit |
| **Daily Reports** | Automated daily report generation via cron job |

### Default Assets (ETFs)

| Ticker | Name | Asset Class |
|--------|------|-------------|
| SPY | S&P 500 | US Equities |
| TLT | Treasury Bonds 20+ Years | Bonds |
| GLD | Gold Trust | Commodities |
| VNQ | Real Estate Investment Trust | Real Estate |

These assets were selected to demonstrate **diversification benefits** through low correlations between asset classes.

---

## 📁 Project Structure
```
project_asset_management_server/
├── portfolio.py                 # Backend module (data fetching, calculations)
├── single_asset.py              # Quant A module
├── single_asset_modelling.py    # Quant A ML module
├── requirements.txt             # Python dependencies
├── README.md                    # Main README
├── README_QUANT_B.md            # This file
├── .gitignore
│
├── app/
│   ├── Home.py                  # Streamlit homepage
│   └── pages/
│       ├── 1_Single_Asset.py    # Quant A interface
│       └── 2_Portfolio.py       # Quant B interface
│
├── scripts/
│   ├── daily_report.py          # Daily report generator
│   └── run_daily_report.sh      # Shell script for cron
│
└── reports/
    ├── report_YYYY-MM-DD.txt    # Daily reports
    └── cron.log                 # Cron execution logs
```

---

## 🔧 Technical Implementation

### Backend (`portfolio.py`)

#### Data Loading
```python
def load_multi_asset(tickers, period, interval):
    """
    Load price data for multiple assets from Yahoo Finance.
    Returns DataFrame with adjusted close prices.
    """
```

#### Portfolio Simulation
```python
def simulate_portfolio(prices, weights, rebalance_frequency):
    """
    Simulate portfolio performance with given weights.
    Supports: 'daily', 'weekly', 'monthly', 'none' rebalancing.
    Returns: equity curve (Series), returns (Series)
    """
```

#### Key Metrics Functions
```python
def portfolio_volatility(prices, weights, annualise=True)
def portfolio_expected_return(prices, weights, annualise=True)
def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
def max_drawdown(equity)
def diversification_ratio(prices, weights)
def correlation_matrix(prices)
```

### Frontend (`app/pages/2_Portfolio.py`)

Built with **Streamlit** and **Plotly** for interactive visualization.

#### Tabs:
1. **Performance** - Cumulative returns chart comparing assets vs portfolio
2. **Correlations** - Heatmap of return correlations
3. **Metrics** - Portfolio and individual asset metrics
4. **Allocation** - Pie chart and weight details
5. **Daily Reports** - View and download generated reports

#### User Controls:
- Asset selection (multi-select, min. 3 assets)
- Historical period (6 months to 5 years)
- Weighting method (equal or custom)
- Rebalancing frequency (daily/weekly/monthly/none)

---

## ⏰ Automated Daily Reports

### Cron Configuration
```bash
# Runs daily at 19:00 UTC (20:00 Paris time)
0 19 * * * /home/ubuntu/project_asset_management_server/scripts/run_daily_report.sh
```

### Report Contents
- Portfolio summary (assets, weights, period)
- Portfolio metrics (return, volatility, Sharpe, max drawdown)
- Asset prices (open, latest, change %)
- Individual asset metrics
- Correlation matrix

### Sample Output
```
================================================================================
                    DAILY PORTFOLIO REPORT - 2026-01-07
================================================================================

PORTFOLIO METRICS:
  - Annual Return:           22.01%
  - Annual Volatility:       10.97%
  - Sharpe Ratio:             2.006
  - Max Drawdown:            -8.12%
  - Final Value (base 1):     1.237
```

---

## 🚀 Deployment

### Infrastructure
- **Cloud Provider**: AWS EC2
- **Instance Type**: t3.micro
- **OS**: Ubuntu 24.04 LTS
- **Region**: Europe (Stockholm) eu-north-1

### Running 24/7
The application runs continuously using `screen`:
```bash
screen -S streamlit
cd ~/project_asset_management_server/app
streamlit run Home.py --server.port 8501 --server.address 0.0.0.0
# Detach: Ctrl+A then D
```

### Security
- SSH access via key pair authentication
- Port 8501 open for web traffic (Security Group)

---

## 📊 Portfolio Theory

### Diversification Ratio
```
DR = (Weighted Average of Individual Volatilities) / Portfolio Volatility
```
- DR > 1 indicates diversification benefit
- Higher DR = better diversification

### Rebalancing Impact
| Frequency | Effect |
|-----------|--------|
| Daily | Maintains target weights precisely |
| Weekly | Good balance of precision and costs |
| Monthly | Lower trading costs, more drift allowed |
| None | Buy and hold, weights drift with returns |

### Correlation Interpretation
| Correlation | Meaning |
|-------------|---------|
| ~1.0 | Assets move together (no diversification) |
| ~0.0 | Assets are independent (good diversification) |
| ~-1.0 | Assets move opposite (best diversification) |

---

## 🛠️ Local Development

### Prerequisites
- Python 3.10+
- pip

### Installation
```bash
git clone https://github.com/NatachaGsn/project_asset_management_server.git
cd project_asset_management_server
git checkout feature/quant_B

python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Run Locally
```bash
cd app
streamlit run Home.py
```

### Test Portfolio Module
```bash
python portfolio.py
```

---

## 📈 Results & Performance

### Portfolio Performance (1 Year)
| Metric | Value |
|--------|-------|
| Annual Return | ~21-22% |
| Annual Volatility | ~11% |
| Sharpe Ratio | ~1.9-2.0 |
| Max Drawdown | ~-8% |
| Diversification Ratio | ~1.55 |

### Asset Correlations
| | GLD | SPY | TLT | VNQ |
|---|---|---|---|---|
| **GLD** | 1.00 | 0.04 | 0.04 | 0.11 |
| **SPY** | 0.04 | 1.00 | 0.10 | 0.64 |
| **TLT** | 0.04 | 0.10 | 1.00 | 0.34 |
| **VNQ** | 0.11 | 0.64 | 0.34 | 1.00 |

**Key Insight**: GLD (Gold) has near-zero correlation with SPY (Stocks) → excellent diversifier!

---

## 👥 Team

| Role | Responsibility |
|------|----------------|
| **Quant A** | Single Asset Analysis (Natacha) |
| **Quant B** | Multi-Asset Portfolio (Margaux) |

---

## 📚 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.12 | Backend logic |
| Streamlit | Web interface |
| Plotly | Interactive charts |
| yfinance | Yahoo Finance API |
| pandas | Data manipulation |
| numpy | Numerical calculations |
| AWS EC2 | Cloud hosting |
| Git/GitHub | Version control |
| cron | Task scheduling |

---

## 📝 Git Workflow

### Branches
- `main` - Production branch
- `feature/quant_A` - Single Asset development
- `feature/quant_B` - Portfolio development

### Commit Convention
```
type(scope): description

feat(quant-b): add portfolio simulation
fix(quant-b): correct Sharpe ratio calculation
refactor(quant-b): translate code to English
docs(quant-b): add README documentation
```

---

## 📄 License

This project was developed for educational purposes as part of the ESILV curriculum.

---

## 🔗 Links

- **Live App**: http://13.60.98.189:8501
- **GitHub**: https://github.com/NatachaGsn/project_asset_management_server
- **Branch**: feature/quant_B
