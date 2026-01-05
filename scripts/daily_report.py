#!/usr/bin/env python3
"""
Daily Report Generator - Quant B Portfolio
Generates a daily report with portfolio metrics at 8pm
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio import (
    get_multi_asset_data,
    calculate_portfolio_weights,
    simulate_portfolio,
    calculate_correlation_matrix,
    calculate_portfolio_volatility,
    calculate_portfolio_expected_return,
    calculate_individual_metrics,
    sharpe_ratio,
    max_drawdown,
    DEFAULT_TICKERS,
)

def generate_report():
    """Generate daily portfolio report."""
    
    # Create reports directory if not exists
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Get current date
    today = datetime.now().strftime("%Y-%m-%d")
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Fetch data
    print(f"[{report_time}] Fetching data...")
    prices = get_multi_asset_data(DEFAULT_TICKERS, period="1y")
    
    # Calculate portfolio metrics
    weights = calculate_portfolio_weights(len(DEFAULT_TICKERS), method="equal")
    portfolio_cumulative, portfolio_returns = simulate_portfolio(prices, weights)
    
    # Metrics
    port_return = calculate_portfolio_expected_return(prices, weights)
    port_vol = calculate_portfolio_volatility(prices, weights)
    port_sharpe = sharpe_ratio(portfolio_returns)
    port_mdd = max_drawdown(portfolio_cumulative)
    
    # Individual metrics
    individual = calculate_individual_metrics(prices)
    
    # Correlation matrix
    corr_matrix = calculate_correlation_matrix(prices)
    
    # Get latest prices
    latest_prices = prices.iloc[-1]
    open_prices = prices.iloc[0]
    
    # Generate report content
    report = f"""
================================================================================
                    DAILY PORTFOLIO REPORT - {today}
================================================================================
Generated at: {report_time}

--------------------------------------------------------------------------------
                              PORTFOLIO SUMMARY
--------------------------------------------------------------------------------
Assets: {', '.join(DEFAULT_TICKERS)}
Weights: Equal Weight ({100/len(DEFAULT_TICKERS):.1f}% each)
Period: 1 Year

PORTFOLIO METRICS:
  - Annualized Return:    {port_return:>10.2%}
  - Annualized Volatility:{port_vol:>10.2%}
  - Sharpe Ratio:         {port_sharpe:>10.3f}
  - Max Drawdown:         {port_mdd:>10.2%}
  - Final Value (base 1): {portfolio_cumulative.iloc[-1]:>10.3f}

--------------------------------------------------------------------------------
                              ASSET PRICES
--------------------------------------------------------------------------------
{"Ticker":<10} {"Open (1Y ago)":<15} {"Latest":<15} {"Change":<10}
{"-"*50}
"""
    
    for ticker in DEFAULT_TICKERS:
        open_p = open_prices[ticker]
        latest_p = latest_prices[ticker]
        change = (latest_p - open_p) / open_p
        report += f"{ticker:<10} ${open_p:<14.2f} ${latest_p:<14.2f} {change:>+.2%}\n"
    
    report += f"""
--------------------------------------------------------------------------------
                           INDIVIDUAL METRICS
--------------------------------------------------------------------------------
{"Ticker":<10} {"Return":<12} {"Volatility":<12} {"Sharpe":<10} {"Max DD":<10}
{"-"*54}
"""
    
    for ticker in individual.index:
        row = individual.loc[ticker]
        report += f"{ticker:<10} {row['Return (Ann.)']:>+.2%}      {row['Volatility (Ann.)']:>.2%}       {row['Sharpe Ratio']:>6.3f}    {row['Max Drawdown']:>+.2%}\n"
    
    report += f"""
--------------------------------------------------------------------------------
                          CORRELATION MATRIX
--------------------------------------------------------------------------------
{corr_matrix.round(3).to_string()}

================================================================================
                              END OF REPORT
================================================================================
"""
    
    # Save report
    report_filename = f"report_{today}.txt"
    report_path = os.path.join(reports_dir, report_filename)
    
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"[{report_time}] Report saved to: {report_path}")
    print(report)
    
    return report_path


if __name__ == "__main__":
    generate_report()

