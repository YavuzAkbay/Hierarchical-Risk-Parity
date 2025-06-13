# 📊 Hierarchical Risk Parity Portfolio Optimizer

This portfolio optimization tool implements Hierarchical Risk Parity (HRP) methodology to create well-diversified investment portfolios. This project combines modern portfolio theory with machine learning clustering techniques to achieve superior risk-adjusted returns through intelligent asset allocation.

## 🚀 Features

- **Advanced HRP Algorithm**: Implementation of Lopez de Prado's Hierarchical Risk Parity methodology
- **Constraint-Based Optimization**: Support for maximum/minimum weight constraints per asset and sector
- **Risk Analysis**: Comprehensive risk metrics including VaR, Sharpe ratio, and maximum drawdown
- **Visual Analytics**: Interactive charts for portfolio weights, correlation matrices, and performance comparison
- **Sector Diversification**: Built-in sector mapping for enhanced diversification control
- **Performance Benchmarking**: Comparison against equal-weight portfolios
- **Real-time Data Integration**: Automatic fetching of historical price data via Yahoo Finance
- **Comprehensive Reporting**: Detailed analysis reports with investment insights

## 📊 Methodology

The Hierarchical Risk Parity approach revolutionizes traditional portfolio optimization by:

1. Correlation-Based Clustering: Groups assets based on correlation distance matrices
2. Hierarchical Tree Construction: Builds asset relationship trees using linkage methods
3. Recursive Bisection: Allocates weights through recursive risk parity principles
4. Risk Budgeting: Ensures balanced risk contribution across asset clusters

### Mathematical Foundation

The HRP algorithm uses correlation distance:

$$d(i,j) = √(0.5 × (1 - ρ(i,j)))$$

Where ρ(i,j) is the correlation between assets i and j.

Risk allocation follows the recursive bisection principle:

$$α = 1 - (σ₀²)/(σ₀² + σ₁²)$$

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hierarchical-risk-parity
cd hierarchical-risk-parity

# Install required packages
pip install yfinance numpy pandas matplotlib scipy seaborn
```

## 📈 Usage

### Basic Portfolio Optimization

```python
from hierarchical_risk_parity import HierarchicalRiskParity

# Define your stock universe
stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'JPM', 'XOM', 'KO', 'JNJ']

# Create and optimize portfolio
hrp = HierarchicalRiskParity(stocks)
hrp.fetch_data()
hrp.calculate_returns()
hrp.optimize_portfolio()

# Generate comprehensive analysis
hrp.visualize_results()
hrp.generate_report()
```

### Advanced Optimization with Constraints

```python
# Portfolio with position and sector limits
hrp = HierarchicalRiskParity(stocks)
hrp.fetch_data()
hrp.calculate_returns()

# Apply sophisticated constraints
hrp.optimize_portfolio_with_constraints(
    max_weight_per_asset=0.15,      # Max 15% per stock
    min_weight_per_asset=0.02,      # Min 2% per stock
    max_weight_per_sector={
        'Technology': 0.30,          # Max 30% in tech
        'Energy': 0.20,             # Max 20% in energy
        'Healthcare': 0.25          # Max 25% in healthcare
    }
)

hrp.visualize_results()
hrp.generate_report()
```

### Convenience Functions

```python
# Quick analysis without constraints
analyze_stocks(['AAPL', 'MSFT', 'GOOGL', 'NVDA'])

# Analysis with comprehensive constraints
analyze_stocks_with_limits(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'JPM'],
    max_individual_weight=0.12,
    max_sector_weights={'Technology': 0.35},
    min_individual_weight=0.03
)
```

## 🔬 Technical Features

Risk Metrics Calculated

- Annual Return: Annualized portfolio performance
- Annual Volatility: Risk measurement via standard deviation
- Sharpe Ratio: Risk-adjusted return metric
- Maximum Drawdown: Worst peak-to-trough decline
- Value at Risk (95%): Potential loss at 95% confidence level
- Risk Contribution: Individual asset risk contributions

Visualization Components

1. Portfolio Weights Chart: Horizontal bar chart of asset allocations
2. Correlation Heatmap: Asset correlation matrix visualization
3. Risk Contribution Analysis: Risk attribution by asset
4. Performance Comparison: HRP vs Equal Weight portfolio performance

## 📊 Model Output

### Portfolio Analysis Report

The system generates comprehensive reports including:

- Portfolio Composition: Detailed weight allocations
- Performance Metrics: Risk-return characteristics
- Investment Insights: Concentration and diversification analysis
- Risk Assessment: Qualitative risk evaluation
- Recommendations: Strategic guidance for investors

### Key Performance Indicators

- Portfolio Concentration: Herfindahl index measurement
- Effective Number of Assets: Diversification effectiveness
- Risk Assessment: Qualitative performance evaluation
- Diversification Level: Portfolio spread analysis


## 🎯 Advantages of HRP

### Over Traditional Mean-Variance Optimization

- No Return Forecasting: Eliminates unreliable return predictions
- Stable Allocations: Reduces turnover and transaction costs
- Robust to Estimation Error: Less sensitive to input parameter changes
- Intuitive Structure: Tree-based allocation provides interpretability

### Over Equal Weighting

- Risk-Aware Allocation: Considers asset risk characteristics
- Correlation Sensitivity: Accounts for asset relationships
- Dynamic Rebalancing: Adapts to changing market conditions
- Superior Risk-Adjusted Returns: Typically achieves better Sharpe ratios

## 🔧 Configuration Options

### Model Parameters

- linkage_method: Clustering method ('single', 'complete', 'average', 'ward')
- start_date: Historical data start date
- end_date: Historical data end date

### Constraint Parameters

- max_weight_per_asset: Maximum allocation per individual asset
- min_weight_per_asset: Minimum allocation per individual asset
- max_weight_per_sector: Dictionary of sector allocation limits

### Risk Parameters

- risk_free_rate: Automatically fetched 10-year Treasury rate
- Custom risk-free rate override available

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request

## 📝 License

This project is licensed under the GPL v3 - see the (https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. **Do not use this for actual trading decisions without proper risk management and professional financial advice.** Past performance does not guarantee future results. Trading stocks involves substantial risk of loss.

## 🙏 Acknowledgments

- **Marcos López de Prado**: Creator of the Hierarchical Risk Parity methodology
- **scipy**: Scientific computing library for clustering algorithms
- **yfinance**: Financial data API
- **matplotlib/seaborn**: Visualization libraries

## 📧 Contact

Yavuz Akbay - akbay.yavuz@gmail.com

---

⭐️ If this project helped with your financial analysis, please consider giving it a star!

**Built with ❤️ for the intersection of mathematics, machine learning and finance**
