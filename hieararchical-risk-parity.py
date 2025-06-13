import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from datetime import datetime, timedelta
import seaborn as sns

class HierarchicalRiskParity:
    def __init__(self, tickers, start_date=None, end_date=None):
        self.tickers = tickers
        self.start_date = start_date or (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.returns = None
        self.weights = None
        self.risk_free_rate = None
        
    def fetch_data(self):
        try:
            print(f"Fetching data for: {', '.join(self.tickers)}")
            stock_data = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False)
            
            if len(self.tickers) == 1:
                if isinstance(stock_data.columns, pd.MultiIndex):
                    self.data = pd.DataFrame({self.tickers[0]: stock_data['Adj Close']})
                else:
                    self.data = pd.DataFrame({self.tickers[0]: stock_data['Adj Close']})
            else:
                if 'Adj Close' in stock_data.columns.levels[0]:
                    self.data = stock_data['Adj Close']
                else:
                    self.data = stock_data['Close']
            
            self.data = self.data.dropna(axis=1, how='all')
            
            missing_tickers = set(self.tickers) - set(self.data.columns)
            if missing_tickers:
                print(f"Warning: Could not fetch data for: {', '.join(missing_tickers)}")
                self.tickers = [t for t in self.tickers if t in self.data.columns]
            
            if self.data.empty or len(self.tickers) == 0:
                print("Error: No valid stock data retrieved")
                return False
            
            try:
                treasury = yf.download('^TNX', start=self.start_date, end=self.end_date, progress=False)
                if not treasury.empty:
                    if isinstance(treasury.columns, pd.MultiIndex):
                        self.risk_free_rate = treasury['Adj Close'].iloc[-1] / 100
                    else:
                        self.risk_free_rate = treasury['Adj Close'].iloc[-1] / 100
                else:
                    print("Warning: Could not fetch risk-free rate, using default 3%")
                    self.risk_free_rate = 0.042
            except:
                print("Warning: Could not fetch risk-free rate, using default 3%")
                self.risk_free_rate = 0.042
            
            print(f"✓ Successfully fetched data for {len(self.tickers)} stocks")
            print(f"✓ Data period: {self.start_date} to {self.end_date}")
            print(f"✓ Data shape: {self.data.shape}")
            print(f"✓ Current risk-free rate: {self.risk_free_rate:.2%}")
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            print("Debugging info:")
            print(f"- Tickers requested: {self.tickers}")
            print(f"- Date range: {self.start_date} to {self.end_date}")
            return False
        return True

    
    def calculate_returns(self):
        if self.data is None:
            print("Error: No data available. Please fetch data first.")
            return False
        
        self.returns = self.data.pct_change().dropna()
        print(f"✓ Calculated returns for {len(self.returns.columns)} stocks")
        return True
    
    def _correlation_distance(self, corr_matrix):
        return np.sqrt(0.5 * (1 - corr_matrix))
    
    def _get_cluster_variance(self, cov_matrix, cluster_items):
        cov_slice = cov_matrix.loc[cluster_items, cluster_items]
        inv_diag = 1.0 / np.diag(cov_slice)
        inv_diag /= inv_diag.sum()
        return np.dot(inv_diag, np.dot(cov_slice, inv_diag))
    
    def _get_quasi_diagonal_order(self, linkage_matrix):
        tree = sch.to_tree(linkage_matrix, rd=False)
        return tree.pre_order()
    
    def _recursive_bisection(self, cov_matrix, ordered_tickers):
        weights = pd.Series(1.0, index=ordered_tickers)
        cluster_items = [ordered_tickers]
        
        while len(cluster_items) > 0:
            cluster_items = [
                items[j:k] for items in cluster_items 
                for j, k in ((0, len(items)//2), (len(items)//2, len(items))) 
                if len(items) > 1
            ]
            
            for i in range(0, len(cluster_items), 2):
                if i + 1 < len(cluster_items):
                    cluster_0 = cluster_items[i]
                    cluster_1 = cluster_items[i + 1]
                    
                    var_0 = self._get_cluster_variance(cov_matrix, cluster_0)
                    var_1 = self._get_cluster_variance(cov_matrix, cluster_1)
                    
                    alpha = 1 - var_0 / (var_0 + var_1)
                    
                    weights[cluster_0] *= alpha
                    weights[cluster_1] *= (1 - alpha)
        
        return weights
    
    def optimize_portfolio(self, linkage_method='single'):
        if self.returns is None:
            print("Error: Returns not calculated. Please calculate returns first.")
            return False
        
        try:
            cov_matrix = self.returns.cov()
            corr_matrix = self.returns.corr()
            
            distance_matrix = self._correlation_distance(corr_matrix)
            
            condensed_distances = ssd.squareform(distance_matrix, checks=False)
            linkage_matrix = sch.linkage(condensed_distances, method=linkage_method)
            
            sort_indices = self._get_quasi_diagonal_order(linkage_matrix)
            ordered_tickers = corr_matrix.index[sort_indices].tolist()
            
            self.weights = self._recursive_bisection(cov_matrix, ordered_tickers)
            
            print(f"✓ HRP optimization completed using {linkage_method} linkage")
            return True
            
        except Exception as e:
            print(f"Error in optimization: {e}")
            return False
    
    def optimize_portfolio_with_constraints(self, max_weight_per_asset=None, max_weight_per_sector=None, 
                                          min_weight_per_asset=None, linkage_method='single'):
        if self.returns is None:
            print("Error: Returns not calculated. Please calculate returns first.")
            return False
        
        try:
            cov_matrix = self.returns.cov()
            corr_matrix = self.returns.corr()
            
            distance_matrix = self._correlation_distance(corr_matrix)
            
            condensed_distances = ssd.squareform(distance_matrix, checks=False)
            linkage_matrix = sch.linkage(condensed_distances, method=linkage_method)
            
            sort_indices = self._get_quasi_diagonal_order(linkage_matrix)
            ordered_tickers = corr_matrix.index[sort_indices].tolist()
            
            initial_weights = self._recursive_bisection(cov_matrix, ordered_tickers)
            
            self.weights = self._apply_constraints(initial_weights, max_weight_per_asset, 
                                                 max_weight_per_sector, min_weight_per_asset)
            
            print(f"✓ HRP optimization with constraints completed using {linkage_method} linkage")
            return True
            
        except Exception as e:
            print(f"Error in constrained optimization: {e}")
            return False

    def _apply_constraints(self, weights, max_weight_per_asset=None, max_weight_per_sector=None, 
                          min_weight_per_asset=None):
        constrained_weights = weights.copy()
        
        if max_weight_per_asset is not None:
            excess_weights = {}
            for ticker in constrained_weights.index:
                if constrained_weights[ticker] > max_weight_per_asset:
                    excess_weights[ticker] = constrained_weights[ticker] - max_weight_per_asset
                    constrained_weights[ticker] = max_weight_per_asset
            
            if excess_weights:
                total_excess = sum(excess_weights.values())
                remaining_tickers = [t for t in constrained_weights.index if t not in excess_weights]
                
                if remaining_tickers:
                    remaining_capacity = sum(max(0, max_weight_per_asset - constrained_weights[t]) 
                                           for t in remaining_tickers)
                    
                    if remaining_capacity > 0:
                        for ticker in remaining_tickers:
                            available_capacity = max(0, max_weight_per_asset - constrained_weights[ticker])
                            redistribution = total_excess * (available_capacity / remaining_capacity)
                            constrained_weights[ticker] += redistribution
        
        if max_weight_per_sector is not None:
            sector_mapping = self._get_sector_mapping()
            sector_weights = {}
            
            for ticker, weight in constrained_weights.items():
                sector = sector_mapping.get(ticker)
                if sector:
                    sector_weights[sector] = sector_weights.get(sector, 0) + weight
            
            for sector, max_weight in max_weight_per_sector.items():
                if sector in sector_weights and sector_weights[sector] > max_weight:
                    reduction_factor = max_weight / sector_weights[sector]
                    for ticker in constrained_weights.index:
                        if sector_mapping.get(ticker) == sector:
                            constrained_weights[ticker] *= reduction_factor
        
        if min_weight_per_asset is not None:
            deficit_weights = {}
            for ticker in constrained_weights.index:
                if constrained_weights[ticker] < min_weight_per_asset:
                    deficit_weights[ticker] = min_weight_per_asset - constrained_weights[ticker]
                    constrained_weights[ticker] = min_weight_per_asset
            
            if deficit_weights:
                total_deficit = sum(deficit_weights.values())
                remaining_tickers = [t for t in constrained_weights.index if t not in deficit_weights]
                
                if remaining_tickers:
                    total_remaining_weight = sum(constrained_weights[t] for t in remaining_tickers)
                    
                    if total_remaining_weight > total_deficit:
                        for ticker in remaining_tickers:
                            reduction_ratio = total_deficit / total_remaining_weight
                            constrained_weights[ticker] *= (1 - reduction_ratio)
        
        return constrained_weights / constrained_weights.sum()
    
    def calculate_portfolio_metrics(self):
        if self.weights is None or self.returns is None:
            print("Error: Portfolio not optimized yet.")
            return None
        
        portfolio_returns = (self.returns * self.weights).sum(axis=1)
        
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility
        
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        var_95 = np.percentile(portfolio_returns, 5)
        
        metrics = {
            'Annual Return': f"{annual_return:.2%}",
            'Annual Volatility': f"{annual_volatility:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.3f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Value at Risk (95%)': f"{var_95:.2%}",
            'Risk-Free Rate': f"{self.risk_free_rate:.2%}"
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, returns):
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def visualize_results(self, figsize=(15, 12)):
        if self.weights is None:
            print("Error: No weights to visualize. Please optimize portfolio first.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        weights_sorted = self.weights.sort_values(ascending=True)
        colors = plt.cm.viridis(np.linspace(0, 1, len(weights_sorted)))
        
        bars = ax1.barh(range(len(weights_sorted)), weights_sorted.values, color=colors)
        ax1.set_yticks(range(len(weights_sorted)))
        ax1.set_yticklabels(weights_sorted.index)
        ax1.set_xlabel('Portfolio Weight')
        ax1.set_title('HRP Portfolio Weights', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1%}', ha='left', va='center', fontsize=10)
        
        corr_matrix = self.returns.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, ax=ax2, cbar_kws={"shrink": .8})
        ax2.set_title('Stock Correlation Matrix', fontsize=14, fontweight='bold')
        
        portfolio_var = (self.returns * self.weights).sum(axis=1).var()
        risk_contributions = []
        for ticker in self.weights.index:
            weight = self.weights[ticker]
            asset_var = self.returns[ticker].var()
            risk_contrib = (weight * asset_var) / portfolio_var
            risk_contributions.append(risk_contrib)
        
        risk_contrib_series = pd.Series(risk_contributions, index=self.weights.index)
        risk_contrib_sorted = risk_contrib_series.sort_values(ascending=True)
        
        ax3.barh(range(len(risk_contrib_sorted)), risk_contrib_sorted.values, 
                color='lightcoral', alpha=0.7)
        ax3.set_yticks(range(len(risk_contrib_sorted)))
        ax3.set_yticklabels(risk_contrib_sorted.index)
        ax3.set_xlabel('Risk Contribution')
        ax3.set_title('Risk Contribution by Asset', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        portfolio_returns = (self.returns * self.weights).sum(axis=1)
        equal_weight_returns = self.returns.mean(axis=1)
        
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        equal_weight_cumulative = (1 + equal_weight_returns).cumprod()
        
        ax4.plot(portfolio_cumulative.index, portfolio_cumulative.values, 
                label='HRP Portfolio', linewidth=2, color='navy')
        ax4.plot(equal_weight_cumulative.index, equal_weight_cumulative.values, 
                label='Equal Weight Portfolio', linewidth=2, color='red', alpha=0.7)
        
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Cumulative Return')
        ax4.set_title('Portfolio Performance Comparison', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        if self.weights is None:
            print("Error: Portfolio not optimized. Please run optimization first.")
            return
        
        metrics = self.calculate_portfolio_metrics()
        if metrics is None:
            return
        
        print("=" * 80)
        print("HIERARCHICAL RISK PARITY PORTFOLIO ANALYSIS REPORT")
        print("=" * 80)
        print(f"Analysis Period: {self.start_date} to {self.end_date}")
        print(f"Number of Assets: {len(self.tickers)}")
        print(f"Assets Analyzed: {', '.join(self.tickers)}")
        print()
        
        print("PORTFOLIO WEIGHTS:")
        print("-" * 40)
        for ticker, weight in self.weights.sort_values(ascending=False).items():
            print(f"{ticker:>8}: {weight:>8.2%}")
        print()
        
        print("PERFORMANCE METRICS:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric:>20}: {value:>12}")
        print()
        
        print("INVESTMENT INSIGHTS:")
        print("-" * 40)
        
        max_weight = self.weights.max()
        min_weight = self.weights.min()
        weight_concentration = (self.weights ** 2).sum()
        effective_assets = 1 / weight_concentration
        
        print(f"• Portfolio Concentration: {weight_concentration:.3f}")
        print(f"• Effective Number of Assets: {effective_assets:.1f}")
        print(f"• Largest Position: {max_weight:.1%}")
        print(f"• Smallest Position: {min_weight:.1%}")
        print()
        
        if float(metrics['Sharpe Ratio']) > 1.0:
            risk_assessment = "Excellent risk-adjusted returns"
        elif float(metrics['Sharpe Ratio']) > 0.5:
            risk_assessment = "Good risk-adjusted returns"
        else:
            risk_assessment = "Below-average risk-adjusted returns"
        
        print(f"• Risk Assessment: {risk_assessment}")
        print(f"• Diversification Level: {'High' if effective_assets > len(self.tickers) * 0.7 else 'Moderate'}")
        print()
        
        print("RECOMMENDATIONS FOR MID-LONG TERM INVESTORS:")
        print("-" * 50)
        print("• HRP provides better diversification than equal weighting")
        print("• Portfolio automatically balances risk across asset clusters")
        print("• Suitable for investors seeking stable, risk-adjusted returns")
        print("• Consider rebalancing quarterly or semi-annually")
        print("• Monitor correlation changes during market stress periods")
        print("=" * 80)

    def _get_sector_mapping(self):
        sector_mapping = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'NVDA': 'Technology',
            'TXN': 'Technology',
            'CSCO': 'Technology',
            'KO': 'Consumer_Staples',
            'PG': 'Consumer_Staples',
            'WMT': 'Consumer_Staples',
            'JNJ': 'Healthcare',
            'AMGN': 'Healthcare',
            'JPM': 'Financials',
            'XOM': 'Energy',
            'CVX': 'Energy',
            'COP': 'Energy',
            'DIS': 'Communication',
            'VZ': 'Communication',
            'LMT': 'Industrials',
            'HD': 'Consumer_Discretionary',
            'MO': 'Consumer_Staples'
        }
        return sector_mapping

def analyze_stocks(tickers, start_date=None, end_date=None):
    print(f"Starting HRP analysis for: {', '.join(tickers)}")
    print("-" * 60)
    
    hrp = HierarchicalRiskParity(tickers, start_date, end_date)
    
    if not hrp.fetch_data():
        return None
    
    if not hrp.calculate_returns():
        return None
    
    if not hrp.optimize_portfolio():
        return None
    
    hrp.visualize_results()
    hrp.generate_report()
    
    return hrp

def analyze_stocks_with_limits(tickers, max_individual_weight=None, max_sector_weights=None, 
                              min_individual_weight=None, start_date=None, end_date=None):
    print(f"Starting HRP analysis with constraints for: {', '.join(tickers)}")
    print("-" * 60)
    
    hrp = HierarchicalRiskParity(tickers, start_date, end_date)
    
    if not hrp.fetch_data() or not hrp.calculate_returns():
        return None
    
    print("\n1. Running unconstrained HRP optimization...")
    hrp.optimize_portfolio()
    unconstrained_weights = hrp.weights.copy()
    
    print("\n2. Running constrained HRP optimization...")
    hrp.optimize_portfolio_with_constraints(
        max_weight_per_asset=max_individual_weight,
        max_weight_per_sector=max_sector_weights,
        min_weight_per_asset=min_individual_weight
    )
    
    print("\n" + "="*80)
    print("ALLOCATION COMPARISON: UNCONSTRAINED vs CONSTRAINED")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        'Unconstrained': unconstrained_weights,
        'Constrained': hrp.weights,
        'Difference': hrp.weights - unconstrained_weights
    }).round(4)
    
    print(comparison_df.to_string())
    print()
    
    if max_individual_weight:
        violations = unconstrained_weights[unconstrained_weights > max_individual_weight]
        if not violations.empty:
            print(f"ORIGINAL VIOLATIONS (max {max_individual_weight:.1%} per asset):")
            for ticker, weight in violations.items():
                print(f"  {ticker}: {weight:.2%} (exceeded by {weight - max_individual_weight:.2%})")
            print()
    
    if max_sector_weights:
        sector_mapping = hrp._get_sector_mapping()
        sector_weights = {}
        
        for ticker, weight in unconstrained_weights.items():
            sector = sector_mapping.get(ticker)
            if sector:
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        sector_violations = {sector: weight for sector, weight in sector_weights.items() 
                           if sector in max_sector_weights and weight > max_sector_weights[sector]}
        
        if sector_violations:
            print("SECTOR VIOLATIONS:")
            for sector, weight in sector_violations.items():
                limit = max_sector_weights[sector]
                print(f"  {sector}: {weight:.2%} (exceeded by {weight - limit:.2%})")
            print()
    
    hrp.visualize_results()
    hrp.generate_report()
    
    return hrp

if __name__ == "__main__":

# Usage Without Constraints:
    '''
    tech_stocks = ['TXN', 'KO', 'VZ', 'COP', 'CSCO', 'LMT', 'MO', 'HD', 'CVX', 'AMGN', 'NVDA', 'MSFT', 'GOOGL']
    
    hrp = HierarchicalRiskParity(tech_stocks)
    hrp.fetch_data()
    hrp.calculate_returns()
    hrp.optimize_portfolio(
        linkage_method='single'
    )
    hrp.visualize_results()
    hrp.generate_report()
    '''

# Usage With Constraints:

    tech_stocks = ['TXN', 'KO', 'VZ', 'COP', 'CSCO', 'LMT', 'MO', 'HD', 'CVX', 'AMGN', 'NVDA', 'MSFT', 'GOOGL']
    
    hrp = HierarchicalRiskParity(tech_stocks)
    hrp.fetch_data()
    hrp.calculate_returns()
    hrp.optimize_portfolio_with_constraints(
        max_weight_per_asset=0.05,
        min_weight_per_asset=0.01,
        max_weight_per_sector={
            'Technology': 0.25,
            'Energy': 0.20,
        },
        linkage_method='single'
    )
    hrp.visualize_results()
    hrp.generate_report()