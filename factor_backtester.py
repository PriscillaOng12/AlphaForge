"""
Factor Model Backtester & Research Framework
==========================================

A comprehensive Python framework for factor-based equity research and backtesting.
Features:
- Multi-source data ingestion (Yahoo Finance, Polygon)
- Classic risk factor computation (Value, Momentum, Quality, Size, Low-Vol)
- Long-short portfolio construction
- Transaction cost modeling
- Bayesian shrinkage and Lasso regularization
- Walk-forward out-of-sample testing
- Comprehensive performance analytics
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import warnings
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import pickle

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration class for backtesting parameters"""
    start_date: str = "2010-01-01"
    end_date: str = "2023-12-31"
    rebalance_freq: str = "M"  # M=Monthly, Q=Quarterly
    lookback_window: int = 252  # Trading days for factor calculation
    transaction_cost: float = 0.0010  # 10 bps
    max_weight: float = 0.05  # Maximum position size
    min_weight: float = -0.05  # Minimum position size
    leverage: float = 1.0  # Portfolio leverage
    min_market_cap: float = 100e6  # Minimum market cap filter
    exclude_financials: bool = True
    exclude_utilities: bool = True


class DataProvider:
    """Handles data ingestion from multiple sources"""
    
    def __init__(self, polygon_api_key: Optional[str] = None):
        self.polygon_api_key = polygon_api_key
        self.cache_dir = Path("data_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_universe(self, index: str = "SP500") -> List[str]:
        """Get stock universe (S&P 500 by default)"""
        if index == "SP500":
            # Get S&P 500 tickers from Wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            try:
                tables = pd.read_html(url)
                sp500_table = tables[0]
                tickers = sp500_table['Symbol'].tolist()
                # Clean tickers (remove dots, etc.)
                tickers = [ticker.replace('.', '-') for ticker in tickers]
                return tickers[:100]  # Limit to first 100 for demo
            except Exception as e:
                logger.warning(f"Failed to fetch S&P 500 list: {e}")
                # Fallback to manual list
                return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
        
        return []
    
    def fetch_yahoo_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        cache_file = self.cache_dir / f"yahoo_data_{start_date}_{end_date}.pkl"
        
        if cache_file.exists():
            logger.info("Loading cached Yahoo Finance data")
            return pd.read_pickle(cache_file)
        
        logger.info(f"Fetching data for {len(tickers)} tickers from Yahoo Finance")
        
        def fetch_single_ticker(ticker):
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(start=start_date, end=end_date)
                if len(data) > 0:
                    data['ticker'] = ticker
                    return data
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
                return None
        
        # Parallel fetching
        all_data = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_single_ticker, ticker): ticker for ticker in tickers}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_data.append(result)
        
        if not all_data:
            raise ValueError("No data fetched successfully")
        
        # Combine all data
        df = pd.concat(all_data, ignore_index=True)
        df.reset_index(inplace=True)
        
        # Save to cache
        df.to_pickle(cache_file)
        logger.info(f"Cached data for {len(df['ticker'].unique())} tickers")
        
        return df
    
    def fetch_polygon_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from Polygon API (requires API key)"""
        if not self.polygon_api_key:
            raise ValueError("Polygon API key required")
        
        # Implementation would go here
        # For now, fallback to Yahoo Finance
        return self.fetch_yahoo_data(tickers, start_date, end_date)


class FactorCalculator:
    """Calculate classic risk factors"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare data for factor calculation"""
        # Ensure Date column is datetime
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Calculate basic metrics
        self.data['returns'] = self.data.groupby('ticker')['Close'].pct_change()
        self.data['log_returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['market_cap'] = self.data['Close'] * self.data['Volume']  # Proxy
        
        # Forward fill missing values
        self.data = self.data.fillna(method='ffill')
        
    def calculate_momentum(self, lookback: int = 252) -> pd.Series:
        """Calculate momentum factor (12-1 month returns)"""
        momentum = self.data.groupby('ticker')['returns'].rolling(
            window=lookback, min_periods=int(lookback * 0.8)
        ).apply(lambda x: np.prod(1 + x) - 1).reset_index(0, drop=True)
        
        return momentum
    
    def calculate_value(self) -> pd.Series:
        """Calculate value factor using P/E proxy"""
        # Use inverse of price momentum as PE proxy
        value = -self.data.groupby('ticker')['Close'].rolling(
            window=60, min_periods=30
        ).apply(lambda x: (x.iloc[-1] / x.mean()) - 1).reset_index(0, drop=True)
        
        return value
    
    def calculate_quality(self) -> pd.Series:
        """Calculate quality factor using return stability"""
        quality = self.data.groupby('ticker')['returns'].rolling(
            window=252, min_periods=126
        ).apply(lambda x: x.mean() / x.std() if x.std() > 0 else 0).reset_index(0, drop=True)
        
        return quality
    
    def calculate_size(self) -> pd.Series:
        """Calculate size factor (log market cap)"""
        size = -np.log(self.data['market_cap'] + 1)  # Negative for small-cap tilt
        return size
    
    def calculate_low_vol(self) -> pd.Series:
        """Calculate low volatility factor"""
        low_vol = -self.data.groupby('ticker')['returns'].rolling(
            window=60, min_periods=30
        ).std().reset_index(0, drop=True)
        
        return low_vol
    
    def calculate_all_factors(self) -> pd.DataFrame:
        """Calculate all factors and return combined DataFrame"""
        logger.info("Calculating all factors...")
        
        factors_df = self.data[['Date', 'ticker', 'Close', 'returns', 'market_cap']].copy()
        
        factors_df['momentum'] = self.calculate_momentum()
        factors_df['value'] = self.calculate_value()
        factors_df['quality'] = self.calculate_quality()
        factors_df['size'] = self.calculate_size()
        factors_df['low_vol'] = self.calculate_low_vol()
        
        # Rank factors (0-1 scale)
        factor_cols = ['momentum', 'value', 'quality', 'size', 'low_vol']
        for col in factor_cols:
            factors_df[f'{col}_rank'] = factors_df.groupby('Date')[col].rank(pct=True)
        
        # Remove rows with insufficient data
        factors_df = factors_df.dropna()
        
        logger.info(f"Calculated factors for {len(factors_df)} observations")
        return factors_df


class PortfolioConstructor:
    """Construct long-short portfolios based on factor signals"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
    def apply_bayesian_shrinkage(self, factor_scores: pd.DataFrame, 
                                shrinkage_factor: float = 0.3) -> pd.DataFrame:
        """Apply Bayesian shrinkage to factor scores"""
        shrunk_scores = factor_scores.copy()
        
        for col in ['momentum', 'value', 'quality', 'size', 'low_vol']:
            if col in shrunk_scores.columns:
                mean_score = shrunk_scores[col].mean()
                shrunk_scores[col] = (1 - shrinkage_factor) * shrunk_scores[col] + \
                                   shrinkage_factor * mean_score
        
        return shrunk_scores
    
    def apply_lasso_regularization(self, factor_scores: pd.DataFrame, 
                                  target_returns: pd.Series) -> pd.DataFrame:
        """Apply Lasso regularization to factor scores"""
        factor_cols = ['momentum', 'value', 'quality', 'size', 'low_vol']
        available_cols = [col for col in factor_cols if col in factor_scores.columns]
        
        if len(available_cols) == 0:
            return factor_scores
        
        # Prepare data
        X = factor_scores[available_cols].fillna(0)
        y = target_returns.fillna(0)
        
        # Align indices
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(X) < 10:  # Not enough data
            return factor_scores
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit Lasso
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X_scaled, y)
        
        # Apply regularization
        regularized_scores = factor_scores.copy()
        for i, col in enumerate(available_cols):
            regularized_scores[col] *= lasso.coef_[i]
        
        return regularized_scores
    
    def construct_portfolio(self, factor_data: pd.DataFrame, 
                          rebalance_date: pd.Timestamp,
                          use_shrinkage: bool = True,
                          use_lasso: bool = True) -> pd.Series:
        """Construct portfolio weights for a given rebalance date"""
        
        # Get data for this rebalance date
        date_data = factor_data[factor_data['Date'] == rebalance_date].copy()
        
        if len(date_data) < 10:  # Not enough stocks
            return pd.Series(dtype=float)
        
        # Apply regularization techniques
        if use_shrinkage:
            date_data = self.apply_bayesian_shrinkage(date_data)
        
        if use_lasso and len(date_data) > 20:
            # Use next period returns as target for Lasso
            future_returns = factor_data[factor_data['Date'] > rebalance_date].groupby('ticker')['returns'].first()
            date_data = self.apply_lasso_regularization(date_data, future_returns)
        
        # Combine factors into composite score
        factor_cols = ['momentum_rank', 'value_rank', 'quality_rank', 'size_rank', 'low_vol_rank']
        available_factor_cols = [col for col in factor_cols if col in date_data.columns]
        
        if not available_factor_cols:
            return pd.Series(dtype=float)
        
        # Equal-weight combination of factors
        date_data['composite_score'] = date_data[available_factor_cols].mean(axis=1)
        
        # Convert to portfolio weights
        date_data['weight'] = 0.0
        
        # Long top quintile, short bottom quintile
        top_quintile = date_data['composite_score'].quantile(0.8)
        bottom_quintile = date_data['composite_score'].quantile(0.2)
        
        long_stocks = date_data[date_data['composite_score'] >= top_quintile]
        short_stocks = date_data[date_data['composite_score'] <= bottom_quintile]
        
        # Assign weights
        if len(long_stocks) > 0:
            long_weight = self.config.leverage / (2 * len(long_stocks))
            date_data.loc[long_stocks.index, 'weight'] = long_weight
        
        if len(short_stocks) > 0:
            short_weight = -self.config.leverage / (2 * len(short_stocks))
            date_data.loc[short_stocks.index, 'weight'] = short_weight
        
        # Apply position limits
        date_data['weight'] = np.clip(date_data['weight'], 
                                    self.config.min_weight, 
                                    self.config.max_weight)
        
        # Return weights as Series
        weights = date_data.set_index('ticker')['weight']
        weights = weights[weights != 0]  # Remove zero weights
        
        return weights


class TransactionCostModel:
    """Model transaction costs"""
    
    def __init__(self, cost_per_trade: float = 0.001):
        self.cost_per_trade = cost_per_trade
    
    def calculate_costs(self, old_weights: pd.Series, new_weights: pd.Series) -> float:
        """Calculate transaction costs based on turnover"""
        # Align indices
        all_tickers = old_weights.index.union(new_weights.index)
        old_aligned = old_weights.reindex(all_tickers, fill_value=0)
        new_aligned = new_weights.reindex(all_tickers, fill_value=0)
        
        # Calculate turnover
        turnover = abs(new_aligned - old_aligned).sum()
        
        # Apply transaction cost
        return turnover * self.cost_per_trade


class PerformanceAnalyzer:
    """Analyze portfolio performance"""
    
    def __init__(self):
        self.results = {}
    
    def calculate_returns(self, weights_series: pd.DataFrame, 
                         returns_data: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns"""
        portfolio_returns = []
        
        for date in weights_series.index:
            date_weights = weights_series.loc[date]
            
            # Get next period returns
            next_date = returns_data[returns_data['Date'] > date]['Date'].min()
            if pd.isna(next_date):
                continue
                
            next_returns = returns_data[returns_data['Date'] == next_date].set_index('ticker')['returns']
            
            # Calculate weighted return
            common_tickers = date_weights.index.intersection(next_returns.index)
            if len(common_tickers) > 0:
                portfolio_return = (date_weights.loc[common_tickers] * 
                                  next_returns.loc[common_tickers]).sum()
                portfolio_returns.append({'Date': next_date, 'return': portfolio_return})
        
        return pd.DataFrame(portfolio_returns).set_index('Date')['return']
    
    def calculate_metrics(self, returns: pd.Series) -> Dict:
        """Calculate comprehensive performance metrics"""
        if len(returns) == 0:
            return {}
        
        # Remove NaN values
        returns = returns.dropna()
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative / rolling_max) - 1
        max_drawdown = drawdown.min()
        
        # Additional metrics
        win_rate = (returns > 0).mean()
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'num_observations': len(returns)
        }
    
    def plot_performance(self, returns: pd.Series, title: str = "Portfolio Performance"):
        """Plot performance charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cumulative returns
        cumulative = (1 + returns).cumprod()
        axes[0, 0].plot(cumulative.index, cumulative.values)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        
        # Rolling Sharpe ratio
        rolling_sharpe = returns.rolling(60).mean() / returns.rolling(60).std() * np.sqrt(252)
        axes[0, 1].plot(rolling_sharpe.index, rolling_sharpe.values)
        axes[0, 1].set_title('Rolling Sharpe Ratio (60-day)')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        
        # Drawdown
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative / rolling_max) - 1
        axes[1, 0].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[1, 0].set_title('Drawdown')
        axes[1, 0].set_ylabel('Drawdown')
        
        # Return distribution
        axes[1, 1].hist(returns, bins=50, alpha=0.7)
        axes[1, 1].set_title('Return Distribution')
        axes[1, 1].set_xlabel('Return')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        plt.show()


class Backtester:
    """Main backtesting engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_provider = DataProvider()
        self.portfolio_constructor = PortfolioConstructor(config)
        self.transaction_cost_model = TransactionCostModel(config.transaction_cost)
        self.performance_analyzer = PerformanceAnalyzer()
        
    def run_backtest(self, tickers: Optional[List[str]] = None, 
                    use_shrinkage: bool = True, 
                    use_lasso: bool = True) -> Dict:
        """Run complete backtest"""
        logger.info("Starting backtest...")
        
        # Get universe if not provided
        if tickers is None:
            tickers = self.data_provider.get_universe()
        
        # Fetch data
        raw_data = self.data_provider.fetch_yahoo_data(
            tickers, self.config.start_date, self.config.end_date
        )
        
        # Calculate factors
        factor_calculator = FactorCalculator(raw_data)
        factor_data = factor_calculator.calculate_all_factors()
        
        # Get rebalance dates
        rebalance_dates = pd.date_range(
            start=self.config.start_date, 
            end=self.config.end_date, 
            freq=self.config.rebalance_freq
        )
        
        # Run backtest
        portfolio_weights = []
        transaction_costs = []
        prev_weights = pd.Series(dtype=float)
        
        for rebalance_date in rebalance_dates:
            # Skip if no data available
            if rebalance_date not in factor_data['Date'].values:
                continue
                
            # Construct portfolio
            weights = self.portfolio_constructor.construct_portfolio(
                factor_data, rebalance_date, use_shrinkage, use_lasso
            )
            
            if len(weights) == 0:
                continue
            
            # Calculate transaction costs
            costs = self.transaction_cost_model.calculate_costs(prev_weights, weights)
            transaction_costs.append(costs)
            
            # Store weights
            portfolio_weights.append({
                'Date': rebalance_date,
                'weights': weights,
                'transaction_costs': costs
            })
            
            prev_weights = weights
        
        # Calculate performance
        if portfolio_weights:
            weights_df = pd.DataFrame(portfolio_weights)
            weights_df = weights_df.set_index('Date')
            
            # Calculate returns
            portfolio_returns = self.performance_analyzer.calculate_returns(
                weights_df['weights'], factor_data
            )
            
            # Adjust for transaction costs
            cost_series = pd.Series(transaction_costs, index=weights_df.index)
            adjusted_returns = portfolio_returns - cost_series
            
            # Calculate metrics
            metrics = self.performance_analyzer.calculate_metrics(adjusted_returns)
            
            # Store results
            results = {
                'returns': adjusted_returns,
                'gross_returns': portfolio_returns,
                'transaction_costs': cost_series,
                'metrics': metrics,
                'factor_data': factor_data,
                'portfolio_weights': weights_df
            }
            
            logger.info(f"Backtest completed. Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            return results
        
        else:
            logger.warning("No valid portfolio weights generated")
            return {}
    
    def walk_forward_analysis(self, tickers: Optional[List[str]] = None,
                            initial_window: int = 252,
                            step_size: int = 21) -> Dict:
        """Perform walk-forward out-of-sample analysis"""
        logger.info("Starting walk-forward analysis...")
        
        # Get universe if not provided
        if tickers is None:
            tickers = self.data_provider.get_universe()
        
        # Fetch data
        raw_data = self.data_provider.fetch_yahoo_data(
            tickers, self.config.start_date, self.config.end_date
        )
        
        # Calculate factors
        factor_calculator = FactorCalculator(raw_data)
        factor_data = factor_calculator.calculate_all_factors()
        
        # Get unique dates
        dates = sorted(factor_data['Date'].unique())
        
        # Walk-forward analysis
        results = []
        
        for i in range(initial_window, len(dates) - step_size, step_size):
            train_end_date = dates[i]
            test_start_date = dates[i + 1]
            test_end_date = dates[min(i + step_size, len(dates) - 1)]
            
            # Training data
            train_data = factor_data[factor_data['Date'] <= train_end_date]
            
            # Test data
            test_data = factor_data[
                (factor_data['Date'] >= test_start_date) & 
                (factor_data['Date'] <= test_end_date)
            ]
            
            if len(test_data) == 0:
                continue
            
            # Construct portfolio on training data
            weights = self.portfolio_constructor.construct_portfolio(
                train_data, train_end_date, use_shrinkage=True, use_lasso=True
            )
            
            if len(weights) == 0:
                continue
            
            # Calculate out-of-sample returns
            test_returns = []
            for test_date in test_data['Date'].unique():
                date_returns = test_data[test_data['Date'] == test_date].set_index('ticker')['returns']
                common_tickers = weights.index.intersection(date_returns.index)
                
                if len(common_tickers) > 0:
                    portfolio_return = (weights.loc[common_tickers] * 
                                      date_returns.loc[common_tickers]).sum()
                    test_returns.append(portfolio_return)
            
            if test_returns:
                results.append({
                    'train_end': train_end_date,
                    'test_start': test_start_date,
                    'test_end': test_end_date,
                    'returns': test_returns,
                    'weights': weights
                })
        
        # Combine results
        all_returns = []
        for result in results:
            all_returns.extend(result['returns'])
        
        if all_returns:
            returns_series = pd.Series(all_returns)
            metrics = self.performance_analyzer.calculate_metrics(returns_series)
            
            logger.info(f"Walk-forward analysis completed. OOS Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            
            return {
                'oos_returns': returns_series,
                'oos_metrics': metrics,
                'detailed_results': results
            }
        
        return {}


# Example usage and research notebook
def main():
    """Main function demonstrating the framework"""
    
    # Configuration
    config = BacktestConfig(
        start_date="2015-01-01",
        end_date="2023-12-31",
        rebalance_freq="M",
        transaction_cost=0.001,
        max_weight=0.05,
        min_weight=-0.05
    )
    
    # Initialize backtester
    backtester = Backtester(config)
    
    # Run basic backtest
    print("Running basic backtest...")
    results = backtester.run_backtest(use_shrinkage=True, use_lasso=True)
    
    if results:
        print("\n=== Backtest Results ===")
        for metric, value in results['metrics'].items():
            print(f"{metric}: {value:.4f}")
        
        # Plot performance
        backtester.performance_analyzer.plot_performance(
            results['returns'], 
            "Factor Model Portfolio Performance"
        )
    
    # Run walk-forward analysis
    print("\nRunning walk-forward analysis...")
    oos_results = backtester.walk_forward_analysis()
    
    if oos_results:
        print("\n=== Out-of-Sample Results ===")
        for metric, value in oos_results['oos_metrics'].items():
            print(f"{metric}: {value:.4f}")
        
        # Plot OOS performance
        backtester.performance_analyzer.plot_performance(
            oos_results['oos_returns'], 
            "Out-of-Sample Performance"
        )
    
    return results, oos_results


if __name__ == "__main__":
    # Run the analysis
    results, oos_results = main()
