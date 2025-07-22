"""
Custom Factors Example
=====================

Demonstrates how to extend the framework with custom factors
and alternative portfolio construction methods.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_backtester import (
    Backtester, BacktestConfig, FactorCalculator, 
    PortfolioConstructor, DataProvider
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class CustomFactorCalculator(FactorCalculator):
    """Extended factor calculator with custom factors"""
    
    def calculate_mean_reversion(self, window: int = 20) -> pd.Series:
        """Calculate mean reversion factor"""
        # Price relative to moving average
        ma = self.data.groupby('ticker')['Close'].rolling(window=window).mean()
        mean_reversion = -(self.data['Close'] / ma - 1)  # Negative for mean reversion
        return mean_reversion.reset_index(0, drop=True)
    
    def calculate_volume_trend(self, window: int = 20) -> pd.Series:
        """Calculate volume trend factor"""
        # Volume relative to average
        vol_ma = self.data.groupby('ticker')['Volume'].rolling(window=window).mean()
        volume_trend = self.data['Volume'] / vol_ma - 1
        return volume_trend.reset_index(0, drop=True)
    
    def calculate_price_acceleration(self, short_window: int = 5, long_window: int = 20) -> pd.Series:
        """Calculate price acceleration factor"""
        # Difference between short and long term momentum
        short_mom = self.data.groupby('ticker')['returns'].rolling(
            window=short_window).apply(lambda x: np.prod(1 + x) - 1)
        long_mom = self.data.groupby('ticker')['returns'].rolling(
            window=long_window).apply(lambda x: np.prod(1 + x) - 1)
        
        acceleration = short_mom - long_mom
        return acceleration.reset_index(0, drop=True)
    
    def calculate_volatility_regime(self, window: int = 60) -> pd.Series:
        """Calculate volatility regime factor"""
        # Current volatility vs historical
        current_vol = self.data.groupby('ticker')['returns'].rolling(
            window=20).std()
        historical_vol = self.data.groupby('ticker')['returns'].rolling(
            window=window).std()
        
        vol_regime = current_vol / historical_vol - 1
        return vol_regime.reset_index(0, drop=True)
    
    def calculate_earnings_momentum(self) -> pd.Series:
        """Calculate earnings momentum proxy using return stability"""
        # Proxy using return consistency as earnings quality indicator
        earnings_momentum = self.data.groupby('ticker')['returns'].rolling(
            window=60, min_periods=30
        ).apply(
            lambda x: x.rolling(20).mean().diff().mean() if len(x) > 20 else 0
        ).reset_index(0, drop=True)
        
        return earnings_momentum
    
    def calculate_all_factors(self) -> pd.DataFrame:
        """Calculate all factors including custom ones"""
        # Get base factors
        factors_df = super().calculate_all_factors()
        
        print("Calculating custom factors...")
        
        # Add custom factors
        factors_df['mean_reversion'] = self.calculate_mean_reversion()
        factors_df['volume_trend'] = self.calculate_volume_trend()
        factors_df['price_acceleration'] = self.calculate_price_acceleration()
        factors_df['volatility_regime'] = self.calculate_volatility_regime()
        factors_df['earnings_momentum'] = self.calculate_earnings_momentum()
        
        # Rank custom factors
        custom_factors = ['mean_reversion', 'volume_trend', 'price_acceleration', 
                         'volatility_regime', 'earnings_momentum']
        
        for col in custom_factors:
            if col in factors_df.columns:
                factors_df[f'{col}_rank'] = factors_df.groupby('Date')[col].rank(pct=True)
        
        # Remove rows with insufficient data
        factors_df = factors_df.dropna()
        
        print(f"Calculated {len(custom_factors)} custom factors")
        return factors_df


class MLPortfolioConstructor(PortfolioConstructor):
    """Machine learning-based portfolio constructor"""
    
    def __init__(self, config, model_type='random_forest'):
        super().__init__(config)
        self.model_type = model_type
        self.scaler = StandardScaler()
        
    def train_ml_model(self, factor_data: pd.DataFrame, target_col: str = 'returns'):
        """Train machine learning model to predict returns"""
        # Prepare features
        feature_cols = [col for col in factor_data.columns 
                       if col.endswith('_rank') or col.endswith('_norm')]
        
        if not feature_cols:
            print("No feature columns found")
            return None
            
        # Get training data
        train_data = factor_data[feature_cols + [target_col, 'ticker', 'Date']].dropna()
        
        if len(train_data) < 100:
            print("Insufficient training data")
            return None
            
        # Prepare features and targets
        X = train_data[feature_cols]
        y = train_data[target_col]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        if self.model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=50,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        model.fit(X_scaled, y)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return model
    
    def construct_ml_portfolio(self, factor_data: pd.DataFrame, 
                              rebalance_date: pd.Timestamp) -> pd.Series:
        """Construct portfolio using ML predictions"""
        
        # Get data for this date
        current_data = factor_data[factor_data['Date'] == rebalance_date].copy()
        
        if len(current_data) < 10:
            return pd.Series(dtype=float)
        
        # Get training data (all data before this date)
        train_data = factor_data[factor_data['Date'] < rebalance_date].copy()
        
        if len(train_data) < 500:  # Need sufficient training data
            print(f"Insufficient training data: {len(train_data)} observations")
            return pd.Series(dtype=float)
        
        # Add forward returns to training data
        train_data['future_returns'] = train_data.groupby('ticker')['returns'].shift(-1)
        train_data = train_data.dropna(subset=['future_returns'])
        
        # Train model
        model = self.train_ml_model(train_data, 'future_returns')
        
        if model is None:
            return pd.Series(dtype=float)
        
        # Make predictions for current period
        feature_cols = [col for col in current_data.columns 
                       if col.endswith('_rank') or col.endswith('_norm')]
        
        if not feature_cols:
            return pd.Series(dtype=float)
            
        X_current = current_data[feature_cols].fillna(0)
        
        if len(X_current) == 0:
            return pd.Series(dtype=float)
            
        X_current_scaled = self.scaler.transform(X_current)
        predictions = model.predict(X_current_scaled)
        
        # Convert predictions to portfolio weights
        current_data['ml_score'] = predictions
        
        # Create long-short portfolio based on predictions
        current_data['weight'] = 0.0
        
        # Long top quintile, short bottom quintile
        top_quintile = current_data['ml_score'].quantile(0.8)
        bottom_quintile = current_data['ml_score'].quantile(0.2)
        
        long_stocks = current_data[current_data['ml_score'] >= top_quintile]
        short_stocks = current_data[current_data['ml_score'] <= bottom_quintile]
        
        # Assign weights
        if len(long_stocks) > 0:
            long_weight = self.config.leverage / (2 * len(long_stocks))
            current_data.loc[long_stocks.index, 'weight'] = long_weight
        
        if len(short_stocks) > 0:
            short_weight = -self.config.leverage / (2 * len(short_stocks))
            current_data.loc[short_stocks.index, 'weight'] = short_weight
        
        # Apply position limits
        current_data['weight'] = np.clip(current_data['weight'], 
                                       self.config.min_weight, 
                                       self.config.max_weight)
        
        # Return weights
        weights = current_data.set_index('ticker')['weight']
        weights = weights[weights != 0]
        
        print(f"ML Portfolio: {len(weights)} positions, "
              f"Long: {(weights > 0).sum()}, Short: {(weights < 0).sum()}")
        
        return weights


class CustomBacktester(Backtester):
    """Custom backtester with extended functionality"""
    
    def __init__(self, config: BacktestConfig, use_ml: bool = False):
        super().__init__(config)
        self.use_ml = use_ml
        
        if use_ml:
            self.portfolio_constructor = MLPortfolioConstructor(config)
    
    def run_custom_backtest(self, tickers=None, factor_weights=None):
        """Run backtest with custom factors and optional ML"""
        
        print("🔬 Running Custom Factor Backtest")
        print("=" * 40)
        
        # Get data
        if tickers is None:
            tickers = self.data_provider.get_universe()[:50]  # Smaller for demo
            
        raw_data = self.data_provider.fetch_yahoo_data(
            tickers, self.config.start_date, self.config.end_date
        )
        
        # Use custom factor calculator
        factor_calculator = CustomFactorCalculator(raw_data)
        factor_data = factor_calculator.calculate_all_factors()
        
        print(f"📊 Factor data shape: {factor_data.shape}")
        print(f"📊 Available factors: {[col for col in factor_data.columns if col.endswith('_rank')]}")
        
        # Analyze factor correlations
        factor_cols = [col for col in factor_data.columns if col.endswith('_rank')]
        correlation_matrix = factor_data[factor_cols].corr()
        
        print("\n📊 Factor Correlation Matrix:")
        print(correlation_matrix.round(3))
        
        # Run backtest with different approaches
        results = {}
        
        # 1. Traditional approach with custom factors
        print("\n🔄 Running traditional approach...")
        traditional_results = self._run_traditional_backtest(factor_data, factor_weights)
        results['traditional'] = traditional_results
        
        # 2. ML approach (if enabled)
        if self.use_ml:
            print("\n🤖 Running ML approach...")
            ml_results = self._run_ml_backtest(factor_data)
            results['ml'] = ml_results
        
        return results
    
    def _run_traditional_backtest(self, factor_data, factor_weights=None):
        """Run traditional factor-based backtest"""
        
        # Get rebalance dates
        rebalance_dates = pd.date_range(
            start=self.config.start_date, 
            end=self.config.end_date, 
            freq=self.config.rebalance_freq
        )
        
        portfolio_weights = []
        prev_weights = pd.Series(dtype=float)
        
        for rebalance_date in rebalance_dates:
            if rebalance_date not in factor_data['Date'].values:
                continue
                
            weights = self.portfolio_constructor.construct_portfolio(
                factor_data, rebalance_date, use_shrinkage=True, use_lasso=True
            )
            
            if len(weights) > 0:
                costs = self.transaction_cost_model.calculate_costs(prev_weights, weights)
                portfolio_weights.append({
                    'Date': rebalance_date,
                    'weights': weights,
                    'transaction_costs': costs
                })
                prev_weights = weights
        
        if portfolio_weights:
            weights_df = pd.DataFrame(portfolio_weights).set_index('Date')
            portfolio_returns = self.performance_analyzer.calculate_returns(
                weights_df['weights'], factor_data
            )
            
            cost_series = pd.Series([p['transaction_costs'] for p in portfolio_weights], 
                                  index=weights_df.index)
            adjusted_returns = portfolio_returns - cost_series
            
            metrics = self.performance_analyzer.calculate_metrics(adjusted_returns)
            
            return {
                'returns': adjusted_returns,
                'metrics': metrics,
                'weights': weights_df
            }
        
        return {}
    
    def _run_ml_backtest(self, factor_data):
        """Run ML-based backtest"""
        
        rebalance_dates = pd.date_range(
            start=self.config.start_date, 
            end=self.config.end_date, 
            freq=self.config.rebalance_freq
        )
        
        portfolio_weights = []
        prev_weights = pd.Series(dtype=float)
        
        for i, rebalance_date in enumerate(rebalance_dates):
            if rebalance_date not in factor_data['Date'].values:
                continue
            
            # Skip first few periods to have enough training data
            if i < 10:
                continue
                
            weights = self.portfolio_constructor.construct_ml_portfolio(
                factor_data, rebalance_date
            )
            
            if len(weights) > 0:
                costs = self.transaction_cost_model.calculate_costs(prev_weights, weights)
                portfolio_weights.append({
                    'Date': rebalance_date,
                    'weights': weights,
                    'transaction_costs': costs
                })
                prev_weights = weights
        
        if portfolio_weights:
            weights_df = pd.DataFrame(portfolio_weights).set_index('Date')
            portfolio_returns = self.performance_analyzer.calculate_returns(
                weights_df['weights'], factor_data
            )
            
            cost_series = pd.Series([p['transaction_costs'] for p in portfolio_weights], 
                                  index=weights_df.index)
            adjusted_returns = portfolio_returns - cost_series
            
            metrics = self.performance_analyzer.calculate_metrics(adjusted_returns)
            
            return {
                'returns': adjusted_returns,
                'metrics': metrics,
                'weights': weights_df
            }
        
        return {}


def main():
    """Main function demonstrating custom factors and ML"""
    
    # Configuration
    config = BacktestConfig(
        start_date="2018-01-01",
        end_date="2023-12-31",
        rebalance_freq="M",
        transaction_cost=0.0015,
        max_weight=0.08,
        min_weight=-0.08
    )
    
    # Test universe
    test_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V',
        'PG', 'UNH', 'HD', 'MA', 'DIS', 'ADBE', 'NFLX', 'CRM', 'CMCSA', 'PEP',
        'KO', 'PFE', 'BAC', 'WMT', 'XOM', 'CVX', 'ABT', 'TMO', 'DHR', 'BMY'
    ]
    
    # Run traditional backtest
    print("🔬 Testing Traditional Approach with Custom Factors")
    traditional_backtester = CustomBacktester(config, use_ml=False)
    traditional_results = traditional_backtester.run_custom_backtest(tickers=test_tickers)
    
    # Run ML backtest
    print("\n🤖 Testing Machine Learning Approach")
    ml_backtester = CustomBacktester(config, use_ml=True)
    ml_results = ml_backtester.run_custom_backtest(tickers=test_tickers)
    
    # Compare results
    print("\n📊 Results Comparison:")
    print("=" * 50)
    
    approaches = ['traditional', 'ml']
    all_results = [traditional_results, ml_results]
    
    comparison_data = []
    
    for approach, results in zip(approaches, all_results):
        if approach in results and results[approach]:
            metrics = results[approach]['metrics']
            comparison_data.append({
                'Approach': approach.title(),
                'Total Return': metrics['total_return'],
                'Annualized Return': metrics['annualized_return'],
                'Volatility': metrics['volatility'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Max Drawdown': metrics['max_drawdown'],
                'Win Rate': metrics['win_rate']
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance comparison
        metrics_to_plot = ['Annualized Return', 'Volatility', 'Sharpe Ratio']
        
        for i, metric in enumerate(metrics_to_plot):
            if i < 3:
                row, col = i // 2, i % 2
                values = comparison_df[metric]
                axes[row, col].bar(comparison_df['Approach'], values, alpha=0.7)
                axes[row, col].set_title(f'{metric} Comparison')
                axes[row, col].set_ylabel(metric)
                axes[row, col].grid(True, alpha=0.3)
        
        # Cumulative returns
        axes[1, 1].set_title('Cumulative Returns Comparison')
        
        for approach, results in zip(approaches, all_results):
            if approach in results and results[approach] and 'returns' in results[approach]:
                cum_returns = (1 + results[approach]['returns']).cumprod()
                axes[1, 1].plot(cum_returns.index, cum_returns.values, 
                               label=approach.title(), linewidth=2)
        
        axes[1, 1].set_ylabel('Cumulative Return')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('custom_factors_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n✅ Custom factors analysis complete!")
        print("📊 Chart saved as 'custom_factors_comparison.png'")
    
    return traditional_results, ml_results

if __name__ == "__main__":
    traditional_results, ml_results = main()
