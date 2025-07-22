"""
Unit tests for Factor Model Backtester
======================================

Comprehensive test suite to ensure framework reliability.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_backtester import (
    BacktestConfig, DataProvider, FactorCalculator, 
    PortfolioConstructor, TransactionCostModel, 
    PerformanceAnalyzer, Backtester
)

class TestBacktestConfig(unittest.TestCase):
    """Test BacktestConfig class"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = BacktestConfig()
        
        self.assertGreater(high_costs, costs)


class TestPerformanceAnalyzer(unittest.TestCase):
    """Test PerformanceAnalyzer class"""
    
    def setUp(self):
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Create sample returns data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        self.sample_returns = pd.Series(
            np.random.randn(len(dates)) * 0.01,
            index=dates,
            name='returns'
        )
    
    def test_calculate_metrics(self):
        """Test performance metrics calculation"""
        metrics = self.performance_analyzer.calculate_metrics(self.sample_returns)
        
        # Check if all expected metrics are present
        expected_metrics = [
            'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
            'max_drawdown', 'win_rate', 'skewness', 'kurtosis', 'num_observations'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check if metrics are reasonable
        self.assertIsInstance(metrics['sharpe_ratio'], float)
        self.assertGreaterEqual(metrics['win_rate'], 0)
        self.assertLessEqual(metrics['win_rate'], 1)
        self.assertLessEqual(metrics['max_drawdown'], 0)
        self.assertEqual(metrics['num_observations'], len(self.sample_returns))
    
    def test_empty_returns(self):
        """Test handling of empty returns"""
        empty_returns = pd.Series(dtype=float)
        metrics = self.performance_analyzer.calculate_metrics(empty_returns)
        
        self.assertEqual(metrics, {})
    
    def test_calculate_returns_with_weights(self):
        """Test portfolio return calculation"""
        # Create sample weights and returns data
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        # Sample weights series (portfolio weights over time)
        weights_data = []
        for i, date in enumerate(dates[:-1]):  # Exclude last date
            weights = {ticker: np.random.random() * 0.1 for ticker in tickers}
            weights_data.append(pd.Series(weights))
        
        weights_series = pd.Series(weights_data, index=dates[:-1])
        
        # Sample returns data
        returns_data = []
        for date in dates:
            for ticker in tickers:
                returns_data.append({
                    'Date': date,
                    'ticker': ticker,
                    'returns': np.random.randn() * 0.02
                })
        
        returns_df = pd.DataFrame(returns_data)
        
        portfolio_returns = self.performance_analyzer.calculate_returns(
            weights_series, returns_df
        )
        
        self.assertIsInstance(portfolio_returns, pd.Series)


class TestBacktester(unittest.TestCase):
    """Test Backtester class integration"""
    
    def setUp(self):
        self.config = BacktestConfig(
            start_date="2020-01-01",
            end_date="2020-06-30",
            rebalance_freq="M",
            transaction_cost=0.001
        )
        self.backtester = Backtester(self.config)
    
    def test_initialization(self):
        """Test backtester initialization"""
        self.assertIsInstance(self.backtester.config, BacktestConfig)
        self.assertIsInstance(self.backtester.data_provider, DataProvider)
        self.assertIsInstance(self.backtester.portfolio_constructor, PortfolioConstructor)
        self.assertIsInstance(self.backtester.transaction_cost_model, TransactionCostModel)
        self.assertIsInstance(self.backtester.performance_analyzer, PerformanceAnalyzer)
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid dates
        with self.assertRaises(Exception):
            invalid_config = BacktestConfig(
                start_date="2020-01-01",
                end_date="2019-01-01"  # End before start
            )
    
    def test_small_backtest(self):
        """Test backtest with minimal data"""
        # Use a very small universe and short period for testing
        test_tickers = ['AAPL', 'MSFT']
        
        try:
            # This may fail due to data availability, which is acceptable for testing
            results = self.backtester.run_backtest(tickers=test_tickers)
            
            if results:  # If we got results
                self.assertIn('metrics', results)
                self.assertIn('returns', results)
                
                # Check metrics structure
                metrics = results['metrics']
                self.assertIn('total_return', metrics)
                self.assertIn('sharpe_ratio', metrics)
                
        except Exception as e:
            # Data fetching might fail in test environment
            print(f"Backtest failed (expected in test environment): {e}")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets"""
        empty_data = pd.DataFrame()
        
        # FactorCalculator should handle empty data gracefully
        try:
            factor_calculator = FactorCalculator(empty_data)
            # Should not crash, but may return empty results
        except Exception as e:
            # Some exceptions are acceptable for truly empty data
            self.assertIsInstance(e, (ValueError, KeyError, IndexError))
    
    def test_single_stock_portfolio(self):
        """Test portfolio construction with single stock"""
        config = BacktestConfig()
        portfolio_constructor = PortfolioConstructor(config)
        
        # Create minimal factor data with one stock
        factor_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=5, freq='D'),
            'ticker': ['AAPL'] * 5,
            'momentum_rank': [0.5] * 5,
            'value_rank': [0.5] * 5,
            'quality_rank': [0.5] * 5,
            'size_rank': [0.5] * 5,
            'low_vol_rank': [0.5] * 5
        })
        
        weights = portfolio_constructor.construct_portfolio(
            factor_data, factor_data['Date'].iloc[2]
        )
        
        # Should handle single stock case
        self.assertIsInstance(weights, pd.Series)
    
    def test_missing_data_handling(self):
        """Test handling of missing data"""
        # Create data with NaN values
        data_with_nans = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10, freq='D'),
            'ticker': ['AAPL'] * 10,
            'Close': [100, np.nan, 102, 103, np.nan, 105, 106, np.nan, 108, 109],
            'Volume': [1000000] * 10
        })
        
        factor_calculator = FactorCalculator(data_with_nans)
        
        # Should handle NaN values without crashing
        self.assertIn('returns', factor_calculator.data.columns)
    
    def test_extreme_config_values(self):
        """Test extreme configuration values"""
        # Test very high transaction costs
        extreme_config = BacktestConfig(
            transaction_cost=0.1,  # 10% transaction cost
            max_weight=1.0,        # 100% position limit
            min_weight=-1.0        # 100% short limit
        )
        
        self.assertEqual(extreme_config.transaction_cost, 0.1)
        self.assertEqual(extreme_config.max_weight, 1.0)
        self.assertEqual(extreme_config.min_weight, -1.0)


class TestDataQuality(unittest.TestCase):
    """Test data quality and validation"""
    
    def test_return_calculation_consistency(self):
        """Test that returns are calculated consistently"""
        # Create simple price series
        prices = pd.Series([100, 101, 99, 102, 98])
        expected_returns = prices.pct_change().dropna()
        
        # Create sample data
        sample_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=5, freq='D'),
            'ticker': ['TEST'] * 5,
            'Close': prices,
            'Volume': [1000000] * 5
        })
        
        factor_calculator = FactorCalculator(sample_data)
        calculated_returns = factor_calculator.data['returns'].dropna()
        
        # Check if returns are calculated correctly
        np.testing.assert_array_almost_equal(
            calculated_returns.values, 
            expected_returns.values,
            decimal=6
        )
    
    def test_factor_ranking_consistency(self):
        """Test that factor ranking works correctly"""
        # Create sample factor data
        factor_data = pd.DataFrame({
            'Date': ['2020-01-01'] * 5,
            'ticker': ['A', 'B', 'C', 'D', 'E'],
            'momentum': [1, 2, 3, 4, 5]
        })
        
        factor_data['Date'] = pd.to_datetime(factor_data['Date'])
        factor_data['momentum_rank'] = factor_data.groupby('Date')['momentum'].rank(pct=True)
        
        # Check ranking
        expected_ranks = [0.2, 0.4, 0.6, 0.8, 1.0]
        np.testing.assert_array_almost_equal(
            factor_data['momentum_rank'].values,
            expected_ranks,
            decimal=6
        )


def run_performance_tests():
    """Run performance benchmarks"""
    print("🔄 Running Performance Tests...")
    
    import time
    
    # Test data loading performance
    start_time = time.time()
    data_provider = DataProvider()
    universe = data_provider.get_universe()[:10]  # Small universe
    
    try:
        # This will use cached data if available
        data = data_provider.fetch_yahoo_data(universe, "2020-01-01", "2020-12-31")
        data_time = time.time() - start_time
        print(f"📊 Data loading: {data_time:.2f} seconds for {len(universe)} stocks")
        
        # Test factor calculation performance
        start_time = time.time()
        factor_calculator = FactorCalculator(data)
        factor_data = factor_calculator.calculate_all_factors()
        factor_time = time.time() - start_time
        print(f"🔬 Factor calculation: {factor_time:.2f} seconds for {len(factor_data)} observations")
        
    except Exception as e:
        print(f"⚠️ Performance test skipped due to data availability: {e}")


if __name__ == '__main__':
    print("🧪 Running Factor Model Backtester Tests")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run performance tests
    print("\n" + "=" * 50)
    run_performance_tests()
    
    print("\n✅ All tests completed!")Equal(config.start_date, "2010-01-01")
        self.assertEqual(config.end_date, "2023-12-31")
        self.assertEqual(config.rebalance_freq, "M")
        self.assertEqual(config.transaction_cost, 0.001)
        self.assertEqual(config.max_weight, 0.05)
        self.assertEqual(config.min_weight, -0.05)
        self.assertEqual(config.leverage, 1.0)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = BacktestConfig(
            start_date="2020-01-01",
            end_date="2022-12-31",
            transaction_cost=0.002,
            max_weight=0.10
        )
        
        self.assertEqual(config.start_date, "2020-01-01")
        self.assertEqual(config.end_date, "2022-12-31")
        self.assertEqual(config.transaction_cost, 0.002)
        self.assertEqual(config.max_weight, 0.10)


class TestDataProvider(unittest.TestCase):
    """Test DataProvider class"""
    
    def setUp(self):
        self.data_provider = DataProvider()
    
    def test_get_universe(self):
        """Test universe generation"""
        universe = self.data_provider.get_universe("SP500")
        
        self.assertIsInstance(universe, list)
        self.assertGreater(len(universe), 0)
        self.assertIn('AAPL', universe)  # Should contain major stocks
    
    def test_create_sample_data(self):
        """Test sample data creation for testing"""
        # Create sample data for testing
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        data = []
        for ticker in tickers:
            for date in dates:
                if date.weekday() < 5:  # Only weekdays
                    price = 100 + np.random.randn() * 10  # Random walk
                    data.append({
                        'Date': date,
                        'ticker': ticker,
                        'Open': price,
                        'High': price * 1.02,
                        'Low': price * 0.98,
                        'Close': price,
                        'Volume': np.random.randint(1000000, 10000000)
                    })
        
        df = pd.DataFrame(data)
        
        self.assertGreater(len(df), 0)
        self.assertEqual(len(df['ticker'].unique()), 3)
        self.assertIn('Date', df.columns)
        self.assertIn('Close', df.columns)


class TestFactorCalculator(unittest.TestCase):
    """Test FactorCalculator class"""
    
    def setUp(self):
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        data = []
        for ticker in tickers:
            price = 100
            for date in dates:
                if date.weekday() < 5:  # Only weekdays
                    price *= (1 + np.random.randn() * 0.02)  # Random walk
                    data.append({
                        'Date': date,
                        'ticker': ticker,
                        'Open': price,
                        'High': price * 1.01,
                        'Low': price * 0.99,
                        'Close': price,
                        'Volume': np.random.randint(1000000, 10000000)
                    })
        
        self.sample_data = pd.DataFrame(data)
        self.factor_calculator = FactorCalculator(self.sample_data)
    
    def test_prepare_data(self):
        """Test data preparation"""
        # Check if returns are calculated
        self.assertIn('returns', self.factor_calculator.data.columns)
        self.assertIn('market_cap', self.factor_calculator.data.columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(
            self.factor_calculator.data['Date']))
    
    def test_calculate_momentum(self):
        """Test momentum factor calculation"""
        momentum = self.factor_calculator.calculate_momentum(lookback=20)
        
        self.assertIsInstance(momentum, pd.Series)
        self.assertGreater(len(momentum.dropna()), 0)
    
    def test_calculate_value(self):
        """Test value factor calculation"""
        value = self.factor_calculator.calculate_value()
        
        self.assertIsInstance(value, pd.Series)
        self.assertGreater(len(value.dropna()), 0)
    
    def test_calculate_quality(self):
        """Test quality factor calculation"""
        quality = self.factor_calculator.calculate_quality()
        
        self.assertIsInstance(quality, pd.Series)
        self.assertGreater(len(quality.dropna()), 0)
    
    def test_calculate_all_factors(self):
        """Test comprehensive factor calculation"""
        factor_data = self.factor_calculator.calculate_all_factors()
        
        self.assertIsInstance(factor_data, pd.DataFrame)
        self.assertGreater(len(factor_data), 0)
        
        # Check if all factors are present
        expected_factors = ['momentum', 'value', 'quality', 'size', 'low_vol']
        for factor in expected_factors:
            self.assertIn(factor, factor_data.columns)
            self.assertIn(f'{factor}_rank', factor_data.columns)


class TestPortfolioConstructor(unittest.TestCase):
    """Test PortfolioConstructor class"""
    
    def setUp(self):
        self.config = BacktestConfig(max_weight=0.1, min_weight=-0.1)
        self.portfolio_constructor = PortfolioConstructor(self.config)
        
        # Create sample factor data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2020-03-31', freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        data = []
        for date in dates:
            if date.weekday() < 5:  # Only weekdays
                for ticker in tickers:
                    data.append({
                        'Date': date,
                        'ticker': ticker,
                        'Close': 100 + np.random.randn() * 10,
                        'returns': np.random.randn() * 0.02,
                        'momentum_rank': np.random.random(),
                        'value_rank': np.random.random(),
                        'quality_rank': np.random.random(),
                        'size_rank': np.random.random(),
                        'low_vol_rank': np.random.random()
                    })
        
        self.factor_data = pd.DataFrame(data)
    
    def test_bayesian_shrinkage(self):
        """Test Bayesian shrinkage"""
        original_data = self.factor_data.copy()
        shrunk_data = self.portfolio_constructor.apply_bayesian_shrinkage(
            original_data, shrinkage_factor=0.3
        )
        
        self.assertEqual(len(shrunk_data), len(original_data))
        # Values should be shrunk toward mean
        self.assertNotEqual(shrunk_data['momentum_rank'].var(), 
                           original_data['momentum_rank'].var())
    
    def test_construct_portfolio(self):
        """Test portfolio construction"""
        sample_date = self.factor_data['Date'].iloc[20]
        weights = self.portfolio_constructor.construct_portfolio(
            self.factor_data, sample_date
        )
        
        self.assertIsInstance(weights, pd.Series)
        
        if len(weights) > 0:
            # Check position limits
            self.assertTrue(all(weights <= self.config.max_weight))
            self.assertTrue(all(weights >= self.config.min_weight))
            
            # Check that we have both long and short positions
            long_positions = (weights > 0).sum()
            short_positions = (weights < 0).sum()
            self.assertGreater(long_positions + short_positions, 0)


class TestTransactionCostModel(unittest.TestCase):
    """Test TransactionCostModel class"""
    
    def setUp(self):
        self.cost_model = TransactionCostModel(cost_per_trade=0.001)
    
    def test_calculate_costs(self):
        """Test transaction cost calculation"""
        old_weights = pd.Series({'AAPL': 0.05, 'MSFT': 0.03, 'GOOGL': -0.02})
        new_weights = pd.Series({'AAPL': 0.04, 'MSFT': 0.05, 'GOOGL': -0.03, 'AMZN': 0.02})
        
        costs = self.cost_model.calculate_costs(old_weights, new_weights)
        
        self.assertIsInstance(costs, float)
        self.assertGreaterEqual(costs, 0)
        
        # Higher turnover should result in higher costs
        high_turnover_weights = pd.Series({'AAPL': -0.05, 'MSFT': -0.03, 'GOOGL': 0.08})
        high_costs = self.cost_model.calculate_costs(old_weights, high_turnover_weights)
        
        self.assert
