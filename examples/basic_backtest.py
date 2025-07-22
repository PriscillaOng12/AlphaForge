"""
Basic Backtest Example
=====================

Simple example demonstrating how to run a basic factor model backtest.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_backtester import Backtester, BacktestConfig
import pandas as pd
import matplotlib.pyplot as plt

def main():
    """Run a basic backtest example"""
    
    print("🚀 Factor Model Backtest - Basic Example")
    print("=" * 50)
    
    # Configuration
    config = BacktestConfig(
        start_date="2018-01-01",
        end_date="2023-12-31",
        rebalance_freq="M",  # Monthly rebalancing
        transaction_cost=0.001,  # 10 bps
        max_weight=0.10,  # 10% maximum position
        min_weight=-0.10,  # 10% maximum short position
        leverage=1.0  # No leverage
    )
    
    print(f"📊 Configuration:")
    print(f"   Period: {config.start_date} to {config.end_date}")
    print(f"   Rebalancing: {config.rebalance_freq}")
    print(f"   Transaction Cost: {config.transaction_cost:.1%}")
    print(f"   Position Limits: {config.min_weight:.1%} to {config.max_weight:.1%}")
    
    # Initialize backtester
    backtester = Backtester(config)
    
    # Define a small universe for quick testing
    test_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
        'META', 'NVDA', 'JPM', 'JNJ', 'V',
        'PG', 'UNH', 'HD', 'MA', 'DIS',
        'ADBE', 'NFLX', 'CRM', 'CMCSA', 'PEP'
    ]
    
    print(f"\n📈 Universe: {len(test_tickers)} stocks")
    
    # Run backtest
    print("\n🔄 Running backtest...")
    results = backtester.run_backtest(
        tickers=test_tickers,
        use_shrinkage=True,
        use_lasso=True
    )
    
    if not results:
        print("❌ Backtest failed")
        return
    
    # Display results
    print("\n📊 Results:")
    print("=" * 30)
    
    metrics = results['metrics']
    
    print(f"📈 Total Return:        {metrics['total_return']:>8.2%}")
    print(f"📈 Annualized Return:   {metrics['annualized_return']:>8.2%}")
    print(f"📊 Volatility:          {metrics['volatility']:>8.2%}")
    print(f"⚡ Sharpe Ratio:        {metrics['sharpe_ratio']:>8.2f}")
    print(f"📉 Maximum Drawdown:    {metrics['max_drawdown']:>8.2%}")
    print(f"🎯 Win Rate:            {metrics['win_rate']:>8.2%}")
    print(f"📐 Skewness:            {metrics['skewness']:>8.2f}")
    print(f"📊 Kurtosis:            {metrics['kurtosis']:>8.2f}")
    print(f"📋 Observations:        {metrics['num_observations']:>8d}")
    
    # Transaction cost analysis
    total_costs = results['transaction_costs'].sum()
    print(f"\n💰 Transaction Costs:")
    print(f"   Total Costs:         {total_costs:.4f}")
    print(f"   Average per Trade:   {results['transaction_costs'].mean():.4f}")
    print(f"   Cost Drag (Annual):  {total_costs / len(results['returns']) * 252:.2%}")
    
    # Performance comparison
    gross_return = results['gross_returns'].sum()
    net_return = results['returns'].sum()
    cost_impact = gross_return - net_return
    
    print(f"\n📊 Performance Breakdown:")
    print(f"   Gross Return:        {gross_return:>8.2%}")
    print(f"   Net Return:          {net_return:>8.2%}")
    print(f"   Cost Impact:         {cost_impact:>8.2%}")
    
    # Simple visualization
    print("\n📊 Creating performance chart...")
    
    # Calculate cumulative returns
    cum_returns = (1 + results['returns']).cumprod()
    cum_gross_returns = (1 + results['gross_returns']).cumprod()
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Cumulative returns
    plt.subplot(2, 2, 1)
    plt.plot(cum_returns.index, cum_returns.values, label='Net Returns', linewidth=2)
    plt.plot(cum_gross_returns.index, cum_gross_returns.values, 
             label='Gross Returns', linewidth=2, alpha=0.7)
    plt.title('Cumulative Returns')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Drawdown
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns / running_max - 1) * 100
    plt.subplot(2, 2, 2)
    plt.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    plt.title('Drawdown')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Rolling Sharpe
    rolling_sharpe = (results['returns'].rolling(60).mean() / 
                     results['returns'].rolling(60).std() * 
                     (252**0.5))
    plt.subplot(2, 2, 3)
    plt.plot(rolling_sharpe.index, rolling_sharpe.values, color='orange')
    plt.title('Rolling Sharpe Ratio (60-day)')
    plt.ylabel('Sharpe Ratio')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Return distribution
    plt.subplot(2, 2, 4)
    plt.hist(results['returns'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Return Distribution')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.axvline(results['returns'].mean(), color='red', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('basic_backtest_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Basic backtest completed successfully!")
    print("📊 Chart saved as 'basic_backtest_results.png'")
    
    return results

if __name__ == "__main__":
    results = main()
