"""
Walk-Forward Analysis Example
============================

Demonstrates out-of-sample testing using walk-forward analysis
with expanding and rolling windows.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factor_backtester import Backtester, BacktestConfig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """Run walk-forward analysis example"""
    
    print("🔄 Walk-Forward Analysis - Out-of-Sample Testing")
    print("=" * 55)
    
    # Configuration
    config = BacktestConfig(
        start_date="2015-01-01",
        end_date="2023-12-31",
        rebalance_freq="M",
        transaction_cost=0.0015,  # 15 bps (higher for realism)
        max_weight=0.08,
        min_weight=-0.08,
        leverage=1.0
    )
    
    print(f"📊 Configuration:")
    print(f"   Period: {config.start_date} to {config.end_date}")
    print(f"   Transaction Cost: {config.transaction_cost:.2%}")
    print(f"   Position Limits: ±{config.max_weight:.1%}")
    
    # Test universe (larger for more robust results)
    test_tickers = [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'ADBE', 'CRM', 'NFLX', 'ORCL',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'DHR', 'BMY', 'AMGN', 'GILD', 'BIIB',
        # Financial
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BLK', 'SPGI', 'ICE',
        # Consumer
        'PG', 'KO', 'PEP', 'WMT', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT', 'LOW',
        # Industrial
        'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'LMT', 'RTX', 'DE', 'EMR'
    ]
    
    print(f"📈 Universe: {len(test_tickers)} stocks")
    
    # Initialize backtester
    backtester = Backtester(config)
    
    # Run standard backtest for comparison
    print("\n🔄 Running full-sample backtest for comparison...")
    full_sample_results = backtester.run_backtest(
        tickers=test_tickers,
        use_shrinkage=True,
        use_lasso=True
    )
    
    # Run walk-forward analysis
    print("\n🔄 Running walk-forward analysis...")
    oos_results = backtester.walk_forward_analysis(
        tickers=test_tickers,
        initial_window=504,  # 2 years initial training
        step_size=21  # Monthly steps
    )
    
    if not oos_results or not full_sample_results:
        print("❌ Analysis failed")
        return
    
    # Compare results
    print("\n📊 Results Comparison:")
    print("=" * 40)
    
    is_metrics = full_sample_results['metrics']
    oos_metrics = oos_results['oos_metrics']
    
    comparison_data = {
        'Metric': [
            'Total Return', 'Annualized Return', 'Volatility', 
            'Sharpe Ratio', 'Max Drawdown', 'Win Rate'
        ],
        'In-Sample': [
            is_metrics['total_return'],
            is_metrics['annualized_return'],
            is_metrics['volatility'],
            is_metrics['sharpe_ratio'],
            is_metrics['max_drawdown'],
            is_metrics['win_rate']
        ],
        'Out-of-Sample': [
            oos_metrics['total_return'],
            oos_metrics['annualized_return'],
            oos_metrics['volatility'],
            oos_metrics['sharpe_ratio'],
            oos_metrics['max_drawdown'],
            oos_metrics['win_rate']
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Calculate degradation
    comparison_df['Degradation'] = (
        (comparison_df['Out-of-Sample'] - comparison_df['In-Sample']) / 
        comparison_df['In-Sample'].abs() * 100
    )
    
    print(f"{'Metric':<20} {'In-Sample':<12} {'Out-of-Sample':<15} {'Degradation':<12}")
    print("-" * 65)
    
    for _, row in comparison_df.iterrows():
        if row['Metric'] in ['Total Return', 'Annualized Return', 'Volatility', 'Max Drawdown', 'Win Rate']:
            is_val = f"{row['In-Sample']:.2%}"
            oos_val = f"{row['Out-of-Sample']:.2%}"
        else:
            is_val = f"{row['In-Sample']:.2f}"
            oos_val = f"{row['Out-of-Sample']:.2f}"
        
        degradation = f"{row['Degradation']:.1f}%"
        print(f"{row['Metric']:<20} {is_val:<12} {oos_val:<15} {degradation:<12}")
    
    # Statistical significance test
    if 'returns' in full_sample_results and 'oos_returns' in oos_results:
        from scipy import stats
        
        is_returns = full_sample_results['returns'].dropna()
        oos_returns = oos_results['oos_returns'].dropna()
        
        # T-test for mean difference
        t_stat, p_value = stats.ttest_ind(is_returns, oos_returns)
        
        print(f"\n📊 Statistical Tests:")
        print(f"   Mean Return Difference: {is_returns.mean() - oos_returns.mean():.4f}")
        print(f"   T-statistic: {t_stat:.2f}")
        print(f"   P-value: {p_value:.4f}")
        print(f"   Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Detailed walk-forward results
    if 'detailed_results' in oos_results:
        detailed = oos_results['detailed_results']
        
        print(f"\n📊 Walk-Forward Details:")
        print(f"   Number of windows: {len(detailed)}")
        print(f"   Average window size: {np.mean([len(r['returns']) for r in detailed]):.1f} observations")
        
        # Window-by-window analysis
        window_metrics = []
        for i, result in enumerate(detailed):
            if result['returns']:
                returns = pd.Series(result['returns'])
                window_metrics.append({
                    'Window': i + 1,
                    'Start': result['test_start'].strftime('%Y-%m'),
                    'End': result['test_end'].strftime('%Y-%m'),
                    'Return': returns.sum(),
                    'Sharpe': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                    'Observations': len(returns)
                })
        
        if window_metrics:
            window_df = pd.DataFrame(window_metrics)
            print(f"\n   Best window: {window_df.loc[window_df['Sharpe'].idxmax(), 'Start']} "
                  f"(Sharpe: {window_df['Sharpe'].max():.2f})")
            print(f"   Worst window: {window_df.loc[window_df['Sharpe'].idxmin(), 'Start']} "
                  f"(Sharpe: {window_df['Sharpe'].min():.2f})")
            print(f"   Positive windows: {(window_df['Return'] > 0).sum()}/{len(window_df)} "
                  f"({(window_df['Return'] > 0).mean():.1%})")
    
    # Visualization
    print("\n📊 Creating analysis charts...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Cumulative returns comparison
    if 'returns' in full_sample_results and 'oos_returns' in oos_results:
        is_cum = (1 + full_sample_results['returns']).cumprod()
        oos_cum = (1 + oos_results['oos_returns']).cumprod()
        
        axes[0, 0].plot(is_cum.index, is_cum.values, label='In-Sample', linewidth=2)
        axes[0, 0].plot(oos_cum.index, oos_cum.values, label='Out-of-Sample', linewidth=2)
        axes[0, 0].set_title('Cumulative Returns Comparison')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Performance metrics comparison
    metrics_to_plot = ['Annualized Return', 'Volatility', 'Sharpe Ratio']
    metric_values = comparison_df[comparison_df['Metric'].isin(metrics_to_plot)]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, metric_values['In-Sample'].values, width, 
                   label='In-Sample', alpha=0.8)
    axes[0, 1].bar(x + width/2, metric_values['Out-of-Sample'].values, width, 
                   label='Out-of-Sample', alpha=0.8)
    axes[0, 1].set_title('Performance Metrics Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metrics_to_plot, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Degradation analysis
    degradation_data = comparison_df[comparison_df['Metric'].isin(metrics_to_plot)]
    colors = ['red' if x < 0 else 'green' for x in degradation_data['Degradation']]
    
    axes[0, 2].bar(range(len(degradation_data)), degradation_data['Degradation'], 
                   color=colors, alpha=0.7)
    axes[0, 2].set_title('Performance Degradation (%)')
    axes[0, 2].set_xticks(range(len(degradation_data)))
    axes[0, 2].set_xticklabels(degradation_data['Metric'], rotation=45)
    axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 2].set_ylabel('Degradation (%)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Rolling performance (if enough data)
    if 'returns' in full_sample_results:
        returns = full_sample_results['returns']
        rolling_returns = returns.rolling(60).sum()  # 60-day rolling returns
        
        axes[1, 0].plot(rolling_returns.index, rolling_returns.values, alpha=0.7)
        axes[1, 0].set_title('Rolling 60-Day Returns (In-Sample)')
        axes[1, 0].set_ylabel('60-Day Return')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Return distributions comparison
    if 'returns' in full_sample_results and 'oos_returns' in oos_results:
        axes[1, 1].hist(full_sample_results['returns'], bins=30, alpha=0.7, 
                       label='In-Sample', density=True)
        axes[1, 1].hist(oos_results['oos_returns'], bins=30, alpha=0.7, 
                       label='Out-of-Sample', density=True)
        axes[1, 1].set_title('Return Distributions')
        axes[1, 1].set_xlabel('Daily Return')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Window-by-window Sharpe ratios
    if window_metrics:
        window_df = pd.DataFrame(window_metrics)
        axes[1, 2].plot(range(len(window_df)), window_df['Sharpe'], 'o-', alpha=0.7)
        axes[1, 2].set_title('Sharpe Ratio by Window')
        axes[1, 2].set_xlabel('Window Number')
        axes[1, 2].set_ylabel('Sharpe Ratio')
        axes[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('walk_forward_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary
    print(f"\n✅ Walk-Forward Analysis Complete!")
    print(f"📊 Chart saved as 'walk_forward_analysis.png'")
    
    # Key insights
    print(f"\n🔍 Key Insights:")
    sharpe_degradation = comparison_df[comparison_df['Metric'] == 'Sharpe Ratio']['Degradation'].iloc[0]
    
    if sharpe_degradation > -20:
        print(f"   ✅ Model shows good out-of-sample performance")
    elif sharpe_degradation > -50:
        print(f"   ⚠️  Model shows moderate degradation out-of-sample")
    else:
        print(f"   ❌ Model shows significant degradation out-of-sample")
    
    print(f"   📊 Sharpe ratio degradation: {sharpe_degradation:.1f}%")
    
    if 'detailed_results' in oos_results and window_metrics:
        positive_windows = (window_df['Return'] > 0).mean()
        print(f"   📈 Positive performance in {positive_windows:.1%} of windows")
    
    return full_sample_results, oos_results

if __name__ == "__main__":
    is_results, oos_results = main()
