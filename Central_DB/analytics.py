"""
Analytics Module
Performance metrics and statistics
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import statistics


class Analytics:
    """Portfolio and strategy analytics"""
    
    def __init__(self, database):
        self.db = database
    
    def get_portfolio_metrics(self) -> Dict:
        """Get current portfolio metrics"""
        snapshot = self.db.get_latest_snapshot()
        
        if not snapshot:
            return {
                'total_capital': 0.0,
                'deployed_capital': 0.0,
                'available_capital': 0.0,
                'total_pnl': 0.0,
                'daily_pnl': 0.0,
                'num_positions': 0,
                'num_trades': 0,
                'utilization': 0.0
            }
        
        utilization = (snapshot['deployed_capital'] / snapshot['total_capital'] * 100 
                      if snapshot['total_capital'] > 0 else 0)
        
        return {
            'total_capital': snapshot['total_capital'],
            'deployed_capital': snapshot['deployed_capital'],
            'available_capital': snapshot['available_capital'],
            'total_pnl': snapshot['total_pnl'],
            'daily_pnl': snapshot['daily_pnl'],
            'num_positions': snapshot['num_positions'],
            'num_trades': snapshot['num_trades'],
            'utilization': utilization
        }
    
    def get_strategy_performance(self, strategy: Optional[str] = None) -> Dict:
        """Get strategy performance metrics"""
        if strategy:
            perf = self.db.get_strategy_performance(strategy)
            return perf if perf else self._empty_performance()
        else:
            return self.db.get_all_strategy_performance()
    
    def _empty_performance(self) -> Dict:
        """Empty performance dict"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0
        }
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe
        sharpe = (avg_return - risk_free_rate / 252) / std_return * (252 ** 0.5)
        return sharpe
    
    def calculate_max_drawdown(self, equity_curve: List[float]) -> Dict:
        """Calculate maximum drawdown"""
        if not equity_curve:
            return {'max_drawdown': 0.0, 'current_drawdown': 0.0}
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        # Current drawdown
        current_peak = max(equity_curve)
        current_dd = (current_peak - equity_curve[-1]) / current_peak if current_peak > 0 else 0
        
        return {
            'max_drawdown': max_dd,
            'current_drawdown': current_dd,
            'max_drawdown_pct': max_dd * 100,
            'current_drawdown_pct': current_dd * 100
        }
    
    def get_daily_returns(self, days: int = 30) -> List[float]:
        """Get daily returns for period"""
        # This would query portfolio snapshots
        # For now, return empty list
        return []
    
    def get_win_rate_by_platform(self) -> Dict[str, float]:
        """Calculate win rate by platform"""
        platforms = ['alpaca', 'polymarket', 'kalshi']
        win_rates = {}
        
        for platform in platforms:
            trades = self.db.get_trade_history(limit=1000)
            platform_trades = [t for t in trades if t['platform'] == platform and t['status'] == 'closed']
            
            if platform_trades:
                winners = sum(1 for t in platform_trades if t.get('pnl', 0) > 0)
                win_rates[platform] = winners / len(platform_trades)
            else:
                win_rates[platform] = 0.0
        
        return win_rates
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        portfolio = self.get_portfolio_metrics()
        all_strategies = self.db.get_all_strategy_performance()
        
        # Calculate aggregate metrics
        total_trades = sum(s['total_trades'] for s in all_strategies)
        total_wins = sum(s['winning_trades'] for s in all_strategies)
        overall_win_rate = total_wins / total_trades if total_trades > 0 else 0.0
        
        return {
            'portfolio': portfolio,
            'overall_stats': {
                'total_trades': total_trades,
                'winning_trades': total_wins,
                'overall_win_rate': overall_win_rate,
                'total_pnl': portfolio['total_pnl']
            },
            'strategies': all_strategies,
            'win_rates_by_platform': self.get_win_rate_by_platform()
        }
    
    def print_summary(self):
        """Print formatted performance summary"""
        summary = self.get_performance_summary()
        
        print("\n" + "="*70)
        print("PORTFOLIO PERFORMANCE SUMMARY")
        print("="*70)
        
        portfolio = summary['portfolio']
        print(f"\nTotal Capital:      ${portfolio['total_capital']:>12,.2f}")
        print(f"Deployed Capital:   ${portfolio['deployed_capital']:>12,.2f}")
        print(f"Available Capital:  ${portfolio['available_capital']:>12,.2f}")
        print(f"Total P&L:          ${portfolio['total_pnl']:>12,.2f}")
        print(f"Daily P&L:          ${portfolio['daily_pnl']:>12,.2f}")
        print(f"Utilization:        {portfolio['utilization']:>12.1f}%")
        
        overall = summary['overall_stats']
        print(f"\n{'='*70}")
        print("OVERALL STATISTICS")
        print(f"{'='*70}")
        print(f"Total Trades:       {overall['total_trades']:>12}")
        print(f"Winning Trades:     {overall['winning_trades']:>12}")
        print(f"Win Rate:           {overall['overall_win_rate']:>12.1%}")
        
        if summary['strategies']:
            print(f"\n{'='*70}")
            print("STRATEGY PERFORMANCE")
            print(f"{'='*70}")
            print(f"{'Strategy':<20} {'Trades':<10} {'Win Rate':<12} {'P&L':<15}")
            print("-"*70)
            
            for strategy in summary['strategies']:
                print(f"{strategy['strategy']:<20} "
                      f"{strategy['total_trades']:<10} "
                      f"{strategy['win_rate']:>11.1%} "
                      f"${strategy['total_pnl']:>13,.2f}")
        
        print("="*70 + "\n")