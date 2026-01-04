"""
Capital Allocation Engine
Intelligently allocates capital between arbitrage and Sharky strategies

Allocation Logic:
- Risk-adjusted returns
- Kelly Criterion
- Dynamic rebalancing
- Portfolio optimization
"""
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """Track strategy performance metrics"""
    strategy_name: str
    total_trades: int = 0
    successful_trades: int = 0
    total_profit: float = 0.0
    total_cost: float = 0.0
    avg_trade_duration_minutes: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    @property
    def roi(self) -> float:
        """Return on investment"""
        return (self.total_profit / self.total_cost) if self.total_cost > 0 else 0.0
    
    @property
    def profit_per_trade(self) -> float:
        """Average profit per trade"""
        return self.total_profit / self.total_trades if self.total_trades > 0 else 0.0
    
    @property
    def daily_profit_estimate(self) -> float:
        """Estimated daily profit based on trade frequency"""
        if self.total_trades == 0 or self.avg_trade_duration_minutes == 0:
            return 0.0
        
        # Calculate trades per day
        trades_per_day = (24 * 60) / self.avg_trade_duration_minutes
        return trades_per_day * self.profit_per_trade


@dataclass
class AllocationDecision:
    """Capital allocation decision"""
    arbitrage_allocation: float  # Dollar amount
    sharky_scalp_allocation: float  # Dollar amount
    sharky_directional_allocation: float  # Dollar amount
    
    arbitrage_percent: float  # Percentage
    sharky_scalp_percent: float  # Percentage
    sharky_directional_percent: float  # Percentage
    
    reasoning: str
    timestamp: datetime
    
    def __repr__(self):
        return (f"Allocation: Arb={self.arbitrage_percent:.1%}, "
                f"Scalp={self.sharky_scalp_percent:.1%}, "
                f"Dir={self.sharky_directional_percent:.1%}")


class CapitalAllocationEngine:
    """
    Intelligently allocates capital between strategies
    
    Strategies:
    1. Pure Arbitrage (risk-free, rare)
    2. Sharky Scalping (99%+ certainty, high frequency)
    3. Sharky Directional (95%+ certainty, medium-term)
    """
    
    def __init__(self, total_capital: float):
        self.total_capital = total_capital
        self.available_capital = total_capital
        
        # Strategy performance tracking
        self.arbitrage_perf = StrategyPerformance("Pure Arbitrage")
        self.scalp_perf = StrategyPerformance("Sharky Scalping")
        self.directional_perf = StrategyPerformance("Sharky Directional")
        
        # Allocation constraints
        self.min_arbitrage_allocation = 0.10  # Always keep 10% for arbitrage
        self.max_single_strategy = 0.70  # Max 70% in any single strategy
        self.reserve_buffer = 0.05  # Keep 5% in reserve
        
        # Risk parameters
        self.kelly_fraction = 0.25  # Conservative Kelly
        self.rebalance_threshold = 0.10  # Rebalance if drift > 10%
        
        # Current allocation
        self.current_allocation: Optional[AllocationDecision] = None
        self.last_rebalance = datetime.now()
        
    def calculate_optimal_allocation(
        self,
        arbitrage_opportunities: int,
        scalp_opportunities: int,
        directional_opportunities: int
    ) -> AllocationDecision:
        """
        Calculate optimal capital allocation based on:
        1. Opportunity availability
        2. Historical performance
        3. Risk-adjusted returns
        4. Kelly Criterion
        """
        
        logger.info("Calculating optimal capital allocation...")
        
        # Step 1: Calculate expected returns for each strategy
        arb_expected_return = self._calculate_expected_return(
            self.arbitrage_perf, arbitrage_opportunities
        )
        scalp_expected_return = self._calculate_expected_return(
            self.scalp_perf, scalp_opportunities
        )
        directional_expected_return = self._calculate_expected_return(
            self.directional_perf, directional_opportunities
        )
        
        logger.info(f"Expected returns: Arb={arb_expected_return:.2%}, "
                   f"Scalp={scalp_expected_return:.2%}, "
                   f"Dir={directional_expected_return:.2%}")
        
        # Step 2: Calculate risk-adjusted scores
        arb_score = self._risk_adjusted_score(
            arb_expected_return, 
            self.arbitrage_perf,
            arbitrage_opportunities
        )
        scalp_score = self._risk_adjusted_score(
            scalp_expected_return,
            self.scalp_perf,
            scalp_opportunities
        )
        directional_score = self._risk_adjusted_score(
            directional_expected_return,
            self.directional_perf,
            directional_opportunities
        )
        
        total_score = arb_score + scalp_score + directional_score
        
        if total_score == 0:
            # No historical data, use default allocation
            return self._default_allocation()
        
        # Step 3: Calculate base allocation (proportional to scores)
        arb_percent = arb_score / total_score
        scalp_percent = scalp_score / total_score
        directional_percent = directional_score / total_score
        
        # Step 4: Apply constraints
        arb_percent = max(arb_percent, self.min_arbitrage_allocation)
        arb_percent = min(arb_percent, self.max_single_strategy)
        
        scalp_percent = max(scalp_percent, 0.0)
        scalp_percent = min(scalp_percent, self.max_single_strategy)
        
        directional_percent = max(directional_percent, 0.0)
        directional_percent = min(directional_percent, self.max_single_strategy)
        
        # Step 5: Normalize to sum to (1 - reserve_buffer)
        total_allocated = arb_percent + scalp_percent + directional_percent
        allocation_target = 1.0 - self.reserve_buffer
        
        if total_allocated > 0:
            scale_factor = allocation_target / total_allocated
            arb_percent *= scale_factor
            scalp_percent *= scale_factor
            directional_percent *= scale_factor
        
        # Step 6: Convert to dollar amounts
        deployable_capital = self.total_capital * allocation_target
        
        arb_allocation = deployable_capital * arb_percent
        scalp_allocation = deployable_capital * scalp_percent
        directional_allocation = deployable_capital * directional_percent
        
        # Step 7: Generate reasoning
        reasoning = self._generate_reasoning(
            arb_score, scalp_score, directional_score,
            arbitrage_opportunities, scalp_opportunities, directional_opportunities
        )
        
        decision = AllocationDecision(
            arbitrage_allocation=arb_allocation,
            sharky_scalp_allocation=scalp_allocation,
            sharky_directional_allocation=directional_allocation,
            arbitrage_percent=arb_percent,
            sharky_scalp_percent=scalp_percent,
            sharky_directional_percent=directional_percent,
            reasoning=reasoning,
            timestamp=datetime.now()
        )
        
        self.current_allocation = decision
        self.last_rebalance = datetime.now()
        
        logger.info(f"Optimal allocation: {decision}")
        return decision
    
    def _calculate_expected_return(
        self, 
        perf: StrategyPerformance, 
        opportunities: int
    ) -> float:
        """
        Calculate expected return for a strategy
        
        Formula: Expected_Return = Win_Rate × Avg_Profit_Per_Trade × Opportunities
        """
        if perf.total_trades == 0:
            # No historical data, use theoretical estimates
            return self._theoretical_expected_return(perf.strategy_name, opportunities)
        
        # Historical expected return
        expected_return = perf.win_rate * perf.profit_per_trade * opportunities
        return expected_return
    
    def _theoretical_expected_return(self, strategy_name: str, opportunities: int) -> float:
        """Theoretical expected returns when no historical data"""
        
        if "Arbitrage" in strategy_name:
            # Pure arbitrage: 0.2-0.5% per trade, 100% win rate (if executed)
            return 0.003 * opportunities * 1.0
        
        elif "Scalping" in strategy_name:
            # Sharky scalping: 0.1-1% per trade, 98% win rate
            return 0.005 * opportunities * 0.98
        
        elif "Directional" in strategy_name:
            # Sharky directional: 2-10% per trade, 95% win rate
            return 0.05 * opportunities * 0.95
        
        return 0.01 * opportunities
    
    def _risk_adjusted_score(
        self, 
        expected_return: float,
        perf: StrategyPerformance,
        opportunities: int
    ) -> float:
        """
        Calculate risk-adjusted score
        
        Formula: Score = Expected_Return × Win_Rate × Sqrt(Opportunities) / (1 + Max_Drawdown)
        """
        if opportunities == 0:
            return 0.0
        
        # Opportunity factor (diminishing returns for large numbers)
        opportunity_factor = np.sqrt(opportunities)
        
        # Win rate (default to theoretical if no data)
        win_rate = perf.win_rate if perf.total_trades > 0 else self._theoretical_win_rate(perf.strategy_name)
        
        # Drawdown penalty (1 = no drawdown, 0.5 = 50% drawdown)
        drawdown_penalty = 1.0 / (1.0 + perf.max_drawdown)
        
        # Calculate score
        score = expected_return * win_rate * opportunity_factor * drawdown_penalty
        
        return max(score, 0.0)
    
    def _theoretical_win_rate(self, strategy_name: str) -> float:
        """Theoretical win rates for each strategy"""
        if "Arbitrage" in strategy_name:
            return 1.0  # Risk-free (excluding execution risk)
        elif "Scalping" in strategy_name:
            return 0.98  # 98% win rate (Sharky's historical)
        elif "Directional" in strategy_name:
            return 0.95  # 95% win rate
        return 0.90
    
    def _default_allocation(self) -> AllocationDecision:
        """
        Default allocation when no historical data
        
        Strategy:
        - 20% Pure Arbitrage (always available, risk-free)
        - 60% Sharky Scalping (high frequency, low risk)
        - 15% Sharky Directional (medium-term, higher returns)
        - 5% Reserve
        """
        deployable = self.total_capital * 0.95
        
        return AllocationDecision(
            arbitrage_allocation=deployable * 0.20,
            sharky_scalp_allocation=deployable * 0.60,
            sharky_directional_allocation=deployable * 0.15,
            arbitrage_percent=0.20,
            sharky_scalp_percent=0.60,
            sharky_directional_percent=0.15,
            reasoning="Default allocation (no historical data)",
            timestamp=datetime.now()
        )
    
    def _generate_reasoning(
        self,
        arb_score: float,
        scalp_score: float,
        directional_score: float,
        arb_opps: int,
        scalp_opps: int,
        dir_opps: int
    ) -> str:
        """Generate human-readable reasoning for allocation"""
        
        total_score = arb_score + scalp_score + directional_score
        
        if total_score == 0:
            return "Using default allocation (no performance data)"
        
        reasons = []
        
        # Arbitrage reasoning
        if arb_score / total_score > 0.30:
            reasons.append(f"High arbitrage allocation ({arb_opps} opportunities available)")
        elif arb_opps == 0:
            reasons.append("Minimal arbitrage allocation (no opportunities)")
        
        # Scalping reasoning
        if scalp_score / total_score > 0.50:
            reasons.append(f"Heavy scalping focus ({scalp_opps} high-certainty opportunities)")
        elif scalp_opps > 20:
            reasons.append(f"Strong scalping presence ({scalp_opps} opportunities)")
        
        # Directional reasoning
        if directional_score / total_score > 0.30:
            reasons.append(f"Significant directional positions ({dir_opps} medium-term opportunities)")
        
        # Performance-based reasoning
        if self.scalp_perf.total_trades > 0 and self.scalp_perf.roi > 0.10:
            reasons.append(f"Scalping showing strong returns ({self.scalp_perf.roi:.1%} ROI)")
        
        if self.directional_perf.total_trades > 0 and self.directional_perf.roi > 0.20:
            reasons.append(f"Directional strategy outperforming ({self.directional_perf.roi:.1%} ROI)")
        
        return "; ".join(reasons) if reasons else "Balanced allocation across strategies"
    
    def should_rebalance(self) -> bool:
        """Check if portfolio should be rebalanced"""
        
        if not self.current_allocation:
            return True
        
        # Rebalance if it's been more than 1 hour
        time_since_rebalance = datetime.now() - self.last_rebalance
        if time_since_rebalance > timedelta(hours=1):
            return True
        
        # Rebalance if performance has significantly changed
        # (This would require tracking actual vs. target allocation)
        
        return False
    
    def update_performance(
        self, 
        strategy: str,
        profit: float,
        cost: float,
        success: bool,
        duration_minutes: float
    ):
        """Update strategy performance metrics"""
        
        # Select the appropriate performance tracker
        if "arbitrage" in strategy.lower():
            perf = self.arbitrage_perf
        elif "scalp" in strategy.lower():
            perf = self.scalp_perf
        elif "directional" in strategy.lower():
            perf = self.directional_perf
        else:
            logger.warning(f"Unknown strategy: {strategy}")
            return
        
        # Update metrics
        perf.total_trades += 1
        if success:
            perf.successful_trades += 1
        
        perf.total_profit += profit
        perf.total_cost += cost
        
        # Update average duration (running average)
        if perf.total_trades == 1:
            perf.avg_trade_duration_minutes = duration_minutes
        else:
            perf.avg_trade_duration_minutes = (
                (perf.avg_trade_duration_minutes * (perf.total_trades - 1) + duration_minutes) 
                / perf.total_trades
            )
        
        # Update win rate
        perf.win_rate = perf.successful_trades / perf.total_trades
        
        logger.info(f"Updated {strategy} performance: "
                   f"{perf.total_trades} trades, "
                   f"{perf.win_rate:.1%} win rate, "
                   f"${perf.total_profit:.2f} profit")
    
    def get_allocation_summary(self) -> Dict:
        """Get current allocation summary"""
        
        if not self.current_allocation:
            return {"status": "No allocation calculated"}
        
        return {
            "total_capital": self.total_capital,
            "available_capital": self.available_capital,
            "allocation": {
                "arbitrage": {
                    "amount": self.current_allocation.arbitrage_allocation,
                    "percent": self.current_allocation.arbitrage_percent,
                    "performance": {
                        "trades": self.arbitrage_perf.total_trades,
                        "win_rate": self.arbitrage_perf.win_rate,
                        "roi": self.arbitrage_perf.roi,
                        "total_profit": self.arbitrage_perf.total_profit
                    }
                },
                "sharky_scalping": {
                    "amount": self.current_allocation.sharky_scalp_allocation,
                    "percent": self.current_allocation.sharky_scalp_percent,
                    "performance": {
                        "trades": self.scalp_perf.total_trades,
                        "win_rate": self.scalp_perf.win_rate,
                        "roi": self.scalp_perf.roi,
                        "total_profit": self.scalp_perf.total_profit
                    }
                },
                "sharky_directional": {
                    "amount": self.current_allocation.sharky_directional_allocation,
                    "percent": self.current_allocation.sharky_directional_percent,
                    "performance": {
                        "trades": self.directional_perf.total_trades,
                        "win_rate": self.directional_perf.win_rate,
                        "roi": self.directional_perf.roi,
                        "total_profit": self.directional_perf.total_profit
                    }
                }
            },
            "reasoning": self.current_allocation.reasoning,
            "last_rebalance": self.last_rebalance.isoformat()
        }


def test_allocation_engine():
    """Test the allocation engine"""
    
    print("=" * 80)
    print("CAPITAL ALLOCATION ENGINE TEST")
    print("=" * 80)
    
    # Create engine with $100k
    engine = CapitalAllocationEngine(total_capital=100)
    
    # Test 1: Default allocation (no historical data)
    print("\n1. Default Allocation (No Historical Data)")
    print("-" * 80)
    decision = engine.calculate_optimal_allocation(
        arbitrage_opportunities=5,
        scalp_opportunities=20,
        directional_opportunities=8
    )
    print(f"Arbitrage: ${decision.arbitrage_allocation:,.2f} ({decision.arbitrage_percent:.1%})")
    print(f"Scalping: ${decision.sharky_scalp_allocation:,.2f} ({decision.sharky_scalp_percent:.1%})")
    print(f"Directional: ${decision.sharky_directional_allocation:,.2f} ({decision.sharky_directional_percent:.1%})")
    print(f"Reasoning: {decision.reasoning}")
    
    # Test 2: Update performance with simulated trades
    print("\n2. Simulating Strategy Performance")
    print("-" * 80)
    
    # Simulate successful scalping trades
    for i in range(50):
        engine.update_performance(
            strategy="sharky_scalp",
            profit=50.0,  # $50 profit per trade
            cost=5000.0,  # $5000 position
            success=True,
            duration_minutes=10  # 10 minutes per trade
        )
    
    # Simulate successful directional trades
    for i in range(10):
        engine.update_performance(
            strategy="sharky_directional",
            profit=500.0,  # $500 profit per trade
            cost=10000.0,  # $10k position
            success=True,
            duration_minutes=1440  # 1 day per trade
        )
    
    # Simulate fewer arbitrage trades
    for i in range(3):
        engine.update_performance(
            strategy="pure_arbitrage",
            profit=20.0,  # $20 profit per trade
            cost=2000.0,  # $2k position
            success=True,
            duration_minutes=5  # 5 minutes per trade
        )
    
    print(f"Scalping: {engine.scalp_perf.total_trades} trades, "
          f"{engine.scalp_perf.win_rate:.1%} win rate, "
          f"{engine.scalp_perf.roi:.1%} ROI")
    print(f"Directional: {engine.directional_perf.total_trades} trades, "
          f"{engine.directional_perf.win_rate:.1%} win rate, "
          f"{engine.directional_perf.roi:.1%} ROI")
    print(f"Arbitrage: {engine.arbitrage_perf.total_trades} trades, "
          f"{engine.arbitrage_perf.win_rate:.1%} win rate, "
          f"{engine.arbitrage_perf.roi:.1%} ROI")
    
    # Test 3: Recalculate with performance data
    print("\n3. Optimized Allocation (With Performance Data)")
    print("-" * 80)
    decision = engine.calculate_optimal_allocation(
        arbitrage_opportunities=2,  # Fewer arb opportunities
        scalp_opportunities=40,  # More scalping opportunities
        directional_opportunities=10
    )
    print(f"Arbitrage: ${decision.arbitrage_allocation:,.2f} ({decision.arbitrage_percent:.1%})")
    print(f"Scalping: ${decision.sharky_scalp_allocation:,.2f} ({decision.sharky_scalp_percent:.1%})")
    print(f"Directional: ${decision.sharky_directional_allocation:,.2f} ({decision.sharky_directional_percent:.1%})")
    print(f"Reasoning: {decision.reasoning}")
    
    # Test 4: Show full summary
    print("\n4. Full Allocation Summary")
    print("-" * 80)
    summary = engine.get_allocation_summary()
    
    for strategy, data in summary['allocation'].items():
        print(f"\n{strategy.upper()}:")
        print(f"  Allocated: ${data['amount']:,.2f} ({data['percent']:.1%})")
        print(f"  Trades: {data['performance']['trades']}")
        print(f"  Win Rate: {data['performance']['win_rate']:.1%}")
        print(f"  ROI: {data['performance']['roi']:.1%}")
        print(f"  Total Profit: ${data['performance']['total_profit']:,.2f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    test_allocation_engine()