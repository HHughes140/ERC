"""
Portfolio Risk Manager for ERC Trading System

Provides:
- Cross-model position limits
- Correlation detection
- Portfolio-level VaR
- Drawdown monitoring
- Dynamic position sizing

CRITICAL: Individual model risks compound!
Correlated positions across models = concentrated risk.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Position representation"""
    position_id: str
    model: str           # 'arbitrage', 'sharky', 'weather', 'ml_engine'
    platform: str
    market_id: str
    side: str           # 'long', 'short', 'yes', 'no'
    entry_price: float
    quantity: float
    current_price: float
    entry_time: datetime
    metadata: Dict = field(default_factory=dict)

    @property
    def notional_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl(self) -> float:
        if self.side in ['long', 'yes']:
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def pnl_pct(self) -> float:
        cost = self.entry_price * self.quantity
        return self.pnl / cost if cost > 0 else 0


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    # Portfolio limits
    max_portfolio_exposure: float = 0.80      # Max 80% of capital deployed
    max_single_position_pct: float = 0.10     # Max 10% in single position
    max_model_exposure_pct: float = 0.30      # Max 30% per model
    max_platform_exposure_pct: float = 0.50   # Max 50% per platform
    max_correlated_exposure_pct: float = 0.20 # Max 20% in correlated positions

    # Risk limits
    max_portfolio_var: float = 0.05           # 5% daily VaR limit
    max_drawdown: float = 0.15                # 15% max drawdown
    stop_trading_drawdown: float = 0.20       # Stop at 20% drawdown

    # Position limits
    max_open_positions: int = 20
    max_positions_per_model: int = 10
    max_positions_per_market: int = 2         # Avoid concentration

    # Correlation
    correlation_threshold: float = 0.7        # Consider correlated above this


@dataclass
class RiskMetrics:
    """Current portfolio risk metrics"""
    total_exposure: float
    exposure_pct: float
    model_exposures: Dict[str, float]
    platform_exposures: Dict[str, float]
    portfolio_var_95: float
    current_drawdown: float
    max_drawdown: float
    sharpe_ratio: float
    num_positions: int
    correlated_groups: List[List[str]]
    risk_score: float  # 0-1, higher = more risk


@dataclass
class RiskDecision:
    """Decision from risk manager"""
    allowed: bool
    reason: str
    max_size: Optional[float] = None
    adjusted_size: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


class PortfolioRiskManager:
    """
    Central risk manager for all trading models.

    Enforces position limits, monitors correlations,
    and provides portfolio-level risk metrics.
    """

    def __init__(self, total_capital: float = 10000.0,
                 limits: Optional[RiskLimits] = None,
                 max_portfolio_exposure: float = 0.80,
                 max_single_position_pct: float = 0.10):
        """
        Args:
            total_capital: Total trading capital (default $10,000)
            limits: Risk limit configuration
            max_portfolio_exposure: Max portfolio exposure (default 80%)
            max_single_position_pct: Max single position % (default 10%)
        """
        self.total_capital = total_capital
        self.max_portfolio_exposure = max_portfolio_exposure
        self.max_single_position_pct = max_single_position_pct
        self.limits = limits or RiskLimits()

        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []

        # Performance tracking
        self.equity_history: List[Tuple[datetime, float]] = []
        self.high_water_mark: float = total_capital
        self.daily_pnl: float = 0
        self.last_reset: datetime = datetime.now()

        # Correlation matrix (market_id -> market_id -> correlation)
        self._correlations: Dict[str, Dict[str, float]] = defaultdict(dict)

    # ========== Position Management ==========

    def add_position(self, position: Position) -> RiskDecision:
        """
        Add position if within risk limits.

        Args:
            position: Position to add

        Returns:
            RiskDecision with approval status
        """
        # Check all limits
        decision = self.check_new_position(position)

        if decision.allowed:
            self.positions[position.position_id] = position
            logger.info(
                f"Position added: {position.position_id} "
                f"({position.model}/{position.market_id})"
            )

        return decision

    def update_position(self, position_id: str, current_price: float) -> None:
        """Update position with current price"""
        if position_id in self.positions:
            self.positions[position_id].current_price = current_price

    def close_position(self, position_id: str, exit_price: float) -> Optional[float]:
        """
        Close position and return realized PnL.

        Args:
            position_id: Position to close
            exit_price: Exit price

        Returns:
            Realized PnL or None if position not found
        """
        if position_id not in self.positions:
            return None

        position = self.positions[position_id]
        position.current_price = exit_price

        pnl = position.pnl
        self.daily_pnl += pnl
        self.closed_positions.append(position)
        del self.positions[position_id]

        logger.info(
            f"Position closed: {position_id} "
            f"PnL: ${pnl:.2f} ({position.pnl_pct:.1%})"
        )

        return pnl

    # ========== Risk Checks ==========

    def check_new_position(self, position: Position) -> RiskDecision:
        """
        Check if new position is within all risk limits.

        Args:
            position: Proposed position

        Returns:
            RiskDecision with detailed reasoning
        """
        warnings = []
        position_value = position.notional_value

        # 1. Check number of positions
        if len(self.positions) >= self.limits.max_open_positions:
            return RiskDecision(
                allowed=False,
                reason=f"Max positions ({self.limits.max_open_positions}) reached"
            )

        # 2. Check positions per model
        model_positions = sum(
            1 for p in self.positions.values() if p.model == position.model
        )
        if model_positions >= self.limits.max_positions_per_model:
            return RiskDecision(
                allowed=False,
                reason=f"Max positions per model ({self.limits.max_positions_per_model}) reached for {position.model}"
            )

        # 3. Check positions in same market
        market_positions = sum(
            1 for p in self.positions.values() if p.market_id == position.market_id
        )
        if market_positions >= self.limits.max_positions_per_market:
            return RiskDecision(
                allowed=False,
                reason=f"Max positions in market {position.market_id} reached"
            )

        # 4. Check total exposure
        current_exposure = self._calculate_total_exposure()
        new_exposure = current_exposure + position_value
        exposure_pct = new_exposure / self.total_capital

        if exposure_pct > self.limits.max_portfolio_exposure:
            max_allowed = (self.limits.max_portfolio_exposure * self.total_capital - current_exposure)
            return RiskDecision(
                allowed=False,
                reason=f"Would exceed max portfolio exposure ({self.limits.max_portfolio_exposure:.0%})",
                max_size=max(0, max_allowed / position.current_price)
            )

        # 5. Check single position size
        position_pct = position_value / self.total_capital
        if position_pct > self.limits.max_single_position_pct:
            max_size = self.limits.max_single_position_pct * self.total_capital / position.current_price
            warnings.append(f"Position size reduced from {position_pct:.1%} to {self.limits.max_single_position_pct:.0%}")
            return RiskDecision(
                allowed=True,
                reason="Position size adjusted to meet limit",
                max_size=max_size,
                adjusted_size=max_size,
                warnings=warnings
            )

        # 6. Check model exposure
        model_exposure = sum(
            p.notional_value for p in self.positions.values() if p.model == position.model
        ) + position_value
        model_exposure_pct = model_exposure / self.total_capital

        if model_exposure_pct > self.limits.max_model_exposure_pct:
            return RiskDecision(
                allowed=False,
                reason=f"Would exceed max exposure for model {position.model} ({self.limits.max_model_exposure_pct:.0%})"
            )

        # 7. Check platform exposure
        platform_exposure = sum(
            p.notional_value for p in self.positions.values() if p.platform == position.platform
        ) + position_value
        platform_exposure_pct = platform_exposure / self.total_capital

        if platform_exposure_pct > self.limits.max_platform_exposure_pct:
            return RiskDecision(
                allowed=False,
                reason=f"Would exceed max exposure for platform {position.platform} ({self.limits.max_platform_exposure_pct:.0%})"
            )

        # 8. Check drawdown
        current_dd = self._calculate_drawdown()
        if current_dd >= self.limits.stop_trading_drawdown:
            return RiskDecision(
                allowed=False,
                reason=f"Trading halted: drawdown {current_dd:.1%} exceeds limit {self.limits.stop_trading_drawdown:.0%}"
            )

        if current_dd >= self.limits.max_drawdown:
            warnings.append(f"Near max drawdown: {current_dd:.1%}")

        # 9. Check correlation with existing positions
        correlated_exposure = self._calculate_correlated_exposure(position)
        if correlated_exposure > self.limits.max_correlated_exposure_pct * self.total_capital:
            warnings.append(f"High correlation with existing positions")

        # All checks passed
        return RiskDecision(
            allowed=True,
            reason="Within all risk limits",
            warnings=warnings
        )

    def get_max_position_size(self, model: str, platform: str,
                              price: float, market_id: str) -> float:
        """
        Get maximum allowed position size for new trade.

        Args:
            model: Trading model name
            platform: Trading platform
            price: Current price
            market_id: Market identifier

        Returns:
            Maximum quantity allowed
        """
        # Start with single position limit
        max_value = self.limits.max_single_position_pct * self.total_capital

        # Check portfolio exposure limit
        current_exposure = self._calculate_total_exposure()
        available_exposure = self.limits.max_portfolio_exposure * self.total_capital - current_exposure
        max_value = min(max_value, available_exposure)

        # Check model exposure limit
        model_exposure = sum(
            p.notional_value for p in self.positions.values() if p.model == model
        )
        available_model = self.limits.max_model_exposure_pct * self.total_capital - model_exposure
        max_value = min(max_value, available_model)

        # Check platform exposure limit
        platform_exposure = sum(
            p.notional_value for p in self.positions.values() if p.platform == platform
        )
        available_platform = self.limits.max_platform_exposure_pct * self.total_capital - platform_exposure
        max_value = min(max_value, available_platform)

        # Apply drawdown adjustment
        drawdown = self._calculate_drawdown()
        if drawdown > 0.05:  # Reduce size if in drawdown
            drawdown_multiplier = 1 - (drawdown / self.limits.max_drawdown)
            max_value *= max(0.3, drawdown_multiplier)

        return max(0, max_value / price)

    # ========== Risk Metrics ==========

    def get_risk_metrics(self) -> RiskMetrics:
        """Calculate current portfolio risk metrics"""
        total_exposure = self._calculate_total_exposure()

        # Model exposures
        model_exposures = defaultdict(float)
        for p in self.positions.values():
            model_exposures[p.model] += p.notional_value

        # Platform exposures
        platform_exposures = defaultdict(float)
        for p in self.positions.values():
            platform_exposures[p.platform] += p.notional_value

        # Drawdown
        current_dd = self._calculate_drawdown()
        max_dd = self._calculate_max_drawdown()

        # VaR estimate (simplified)
        var_95 = self._estimate_var()

        # Sharpe (simplified)
        sharpe = self._calculate_sharpe()

        # Correlated groups
        correlated = self._find_correlated_groups()

        # Risk score (0-1)
        risk_score = self._calculate_risk_score(
            total_exposure / self.total_capital,
            current_dd, var_95, len(correlated)
        )

        return RiskMetrics(
            total_exposure=total_exposure,
            exposure_pct=total_exposure / self.total_capital,
            model_exposures=dict(model_exposures),
            platform_exposures=dict(platform_exposures),
            portfolio_var_95=var_95,
            current_drawdown=current_dd,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            num_positions=len(self.positions),
            correlated_groups=correlated,
            risk_score=risk_score
        )

    def _calculate_total_exposure(self) -> float:
        """Calculate total portfolio exposure"""
        return sum(p.notional_value for p in self.positions.values())

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from high water mark"""
        current_equity = self.total_capital + sum(p.pnl for p in self.positions.values())

        # Update high water mark
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity

        if self.high_water_mark == 0:
            return 0

        return (self.high_water_mark - current_equity) / self.high_water_mark

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity history"""
        if len(self.equity_history) < 2:
            return self._calculate_drawdown()

        equities = [e[1] for e in self.equity_history]
        peak = equities[0]
        max_dd = 0

        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    def _estimate_var(self, confidence: float = 0.95) -> float:
        """Estimate portfolio VaR"""
        if not self.positions:
            return 0

        # Simplified VaR using position volatility assumption
        # In production, use historical returns
        total_exposure = self._calculate_total_exposure()
        assumed_daily_vol = 0.02  # 2% daily volatility assumption

        # Normal distribution quantile
        z_score = 1.645 if confidence == 0.95 else 2.326

        var = total_exposure * assumed_daily_vol * z_score

        return var / self.total_capital

    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio from recent returns"""
        if len(self.equity_history) < 10:
            return 0

        equities = [e[1] for e in self.equity_history[-252:]]  # Last year
        returns = np.diff(equities) / equities[:-1]

        if len(returns) == 0 or np.std(returns) == 0:
            return 0

        return np.mean(returns) / np.std(returns) * np.sqrt(252)

    def _calculate_correlated_exposure(self, new_position: Position) -> float:
        """Calculate exposure to positions correlated with new position"""
        correlated_exposure = new_position.notional_value

        for pos in self.positions.values():
            correlation = self._get_correlation(new_position.market_id, pos.market_id)
            if abs(correlation) > self.limits.correlation_threshold:
                correlated_exposure += pos.notional_value * abs(correlation)

        return correlated_exposure

    def _find_correlated_groups(self) -> List[List[str]]:
        """Find groups of correlated positions"""
        if len(self.positions) < 2:
            return []

        position_list = list(self.positions.values())
        visited = set()
        groups = []

        for i, pos in enumerate(position_list):
            if pos.position_id in visited:
                continue

            group = [pos.position_id]
            visited.add(pos.position_id)

            for other in position_list[i+1:]:
                if other.position_id in visited:
                    continue

                correlation = self._get_correlation(pos.market_id, other.market_id)
                if abs(correlation) > self.limits.correlation_threshold:
                    group.append(other.position_id)
                    visited.add(other.position_id)

            if len(group) > 1:
                groups.append(group)

        return groups

    def _get_correlation(self, market_id_1: str, market_id_2: str) -> float:
        """Get or estimate correlation between markets"""
        if market_id_1 == market_id_2:
            return 1.0

        # Check cached correlation
        if market_id_1 in self._correlations:
            if market_id_2 in self._correlations[market_id_1]:
                return self._correlations[market_id_1][market_id_2]

        # Simple heuristic: same platform/category = higher correlation
        # In production, compute from historical data
        return 0.3  # Default assumption

    def set_correlation(self, market_id_1: str, market_id_2: str, correlation: float):
        """Set correlation between markets"""
        self._correlations[market_id_1][market_id_2] = correlation
        self._correlations[market_id_2][market_id_1] = correlation

    def _calculate_risk_score(self, exposure_pct: float, drawdown: float,
                              var: float, num_correlated_groups: int) -> float:
        """Calculate overall risk score (0-1)"""
        # Weighted factors
        exposure_score = min(exposure_pct / self.limits.max_portfolio_exposure, 1.0) * 0.3
        drawdown_score = min(drawdown / self.limits.max_drawdown, 1.0) * 0.3
        var_score = min(var / self.limits.max_portfolio_var, 1.0) * 0.25
        correlation_score = min(num_correlated_groups / 5, 1.0) * 0.15

        return exposure_score + drawdown_score + var_score + correlation_score

    # ========== Equity Tracking ==========

    def record_equity(self):
        """Record current equity for tracking"""
        current_equity = self.total_capital + sum(p.pnl for p in self.positions.values())
        self.equity_history.append((datetime.now(), current_equity))

        # Keep last 252 trading days
        if len(self.equity_history) > 252 * 8:  # 8 snapshots per day
            self.equity_history = self.equity_history[-252*8:]

    def reset_daily_pnl(self):
        """Reset daily PnL tracking"""
        self.daily_pnl = 0
        self.last_reset = datetime.now()


def test_portfolio_risk():
    """Test portfolio risk manager"""
    print("=== Portfolio Risk Manager Test ===\n")

    # Create manager with $100k capital
    manager = PortfolioRiskManager(total_capital=100000)

    # Test adding positions
    positions = [
        Position(
            position_id="pos1", model="arbitrage", platform="polymarket",
            market_id="mkt1", side="yes", entry_price=0.50,
            quantity=100, current_price=0.52, entry_time=datetime.now()
        ),
        Position(
            position_id="pos2", model="sharky", platform="polymarket",
            market_id="mkt2", side="yes", entry_price=0.95,
            quantity=50, current_price=0.96, entry_time=datetime.now()
        ),
        Position(
            position_id="pos3", model="arbitrage", platform="kalshi",
            market_id="mkt3", side="no", entry_price=0.45,
            quantity=200, current_price=0.44, entry_time=datetime.now()
        ),
    ]

    print("Adding positions:")
    for pos in positions:
        decision = manager.add_position(pos)
        print(f"  {pos.position_id}: {decision.reason}")
        if decision.warnings:
            print(f"    Warnings: {decision.warnings}")

    # Get risk metrics
    metrics = manager.get_risk_metrics()
    print(f"\nRisk Metrics:")
    print(f"  Total Exposure: ${metrics.total_exposure:.2f} ({metrics.exposure_pct:.1%})")
    print(f"  Positions: {metrics.num_positions}")
    print(f"  Model Exposures: {metrics.model_exposures}")
    print(f"  VaR (95%): {metrics.portfolio_var_95:.2%}")
    print(f"  Current Drawdown: {metrics.current_drawdown:.2%}")
    print(f"  Risk Score: {metrics.risk_score:.2f}")

    # Test max position size
    max_size = manager.get_max_position_size(
        model="sharky", platform="polymarket", price=0.90, market_id="mkt4"
    )
    print(f"\nMax position size for new Sharky trade: {max_size:.0f} shares")

    # Test oversized position rejection
    big_position = Position(
        position_id="pos_big", model="ml_engine", platform="alpaca",
        market_id="SPY", side="long", entry_price=450.0,
        quantity=100, current_price=450.0, entry_time=datetime.now()
    )
    decision = manager.check_new_position(big_position)
    print(f"\nBig position check: {decision.allowed} - {decision.reason}")


if __name__ == "__main__":
    test_portfolio_risk()
