"""
Risk Factors Module
Calculates and monitors risk metrics for the trading system
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import math

from config import ERCConfig

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Risk metrics for a portfolio or strategy"""
    var_95: float = 0.0          # Value at Risk (95% confidence)
    var_99: float = 0.0          # Value at Risk (99% confidence)
    max_drawdown: float = 0.0    # Maximum drawdown
    current_drawdown: float = 0.0
    volatility: float = 0.0      # Portfolio volatility
    sharpe_ratio: float = 0.0    # Risk-adjusted return
    sortino_ratio: float = 0.0   # Downside risk-adjusted return
    beta: float = 1.0            # Market sensitivity

    def to_dict(self) -> Dict:
        return {
            'var_95': self.var_95,
            'var_99': self.var_99,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'beta': self.beta
        }


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_daily_loss: float = 50.0
    max_drawdown: float = 100.0
    max_position_size: float = 100.0
    max_positions: int = 10
    max_concentration: float = 0.25  # Max % in single position
    max_leverage: float = 1.0

    def to_dict(self) -> Dict:
        return {
            'max_daily_loss': self.max_daily_loss,
            'max_drawdown': self.max_drawdown,
            'max_position_size': self.max_position_size,
            'max_positions': self.max_positions,
            'max_concentration': self.max_concentration,
            'max_leverage': self.max_leverage
        }


class RiskAnalyzer:
    """
    Risk analysis and monitoring

    Features:
    - VaR (Value at Risk) calculation
    - Drawdown tracking
    - Position sizing recommendations
    - Risk limit monitoring
    - Trade approval based on risk
    """

    def __init__(self, config: ERCConfig):
        self.config = config

        # Risk limits from config
        self.limits = RiskLimits(
            max_daily_loss=config.MAX_DAILY_LOSS,
            max_drawdown=config.MAX_DRAWDOWN,
            max_position_size=config.MAX_POSITION_SIZE,
            max_positions=config.MAX_CONCURRENT_POSITIONS
        )

        # Current state
        self.metrics = RiskMetrics()

        # Historical data for calculations
        self.return_history: List[float] = []
        self.equity_history: List[float] = []
        self.peak_equity: float = config.TOTAL_CAPITAL

        logger.info("Risk Analyzer initialized")

    def update_metrics(self, current_equity: float, daily_return: float = 0.0):
        """Update risk metrics with new data"""
        # Track equity
        self.equity_history.append(current_equity)
        if daily_return != 0:
            self.return_history.append(daily_return)

        # Keep limited history
        if len(self.return_history) > 252:  # ~1 year of trading days
            self.return_history = self.return_history[-252:]
        if len(self.equity_history) > 252:
            self.equity_history = self.equity_history[-252:]

        # Update peak and drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        self.metrics.current_drawdown = self.peak_equity - current_equity
        self.metrics.max_drawdown = max(
            self.metrics.max_drawdown,
            self.metrics.current_drawdown
        )

        # Calculate volatility if we have enough data
        if len(self.return_history) >= 20:
            self.metrics.volatility = self._calculate_volatility()
            self.metrics.var_95 = self._calculate_var(0.95)
            self.metrics.var_99 = self._calculate_var(0.99)
            self.metrics.sharpe_ratio = self._calculate_sharpe()
            self.metrics.sortino_ratio = self._calculate_sortino()

    def _calculate_volatility(self) -> float:
        """Calculate annualized volatility"""
        if len(self.return_history) < 2:
            return 0.0

        returns = self.return_history[-20:]  # Use last 20 periods
        mean_return = sum(returns) / len(returns)

        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = math.sqrt(variance)

        # Annualize (assuming daily data)
        return std_dev * math.sqrt(252)

    def _calculate_var(self, confidence: float) -> float:
        """Calculate Value at Risk"""
        if len(self.return_history) < 20:
            return 0.0

        returns = sorted(self.return_history)
        index = int((1 - confidence) * len(returns))

        return abs(returns[max(0, index)]) * self.peak_equity

    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.return_history) < 20 or self.metrics.volatility == 0:
            return 0.0

        mean_return = sum(self.return_history) / len(self.return_history)
        annual_return = mean_return * 252
        risk_free = self.config.RISK_FREE_RATE

        return (annual_return - risk_free) / self.metrics.volatility

    def _calculate_sortino(self) -> float:
        """Calculate Sortino ratio (downside risk only)"""
        if len(self.return_history) < 20:
            return 0.0

        mean_return = sum(self.return_history) / len(self.return_history)
        negative_returns = [r for r in self.return_history if r < 0]

        if not negative_returns:
            return float('inf')

        downside_variance = sum(r ** 2 for r in negative_returns) / len(negative_returns)
        downside_std = math.sqrt(downside_variance) * math.sqrt(252)

        if downside_std == 0:
            return float('inf')

        annual_return = mean_return * 252
        risk_free = self.config.RISK_FREE_RATE

        return (annual_return - risk_free) / downside_std

    def check_limits(self, portfolio_state: Dict) -> Dict:
        """
        Check if portfolio state violates any risk limits

        Returns:
            Dict with 'breach' bool and details
        """
        breaches = []
        severity = 'normal'

        # Check daily loss
        daily_pnl = portfolio_state.get('daily_pnl', 0)
        if daily_pnl < -self.limits.max_daily_loss:
            breaches.append({
                'type': 'daily_loss',
                'current': abs(daily_pnl),
                'limit': self.limits.max_daily_loss,
                'severity': 'critical'
            })
            severity = 'critical'
        elif daily_pnl < -self.limits.max_daily_loss * 0.8:
            breaches.append({
                'type': 'daily_loss_warning',
                'current': abs(daily_pnl),
                'limit': self.limits.max_daily_loss,
                'severity': 'warning'
            })
            if severity == 'normal':
                severity = 'warning'

        # Check drawdown
        max_dd = portfolio_state.get('max_drawdown', self.metrics.max_drawdown)
        if max_dd >= self.limits.max_drawdown:
            breaches.append({
                'type': 'max_drawdown',
                'current': max_dd,
                'limit': self.limits.max_drawdown,
                'severity': 'critical'
            })
            severity = 'critical'
        elif max_dd >= self.limits.max_drawdown * 0.8:
            breaches.append({
                'type': 'drawdown_warning',
                'current': max_dd,
                'limit': self.limits.max_drawdown,
                'severity': 'warning'
            })
            if severity == 'normal':
                severity = 'warning'

        # Check position count
        open_positions = portfolio_state.get('open_positions', 0)
        if open_positions >= self.limits.max_positions:
            breaches.append({
                'type': 'position_limit',
                'current': open_positions,
                'limit': self.limits.max_positions,
                'severity': 'warning'
            })
            if severity == 'normal':
                severity = 'warning'

        return {
            'breach': len(breaches) > 0,
            'breaches': breaches,
            'severity': severity,
            'reason': breaches[0]['type'] if breaches else None
        }

    def approve_trade(self, opportunity: Any, capital: float = None) -> bool:
        """
        Approve or reject a trade based on risk parameters

        Args:
            opportunity: Trading opportunity object
            capital: Proposed capital to deploy

        Returns:
            True if trade is approved
        """
        # Check position size limit
        if capital and capital > self.limits.max_position_size:
            logger.debug(f"Trade rejected: Size ${capital:.2f} exceeds limit ${self.limits.max_position_size:.2f}")
            return False

        # Check current drawdown
        if self.metrics.current_drawdown >= self.limits.max_drawdown * 0.9:
            logger.warning("Trade rejected: Near max drawdown")
            return False

        # Check risk score if available
        risk_score = getattr(opportunity, 'risk_score', 0)
        if risk_score > 0.8:  # High risk
            logger.debug(f"Trade rejected: High risk score {risk_score:.2f}")
            return False

        return True

    def calculate_position_size(self, opportunity: Any,
                               available_capital: float,
                               strategy: str = None) -> float:
        """
        Calculate recommended position size based on risk parameters

        Uses Kelly Criterion with modifications for prediction markets
        """
        # Get opportunity metrics
        win_prob = getattr(opportunity, 'certainty', 0.5)
        profit_potential = getattr(opportunity, 'profit_potential', 0.1)
        risk_score = getattr(opportunity, 'risk_score', 0.5)

        # Calculate edge
        edge = win_prob * profit_potential - (1 - win_prob)

        if edge <= 0:
            return 0.0

        # Kelly fraction (with half-Kelly for safety)
        if profit_potential > 0:
            kelly = edge / profit_potential
            kelly = kelly * 0.5  # Half-Kelly
        else:
            kelly = 0.1

        # Apply risk adjustments
        risk_multiplier = 1.0 - risk_score
        kelly *= risk_multiplier

        # Apply strategy-specific limits
        strategy_limits = {
            'arbitrage': 0.25,
            'scalping': 0.15,
            'directional': 0.10,
            'sharky': 0.15,
        }
        max_fraction = strategy_limits.get(strategy, 0.10)
        kelly = min(kelly, max_fraction)

        # Calculate position size
        position_size = available_capital * kelly

        # Apply absolute limits
        position_size = min(position_size, self.limits.max_position_size)
        position_size = min(position_size, available_capital * self.limits.max_concentration)

        return max(0, position_size)

    def get_risk_report(self) -> Dict:
        """Generate risk report"""
        return {
            'metrics': self.metrics.to_dict(),
            'limits': self.limits.to_dict(),
            'status': 'normal' if self.metrics.current_drawdown < self.limits.max_drawdown * 0.5 else 'elevated',
            'utilization': {
                'drawdown': self.metrics.current_drawdown / self.limits.max_drawdown if self.limits.max_drawdown > 0 else 0,
                'var_95': self.metrics.var_95 / self.limits.max_daily_loss if self.limits.max_daily_loss > 0 else 0
            }
        }

    def print_report(self):
        """Print risk report"""
        report = self.get_risk_report()

        print("\n" + "=" * 50)
        print("RISK REPORT")
        print("=" * 50)
        print(f"Status:           {report['status'].upper()}")
        print(f"Current Drawdown: ${self.metrics.current_drawdown:,.2f}")
        print(f"Max Drawdown:     ${self.metrics.max_drawdown:,.2f}")
        print(f"VaR (95%):        ${self.metrics.var_95:,.2f}")
        print(f"Volatility:       {self.metrics.volatility:.2%}")
        print(f"Sharpe Ratio:     {self.metrics.sharpe_ratio:.2f}")
        print(f"Sortino Ratio:    {self.metrics.sortino_ratio:.2f}")
        print("=" * 50 + "\n")
