"""
Time Decay Module
Calculates time decay effects for prediction market positions
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class TimeDecayMetrics:
    """Time decay metrics for a market/position"""
    market_id: str
    days_to_expiry: float
    hours_to_expiry: float
    decay_factor: float  # 0 to 1, higher = more time value remaining
    urgency_score: float  # 0 to 1, higher = more urgent
    theta: float  # Daily time decay rate
    optimal_hold_period: float  # Recommended hold time in hours

    def to_dict(self) -> Dict:
        return {
            'market_id': self.market_id,
            'days_to_expiry': self.days_to_expiry,
            'hours_to_expiry': self.hours_to_expiry,
            'decay_factor': self.decay_factor,
            'urgency_score': self.urgency_score,
            'theta': self.theta,
            'optimal_hold_period': self.optimal_hold_period
        }


class TimeDecayCalculator:
    """
    Calculates time decay for prediction market positions

    Features:
    - Time value decay modeling
    - Optimal entry/exit timing
    - Urgency scoring for near-expiry markets
    - Hold period recommendations
    """

    def __init__(self):
        # Cache of calculated metrics
        self.metrics_cache: Dict[str, TimeDecayMetrics] = {}

        # Configuration
        self.urgency_threshold_hours = 24  # High urgency within 24 hours
        self.min_hold_hours = 1  # Minimum recommended hold
        self.max_hold_hours = 168  # Maximum recommended hold (1 week)

        logger.info("Time Decay Calculator initialized")

    def calculate_decay(self, market_id: str, expiry_time: datetime,
                       current_price: float = 0.5) -> TimeDecayMetrics:
        """
        Calculate time decay metrics for a market

        Args:
            market_id: Market identifier
            expiry_time: Market expiration datetime
            current_price: Current market price (0 to 1)

        Returns:
            TimeDecayMetrics
        """
        now = datetime.now()

        # Calculate time remaining
        time_remaining = expiry_time - now
        hours_to_expiry = max(0, time_remaining.total_seconds() / 3600)
        days_to_expiry = hours_to_expiry / 24

        # Calculate decay factor (exponential decay model)
        # Markets lose time value faster as they approach expiry
        if hours_to_expiry <= 0:
            decay_factor = 0.0
        else:
            # Half-life model: value decays to 50% at certain point
            half_life_hours = 168  # 1 week
            decay_factor = math.exp(-0.693 * (1 / hours_to_expiry) * half_life_hours) if hours_to_expiry > 0 else 0

        # Normalize decay factor
        decay_factor = max(0.0, min(1.0, decay_factor))

        # Calculate urgency score
        if hours_to_expiry <= 0:
            urgency_score = 1.0
        elif hours_to_expiry <= self.urgency_threshold_hours:
            urgency_score = 1.0 - (hours_to_expiry / self.urgency_threshold_hours)
        else:
            urgency_score = 0.0

        # Calculate theta (daily time decay rate)
        # Theta increases as expiry approaches
        if days_to_expiry > 0:
            base_theta = 0.01  # 1% per day base decay
            theta = base_theta * (1 + 1 / max(days_to_expiry, 0.1))
        else:
            theta = 1.0  # Full decay at expiry

        theta = min(theta, 1.0)

        # Calculate optimal hold period
        optimal_hold = self._calculate_optimal_hold(
            hours_to_expiry=hours_to_expiry,
            current_price=current_price,
            decay_factor=decay_factor
        )

        metrics = TimeDecayMetrics(
            market_id=market_id,
            days_to_expiry=days_to_expiry,
            hours_to_expiry=hours_to_expiry,
            decay_factor=decay_factor,
            urgency_score=urgency_score,
            theta=theta,
            optimal_hold_period=optimal_hold
        )

        # Cache
        self.metrics_cache[market_id] = metrics

        return metrics

    def _calculate_optimal_hold(self, hours_to_expiry: float,
                                current_price: float,
                                decay_factor: float) -> float:
        """Calculate optimal hold period in hours"""
        if hours_to_expiry <= 0:
            return 0

        # Base hold time depends on time to expiry
        if hours_to_expiry < 24:
            # Very short-term: hold until near expiry
            base_hold = hours_to_expiry * 0.8
        elif hours_to_expiry < 168:
            # Medium-term: partial hold
            base_hold = hours_to_expiry * 0.5
        else:
            # Long-term: cap at 1 week
            base_hold = 168

        # Adjust for price extremity
        # Extreme prices (near 0 or 1) suggest holding longer
        price_deviation = abs(current_price - 0.5) * 2  # 0 to 1
        hold_multiplier = 1.0 + price_deviation * 0.5

        optimal = base_hold * hold_multiplier

        # Apply limits
        optimal = max(self.min_hold_hours, min(optimal, self.max_hold_hours))
        optimal = min(optimal, hours_to_expiry * 0.9)  # Don't hold past 90% of expiry

        return optimal

    def get_time_value_premium(self, current_price: float,
                               hours_to_expiry: float) -> float:
        """
        Estimate time value premium in a market price

        Args:
            current_price: Current market price (0 to 1)
            hours_to_expiry: Hours until expiry

        Returns:
            Estimated time value premium (0 to 1)
        """
        if hours_to_expiry <= 0:
            return 0.0

        # Intrinsic value estimate (binary outcome)
        if current_price >= 0.5:
            intrinsic = (current_price - 0.5) * 2  # 0 to 1 scale
        else:
            intrinsic = (0.5 - current_price) * 2

        # Time value is what's left after intrinsic
        time_value = 1.0 - intrinsic

        # Time value decays as expiry approaches
        if hours_to_expiry < 24:
            time_value *= hours_to_expiry / 24
        elif hours_to_expiry < 168:
            time_value *= 0.8 + (hours_to_expiry / 168) * 0.2

        return max(0.0, min(1.0, time_value))

    def should_avoid_entry(self, market: Any, strategy: str = None) -> bool:
        """
        Check if time decay suggests avoiding entry

        Args:
            market: Market object with expiry info
            strategy: Trading strategy type

        Returns:
            True if entry should be avoided
        """
        # Get expiry time
        expiry = getattr(market, 'end_date', None)
        if not expiry:
            expiry = getattr(market, 'expiry', None)
        if not expiry:
            return False  # Can't evaluate, allow

        if isinstance(expiry, str):
            try:
                expiry = datetime.fromisoformat(expiry.replace('Z', '+00:00'))
            except ValueError:
                return False

        market_id = getattr(market, 'condition_id', getattr(market, 'market_id', ''))
        current_price = 0.5
        if hasattr(market, 'outcome_prices') and market.outcome_prices:
            current_price = market.outcome_prices[0]

        metrics = self.calculate_decay(market_id, expiry, current_price)

        # Strategy-specific rules
        if strategy == 'arbitrage':
            # Arbitrage should avoid very short-term markets (execution risk)
            if metrics.hours_to_expiry < 1:
                return True
        elif strategy in ['scalping', 'sharky']:
            # Scalping needs some time to exit
            if metrics.hours_to_expiry < 0.5:
                return True
        elif strategy == 'directional':
            # Directional needs more time
            if metrics.hours_to_expiry < 4:
                return True

        # High urgency with uncertain price suggests avoid
        if metrics.urgency_score > 0.9 and 0.3 < current_price < 0.7:
            return True

        return False

    def get_exit_recommendation(self, market_id: str,
                               entry_time: datetime,
                               expiry_time: datetime,
                               entry_price: float,
                               current_price: float) -> Dict:
        """
        Get exit timing recommendation

        Returns:
            Dict with 'should_exit', 'urgency', 'reason'
        """
        now = datetime.now()
        hours_held = (now - entry_time).total_seconds() / 3600
        hours_to_expiry = max(0, (expiry_time - now).total_seconds() / 3600)

        metrics = self.calculate_decay(market_id, expiry_time, current_price)

        # Check if past optimal hold
        if hours_held > metrics.optimal_hold_period:
            return {
                'should_exit': True,
                'urgency': 0.7,
                'reason': 'Past optimal hold period'
            }

        # Check if very close to expiry
        if hours_to_expiry < 1:
            return {
                'should_exit': True,
                'urgency': 0.9,
                'reason': 'Very close to expiry'
            }

        # Check if profitable and time running out
        profit = current_price - entry_price
        if profit > 0.01 and hours_to_expiry < 4:
            return {
                'should_exit': True,
                'urgency': 0.6,
                'reason': 'Lock in profit before expiry'
            }

        # Check theta decay
        if metrics.theta > 0.1 and hours_to_expiry < 24:
            return {
                'should_exit': True,
                'urgency': 0.5,
                'reason': 'High time decay'
            }

        return {
            'should_exit': False,
            'urgency': metrics.urgency_score,
            'reason': None
        }

    def get_cached_metrics(self, market_id: str) -> Optional[TimeDecayMetrics]:
        """Get cached metrics for a market"""
        return self.metrics_cache.get(market_id)

    def clear_expired(self):
        """Clear expired markets from cache"""
        now = datetime.now()
        expired = [
            market_id for market_id, metrics in self.metrics_cache.items()
            if metrics.hours_to_expiry <= 0
        ]
        for market_id in expired:
            del self.metrics_cache[market_id]

    def print_report(self):
        """Print time decay report"""
        print("\n" + "=" * 50)
        print("TIME DECAY REPORT")
        print("=" * 50)
        print(f"Tracked Markets: {len(self.metrics_cache)}")

        if self.metrics_cache:
            # Sort by urgency
            sorted_markets = sorted(
                self.metrics_cache.items(),
                key=lambda x: x[1].urgency_score,
                reverse=True
            )

            print("\nMost Urgent Markets:")
            for market_id, metrics in sorted_markets[:5]:
                print(f"  {market_id[:30]}")
                print(f"    Hours to expiry: {metrics.hours_to_expiry:.1f}")
                print(f"    Urgency: {metrics.urgency_score:.2f}")
                print(f"    Theta: {metrics.theta:.3f}")

        print("=" * 50 + "\n")
