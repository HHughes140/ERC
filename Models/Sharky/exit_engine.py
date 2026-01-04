"""
Dynamic Exit Engine for Sharky Scanner

Determines optimal exit timing for near-certainty positions based on:
1. Profit targets (net of fees)
2. Stop-loss thresholds
3. Certainty degradation
4. EV comparison (exit now vs hold to resolution)
5. Volatility/time-based triggers

CRITICAL: High-certainty positions still need exit logic!
A 97% certainty market that drops to 92% should trigger review.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Reasons for triggering an exit"""
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    CERTAINTY_DROP = "certainty_drop"
    EV_COMPARISON = "ev_comparison"
    TIME_DECAY = "time_decay"
    VOLATILITY_SPIKE = "volatility_spike"
    MANUAL = "manual"
    RESOLUTION = "resolution"
    HOLD = "hold"  # No exit triggered


@dataclass
class Position:
    """Represents an open position"""
    market_id: str
    platform: str
    entry_price: float
    entry_time: datetime
    shares: float
    side: str  # 'YES' or 'NO'
    entry_certainty: float
    market_type: str
    resolution_time: Optional[datetime] = None

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.shares


@dataclass
class ExitSignal:
    """Signal from exit engine"""
    should_exit: bool
    reason: ExitReason
    urgency: float  # 0-1, higher = more urgent
    expected_pnl: float
    confidence: float
    details: Dict


@dataclass
class ExitParameters:
    """Configurable exit parameters"""
    # Profit targets (net of fees)
    profit_target_pct: float = 0.02  # 2% net profit target
    aggressive_profit_target: float = 0.03  # 3% for quick exit

    # Stop loss
    stop_loss_pct: float = 0.03  # 3% loss triggers exit
    max_loss_dollars: float = 50.0  # Absolute dollar limit

    # Certainty thresholds
    min_certainty: float = 0.90  # Exit if certainty drops below this
    certainty_drop_threshold: float = 0.05  # 5% drop from entry triggers review

    # Time-based
    min_hold_minutes: float = 5.0  # Don't exit immediately
    panic_exit_minutes: float = 30.0  # Exit before resolution if too close

    # Volatility
    max_spread_pct: float = 0.05  # Exit if spread blows out

    # EV comparison
    ev_exit_threshold: float = 1.1  # Exit if exit_ev > 1.1 * hold_ev


class DynamicExitEngine:
    """
    Determines when to exit positions based on multiple factors.

    Key insight: Even high-certainty positions need dynamic management.
    The market can give information about changing certainty.
    """

    def __init__(self, params: Optional[ExitParameters] = None):
        self.params = params or ExitParameters()
        self.fee_rates = {
            'polymarket': {
                'trade': 0.02,  # 2% on trades
                'resolution': 0.00  # No fee if held to resolution
            },
            'kalshi': {
                'trade': 0.01,  # 1% on trades (estimate)
                'resolution': 0.07  # 7% of profit at resolution
            }
        }

    def evaluate_exit(self, position: Position,
                      current_price: float,
                      current_certainty: float,
                      bid_price: float,
                      ask_price: float,
                      time_to_resolution: Optional[timedelta] = None) -> ExitSignal:
        """
        Evaluate whether to exit a position.

        Args:
            position: Current position details
            current_price: Current market price for our side
            current_certainty: Current estimated certainty
            bid_price: Best bid (what we'd get if selling)
            ask_price: Best ask (current market ask)
            time_to_resolution: Time until market resolves

        Returns:
            ExitSignal with recommendation
        """
        signals = []

        # Calculate basic metrics
        exit_price = bid_price  # We sell at bid
        gross_pnl = (exit_price - position.entry_price) * position.shares

        # Calculate fees
        fees = self._calculate_exit_fees(position, exit_price, gross_pnl)
        net_pnl = gross_pnl - fees
        net_pnl_pct = net_pnl / position.cost_basis if position.cost_basis > 0 else 0

        # Current spread
        spread_pct = (ask_price - bid_price) / ((ask_price + bid_price) / 2) if (ask_price + bid_price) > 0 else 0

        # Time held
        time_held = datetime.now() - position.entry_time

        # Check each exit condition

        # 1. Profit target
        profit_signal = self._check_profit_target(net_pnl_pct, time_held)
        if profit_signal:
            signals.append(profit_signal)

        # 2. Stop loss
        stop_signal = self._check_stop_loss(net_pnl, net_pnl_pct, position.cost_basis)
        if stop_signal:
            signals.append(stop_signal)

        # 3. Certainty drop
        certainty_signal = self._check_certainty_drop(
            position.entry_certainty, current_certainty
        )
        if certainty_signal:
            signals.append(certainty_signal)

        # 4. EV comparison
        ev_signal = self._check_ev_comparison(
            position, exit_price, current_certainty, net_pnl
        )
        if ev_signal:
            signals.append(ev_signal)

        # 5. Time decay / approaching resolution
        if time_to_resolution:
            time_signal = self._check_time_decay(
                time_to_resolution, net_pnl_pct, current_certainty
            )
            if time_signal:
                signals.append(time_signal)

        # 6. Volatility spike
        volatility_signal = self._check_volatility(spread_pct)
        if volatility_signal:
            signals.append(volatility_signal)

        # Aggregate signals
        if not signals:
            return ExitSignal(
                should_exit=False,
                reason=ExitReason.HOLD,
                urgency=0.0,
                expected_pnl=net_pnl,
                confidence=current_certainty,
                details={
                    'net_pnl': net_pnl,
                    'net_pnl_pct': net_pnl_pct,
                    'spread_pct': spread_pct,
                    'time_held_minutes': time_held.total_seconds() / 60
                }
            )

        # Return highest urgency signal
        best_signal = max(signals, key=lambda s: s.urgency)
        return best_signal

    def _calculate_exit_fees(self, position: Position,
                             exit_price: float, gross_pnl: float) -> float:
        """Calculate fees for exiting via trade"""
        platform = position.platform.lower()
        fee_structure = self.fee_rates.get(platform, self.fee_rates['polymarket'])

        # Trade exit fee (on revenue)
        exit_revenue = exit_price * position.shares
        trade_fee = exit_revenue * fee_structure['trade']

        return trade_fee

    def _calculate_resolution_fees(self, position: Position,
                                   gross_pnl: float, won: bool) -> float:
        """Calculate fees if holding to resolution"""
        platform = position.platform.lower()
        fee_structure = self.fee_rates.get(platform, self.fee_rates['polymarket'])

        if won and gross_pnl > 0:
            # Fee on profit at resolution
            return gross_pnl * fee_structure['resolution']
        return 0

    def _check_profit_target(self, net_pnl_pct: float,
                             time_held: timedelta) -> Optional[ExitSignal]:
        """Check if profit target is hit"""
        # Don't exit too quickly
        if time_held.total_seconds() < self.params.min_hold_minutes * 60:
            return None

        if net_pnl_pct >= self.params.aggressive_profit_target:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.PROFIT_TARGET,
                urgency=0.9,
                expected_pnl=net_pnl_pct,
                confidence=0.95,
                details={'target_type': 'aggressive', 'pnl_pct': net_pnl_pct}
            )

        if net_pnl_pct >= self.params.profit_target_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.PROFIT_TARGET,
                urgency=0.7,
                expected_pnl=net_pnl_pct,
                confidence=0.9,
                details={'target_type': 'standard', 'pnl_pct': net_pnl_pct}
            )

        return None

    def _check_stop_loss(self, net_pnl: float, net_pnl_pct: float,
                         cost_basis: float) -> Optional[ExitSignal]:
        """Check if stop loss is triggered"""
        # Percentage-based stop
        if net_pnl_pct <= -self.params.stop_loss_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                urgency=1.0,  # Highest urgency
                expected_pnl=net_pnl,
                confidence=0.99,
                details={'trigger': 'percentage', 'loss_pct': net_pnl_pct}
            )

        # Dollar-based stop
        if net_pnl <= -self.params.max_loss_dollars:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.STOP_LOSS,
                urgency=1.0,
                expected_pnl=net_pnl,
                confidence=0.99,
                details={'trigger': 'dollar', 'loss_dollars': net_pnl}
            )

        return None

    def _check_certainty_drop(self, entry_certainty: float,
                              current_certainty: float) -> Optional[ExitSignal]:
        """Check if certainty has dropped significantly"""
        # Below minimum threshold
        if current_certainty < self.params.min_certainty:
            urgency = (self.params.min_certainty - current_certainty) / 0.1
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.CERTAINTY_DROP,
                urgency=min(0.95, 0.7 + urgency * 0.25),
                expected_pnl=0,  # Unknown
                confidence=current_certainty,
                details={
                    'entry_certainty': entry_certainty,
                    'current_certainty': current_certainty,
                    'trigger': 'below_minimum'
                }
            )

        # Significant drop from entry
        certainty_drop = entry_certainty - current_certainty
        if certainty_drop >= self.params.certainty_drop_threshold:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.CERTAINTY_DROP,
                urgency=0.6 + certainty_drop,  # Higher drop = higher urgency
                expected_pnl=0,
                confidence=current_certainty,
                details={
                    'entry_certainty': entry_certainty,
                    'current_certainty': current_certainty,
                    'drop': certainty_drop,
                    'trigger': 'significant_drop'
                }
            )

        return None

    def _check_ev_comparison(self, position: Position, exit_price: float,
                             current_certainty: float,
                             current_net_pnl: float) -> Optional[ExitSignal]:
        """
        Compare EV of exiting now vs holding to resolution.

        Exit now: Lock in current_net_pnl
        Hold: certainty * win_pnl + (1 - certainty) * loss_pnl
        """
        # What we get if we win at resolution
        win_pnl_gross = (1.0 - position.entry_price) * position.shares
        win_fees = self._calculate_resolution_fees(position, win_pnl_gross, won=True)
        win_pnl_net = win_pnl_gross - win_fees

        # What we lose if we lose at resolution
        loss_pnl = -position.cost_basis

        # Expected value of holding
        hold_ev = current_certainty * win_pnl_net + (1 - current_certainty) * loss_pnl

        # Expected value of exiting now
        exit_ev = current_net_pnl

        # Compare
        if exit_ev > 0 and hold_ev > 0:
            ev_ratio = exit_ev / hold_ev
            if ev_ratio > self.params.ev_exit_threshold:
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.EV_COMPARISON,
                    urgency=0.5 + (ev_ratio - 1) * 0.3,
                    expected_pnl=exit_ev,
                    confidence=0.8,
                    details={
                        'exit_ev': exit_ev,
                        'hold_ev': hold_ev,
                        'ev_ratio': ev_ratio,
                        'certainty': current_certainty
                    }
                )

        # Exit is profitable but hold is negative EV
        if exit_ev > 0 and hold_ev < 0:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.EV_COMPARISON,
                urgency=0.85,
                expected_pnl=exit_ev,
                confidence=0.9,
                details={
                    'exit_ev': exit_ev,
                    'hold_ev': hold_ev,
                    'reason': 'hold_ev_negative'
                }
            )

        return None

    def _check_time_decay(self, time_to_resolution: timedelta,
                          net_pnl_pct: float,
                          current_certainty: float) -> Optional[ExitSignal]:
        """
        Check time-based exit triggers.

        Near resolution, volatility increases and our ability to exit decreases.
        If certainty is not high enough, better to exit while we can.
        """
        minutes_to_resolution = time_to_resolution.total_seconds() / 60

        # Panic exit threshold
        if minutes_to_resolution < self.params.panic_exit_minutes:
            # Only if certainty is questionable
            if current_certainty < 0.95:
                urgency = (self.params.panic_exit_minutes - minutes_to_resolution) / self.params.panic_exit_minutes
                return ExitSignal(
                    should_exit=True,
                    reason=ExitReason.TIME_DECAY,
                    urgency=0.6 + urgency * 0.35,
                    expected_pnl=net_pnl_pct,
                    confidence=current_certainty,
                    details={
                        'minutes_to_resolution': minutes_to_resolution,
                        'certainty': current_certainty,
                        'trigger': 'approaching_resolution'
                    }
                )

        return None

    def _check_volatility(self, spread_pct: float) -> Optional[ExitSignal]:
        """Check if spread has blown out (market stress)"""
        if spread_pct > self.params.max_spread_pct:
            return ExitSignal(
                should_exit=True,
                reason=ExitReason.VOLATILITY_SPIKE,
                urgency=0.75,
                expected_pnl=0,
                confidence=0.7,
                details={
                    'spread_pct': spread_pct,
                    'max_allowed': self.params.max_spread_pct
                }
            )

        return None

    def calculate_optimal_exit_price(self, position: Position,
                                     current_certainty: float,
                                     bid: float, ask: float) -> Dict:
        """
        Calculate the optimal limit order price for exiting.

        Trade-off: Higher price = better fill but lower probability of execution.
        """
        spread = ask - bid
        mid = (ask + bid) / 2

        # Base target: middle of spread
        base_target = mid

        # Adjust based on certainty
        # High certainty = more patient, can wait for better price
        # Low certainty = more urgent, willing to hit bid
        patience_factor = (current_certainty - 0.9) / 0.1  # 0-1 for 90-100% certainty
        patience_factor = max(0, min(1, patience_factor))

        # Optimal price between bid and mid
        optimal_price = bid + spread * 0.5 * patience_factor

        # Expected fill probability
        fill_probability = 1 - patience_factor * 0.3

        return {
            'bid': bid,
            'ask': ask,
            'mid': mid,
            'optimal_price': optimal_price,
            'fill_probability': fill_probability,
            'patience_factor': patience_factor,
            'recommendation': 'limit' if patience_factor > 0.3 else 'market'
        }

    def batch_evaluate(self, positions: List[Tuple[Position, Dict]]) -> List[ExitSignal]:
        """
        Evaluate multiple positions at once.

        Args:
            positions: List of (Position, market_data) tuples
                      market_data should have: current_price, bid, ask, certainty, time_to_resolution

        Returns:
            List of ExitSignals
        """
        signals = []

        for position, market_data in positions:
            signal = self.evaluate_exit(
                position=position,
                current_price=market_data.get('current_price', position.entry_price),
                current_certainty=market_data.get('certainty', position.entry_certainty),
                bid_price=market_data.get('bid', market_data.get('current_price', position.entry_price) * 0.98),
                ask_price=market_data.get('ask', market_data.get('current_price', position.entry_price) * 1.02),
                time_to_resolution=market_data.get('time_to_resolution')
            )
            signals.append(signal)

        return signals


def test_exit_engine():
    """Test the dynamic exit engine"""
    engine = DynamicExitEngine()

    # Create test position
    position = Position(
        market_id="test_market",
        platform="polymarket",
        entry_price=0.95,
        entry_time=datetime.now() - timedelta(hours=1),
        shares=100,
        side="YES",
        entry_certainty=0.97,
        market_type="election",
        resolution_time=datetime.now() + timedelta(hours=24)
    )

    print("=== Dynamic Exit Engine Test ===\n")

    # Test 1: Profitable position
    signal = engine.evaluate_exit(
        position=position,
        current_price=0.97,
        current_certainty=0.97,
        bid_price=0.96,
        ask_price=0.98,
        time_to_resolution=timedelta(hours=24)
    )
    print(f"Test 1 - Profitable: should_exit={signal.should_exit}, reason={signal.reason.value}")
    print(f"  Details: {signal.details}\n")

    # Test 2: Certainty dropped
    signal = engine.evaluate_exit(
        position=position,
        current_price=0.92,
        current_certainty=0.88,
        bid_price=0.91,
        ask_price=0.93,
        time_to_resolution=timedelta(hours=24)
    )
    print(f"Test 2 - Certainty drop: should_exit={signal.should_exit}, reason={signal.reason.value}")
    print(f"  Details: {signal.details}\n")

    # Test 3: Stop loss
    signal = engine.evaluate_exit(
        position=position,
        current_price=0.90,
        current_certainty=0.92,
        bid_price=0.89,
        ask_price=0.91,
        time_to_resolution=timedelta(hours=24)
    )
    print(f"Test 3 - Stop loss: should_exit={signal.should_exit}, reason={signal.reason.value}")
    print(f"  Details: {signal.details}\n")

    # Test 4: Near resolution with uncertain outcome
    signal = engine.evaluate_exit(
        position=position,
        current_price=0.94,
        current_certainty=0.93,
        bid_price=0.93,
        ask_price=0.95,
        time_to_resolution=timedelta(minutes=20)
    )
    print(f"Test 4 - Near resolution: should_exit={signal.should_exit}, reason={signal.reason.value}")
    print(f"  Details: {signal.details}\n")


if __name__ == "__main__":
    test_exit_engine()
