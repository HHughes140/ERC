"""
Historical Calibration Module

Provides independent probability estimates based on historical market outcomes.
Tracks actual resolution outcomes by price bucket to calibrate certainty estimates.

Key Insight: Market prices are NOT true probabilities. A market trading at 97%
may historically resolve YES only 85% of the time due to:
- Market inefficiency
- Liquidity effects
- Last-minute volatility
- Mispricing of tail risks

This module builds a calibration curve from actual outcomes to correct for these biases.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# Directory for calibration data
CALIBRATION_DIR = Path(__file__).parent.parent / "Central_DB" / "calibration"
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class OutcomeRecord:
    """Record of a market outcome for calibration"""
    market_id: str
    entry_price: float
    entry_time: str
    resolution_time: str
    resolution_price: float  # 1.0 if won, 0.0 if lost
    market_type: str  # 'directional', 'range', 'sports', 'extreme'
    platform: str  # 'polymarket', 'kalshi'
    won: bool


@dataclass
class CalibrationBucket:
    """Statistics for a price bucket"""
    wins: int = 0
    total: int = 0
    total_profit: float = 0.0
    avg_hold_time_hours: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total > 0 else 0.5

    @property
    def calibrated_probability(self) -> float:
        """
        Bayesian estimate with weak uniform prior (Beta(2,2))
        This prevents extreme estimates with few samples
        """
        alpha, beta = 2.0, 2.0  # Weak prior
        return (self.wins + alpha) / (self.total + alpha + beta)

    @property
    def confidence(self) -> float:
        """How confident are we in this calibration (based on sample size)"""
        # Reaches 90% confidence around 50 samples
        return min(self.total / 50, 1.0)


class HistoricalCalibrator:
    """
    Tracks market outcomes by price bucket to build calibration curves.

    Price buckets are 5-cent increments: [0-5), [5-10), ..., [95-100)

    Usage:
        calibrator = HistoricalCalibrator()

        # Record outcomes as markets resolve
        calibrator.record_outcome(market_id, 0.97, True, 'directional', 'polymarket')

        # Get calibrated certainty
        calibrated_p = calibrator.get_calibrated_certainty(0.97, 'directional')
    """

    def __init__(self, data_file: Optional[Path] = None):
        self.data_file = data_file or (CALIBRATION_DIR / "calibration_data.json")

        # Price buckets (5-cent increments)
        self.bucket_size = 5  # cents
        self.buckets: Dict[int, CalibrationBucket] = {
            i: CalibrationBucket() for i in range(0, 100, self.bucket_size)
        }

        # Per-market-type calibration
        self.type_buckets: Dict[str, Dict[int, CalibrationBucket]] = defaultdict(
            lambda: {i: CalibrationBucket() for i in range(0, 100, self.bucket_size)}
        )

        # Per-platform calibration
        self.platform_buckets: Dict[str, Dict[int, CalibrationBucket]] = defaultdict(
            lambda: {i: CalibrationBucket() for i in range(0, 100, self.bucket_size)}
        )

        # Historical records for auditing
        self.records: List[OutcomeRecord] = []

        self._load()

    def _price_to_bucket(self, price: float) -> int:
        """Convert price (0-1) to bucket key (0-95)"""
        cents = int(price * 100)
        bucket = (cents // self.bucket_size) * self.bucket_size
        return min(max(bucket, 0), 100 - self.bucket_size)

    def record_outcome(self, market_id: str, entry_price: float, won: bool,
                       market_type: str, platform: str,
                       entry_time: Optional[str] = None,
                       resolution_time: Optional[str] = None,
                       profit: float = 0.0) -> None:
        """
        Record a market outcome for calibration.

        Args:
            market_id: Unique market identifier
            entry_price: Price at entry (0-1)
            won: Whether the position won (resolved in our favor)
            market_type: Type of market ('directional', 'range', 'sports', 'extreme')
            platform: Trading platform ('polymarket', 'kalshi')
            entry_time: ISO timestamp of entry (optional)
            resolution_time: ISO timestamp of resolution (optional)
            profit: Actual profit/loss (optional, for tracking)
        """
        bucket_key = self._price_to_bucket(entry_price)

        # Update global bucket
        self.buckets[bucket_key].total += 1
        if won:
            self.buckets[bucket_key].wins += 1
        self.buckets[bucket_key].total_profit += profit

        # Update market-type specific bucket
        self.type_buckets[market_type][bucket_key].total += 1
        if won:
            self.type_buckets[market_type][bucket_key].wins += 1
        self.type_buckets[market_type][bucket_key].total_profit += profit

        # Update platform-specific bucket
        self.platform_buckets[platform][bucket_key].total += 1
        if won:
            self.platform_buckets[platform][bucket_key].wins += 1
        self.platform_buckets[platform][bucket_key].total_profit += profit

        # Store record
        record = OutcomeRecord(
            market_id=market_id,
            entry_price=entry_price,
            entry_time=entry_time or datetime.now().isoformat(),
            resolution_time=resolution_time or datetime.now().isoformat(),
            resolution_price=1.0 if won else 0.0,
            market_type=market_type,
            platform=platform,
            won=won
        )
        self.records.append(record)

        # Auto-save periodically
        if len(self.records) % 10 == 0:
            self._save()

        logger.info(f"Recorded outcome: {market_type}/{platform} @ {entry_price:.2f} -> {'WIN' if won else 'LOSS'}")

    def get_calibrated_certainty(self, market_price: float,
                                  market_type: Optional[str] = None,
                                  platform: Optional[str] = None) -> Dict:
        """
        Get calibrated probability estimate based on historical outcomes.

        Returns dict with:
            - calibrated_probability: Adjusted probability based on history
            - raw_probability: Original market price
            - adjustment: How much we adjusted
            - confidence: How confident we are in the calibration
            - sample_size: Number of historical samples in this bucket
        """
        bucket_key = self._price_to_bucket(market_price)

        # Get the most specific bucket available
        if market_type and market_type in self.type_buckets:
            type_bucket = self.type_buckets[market_type][bucket_key]
            if type_bucket.total >= 5:
                # Have enough type-specific data
                bucket = type_bucket
            else:
                # Fall back to global
                bucket = self.buckets[bucket_key]
        else:
            bucket = self.buckets[bucket_key]

        calibrated_p = bucket.calibrated_probability

        # Blend with market price if insufficient data
        # More data = more weight to calibration, less data = more weight to market
        blend_weight = bucket.confidence

        if bucket.total < 5:
            # Very few samples - mostly trust market price
            final_p = 0.2 * calibrated_p + 0.8 * market_price
            confidence = 0.2
        elif bucket.total < 20:
            # Some samples - blend
            final_p = blend_weight * calibrated_p + (1 - blend_weight) * market_price
            confidence = 0.5
        else:
            # Good sample size - trust calibration more
            final_p = 0.7 * calibrated_p + 0.3 * market_price
            confidence = blend_weight

        adjustment = final_p - market_price

        return {
            'calibrated_probability': final_p,
            'raw_probability': market_price,
            'adjustment': adjustment,
            'confidence': confidence,
            'sample_size': bucket.total,
            'historical_win_rate': bucket.win_rate,
            'bucket': f"{bucket_key}-{bucket_key + self.bucket_size}%"
        }

    def get_bucket_stats(self) -> Dict[str, Dict]:
        """Get statistics for all price buckets"""
        stats = {}
        for bucket_key, bucket in self.buckets.items():
            if bucket.total > 0:
                stats[f"{bucket_key}-{bucket_key + self.bucket_size}%"] = {
                    'wins': bucket.wins,
                    'total': bucket.total,
                    'win_rate': bucket.win_rate,
                    'calibrated_p': bucket.calibrated_probability,
                    'total_profit': bucket.total_profit
                }
        return stats

    def get_expected_edge(self, market_price: float,
                          market_type: Optional[str] = None) -> float:
        """
        Calculate expected edge based on calibration.

        Edge = calibrated_probability - market_probability

        Positive edge means historically this price bucket wins more than market suggests.
        Negative edge means it wins less.
        """
        result = self.get_calibrated_certainty(market_price, market_type)
        return result['calibrated_probability'] - result['raw_probability']

    def should_trade(self, market_price: float, market_type: Optional[str] = None,
                     min_edge: float = 0.02, min_confidence: float = 0.3) -> Tuple[bool, str]:
        """
        Determine if a trade should be taken based on calibration.

        Args:
            market_price: Current market price
            market_type: Type of market
            min_edge: Minimum edge required (default 2%)
            min_confidence: Minimum calibration confidence

        Returns:
            (should_trade, reason)
        """
        result = self.get_calibrated_certainty(market_price, market_type)
        edge = self.get_expected_edge(market_price, market_type)

        if result['confidence'] < min_confidence:
            return False, f"Insufficient calibration data (confidence: {result['confidence']:.0%})"

        if edge < min_edge:
            return False, f"Edge too small ({edge:.1%} < {min_edge:.1%})"

        if result['calibrated_probability'] < 0.90:
            return False, f"Calibrated probability too low ({result['calibrated_probability']:.1%})"

        return True, f"Edge: {edge:.1%}, Calibrated P: {result['calibrated_probability']:.1%}"

    def _save(self) -> None:
        """Save calibration data to disk"""
        data = {
            'buckets': {
                str(k): asdict(v) for k, v in self.buckets.items()
            },
            'type_buckets': {
                t: {str(k): asdict(v) for k, v in buckets.items()}
                for t, buckets in self.type_buckets.items()
            },
            'platform_buckets': {
                p: {str(k): asdict(v) for k, v in buckets.items()}
                for p, buckets in self.platform_buckets.items()
            },
            'records': [asdict(r) for r in self.records[-1000:]],  # Keep last 1000
            'last_updated': datetime.now().isoformat()
        }

        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved calibration data to {self.data_file}")

    def _load(self) -> None:
        """Load calibration data from disk"""
        if not self.data_file.exists():
            logger.info("No calibration data found, starting fresh")
            return

        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)

            # Load global buckets
            for k, v in data.get('buckets', {}).items():
                bucket_key = int(k)
                self.buckets[bucket_key] = CalibrationBucket(**v)

            # Load type-specific buckets
            for t, buckets in data.get('type_buckets', {}).items():
                for k, v in buckets.items():
                    bucket_key = int(k)
                    self.type_buckets[t][bucket_key] = CalibrationBucket(**v)

            # Load platform-specific buckets
            for p, buckets in data.get('platform_buckets', {}).items():
                for k, v in buckets.items():
                    bucket_key = int(k)
                    self.platform_buckets[p][bucket_key] = CalibrationBucket(**v)

            # Load records
            self.records = [OutcomeRecord(**r) for r in data.get('records', [])]

            total_samples = sum(b.total for b in self.buckets.values())
            logger.info(f"Loaded calibration data: {total_samples} historical samples")

        except Exception as e:
            logger.error(f"Error loading calibration data: {e}")


class FeeAdjustedProfitCalculator:
    """
    Calculates actual profit after all fees and costs.

    Critical insight: A 97% position with 2% trading fees = 99% cost,
    yielding only 1% profit if it wins. After accounting for losses,
    this may be NEGATIVE expected value!
    """

    # Platform fee structures (as of 2024)
    FEE_STRUCTURES = {
        'polymarket': {
            'entry_fee': 0.00,        # No entry fee
            'exit_trade_fee': 0.02,   # ~2% for trading out
            'exit_resolution_fee': 0.00,  # No fee on resolution
            'spread_cost': 0.01,      # Estimated bid-ask spread
        },
        'kalshi': {
            'entry_fee': 0.00,        # No entry fee
            'exit_trade_fee': 0.01,   # ~1% trading fee
            'exit_resolution_fee': 0.07,  # 7% of profit on resolution
            'spread_cost': 0.01,      # Estimated bid-ask spread
        }
    }

    def __init__(self, platform: str = 'polymarket'):
        self.platform = platform
        self.fees = self.FEE_STRUCTURES.get(platform, self.FEE_STRUCTURES['polymarket'])

    def calculate_net_profit(self, entry_price: float,
                             exit_price: float = 1.0,
                             exit_via_resolution: bool = True,
                             position_size: float = 100.0) -> Dict:
        """
        Calculate net profit after all fees.

        Args:
            entry_price: Price paid per share (0-1)
            exit_price: Price received on exit (1.0 if won via resolution)
            exit_via_resolution: True if holding to resolution, False if trading out
            position_size: Dollar amount of position

        Returns:
            Dict with profit breakdown
        """
        shares = position_size / entry_price

        # Gross profit
        gross_profit = (exit_price - entry_price) * shares

        # Fee calculations
        entry_cost = position_size * self.fees['entry_fee']
        spread_cost = position_size * self.fees['spread_cost']

        if exit_via_resolution:
            # Holding to resolution
            if gross_profit > 0:
                exit_fee = gross_profit * self.fees['exit_resolution_fee']
            else:
                exit_fee = 0
        else:
            # Trading out early
            exit_value = exit_price * shares
            exit_fee = exit_value * self.fees['exit_trade_fee']

        total_fees = entry_cost + spread_cost + exit_fee
        net_profit = gross_profit - total_fees
        net_profit_pct = net_profit / position_size

        # Calculate breakeven price
        if exit_via_resolution:
            # Need to win enough to cover all costs
            # If we pay 97c and there's 2% spread cost, we need ~99c to break even
            breakeven = entry_price + (self.fees['entry_fee'] + self.fees['spread_cost']) * entry_price
            if self.fees['exit_resolution_fee'] > 0:
                # Account for profit tax on resolution
                breakeven = breakeven / (1 - self.fees['exit_resolution_fee'])
        else:
            breakeven = entry_price * (1 + self.fees['entry_fee'] + 2 * self.fees['exit_trade_fee'] + self.fees['spread_cost'])

        return {
            'gross_profit': gross_profit,
            'total_fees': total_fees,
            'net_profit': net_profit,
            'net_profit_pct': net_profit_pct,
            'is_profitable': net_profit > 0,
            'breakeven_price': min(breakeven, 1.0),
            'fee_breakdown': {
                'entry_fee': entry_cost,
                'spread_cost': spread_cost,
                'exit_fee': exit_fee
            }
        }

    def get_minimum_edge(self) -> float:
        """
        Calculate minimum edge needed to be profitable.

        For scalping strategy holding to resolution:
        Need edge > all fees + some buffer for losses
        """
        # Total cost of round trip if we trade out
        trade_out_cost = (
            self.fees['entry_fee'] +
            self.fees['exit_trade_fee'] +
            self.fees['spread_cost']
        )

        # Resolution cost
        resolution_cost = self.fees['spread_cost']  # Just spread on entry

        # Use the lower of the two + buffer
        min_edge = min(trade_out_cost, resolution_cost) + 0.005  # 0.5% buffer

        return min_edge

    def expected_value(self, entry_price: float,
                       win_probability: float,
                       position_size: float = 100.0) -> Dict:
        """
        Calculate expected value of a position.

        Args:
            entry_price: Price per share
            win_probability: Calibrated probability of winning
            position_size: Dollar amount

        Returns:
            Expected value and recommendation
        """
        # Scenario 1: Win (position resolves YES)
        win_result = self.calculate_net_profit(entry_price, 1.0, True, position_size)

        # Scenario 2: Lose (position resolves NO)
        loss_result = self.calculate_net_profit(entry_price, 0.0, True, position_size)

        # Expected value
        ev = (win_probability * win_result['net_profit'] +
              (1 - win_probability) * loss_result['net_profit'])

        ev_pct = ev / position_size

        return {
            'expected_value': ev,
            'expected_value_pct': ev_pct,
            'win_profit': win_result['net_profit'],
            'loss_amount': loss_result['net_profit'],  # Will be negative
            'breakeven_probability': -loss_result['net_profit'] / (win_result['net_profit'] - loss_result['net_profit']),
            'is_positive_ev': ev > 0,
            'recommendation': 'TRADE' if ev > position_size * 0.005 else 'SKIP'  # Need 0.5% EV
        }


# Convenience functions
_calibrator: Optional[HistoricalCalibrator] = None

def get_calibrator() -> HistoricalCalibrator:
    """Get or create the global calibrator instance"""
    global _calibrator
    if _calibrator is None:
        _calibrator = HistoricalCalibrator()
    return _calibrator


def calibrated_certainty(market_price: float,
                         market_type: Optional[str] = None) -> float:
    """Convenience function to get calibrated certainty"""
    calibrator = get_calibrator()
    result = calibrator.get_calibrated_certainty(market_price, market_type)
    return result['calibrated_probability']


def record_trade_outcome(market_id: str, entry_price: float, won: bool,
                         market_type: str, platform: str) -> None:
    """Convenience function to record an outcome"""
    calibrator = get_calibrator()
    calibrator.record_outcome(market_id, entry_price, won, market_type, platform)
