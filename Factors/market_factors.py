"""
Market Factors Module
Analyzes market-wide factors affecting prediction market pricing
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification"""
    LOW_VOLATILITY = "low_volatility"
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"


class LiquidityLevel(Enum):
    """Liquidity classification"""
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


@dataclass
class MarketState:
    """Current market state"""
    regime: MarketRegime = MarketRegime.NORMAL
    liquidity: LiquidityLevel = LiquidityLevel.NORMAL
    volatility_index: float = 0.0
    avg_spread: float = 0.0
    market_depth: float = 0.0
    activity_level: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'regime': self.regime.value,
            'liquidity': self.liquidity.value,
            'volatility_index': self.volatility_index,
            'avg_spread': self.avg_spread,
            'market_depth': self.market_depth,
            'activity_level': self.activity_level
        }


@dataclass
class MarketFactor:
    """Individual market factor"""
    name: str
    value: float
    impact: float  # -1 to 1, negative = bearish, positive = bullish
    confidence: float  # 0 to 1
    source: str = ""

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'value': self.value,
            'impact': self.impact,
            'confidence': self.confidence,
            'source': self.source
        }


class MarketFactorAnalyzer:
    """
    Analyzes market-wide factors for prediction markets

    Features:
    - Market regime detection
    - Liquidity analysis
    - Spread analysis
    - Activity level monitoring
    - Factor scoring for opportunities
    """

    def __init__(self):
        self.state = MarketState()
        self.factors: List[MarketFactor] = []

        # Historical data
        self.volatility_history: List[float] = []
        self.spread_history: List[float] = []
        self.volume_history: List[float] = []

        # Thresholds
        self.high_vol_threshold = 0.3
        self.low_vol_threshold = 0.1
        self.low_liquidity_threshold = 1000

        logger.info("Market Factor Analyzer initialized")

    def update(self, markets: List[Any]):
        """
        Update market factors from current market data

        Args:
            markets: List of market objects with price/volume data
        """
        if not markets:
            return

        # Calculate aggregate metrics
        spreads = []
        volumes = []
        liquidities = []

        for market in markets:
            # Extract metrics
            if hasattr(market, 'outcome_prices') and len(market.outcome_prices) >= 2:
                yes_price = market.outcome_prices[0]
                no_price = market.outcome_prices[1]
                spread = abs(1.0 - yes_price - no_price)
                spreads.append(spread)

            if hasattr(market, 'volume'):
                volumes.append(float(market.volume))

            if hasattr(market, 'liquidity'):
                liquidities.append(float(market.liquidity))

        # Update state
        if spreads:
            self.state.avg_spread = sum(spreads) / len(spreads)
            self.spread_history.append(self.state.avg_spread)

        if volumes:
            total_volume = sum(volumes)
            self.volume_history.append(total_volume)
            self.state.activity_level = total_volume

        if liquidities:
            avg_liquidity = sum(liquidities) / len(liquidities)
            self.state.market_depth = avg_liquidity

            # Classify liquidity
            if avg_liquidity < self.low_liquidity_threshold:
                self.state.liquidity = LiquidityLevel.VERY_LOW
            elif avg_liquidity < self.low_liquidity_threshold * 5:
                self.state.liquidity = LiquidityLevel.LOW
            elif avg_liquidity < self.low_liquidity_threshold * 20:
                self.state.liquidity = LiquidityLevel.NORMAL
            else:
                self.state.liquidity = LiquidityLevel.HIGH

        # Calculate volatility from spread changes
        if len(self.spread_history) >= 10:
            recent_spreads = self.spread_history[-10:]
            mean_spread = sum(recent_spreads) / len(recent_spreads)
            variance = sum((s - mean_spread) ** 2 for s in recent_spreads) / len(recent_spreads)
            self.state.volatility_index = variance ** 0.5

            # Classify regime
            if self.state.volatility_index < self.low_vol_threshold:
                self.state.regime = MarketRegime.LOW_VOLATILITY
            elif self.state.volatility_index > self.high_vol_threshold:
                self.state.regime = MarketRegime.HIGH_VOLATILITY
            else:
                self.state.regime = MarketRegime.NORMAL

        # Update factors
        self._update_factors()

        # Keep limited history
        max_history = 100
        if len(self.spread_history) > max_history:
            self.spread_history = self.spread_history[-max_history:]
        if len(self.volume_history) > max_history:
            self.volume_history = self.volume_history[-max_history:]

    def _update_factors(self):
        """Update market factors"""
        self.factors = []

        # Liquidity factor
        liquidity_impact = {
            LiquidityLevel.VERY_LOW: -0.5,
            LiquidityLevel.LOW: -0.2,
            LiquidityLevel.NORMAL: 0.0,
            LiquidityLevel.HIGH: 0.2,
        }

        self.factors.append(MarketFactor(
            name="liquidity",
            value=self.state.market_depth,
            impact=liquidity_impact.get(self.state.liquidity, 0),
            confidence=0.8,
            source="market_depth"
        ))

        # Volatility factor
        vol_impact = -0.3 if self.state.regime == MarketRegime.HIGH_VOLATILITY else 0.0

        self.factors.append(MarketFactor(
            name="volatility",
            value=self.state.volatility_index,
            impact=vol_impact,
            confidence=0.7,
            source="spread_volatility"
        ))

        # Spread factor
        spread_impact = -0.4 if self.state.avg_spread > 0.05 else 0.0

        self.factors.append(MarketFactor(
            name="spread",
            value=self.state.avg_spread,
            impact=spread_impact,
            confidence=0.9,
            source="bid_ask_spread"
        ))

    def get_market_score(self) -> float:
        """
        Get overall market score (-1 to 1)
        Negative = unfavorable, Positive = favorable
        """
        if not self.factors:
            return 0.0

        weighted_sum = sum(f.impact * f.confidence for f in self.factors)
        total_confidence = sum(f.confidence for f in self.factors)

        if total_confidence == 0:
            return 0.0

        return weighted_sum / total_confidence

    def score_opportunity(self, opportunity: Any) -> float:
        """
        Score an opportunity based on market factors

        Returns:
            Score from 0 to 1 (higher = better)
        """
        base_score = 0.5

        # Adjust for market state
        market_adjustment = self.get_market_score() * 0.2

        # Adjust for liquidity
        opp_liquidity = getattr(opportunity, 'liquidity', 0)
        if hasattr(opportunity, 'market'):
            opp_liquidity = getattr(opportunity.market, 'liquidity', 0)

        if opp_liquidity < self.low_liquidity_threshold:
            liquidity_penalty = -0.2
        elif opp_liquidity > self.low_liquidity_threshold * 10:
            liquidity_penalty = 0.1
        else:
            liquidity_penalty = 0.0

        # Adjust for spread
        if self.state.avg_spread > 0.03:
            spread_penalty = -0.1
        else:
            spread_penalty = 0.0

        final_score = base_score + market_adjustment + liquidity_penalty + spread_penalty

        return max(0.0, min(1.0, final_score))

    def should_trade(self) -> bool:
        """Check if market conditions are favorable for trading"""
        # Don't trade in very low liquidity
        if self.state.liquidity == LiquidityLevel.VERY_LOW:
            return False

        # Reduce trading in high volatility
        if self.state.regime == MarketRegime.HIGH_VOLATILITY:
            return False

        # Check spreads
        if self.state.avg_spread > 0.10:  # 10% spread is too high
            return False

        return True

    def get_state(self) -> Dict:
        """Get current market state"""
        return {
            'state': self.state.to_dict(),
            'factors': [f.to_dict() for f in self.factors],
            'market_score': self.get_market_score(),
            'should_trade': self.should_trade()
        }

    def print_report(self):
        """Print market factor report"""
        state = self.get_state()

        print("\n" + "=" * 50)
        print("MARKET FACTORS REPORT")
        print("=" * 50)
        print(f"Regime:           {self.state.regime.value}")
        print(f"Liquidity:        {self.state.liquidity.value}")
        print(f"Avg Spread:       {self.state.avg_spread:.4f}")
        print(f"Volatility Index: {self.state.volatility_index:.4f}")
        print(f"Market Score:     {state['market_score']:.2f}")
        print(f"Should Trade:     {state['should_trade']}")

        if self.factors:
            print("\nFactors:")
            for f in self.factors:
                print(f"  {f.name}: {f.value:.4f} (impact: {f.impact:+.2f})")

        print("=" * 50 + "\n")
