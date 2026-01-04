"""
Sentiment Analysis Module
Analyzes market sentiment from various sources
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SentimentLevel(Enum):
    """Sentiment classification"""
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


@dataclass
class SentimentSignal:
    """Individual sentiment signal"""
    source: str
    value: float  # -1 to 1
    confidence: float  # 0 to 1
    timestamp: datetime
    metadata: Dict = None

    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'value': self.value,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata or {}
        }


@dataclass
class MarketSentiment:
    """Aggregated sentiment for a market"""
    market_id: str
    overall_sentiment: float  # -1 to 1
    sentiment_level: SentimentLevel
    signal_count: int
    confidence: float
    last_updated: datetime

    def to_dict(self) -> Dict:
        return {
            'market_id': self.market_id,
            'overall_sentiment': self.overall_sentiment,
            'sentiment_level': self.sentiment_level.value,
            'signal_count': self.signal_count,
            'confidence': self.confidence,
            'last_updated': self.last_updated.isoformat()
        }


class SentimentAnalyzer:
    """
    Analyzes market sentiment from multiple sources

    Features:
    - Price-based sentiment (momentum)
    - Volume-based sentiment
    - Spread-based sentiment
    - Combined sentiment scoring
    """

    def __init__(self):
        # Sentiment by market
        self.market_sentiments: Dict[str, MarketSentiment] = {}

        # Signals by market
        self.signals: Dict[str, List[SentimentSignal]] = {}

        # Global sentiment
        self.global_sentiment: float = 0.0
        self.global_confidence: float = 0.0

        # Configuration
        self.signal_expiry_hours = 24
        self.min_signals_for_confidence = 3

        logger.info("Sentiment Analyzer initialized")

    def record_signal(self, market_id: str, source: str, value: float,
                     confidence: float = 0.5, metadata: Dict = None):
        """
        Record a sentiment signal

        Args:
            market_id: Market identifier
            source: Signal source (price, volume, spread, external)
            value: Sentiment value (-1 to 1)
            confidence: Signal confidence (0 to 1)
            metadata: Additional metadata
        """
        signal = SentimentSignal(
            source=source,
            value=max(-1.0, min(1.0, value)),
            confidence=max(0.0, min(1.0, confidence)),
            timestamp=datetime.now(),
            metadata=metadata
        )

        if market_id not in self.signals:
            self.signals[market_id] = []

        self.signals[market_id].append(signal)

        # Update market sentiment
        self._update_market_sentiment(market_id)

    def _update_market_sentiment(self, market_id: str):
        """Update aggregated sentiment for a market"""
        if market_id not in self.signals:
            return

        # Get recent signals
        cutoff = datetime.now() - timedelta(hours=self.signal_expiry_hours)
        recent_signals = [s for s in self.signals[market_id] if s.timestamp > cutoff]

        if not recent_signals:
            return

        # Calculate weighted average
        weighted_sum = sum(s.value * s.confidence for s in recent_signals)
        total_confidence = sum(s.confidence for s in recent_signals)

        if total_confidence == 0:
            return

        overall = weighted_sum / total_confidence
        avg_confidence = total_confidence / len(recent_signals)

        # Adjust confidence based on signal count
        if len(recent_signals) < self.min_signals_for_confidence:
            avg_confidence *= len(recent_signals) / self.min_signals_for_confidence

        # Determine sentiment level
        if overall <= -0.6:
            level = SentimentLevel.VERY_BEARISH
        elif overall <= -0.2:
            level = SentimentLevel.BEARISH
        elif overall >= 0.6:
            level = SentimentLevel.VERY_BULLISH
        elif overall >= 0.2:
            level = SentimentLevel.BULLISH
        else:
            level = SentimentLevel.NEUTRAL

        self.market_sentiments[market_id] = MarketSentiment(
            market_id=market_id,
            overall_sentiment=overall,
            sentiment_level=level,
            signal_count=len(recent_signals),
            confidence=avg_confidence,
            last_updated=datetime.now()
        )

        # Cleanup old signals
        self.signals[market_id] = recent_signals

    def analyze_market(self, market: Any) -> Optional[MarketSentiment]:
        """
        Analyze sentiment for a market based on its data

        Args:
            market: Market object with price/volume data

        Returns:
            MarketSentiment or None
        """
        market_id = getattr(market, 'condition_id', getattr(market, 'market_id', ''))
        if not market_id:
            return None

        # Price-based sentiment (momentum)
        if hasattr(market, 'outcome_prices') and len(market.outcome_prices) >= 2:
            yes_price = market.outcome_prices[0]
            no_price = market.outcome_prices[1]

            # Price momentum sentiment
            # Prices near extremes indicate strong sentiment
            if yes_price > 0.8:
                price_sentiment = 0.6 + (yes_price - 0.8) * 2
            elif yes_price < 0.2:
                price_sentiment = -0.6 - (0.2 - yes_price) * 2
            else:
                price_sentiment = (yes_price - 0.5) * 2

            self.record_signal(market_id, 'price', price_sentiment, confidence=0.7)

            # Spread-based sentiment (market efficiency)
            spread = abs(1.0 - yes_price - no_price)
            if spread > 0.05:
                # Wide spread indicates uncertainty
                spread_sentiment = 0.0
                spread_confidence = 0.3
            else:
                # Tight spread indicates confident market
                spread_sentiment = price_sentiment * 0.5
                spread_confidence = 0.6

            self.record_signal(market_id, 'spread', spread_sentiment, confidence=spread_confidence)

        # Volume-based sentiment
        if hasattr(market, 'volume'):
            volume = float(market.volume)
            if hasattr(market, 'avg_volume'):
                avg_volume = float(market.avg_volume)
                volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

                # High volume with price direction
                if volume_ratio > 1.5:
                    # High volume amplifies price sentiment
                    volume_sentiment = getattr(self.market_sentiments.get(market_id),
                                              'overall_sentiment', 0) * 0.3
                    self.record_signal(market_id, 'volume', volume_sentiment, confidence=0.5)

        return self.market_sentiments.get(market_id)

    def get_market_sentiment(self, market_id: str) -> Optional[MarketSentiment]:
        """Get current sentiment for a market"""
        return self.market_sentiments.get(market_id)

    def get_sentiment_score(self, market_id: str) -> float:
        """Get sentiment score for a market (-1 to 1)"""
        sentiment = self.market_sentiments.get(market_id)
        if sentiment:
            return sentiment.overall_sentiment
        return 0.0

    def is_sentiment_aligned(self, market_id: str, side: str) -> bool:
        """
        Check if sentiment aligns with proposed trade side

        Args:
            market_id: Market identifier
            side: Trade side ('yes', 'no', 'buy', 'sell')

        Returns:
            True if sentiment supports the trade
        """
        sentiment = self.get_sentiment_score(market_id)

        if side.lower() in ['yes', 'buy', 'long']:
            return sentiment >= 0
        elif side.lower() in ['no', 'sell', 'short']:
            return sentiment <= 0

        return True

    def update_global_sentiment(self):
        """Update global market sentiment"""
        if not self.market_sentiments:
            self.global_sentiment = 0.0
            self.global_confidence = 0.0
            return

        # Calculate weighted average across all markets
        weighted_sum = sum(
            s.overall_sentiment * s.confidence
            for s in self.market_sentiments.values()
        )
        total_confidence = sum(s.confidence for s in self.market_sentiments.values())

        if total_confidence > 0:
            self.global_sentiment = weighted_sum / total_confidence
            self.global_confidence = total_confidence / len(self.market_sentiments)
        else:
            self.global_sentiment = 0.0
            self.global_confidence = 0.0

    def get_global_sentiment_level(self) -> SentimentLevel:
        """Get global sentiment classification"""
        if self.global_sentiment <= -0.6:
            return SentimentLevel.VERY_BEARISH
        elif self.global_sentiment <= -0.2:
            return SentimentLevel.BEARISH
        elif self.global_sentiment >= 0.6:
            return SentimentLevel.VERY_BULLISH
        elif self.global_sentiment >= 0.2:
            return SentimentLevel.BULLISH
        else:
            return SentimentLevel.NEUTRAL

    def get_state(self) -> Dict:
        """Get current sentiment state"""
        self.update_global_sentiment()

        return {
            'global_sentiment': self.global_sentiment,
            'global_level': self.get_global_sentiment_level().value,
            'global_confidence': self.global_confidence,
            'tracked_markets': len(self.market_sentiments),
            'markets': {k: v.to_dict() for k, v in self.market_sentiments.items()}
        }

    def print_report(self):
        """Print sentiment report"""
        state = self.get_state()

        print("\n" + "=" * 50)
        print("SENTIMENT REPORT")
        print("=" * 50)
        print(f"Global Sentiment:  {state['global_sentiment']:+.2f}")
        print(f"Global Level:      {state['global_level']}")
        print(f"Confidence:        {state['global_confidence']:.2f}")
        print(f"Tracked Markets:   {state['tracked_markets']}")

        if self.market_sentiments:
            print("\nTop Bullish Markets:")
            bullish = sorted(
                self.market_sentiments.items(),
                key=lambda x: x[1].overall_sentiment,
                reverse=True
            )[:3]
            for market_id, sent in bullish:
                print(f"  {market_id[:30]}: {sent.overall_sentiment:+.2f}")

            print("\nTop Bearish Markets:")
            bearish = sorted(
                self.market_sentiments.items(),
                key=lambda x: x[1].overall_sentiment
            )[:3]
            for market_id, sent in bearish:
                print(f"  {market_id[:30]}: {sent.overall_sentiment:+.2f}")

        print("=" * 50 + "\n")
