"""
Sentiment Signals Integration for ERC Trading System

Provides sentiment analysis from:
- Twitter/X API
- News headlines
- Reddit (for crypto/stocks)
- Custom sources

CRITICAL: Sentiment is a leading indicator for prediction markets.
Social buzz often precedes price moves.
"""

import re
import asyncio
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SentimentSignal:
    """Single sentiment data point"""
    source: str           # 'twitter', 'news', 'reddit'
    text: str
    timestamp: datetime
    sentiment_score: float  # -1 to +1
    confidence: float       # 0 to 1
    engagement: int        # Likes, retweets, etc.
    keywords: List[str] = field(default_factory=list)


@dataclass
class MarketSentiment:
    """Aggregated sentiment for a market"""
    market_id: str
    keywords: List[str]
    overall_score: float    # -1 to +1
    confidence: float       # 0 to 1
    signal_count: int
    bullish_pct: float
    bearish_pct: float
    neutral_pct: float
    trending_score: float   # 0 to 1 (how much buzz)
    change_1h: float       # Sentiment change in last hour
    change_24h: float      # Sentiment change in last 24h
    sources: Dict[str, float]  # Score by source
    last_updated: datetime


class SimpleSentimentAnalyzer:
    """
    Simple rule-based sentiment analyzer.

    Uses keyword matching and basic NLP patterns.
    For production, consider using transformers or sentiment APIs.
    """

    # Positive sentiment words
    POSITIVE_WORDS = {
        'win', 'wins', 'winning', 'won', 'victory', 'bullish', 'up', 'rise',
        'rising', 'surge', 'surging', 'pump', 'moon', 'good', 'great', 'best',
        'positive', 'confident', 'confirmed', 'success', 'successful', 'ahead',
        'leads', 'leading', 'likely', 'probable', 'strong', 'higher', 'increase',
        'increased', 'rally', 'breakout', 'breakthrough', 'approve', 'approved'
    }

    # Negative sentiment words
    NEGATIVE_WORDS = {
        'lose', 'loses', 'losing', 'lost', 'defeat', 'bearish', 'down', 'fall',
        'falling', 'crash', 'crashing', 'dump', 'bad', 'worst', 'negative',
        'uncertain', 'denied', 'fail', 'failure', 'behind', 'trails', 'unlikely',
        'improbable', 'weak', 'lower', 'decrease', 'decreased', 'selloff',
        'reject', 'rejected', 'scandal', 'problem', 'crisis'
    }

    # Intensifiers
    INTENSIFIERS = {
        'very', 'extremely', 'highly', 'strongly', 'definitely', 'absolutely',
        'certainly', 'clearly', 'obviously', 'significantly', 'massively'
    }

    # Negators
    NEGATORS = {'not', "n't", 'no', 'never', 'neither', 'nobody', 'nothing'}

    def analyze(self, text: str) -> tuple:
        """
        Analyze text sentiment.

        Returns:
            (sentiment_score, confidence)
            sentiment_score: -1 to +1
            confidence: 0 to 1
        """
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)

        positive_count = 0
        negative_count = 0
        intensifier_next = False
        negate_next = False

        for i, word in enumerate(words):
            # Check for intensifiers
            if word in self.INTENSIFIERS:
                intensifier_next = True
                continue

            # Check for negators
            if word in self.NEGATORS:
                negate_next = True
                continue

            # Calculate word contribution
            multiplier = 1.5 if intensifier_next else 1.0

            if word in self.POSITIVE_WORDS:
                if negate_next:
                    negative_count += multiplier
                else:
                    positive_count += multiplier
            elif word in self.NEGATIVE_WORDS:
                if negate_next:
                    positive_count += multiplier
                else:
                    negative_count += multiplier

            intensifier_next = False
            negate_next = False

        total = positive_count + negative_count

        if total == 0:
            return 0.0, 0.3  # Neutral with low confidence

        sentiment = (positive_count - negative_count) / total
        confidence = min(total / 10, 1.0)  # More words = higher confidence

        return sentiment, confidence


class SentimentAggregator:
    """
    Aggregates sentiment signals for markets.

    Tracks sentiment over time and detects changes.
    """

    def __init__(self, analyzer: Optional[SimpleSentimentAnalyzer] = None,
                 decay_hours: float = 24):
        """
        Args:
            analyzer: Sentiment analyzer to use
            decay_hours: Hours after which old signals decay
        """
        self.analyzer = analyzer or SimpleSentimentAnalyzer()
        self.decay_hours = decay_hours

        # Store signals by market
        self._signals: Dict[str, List[SentimentSignal]] = defaultdict(list)

        # Market keyword mapping
        self._market_keywords: Dict[str, List[str]] = {}

    def set_market_keywords(self, market_id: str, keywords: List[str]):
        """Set keywords to track for a market"""
        self._market_keywords[market_id] = [k.lower() for k in keywords]

    def add_signal(self, signal: SentimentSignal):
        """Add a sentiment signal"""
        # Find matching markets
        for market_id, keywords in self._market_keywords.items():
            if any(kw in signal.text.lower() for kw in keywords):
                signal.keywords = [kw for kw in keywords if kw in signal.text.lower()]
                self._signals[market_id].append(signal)

    def process_text(self, text: str, source: str,
                     engagement: int = 0) -> Optional[SentimentSignal]:
        """
        Process text and create sentiment signal.

        Args:
            text: Text to analyze
            source: Source of text
            engagement: Engagement metric

        Returns:
            SentimentSignal or None if no matching market
        """
        sentiment, confidence = self.analyzer.analyze(text)

        signal = SentimentSignal(
            source=source,
            text=text,
            timestamp=datetime.now(),
            sentiment_score=sentiment,
            confidence=confidence,
            engagement=engagement
        )

        self.add_signal(signal)
        return signal

    def get_market_sentiment(self, market_id: str) -> Optional[MarketSentiment]:
        """
        Get aggregated sentiment for a market.

        Args:
            market_id: Market to get sentiment for

        Returns:
            MarketSentiment or None if no data
        """
        signals = self._signals.get(market_id, [])

        if not signals:
            return None

        # Filter to recent signals
        cutoff = datetime.now() - timedelta(hours=self.decay_hours)
        recent_signals = [s for s in signals if s.timestamp > cutoff]

        if not recent_signals:
            return None

        # Calculate time-weighted scores
        now = datetime.now()
        weighted_scores = []
        weights = []
        source_scores = defaultdict(list)

        for signal in recent_signals:
            age_hours = (now - signal.timestamp).total_seconds() / 3600
            time_weight = np.exp(-age_hours / self.decay_hours)
            engagement_weight = np.log1p(signal.engagement) / 10

            weight = time_weight * signal.confidence * (1 + engagement_weight)
            weighted_scores.append(signal.sentiment_score * weight)
            weights.append(weight)
            source_scores[signal.source].append(signal.sentiment_score)

        total_weight = sum(weights)
        overall_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0

        # Calculate sentiment distribution
        bullish = sum(1 for s in recent_signals if s.sentiment_score > 0.1)
        bearish = sum(1 for s in recent_signals if s.sentiment_score < -0.1)
        neutral = len(recent_signals) - bullish - bearish

        total = len(recent_signals)
        bullish_pct = bullish / total
        bearish_pct = bearish / total
        neutral_pct = neutral / total

        # Trending score (signal volume)
        hour_ago = now - timedelta(hours=1)
        recent_hour = sum(1 for s in recent_signals if s.timestamp > hour_ago)
        trending = min(recent_hour / 10, 1.0)

        # Calculate changes
        change_1h = self._calculate_change(recent_signals, hours=1)
        change_24h = self._calculate_change(recent_signals, hours=24)

        # Source breakdown
        sources = {
            source: np.mean(scores) if scores else 0
            for source, scores in source_scores.items()
        }

        return MarketSentiment(
            market_id=market_id,
            keywords=self._market_keywords.get(market_id, []),
            overall_score=overall_score,
            confidence=min(total / 20, 1.0),
            signal_count=total,
            bullish_pct=bullish_pct,
            bearish_pct=bearish_pct,
            neutral_pct=neutral_pct,
            trending_score=trending,
            change_1h=change_1h,
            change_24h=change_24h,
            sources=sources,
            last_updated=now
        )

    def _calculate_change(self, signals: List[SentimentSignal],
                          hours: int) -> float:
        """Calculate sentiment change over period"""
        now = datetime.now()
        cutoff = now - timedelta(hours=hours)
        mid = now - timedelta(hours=hours/2)

        recent = [s.sentiment_score for s in signals if s.timestamp > mid]
        older = [s.sentiment_score for s in signals if cutoff < s.timestamp <= mid]

        if not recent or not older:
            return 0.0

        return np.mean(recent) - np.mean(older)

    def get_all_sentiments(self) -> Dict[str, MarketSentiment]:
        """Get sentiment for all tracked markets"""
        return {
            market_id: self.get_market_sentiment(market_id)
            for market_id in self._market_keywords
            if self.get_market_sentiment(market_id) is not None
        }

    def cleanup_old_signals(self):
        """Remove signals older than decay period"""
        cutoff = datetime.now() - timedelta(hours=self.decay_hours * 2)

        for market_id in self._signals:
            self._signals[market_id] = [
                s for s in self._signals[market_id]
                if s.timestamp > cutoff
            ]


class SentimentIntegration:
    """
    High-level sentiment integration for trading.

    Provides trading signals based on sentiment.
    """

    def __init__(self, aggregator: Optional[SentimentAggregator] = None):
        self.aggregator = aggregator or SentimentAggregator()

    def setup_market(self, market_id: str, question: str,
                     additional_keywords: Optional[List[str]] = None):
        """
        Setup sentiment tracking for a market.

        Automatically extracts keywords from question.
        """
        # Extract keywords from question
        words = re.findall(r'\b[A-Z][a-z]+\b', question)  # Capitalized words
        numbers = re.findall(r'\b\d{4}\b', question)      # Years
        keywords = list(set(words + numbers))

        if additional_keywords:
            keywords.extend(additional_keywords)

        self.aggregator.set_market_keywords(market_id, keywords)

    def get_trading_signal(self, market_id: str,
                           market_price: float) -> Optional[Dict]:
        """
        Get trading signal based on sentiment.

        Args:
            market_id: Market identifier
            market_price: Current market price (0-1)

        Returns:
            Trading signal dict or None
        """
        sentiment = self.aggregator.get_market_sentiment(market_id)

        if sentiment is None or sentiment.confidence < 0.3:
            return None

        # Convert sentiment to implied probability adjustment
        # Positive sentiment suggests higher probability
        sentiment_adjustment = sentiment.overall_score * 0.10  # Max 10% adjustment

        # Trend momentum
        if sentiment.change_1h > 0.2 and sentiment.change_24h > 0.1:
            momentum = 'bullish'
        elif sentiment.change_1h < -0.2 and sentiment.change_24h < -0.1:
            momentum = 'bearish'
        else:
            momentum = 'neutral'

        # Generate signal
        implied_prob = market_price + sentiment_adjustment

        # Determine action
        edge = implied_prob - market_price

        if abs(edge) < 0.05 or sentiment.confidence < 0.5:
            action = 'hold'
        elif edge > 0.05:
            action = 'buy'
        else:
            action = 'sell'

        return {
            'market_id': market_id,
            'action': action,
            'sentiment_score': sentiment.overall_score,
            'sentiment_confidence': sentiment.confidence,
            'implied_adjustment': sentiment_adjustment,
            'momentum': momentum,
            'bullish_pct': sentiment.bullish_pct,
            'bearish_pct': sentiment.bearish_pct,
            'trending': sentiment.trending_score,
            'signal_count': sentiment.signal_count
        }


def test_sentiment():
    """Test sentiment analysis"""
    print("=== Sentiment Analysis Test ===\n")

    # Test analyzer
    analyzer = SimpleSentimentAnalyzer()

    test_texts = [
        "Trump is winning by a huge margin!",
        "The market is crashing, very bearish outlook.",
        "Election results are uncertain at this point.",
        "Not looking good for the incumbent.",
        "Extremely bullish on Bitcoin reaching new highs!",
    ]

    print("Text Analysis:")
    for text in test_texts:
        score, conf = analyzer.analyze(text)
        sentiment = "Bullish" if score > 0.1 else "Bearish" if score < -0.1 else "Neutral"
        print(f"  \"{text[:50]}...\"")
        print(f"    -> {sentiment} (score: {score:.2f}, conf: {conf:.2f})\n")

    # Test aggregator
    print("\n--- Aggregator Test ---")
    aggregator = SentimentAggregator()

    # Setup market
    aggregator.set_market_keywords('election_2024', ['trump', 'biden', 'election', '2024'])

    # Add signals
    signals = [
        ("Trump rally draws massive crowd, supporters confident", 'twitter', 1500),
        ("Poll shows tight race, Biden closing gap", 'news', 0),
        ("Election betting markets surge for Trump", 'twitter', 3000),
        ("Negative headlines for Trump campaign", 'news', 0),
        ("Bullish sentiment on Polymarket for Trump", 'twitter', 500),
    ]

    for text, source, engagement in signals:
        aggregator.process_text(text, source, engagement)

    # Get sentiment
    sentiment = aggregator.get_market_sentiment('election_2024')

    if sentiment:
        print(f"\nMarket Sentiment for 'election_2024':")
        print(f"  Overall Score: {sentiment.overall_score:.2f}")
        print(f"  Confidence: {sentiment.confidence:.2f}")
        print(f"  Bullish: {sentiment.bullish_pct:.1%}")
        print(f"  Bearish: {sentiment.bearish_pct:.1%}")
        print(f"  Trending: {sentiment.trending_score:.2f}")
        print(f"  Sources: {sentiment.sources}")

    # Test integration
    print("\n--- Trading Signal ---")
    integration = SentimentIntegration(aggregator)
    signal = integration.get_trading_signal('election_2024', market_price=0.55)

    if signal:
        print(f"Action: {signal['action'].upper()}")
        print(f"Momentum: {signal['momentum']}")
        print(f"Implied adjustment: {signal['implied_adjustment']:+.1%}")


if __name__ == "__main__":
    test_sentiment()
