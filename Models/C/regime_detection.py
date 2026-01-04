"""
Market Regime Detection Module

Identifies market regimes and adapts trading parameters accordingly.

Regimes:
1. Trending (Hurst > 0.6) - momentum strategies work well
2. Mean-reverting (Hurst < 0.4) - mean reversion works well
3. High volatility - reduce position sizes
4. Low volatility - can increase confidence

CRITICAL: One-size-fits-all parameters don't work.
Trending markets need different params than choppy markets.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CHOPPY = "choppy"
    UNKNOWN = "unknown"


@dataclass
class RegimeParameters:
    """Trading parameters for a regime"""
    k_neighbors: int          # KNN neighbors
    confidence_threshold: float  # Min confidence to trade
    position_size_mult: float    # Position size multiplier
    stop_loss_mult: float        # Stop loss multiplier
    take_profit_mult: float      # Take profit multiplier
    prefer_momentum: bool        # Weight momentum vs mean reversion


# Optimal parameters by regime
REGIME_PARAMS = {
    MarketRegime.TRENDING_UP: RegimeParameters(
        k_neighbors=5,
        confidence_threshold=0.5,
        position_size_mult=1.2,
        stop_loss_mult=1.5,
        take_profit_mult=2.0,
        prefer_momentum=True
    ),
    MarketRegime.TRENDING_DOWN: RegimeParameters(
        k_neighbors=5,
        confidence_threshold=0.5,
        position_size_mult=1.0,
        stop_loss_mult=1.2,
        take_profit_mult=1.5,
        prefer_momentum=True
    ),
    MarketRegime.MEAN_REVERTING: RegimeParameters(
        k_neighbors=12,
        confidence_threshold=0.6,
        position_size_mult=1.0,
        stop_loss_mult=1.0,
        take_profit_mult=1.0,
        prefer_momentum=False
    ),
    MarketRegime.HIGH_VOLATILITY: RegimeParameters(
        k_neighbors=8,
        confidence_threshold=0.7,
        position_size_mult=0.5,
        stop_loss_mult=2.0,
        take_profit_mult=2.5,
        prefer_momentum=True
    ),
    MarketRegime.LOW_VOLATILITY: RegimeParameters(
        k_neighbors=10,
        confidence_threshold=0.4,
        position_size_mult=1.5,
        stop_loss_mult=0.8,
        take_profit_mult=1.2,
        prefer_momentum=False
    ),
    MarketRegime.CHOPPY: RegimeParameters(
        k_neighbors=15,
        confidence_threshold=0.75,
        position_size_mult=0.3,
        stop_loss_mult=1.5,
        take_profit_mult=1.0,
        prefer_momentum=False
    ),
    MarketRegime.UNKNOWN: RegimeParameters(
        k_neighbors=8,
        confidence_threshold=0.6,
        position_size_mult=0.8,
        stop_loss_mult=1.2,
        take_profit_mult=1.5,
        prefer_momentum=False
    ),
}


@dataclass
class RegimeAnalysis:
    """Complete regime analysis result"""
    regime: MarketRegime
    confidence: float
    hurst_exponent: float
    volatility_percentile: float
    trend_strength: float
    parameters: RegimeParameters
    indicators: Dict


class RegimeDetector:
    """
    Detects market regimes using multiple indicators.

    Methods:
    1. Hurst exponent - trending vs mean-reverting
    2. Volatility percentile - high vs low vol
    3. ADX - trend strength
    4. Return autocorrelation - regime persistence
    """

    def __init__(self, lookback: int = 100, vol_lookback: int = 252):
        """
        Args:
            lookback: Bars for regime calculation
            vol_lookback: Bars for volatility percentile
        """
        self.lookback = lookback
        self.vol_lookback = vol_lookback

    def calculate_hurst(self, prices: np.ndarray) -> float:
        """
        Calculate Hurst exponent using R/S analysis.

        H > 0.5 = trending (persistent)
        H = 0.5 = random walk
        H < 0.5 = mean-reverting (anti-persistent)
        """
        if len(prices) < 20:
            return 0.5

        # Log returns
        returns = np.diff(np.log(prices))

        # R/S analysis
        n = len(returns)
        max_k = min(n // 2, 50)

        rs_values = []
        k_values = []

        for k in range(10, max_k):
            # Split into k-sized chunks
            n_chunks = n // k

            if n_chunks < 2:
                continue

            rs_chunk = []
            for i in range(n_chunks):
                chunk = returns[i*k:(i+1)*k]
                mean = np.mean(chunk)

                # Cumulative deviation from mean
                cum_dev = np.cumsum(chunk - mean)

                # Range
                R = np.max(cum_dev) - np.min(cum_dev)

                # Standard deviation
                S = np.std(chunk, ddof=1)

                if S > 0:
                    rs_chunk.append(R / S)

            if rs_chunk:
                rs_values.append(np.mean(rs_chunk))
                k_values.append(k)

        if len(k_values) < 2:
            return 0.5

        # Linear regression in log space to get H
        log_k = np.log(k_values)
        log_rs = np.log(rs_values)

        # Hurst exponent is slope of log-log regression
        slope, _ = np.polyfit(log_k, log_rs, 1)

        return np.clip(slope, 0.1, 0.9)

    def calculate_volatility_percentile(self, close: pd.Series) -> float:
        """
        Calculate current volatility relative to historical.

        Returns percentile (0-1) where 1 = extreme high vol.
        """
        returns = close.pct_change().dropna()

        if len(returns) < 20:
            return 0.5

        # Rolling volatility
        current_vol = returns.iloc[-20:].std()
        historical_vols = returns.rolling(20).std().dropna()

        if len(historical_vols) < 10:
            return 0.5

        # Percentile rank
        percentile = (historical_vols < current_vol).mean()

        return percentile

    def calculate_trend_strength(self, high: pd.Series, low: pd.Series,
                                 close: pd.Series, period: int = 14) -> float:
        """
        Calculate ADX-based trend strength.

        Returns value 0-1 where 1 = strong trend.
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(span=period, adjust=False).mean()

        current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25

        # Normalize to 0-1 (ADX > 50 is very strong)
        return min(current_adx / 50, 1.0)

    def calculate_trend_direction(self, close: pd.Series, period: int = 20) -> float:
        """
        Calculate trend direction.

        Returns: -1 to 1 where 1 = strong uptrend.
        """
        if len(close) < period:
            return 0

        sma = close.rolling(period).mean()
        current = close.iloc[-1]
        current_sma = sma.iloc[-1]

        if pd.isna(current_sma):
            return 0

        deviation = (current - current_sma) / current_sma

        # Also consider slope of SMA
        sma_slope = (sma.iloc[-1] - sma.iloc[-period]) / sma.iloc[-period] if sma.iloc[-period] != 0 else 0

        direction = (deviation + sma_slope * 10) / 2

        return np.clip(direction, -1, 1)

    def calculate_autocorrelation(self, returns: pd.Series, lag: int = 1) -> float:
        """
        Calculate return autocorrelation.

        Positive = trending persistence
        Negative = mean reversion
        """
        if len(returns) < lag + 10:
            return 0

        returns = returns.dropna()
        autocorr = returns.autocorr(lag)

        return autocorr if not pd.isna(autocorr) else 0

    def detect_regime(self, ohlcv: pd.DataFrame) -> RegimeAnalysis:
        """
        Detect current market regime.

        Args:
            ohlcv: DataFrame with OHLCV data

        Returns:
            RegimeAnalysis with regime classification and parameters
        """
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']

        # Calculate indicators
        hurst = self.calculate_hurst(close.values[-self.lookback:])
        vol_pctl = self.calculate_volatility_percentile(close)
        trend_strength = self.calculate_trend_strength(high, low, close)
        trend_direction = self.calculate_trend_direction(close)

        returns = close.pct_change()
        autocorr = self.calculate_autocorrelation(returns)

        # Classify regime
        regime, confidence = self._classify_regime(
            hurst, vol_pctl, trend_strength, trend_direction, autocorr
        )

        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            hurst_exponent=hurst,
            volatility_percentile=vol_pctl,
            trend_strength=trend_strength,
            parameters=REGIME_PARAMS[regime],
            indicators={
                'hurst': hurst,
                'volatility_percentile': vol_pctl,
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'autocorrelation': autocorr
            }
        )

    def _classify_regime(self, hurst: float, vol_pctl: float,
                        trend_strength: float, trend_direction: float,
                        autocorr: float) -> Tuple[MarketRegime, float]:
        """
        Classify regime based on indicators.

        Returns (regime, confidence).
        """
        # High volatility takes precedence
        if vol_pctl > 0.85:
            return MarketRegime.HIGH_VOLATILITY, vol_pctl

        # Low volatility
        if vol_pctl < 0.15:
            return MarketRegime.LOW_VOLATILITY, 1 - vol_pctl

        # Trending vs mean-reverting based on Hurst
        if hurst > 0.6:
            # Strong trending
            if trend_direction > 0.2 and trend_strength > 0.5:
                confidence = (hurst - 0.5) * 2 * trend_strength
                return MarketRegime.TRENDING_UP, min(confidence, 0.95)
            elif trend_direction < -0.2 and trend_strength > 0.5:
                confidence = (hurst - 0.5) * 2 * trend_strength
                return MarketRegime.TRENDING_DOWN, min(confidence, 0.95)

        if hurst < 0.4:
            # Mean-reverting
            confidence = (0.5 - hurst) * 2
            return MarketRegime.MEAN_REVERTING, min(confidence, 0.95)

        # Choppy market (low trend strength, moderate vol)
        if trend_strength < 0.3 and 0.4 <= hurst <= 0.6:
            return MarketRegime.CHOPPY, 0.6

        # Default
        return MarketRegime.UNKNOWN, 0.5

    def get_adapted_parameters(self, ohlcv: pd.DataFrame) -> Dict:
        """
        Get trading parameters adapted to current regime.

        Convenience method that returns a parameter dict.
        """
        analysis = self.detect_regime(ohlcv)
        params = analysis.parameters

        return {
            'k_neighbors': params.k_neighbors,
            'confidence_threshold': params.confidence_threshold,
            'position_size_multiplier': params.position_size_mult,
            'stop_loss_multiplier': params.stop_loss_mult,
            'take_profit_multiplier': params.take_profit_mult,
            'prefer_momentum': params.prefer_momentum,
            'regime': analysis.regime.value,
            'regime_confidence': analysis.confidence,
            'indicators': analysis.indicators
        }


class AdaptiveParameterManager:
    """
    Manages parameter adaptation based on regime changes.

    Provides smooth transitions between parameter sets.
    """

    def __init__(self, detector: Optional[RegimeDetector] = None,
                 transition_speed: float = 0.1):
        """
        Args:
            detector: Regime detector instance
            transition_speed: How fast to adapt parameters (0-1)
        """
        self.detector = detector or RegimeDetector()
        self.transition_speed = transition_speed

        # Current smoothed parameters
        self.current_params = {
            'k_neighbors': 8,
            'confidence_threshold': 0.6,
            'position_size_mult': 1.0,
            'stop_loss_mult': 1.2,
            'take_profit_mult': 1.5,
        }

        self.regime_history: List[MarketRegime] = []

    def update(self, ohlcv: pd.DataFrame) -> Dict:
        """
        Update parameters based on current market data.

        Smoothly transitions between regimes.
        """
        analysis = self.detector.detect_regime(ohlcv)
        target_params = analysis.parameters

        self.regime_history.append(analysis.regime)
        if len(self.regime_history) > 100:
            self.regime_history.pop(0)

        # Smooth transition
        speed = self.transition_speed * analysis.confidence

        self.current_params['k_neighbors'] = int(
            (1 - speed) * self.current_params['k_neighbors'] +
            speed * target_params.k_neighbors
        )
        self.current_params['confidence_threshold'] = (
            (1 - speed) * self.current_params['confidence_threshold'] +
            speed * target_params.confidence_threshold
        )
        self.current_params['position_size_mult'] = (
            (1 - speed) * self.current_params['position_size_mult'] +
            speed * target_params.position_size_mult
        )

        return {
            **self.current_params,
            'regime': analysis.regime,
            'regime_confidence': analysis.confidence,
            'prefer_momentum': target_params.prefer_momentum
        }

    def get_regime_distribution(self) -> Dict[str, float]:
        """Get distribution of recent regimes"""
        if not self.regime_history:
            return {}

        counts = {}
        for regime in self.regime_history:
            counts[regime.value] = counts.get(regime.value, 0) + 1

        total = len(self.regime_history)
        return {k: v / total for k, v in counts.items()}


def test_regime_detection():
    """Test the regime detection module"""
    print("=== Regime Detection Test ===\n")

    # Generate different market conditions
    np.random.seed(42)

    def generate_trending(n, drift):
        returns = drift + np.random.randn(n) * 0.01
        return 100 * np.exp(np.cumsum(returns))

    def generate_mean_reverting(n, strength=0.1):
        prices = [100]
        for _ in range(n - 1):
            ret = -strength * (prices[-1] - 100) / 100 + np.random.randn() * 0.01
            prices.append(prices[-1] * (1 + ret))
        return np.array(prices)

    def generate_volatile(n, vol=0.03):
        returns = np.random.randn(n) * vol
        return 100 * np.exp(np.cumsum(returns))

    detector = RegimeDetector(lookback=50)

    # Test trending up
    print("Testing Trending Up Market:")
    prices = generate_trending(200, drift=0.002)
    ohlcv = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * (1 + np.abs(np.random.randn(200) * 0.005)),
        'low': prices * (1 - np.abs(np.random.randn(200) * 0.005)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 200)
    })
    analysis = detector.detect_regime(ohlcv)
    print(f"  Regime: {analysis.regime.value}")
    print(f"  Hurst: {analysis.hurst_exponent:.3f}")
    print(f"  Confidence: {analysis.confidence:.2f}")
    print(f"  Recommended k: {analysis.parameters.k_neighbors}")

    # Test mean-reverting
    print("\nTesting Mean-Reverting Market:")
    prices = generate_mean_reverting(200, strength=0.2)
    ohlcv = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * (1 + np.abs(np.random.randn(200) * 0.005)),
        'low': prices * (1 - np.abs(np.random.randn(200) * 0.005)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 200)
    })
    analysis = detector.detect_regime(ohlcv)
    print(f"  Regime: {analysis.regime.value}")
    print(f"  Hurst: {analysis.hurst_exponent:.3f}")
    print(f"  Confidence: {analysis.confidence:.2f}")
    print(f"  Recommended k: {analysis.parameters.k_neighbors}")

    # Test high volatility
    print("\nTesting High Volatility Market:")
    prices = generate_volatile(200, vol=0.05)
    ohlcv = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * (1 + np.abs(np.random.randn(200) * 0.02)),
        'low': prices * (1 - np.abs(np.random.randn(200) * 0.02)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 200)
    })
    analysis = detector.detect_regime(ohlcv)
    print(f"  Regime: {analysis.regime.value}")
    print(f"  Vol Percentile: {analysis.volatility_percentile:.2f}")
    print(f"  Confidence: {analysis.confidence:.2f}")
    print(f"  Position Mult: {analysis.parameters.position_size_mult}")


if __name__ == "__main__":
    test_regime_detection()
