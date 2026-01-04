"""
Enhanced Technical Indicators for ML Trading Engine

Provides a comprehensive set of technical indicators for the ML classifier.
Each indicator is normalized and weighted for ensemble use.

Key Features:
1. All indicators are properly normalized (0-1 or z-scored)
2. Explicit lookback periods (no lookahead)
3. Weight recommendations based on predictive power
4. Missing value handling

Current Features (5): RSI, ADX, Nadaraya-Watson, CCI, WT
Enhanced Features (11+): Adding MACD, Bollinger, Volume Profile, OBV, ATR, Divergence
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class IndicatorConfig:
    """Configuration for an indicator"""
    name: str
    weight: float  # Recommended weight for ensemble (0-1)
    lookback: int  # Required bars of history
    description: str


class TechnicalIndicators:
    """
    Comprehensive technical indicator calculator.

    All methods return normalized values suitable for ML consumption.
    No lookahead bias - all calculations use only past data.
    """

    # Indicator metadata with recommended weights
    INDICATORS = {
        'rsi': IndicatorConfig('RSI', 0.85, 14, 'Relative Strength Index'),
        'adx': IndicatorConfig('ADX', 0.75, 20, 'Average Directional Index'),
        'cci': IndicatorConfig('CCI', 0.70, 20, 'Commodity Channel Index'),
        'wt': IndicatorConfig('WaveTrend', 0.80, 21, 'WaveTrend Oscillator'),
        'macd': IndicatorConfig('MACD', 0.90, 34, 'MACD Histogram'),
        'bbands': IndicatorConfig('Bollinger %B', 0.80, 20, 'Bollinger Band Position'),
        'vwap_dev': IndicatorConfig('VWAP Dev', 0.70, 50, 'VWAP Deviation'),
        'obv': IndicatorConfig('OBV', 0.60, 20, 'On-Balance Volume Normalized'),
        'atr': IndicatorConfig('ATR', 0.70, 14, 'Average True Range Normalized'),
        'divergence': IndicatorConfig('Divergence', 0.50, 14, 'RSI-Price Divergence'),
        'volume_profile': IndicatorConfig('Vol Profile', 0.65, 20, 'Volume Profile Position'),
    }

    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: Whether to normalize outputs to 0-1 range
        """
        self.normalize = normalize

    # ========== Original Indicators (from start.py) ==========

    def rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index - momentum oscillator.

        Output: 0-100, normalized to 0-1 if enabled
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        if self.normalize:
            rsi = rsi / 100

        return rsi.fillna(0.5)

    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> pd.Series:
        """
        Average Directional Index - trend strength indicator.

        Output: 0-100, normalized to 0-1 if enabled
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

        if self.normalize:
            adx = adx / 100

        return adx.fillna(0)

    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 20) -> pd.Series:
        """
        Commodity Channel Index - momentum/trend indicator.

        Output: Typically -200 to +200, normalized to 0-1 using sigmoid
        """
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())

        cci = (typical_price - sma) / (0.015 * mad)

        if self.normalize:
            # Sigmoid normalization for unbounded indicator
            cci = 1 / (1 + np.exp(-cci / 100))

        return cci.fillna(0.5)

    def wave_trend(self, high: pd.Series, low: pd.Series, close: pd.Series,
                   n1: int = 10, n2: int = 21) -> pd.Series:
        """
        WaveTrend Oscillator - momentum with smoothing.

        Output: Typically -100 to +100, normalized to 0-1
        """
        hlc3 = (high + low + close) / 3
        esa = hlc3.ewm(span=n1, adjust=False).mean()
        d = abs(hlc3 - esa).ewm(span=n1, adjust=False).mean()
        ci = (hlc3 - esa) / (0.015 * d)
        wt1 = ci.ewm(span=n2, adjust=False).mean()

        if self.normalize:
            # Clip and normalize
            wt1 = np.clip(wt1, -100, 100)
            wt1 = (wt1 + 100) / 200

        return wt1.fillna(0.5)

    # ========== NEW Enhanced Indicators ==========

    def macd_histogram(self, close: pd.Series, fast: int = 12,
                       slow: int = 26, signal: int = 9) -> pd.Series:
        """
        MACD Histogram - trend momentum indicator.

        Measures the difference between MACD line and signal line.
        Positive = bullish momentum, Negative = bearish momentum.

        Output: Normalized to 0-1 using z-score sigmoid
        """
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        if self.normalize:
            # Normalize by price to make comparable across assets
            histogram_pct = histogram / close.rolling(20).mean()
            # Z-score and sigmoid
            mean = histogram_pct.rolling(100).mean()
            std = histogram_pct.rolling(100).std()
            z_score = (histogram_pct - mean) / std.replace(0, np.nan)
            histogram = 1 / (1 + np.exp(-z_score))

        return histogram.fillna(0.5)

    def bollinger_percent_b(self, close: pd.Series, period: int = 20,
                            num_std: float = 2.0) -> pd.Series:
        """
        Bollinger Band %B - position within bands.

        %B = (Price - Lower Band) / (Upper Band - Lower Band)
        0 = at lower band, 0.5 = at middle, 1 = at upper band
        Can exceed 0-1 range when price outside bands.

        Output: Clipped to 0-1
        """
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()

        upper = sma + num_std * std
        lower = sma - num_std * std

        percent_b = (close - lower) / (upper - lower).replace(0, np.nan)

        if self.normalize:
            percent_b = np.clip(percent_b, 0, 1)

        return percent_b.fillna(0.5)

    def vwap_deviation(self, high: pd.Series, low: pd.Series, close: pd.Series,
                       volume: pd.Series, period: int = 50) -> pd.Series:
        """
        VWAP Deviation - distance from volume-weighted average price.

        Positive = price above VWAP (bullish), Negative = below (bearish)

        Output: Normalized deviation in standard deviations, then sigmoid
        """
        typical_price = (high + low + close) / 3

        # Rolling VWAP
        cum_vol = volume.rolling(window=period).sum()
        cum_tp_vol = (typical_price * volume).rolling(window=period).sum()
        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)

        # Deviation from VWAP
        deviation = (close - vwap) / vwap

        if self.normalize:
            # Normalize using rolling statistics
            mean = deviation.rolling(100).mean()
            std = deviation.rolling(100).std()
            z_score = (deviation - mean) / std.replace(0, np.nan)
            deviation = 1 / (1 + np.exp(-z_score * 2))

        return deviation.fillna(0.5)

    def obv_normalized(self, close: pd.Series, volume: pd.Series,
                       period: int = 20) -> pd.Series:
        """
        On-Balance Volume (normalized) - volume flow indicator.

        Tracks cumulative volume based on price direction.

        Output: Rate of change normalized to 0-1
        """
        # Calculate OBV
        direction = np.sign(close.diff())
        obv = (volume * direction).cumsum()

        # Normalize using rate of change
        obv_roc = obv.diff(period) / obv.shift(period).abs().replace(0, np.nan)

        if self.normalize:
            # Clip extreme values and scale
            obv_roc = np.clip(obv_roc, -1, 1)
            obv_roc = (obv_roc + 1) / 2

        return obv_roc.fillna(0.5)

    def atr_normalized(self, high: pd.Series, low: pd.Series, close: pd.Series,
                       period: int = 14) -> pd.Series:
        """
        Average True Range (normalized) - volatility indicator.

        Higher values = higher volatility.

        Output: ATR as percentage of price, normalized to 0-1
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(span=period, adjust=False).mean()

        # Normalize as percentage of price
        atr_pct = atr / close

        if self.normalize:
            # Scale to typical range (0-5% ATR)
            atr_pct = np.clip(atr_pct / 0.05, 0, 1)

        return atr_pct.fillna(0)

    def rsi_price_divergence(self, close: pd.Series, period: int = 14,
                             lookback: int = 10) -> pd.Series:
        """
        RSI-Price Divergence detector.

        Detects when price and RSI are moving in opposite directions,
        which often precedes reversals.

        Output: -1 to +1, normalized to 0-1
        - Bullish divergence (price down, RSI up): > 0.5
        - Bearish divergence (price up, RSI down): < 0.5
        """
        rsi_values = self.rsi(close, period)

        # Calculate slopes
        price_slope = close.diff(lookback) / close.shift(lookback)
        rsi_slope = rsi_values.diff(lookback)

        # Divergence score
        # Negative when both moving same direction, positive when diverging
        price_direction = np.sign(price_slope)
        rsi_direction = np.sign(rsi_slope)

        # Divergence strength
        divergence = -price_direction * rsi_direction * (
            abs(price_slope) * abs(rsi_slope)
        ).clip(upper=0.1)

        if self.normalize:
            # Scale to 0-1
            divergence = (divergence + 0.1) / 0.2
            divergence = np.clip(divergence, 0, 1)

        return divergence.fillna(0.5)

    def volume_profile_position(self, close: pd.Series, volume: pd.Series,
                                period: int = 20, bins: int = 10) -> pd.Series:
        """
        Volume Profile Position - where price sits relative to volume distribution.

        Identifies high-volume price nodes (support/resistance).

        Output: 0-1, position in volume distribution
        """
        result = pd.Series(index=close.index, dtype=float)

        for i in range(period, len(close)):
            window_close = close.iloc[i-period:i]
            window_volume = volume.iloc[i-period:i]

            # Create volume profile
            price_range = window_close.max() - window_close.min()
            if price_range == 0:
                result.iloc[i] = 0.5
                continue

            # Bin prices and sum volumes
            bin_edges = np.linspace(window_close.min(), window_close.max(), bins + 1)
            bin_indices = np.digitize(window_close, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, bins - 1)

            volume_per_bin = np.zeros(bins)
            for j, idx in enumerate(bin_indices):
                volume_per_bin[idx] += window_volume.iloc[j]

            # Find current price position relative to high-volume area
            current_price = close.iloc[i]
            total_vol = volume_per_bin.sum()

            if total_vol == 0:
                result.iloc[i] = 0.5
                continue

            # Cumulative volume below current price
            current_bin = np.digitize([current_price], bin_edges)[0] - 1
            current_bin = np.clip(current_bin, 0, bins - 1)
            vol_below = volume_per_bin[:current_bin + 1].sum()

            result.iloc[i] = vol_below / total_vol

        return result.fillna(0.5)

    # ========== Composite Feature Generator ==========

    def generate_all_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all technical indicators from OHLCV data.

        Args:
            ohlcv: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            DataFrame with all indicators as columns
        """
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        volume = ohlcv.get('volume', pd.Series(1, index=ohlcv.index))

        features = pd.DataFrame(index=ohlcv.index)

        # Original indicators
        features['rsi'] = self.rsi(close)
        features['adx'] = self.adx(high, low, close)
        features['cci'] = self.cci(high, low, close)
        features['wt'] = self.wave_trend(high, low, close)

        # New indicators
        features['macd'] = self.macd_histogram(close)
        features['bbands'] = self.bollinger_percent_b(close)
        features['vwap_dev'] = self.vwap_deviation(high, low, close, volume)
        features['obv'] = self.obv_normalized(close, volume)
        features['atr'] = self.atr_normalized(high, low, close)
        features['divergence'] = self.rsi_price_divergence(close)
        features['vol_profile'] = self.volume_profile_position(close, volume)

        return features

    def get_feature_weights(self) -> Dict[str, float]:
        """Get recommended weights for each feature"""
        return {
            'rsi': 0.85,
            'adx': 0.75,
            'cci': 0.70,
            'wt': 0.80,
            'macd': 0.90,
            'bbands': 0.80,
            'vwap_dev': 0.70,
            'obv': 0.60,
            'atr': 0.70,
            'divergence': 0.50,
            'vol_profile': 0.65,
        }

    def weighted_feature_array(self, ohlcv: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate weighted feature array for ML consumption.

        Returns:
            Tuple of (features array, weights array)
        """
        features = self.generate_all_features(ohlcv)
        weights = self.get_feature_weights()

        # Order features by weight (highest first)
        sorted_features = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        feature_cols = [f[0] for f in sorted_features]
        weight_values = np.array([f[1] for f in sorted_features])

        return features[feature_cols].values, weight_values


def calculate_lorentzian_distance(x1: np.ndarray, x2: np.ndarray,
                                  weights: Optional[np.ndarray] = None) -> float:
    """
    Calculate Lorentzian distance between two feature vectors.

    Lorentzian distance: sum(log(1 + |xi - yi|))
    More robust to outliers than Euclidean distance.

    Args:
        x1, x2: Feature vectors
        weights: Optional feature weights

    Returns:
        Lorentzian distance
    """
    diff = np.abs(x1 - x2)
    log_diff = np.log(1 + diff)

    if weights is not None:
        log_diff = log_diff * weights

    return np.sum(log_diff)


class FeatureSelector:
    """
    Feature selection utilities for ML models.

    Helps identify the most predictive features for a given target.
    """

    @staticmethod
    def correlation_filter(features: pd.DataFrame, target: pd.Series,
                           threshold: float = 0.1) -> List[str]:
        """Select features with sufficient correlation to target"""
        correlations = features.corrwith(target).abs()
        selected = correlations[correlations > threshold].index.tolist()
        return selected

    @staticmethod
    def remove_multicollinear(features: pd.DataFrame,
                              threshold: float = 0.9) -> List[str]:
        """Remove highly correlated features to reduce redundancy"""
        corr_matrix = features.corr().abs()

        # Upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features with correlation > threshold
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

        return [col for col in features.columns if col not in to_drop]


def test_indicators():
    """Test the indicator calculations"""
    # Generate sample data
    np.random.seed(42)
    n = 200

    # Random walk with drift
    returns = np.random.randn(n) * 0.02 + 0.0005
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    volume = np.random.randint(1000, 10000, n).astype(float)

    ohlcv = pd.DataFrame({
        'open': close * 0.999,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    # Calculate indicators
    indicators = TechnicalIndicators(normalize=True)
    features = indicators.generate_all_features(ohlcv)

    print("=== Technical Indicators Test ===\n")
    print(f"Generated {len(features.columns)} features for {len(features)} bars\n")

    print("Feature Statistics:")
    print("-" * 60)
    for col in features.columns:
        values = features[col].dropna()
        print(f"{col:15s}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")

    print("\n" + "-" * 60)
    print("Feature Weights:")
    weights = indicators.get_feature_weights()
    for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:15s}: {weight:.2f}")

    # Test Lorentzian distance
    print("\n" + "-" * 60)
    print("Lorentzian Distance Test:")
    x1 = features.iloc[100].values
    x2 = features.iloc[101].values
    x3 = features.iloc[150].values

    dist_close = calculate_lorentzian_distance(x1, x2)
    dist_far = calculate_lorentzian_distance(x1, x3)
    print(f"  Adjacent bars: {dist_close:.4f}")
    print(f"  Distant bars:  {dist_far:.4f}")


if __name__ == "__main__":
    test_indicators()
