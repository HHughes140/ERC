"""
Tests for Factors/calibration.py
"""
import pytest


class TestHistoricalCalibrator:
    """Tests for HistoricalCalibrator"""

    def test_calibrator_initialization(self):
        """Test calibrator initializes correctly"""
        from Factors.calibration import HistoricalCalibrator

        calibrator = HistoricalCalibrator()
        assert calibrator is not None

    def test_calibrated_certainty(self):
        """Test certainty calibration"""
        from Factors.calibration import HistoricalCalibrator

        calibrator = HistoricalCalibrator()

        result = calibrator.get_calibrated_certainty(
            raw_price=0.95,
            market_type='directional',
            platform='polymarket'
        )

        assert 'calibrated_probability' in result
        assert 'confidence' in result
        assert 0 <= result['calibrated_probability'] <= 1
        assert 0 <= result['confidence'] <= 1

    def test_price_bucket_lookup(self):
        """Test price bucket lookup"""
        from Factors.calibration import HistoricalCalibrator

        calibrator = HistoricalCalibrator()

        # Different prices should map to different buckets
        low_result = calibrator.get_calibrated_certainty(0.50, 'directional', 'polymarket')
        high_result = calibrator.get_calibrated_certainty(0.95, 'directional', 'polymarket')

        # Both should return valid results
        assert low_result['calibrated_probability'] > 0
        assert high_result['calibrated_probability'] > 0


class TestFeeAdjustedProfitCalculator:
    """Tests for FeeAdjustedProfitCalculator"""

    def test_polymarket_fees(self):
        """Test Polymarket fee calculation"""
        from Factors.calibration import FeeAdjustedProfitCalculator

        calc = FeeAdjustedProfitCalculator('polymarket')

        result = calc.calculate_net_profit(
            entry_price=0.95,
            exit_price=1.0,
            won=True
        )

        assert 'net_profit_pct' in result
        assert 'gross_profit_pct' in result
        assert result['net_profit_pct'] <= result['gross_profit_pct']

    def test_kalshi_fees(self):
        """Test Kalshi fee calculation"""
        from Factors.calibration import FeeAdjustedProfitCalculator

        calc = FeeAdjustedProfitCalculator('kalshi')

        result = calc.calculate_net_profit(
            entry_price=0.95,
            exit_price=1.0,
            won=True
        )

        assert 'net_profit_pct' in result
        # Kalshi has different fee structure
        assert result['fees_paid'] >= 0

    def test_expected_value(self):
        """Test expected value calculation"""
        from Factors.calibration import FeeAdjustedProfitCalculator

        calc = FeeAdjustedProfitCalculator('polymarket')

        result = calc.expected_value(
            entry_price=0.95,
            win_probability=0.97
        )

        assert 'expected_value' in result
        assert 'is_positive_ev' in result

    def test_negative_ev_detection(self):
        """Test detection of negative EV trades"""
        from Factors.calibration import FeeAdjustedProfitCalculator

        calc = FeeAdjustedProfitCalculator('polymarket')

        # Low certainty at high price = negative EV
        result = calc.expected_value(
            entry_price=0.99,
            win_probability=0.90
        )

        assert result['is_positive_ev'] is False


class TestGetCalibrator:
    """Tests for get_calibrator singleton"""

    def test_singleton_pattern(self):
        """Test calibrator returns same instance"""
        from Factors.calibration import get_calibrator

        cal1 = get_calibrator()
        cal2 = get_calibrator()

        assert cal1 is cal2
