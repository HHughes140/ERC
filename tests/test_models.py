"""
Tests for trading model modules
"""
import pytest


class TestArbitrageDetector:
    """Tests for Models/Polymarket/math_models.py"""

    def test_detector_initialization(self):
        """Test arbitrage detector initializes"""
        from Models.Polymarket.math_models import ArbitrageDetector

        detector = ArbitrageDetector(transaction_cost=0.02)
        assert detector is not None

    def test_single_market_arbitrage(self):
        """Test single market arbitrage detection"""
        from Models.Polymarket.math_models import ArbitrageDetector

        detector = ArbitrageDetector(transaction_cost=0.02)

        # Prices that sum to less than 1 = arbitrage opportunity
        result = detector.detect_single_market(yes_price=0.45, no_price=0.50)

        assert result is not None
        assert 'guaranteed_profit' in result
        assert result['guaranteed_profit'] > 0

    def test_no_arbitrage_when_prices_fair(self):
        """Test no arbitrage when prices are fair"""
        from Models.Polymarket.math_models import ArbitrageDetector

        detector = ArbitrageDetector(transaction_cost=0.02)

        # Prices that sum to 1 = no arbitrage
        result = detector.detect_single_market(yes_price=0.50, no_price=0.50)

        # Either None or minimal profit
        if result:
            assert result['guaranteed_profit'] <= 0.02


class TestOrderBookSimulator:
    """Tests for Models/Polymarket/order_book_simulator.py"""

    def test_simulator_initialization(self):
        """Test order book simulator initializes"""
        from Models.Polymarket.order_book_simulator import OrderBookSimulator

        sim = OrderBookSimulator()
        assert sim is not None

    def test_execution_estimation(self):
        """Test execution estimation"""
        from Models.Polymarket.order_book_simulator import OrderBookSimulator

        sim = OrderBookSimulator()

        result = sim.estimate_execution(
            prices={'yes': 0.50, 'no': 0.50},
            target_shares=100,
            liquidity=10000
        )

        assert result is not None
        assert 'estimated_slippage' in result


class TestSemanticMatcher:
    """Tests for Models/Polymarket/semantic_matcher.py"""

    def test_matcher_initialization(self):
        """Test semantic matcher initializes"""
        from Models.Polymarket.semantic_matcher import SemanticMatcher

        matcher = SemanticMatcher()
        assert matcher is not None

    def test_similar_markets_match(self):
        """Test similar markets get high similarity score"""
        from Models.Polymarket.semantic_matcher import SemanticMatcher

        matcher = SemanticMatcher()

        result = matcher.match_markets(
            "Will Bitcoin exceed $100,000 by December 2025?",
            "Will BTC price be above $100k by end of 2025?"
        )

        assert result['similarity'] > 0.5

    def test_different_markets_low_score(self):
        """Test different markets get low similarity score"""
        from Models.Polymarket.semantic_matcher import SemanticMatcher

        matcher = SemanticMatcher()

        result = matcher.match_markets(
            "Will Bitcoin exceed $100,000 by December 2025?",
            "Will it rain in New York tomorrow?"
        )

        assert result['similarity'] < 0.5


class TestExitEngine:
    """Tests for Models/Sharky/exit_engine.py"""

    def test_exit_engine_initialization(self):
        """Test exit engine initializes"""
        from Models.Sharky.exit_engine import DynamicExitEngine

        engine = DynamicExitEngine()
        assert engine is not None

    def test_profit_target_exit(self):
        """Test profit target triggers exit"""
        from Models.Sharky.exit_engine import DynamicExitEngine
        from datetime import datetime, timedelta

        engine = DynamicExitEngine()

        signal = engine.check_exit(
            position_id='test_001',
            entry_price=0.95,
            current_price=0.98,  # 3% profit
            entry_certainty=0.97,
            current_certainty=0.97,
            entry_time=datetime.now() - timedelta(hours=1)
        )

        assert signal is not None
        assert signal.should_exit is True
        assert 'profit' in signal.reason.lower()

    def test_stop_loss_exit(self):
        """Test stop loss triggers exit"""
        from Models.Sharky.exit_engine import DynamicExitEngine
        from datetime import datetime, timedelta

        engine = DynamicExitEngine()

        signal = engine.check_exit(
            position_id='test_002',
            entry_price=0.95,
            current_price=0.90,  # 5% loss
            entry_certainty=0.97,
            current_certainty=0.92,
            entry_time=datetime.now() - timedelta(hours=1)
        )

        assert signal is not None
        assert signal.should_exit is True

    def test_no_exit_when_in_range(self):
        """Test no exit when position is within thresholds"""
        from Models.Sharky.exit_engine import DynamicExitEngine
        from datetime import datetime, timedelta

        engine = DynamicExitEngine()

        signal = engine.check_exit(
            position_id='test_003',
            entry_price=0.95,
            current_price=0.96,  # 1% profit - within range
            entry_certainty=0.97,
            current_certainty=0.96,
            entry_time=datetime.now() - timedelta(minutes=30)
        )

        # Should either be None or indicate no exit
        if signal:
            assert signal.should_exit is False
