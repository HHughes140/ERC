"""
Tests for Infrastructure modules
"""
import pytest
import time
from datetime import datetime


class TestCacheManager:
    """Tests for Infrastructure/cache.py"""

    def test_cache_set_get(self):
        """Test basic cache set and get"""
        from Infrastructure.cache import CacheManager

        cache = CacheManager()
        cache.set('test_key', {'value': 123}, namespace='test', ttl=60)

        result = cache.get('test_key', namespace='test')
        assert result == {'value': 123}

    def test_cache_expiration(self):
        """Test cache TTL expiration"""
        from Infrastructure.cache import CacheManager

        cache = CacheManager()
        cache.set('expire_key', 'value', namespace='test', ttl=1)

        # Should exist immediately
        assert cache.get('expire_key', namespace='test') == 'value'

        # Wait for expiration
        time.sleep(1.5)
        assert cache.get('expire_key', namespace='test') is None

    def test_cache_namespaces(self):
        """Test cache namespace isolation"""
        from Infrastructure.cache import CacheManager

        cache = CacheManager()
        cache.set('shared_key', 'ns1_value', namespace='ns1', ttl=60)
        cache.set('shared_key', 'ns2_value', namespace='ns2', ttl=60)

        assert cache.get('shared_key', namespace='ns1') == 'ns1_value'
        assert cache.get('shared_key', namespace='ns2') == 'ns2_value'


class TestRateLimiter:
    """Tests for Infrastructure/error_handling.py RateLimiter"""

    def test_rate_limiter_allows_requests(self):
        """Test rate limiter allows requests within limit"""
        from Infrastructure.error_handling import RateLimiter

        limiter = RateLimiter(rate=10, per=1.0)

        # Should allow 10 requests
        for _ in range(10):
            assert limiter.acquire() is True

    def test_rate_limiter_blocks_excess(self):
        """Test rate limiter blocks excess requests"""
        from Infrastructure.error_handling import RateLimiter

        limiter = RateLimiter(rate=2, per=1.0)

        # First 2 should pass
        assert limiter.acquire() is True
        assert limiter.acquire() is True

        # Third should fail (non-blocking)
        assert limiter.acquire(block=False) is False


class TestRetryHandler:
    """Tests for Infrastructure/error_handling.py RetryHandler"""

    def test_retry_success(self):
        """Test retry handler with successful function"""
        from Infrastructure.error_handling import RetryHandler

        handler = RetryHandler(max_retries=3)

        call_count = 0

        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = handler.execute(success_func)
        assert result == "success"
        assert call_count == 1

    def test_retry_eventual_success(self):
        """Test retry handler with eventual success"""
        from Infrastructure.error_handling import RetryHandler

        handler = RetryHandler(max_retries=3, base_delay=0.1)

        call_count = 0

        def eventual_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = handler.execute(eventual_success)
        assert result == "success"
        assert call_count == 3


class TestPortfolioRiskManager:
    """Tests for Infrastructure/portfolio_risk.py"""

    def test_risk_manager_initialization(self):
        """Test risk manager initializes correctly"""
        from Infrastructure.portfolio_risk import PortfolioRiskManager

        manager = PortfolioRiskManager(
            max_portfolio_exposure=0.8,
            max_single_position_pct=0.1
        )

        assert manager.max_portfolio_exposure == 0.8
        assert manager.max_single_position_pct == 0.1

    def test_position_check(self):
        """Test position risk checking"""
        from Infrastructure.portfolio_risk import PortfolioRiskManager

        manager = PortfolioRiskManager(
            max_portfolio_exposure=0.8,
            max_single_position_pct=0.1,
            total_capital=1000
        )

        # Small position should be approved
        result = manager.check_position(
            model='test',
            platform='polymarket',
            market_id='test_001',
            proposed_size=50
        )
        assert result['approved'] is True


class TestOrderExecution:
    """Tests for Infrastructure/order_execution.py"""

    def test_execution_engine_initialization(self):
        """Test execution engine initializes"""
        from Infrastructure.order_execution import ExecutionEngine

        engine = ExecutionEngine()
        assert engine is not None

    def test_slippage_estimation(self):
        """Test slippage estimation"""
        from Infrastructure.order_execution import ExecutionEngine

        engine = ExecutionEngine()

        slippage = engine.estimate_slippage(
            order_size=100,
            liquidity=10000,
            spread=0.02
        )

        assert 0 <= slippage <= 0.1  # Reasonable slippage range


class TestPaperTrading:
    """Tests for Infrastructure/paper_trading.py"""

    def test_paper_trading_initialization(self):
        """Test paper trading engine initializes"""
        from Infrastructure.paper_trading import PaperTradingEngine

        engine = PaperTradingEngine(initial_capital=10000)
        assert engine.get_balance() == 10000

    def test_simulated_order(self):
        """Test simulated order execution"""
        from Infrastructure.paper_trading import PaperTradingEngine

        engine = PaperTradingEngine(initial_capital=10000)

        result = engine.execute_order(
            symbol='TEST_MARKET',
            side='buy',
            quantity=100,
            price=0.50
        )

        assert result['filled'] is True
        assert result['quantity'] == 100
