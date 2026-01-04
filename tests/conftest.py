"""
Pytest fixtures for ERC Trading System tests
"""
import pytest
import sys
import os
from pathlib import Path
from datetime import datetime
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    from Central_DB.database import Database

    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    db = Database(db_path=db_path)
    yield db

    db.close()
    try:
        os.unlink(db_path)
    except:
        pass


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    class MockConfig:
        MIN_PROFIT_THRESHOLD = 0.02
        POLYMARKET_API_URL = 'https://api.polymarket.com'
        GAMMA_API_URL = 'https://gamma-api.polymarket.com'
        KALSHI_API_URL = 'https://api.elections.kalshi.com'
        POLYMARKET_API_KEY = 'test_key'
        POLYMARKET_SECRET = 'test_secret'
        POLYMARKET_PASSPHRASE = 'test_pass'
        KALSHI_API_KEY = 'test_key'
        KALSHI_PRIVATE_KEY = ''
        KALSHI_ENABLED = False

        # Strategy allocations
        ARBITRAGE_ALLOCATION = 0.40
        SHARKY_ALLOCATION = 0.30
        WEATHER_ALLOCATION = 0.20
        ML_ALLOCATION = 0.10

        # Risk limits
        MAX_POSITION_SIZE = 100
        MAX_PORTFOLIO_EXPOSURE = 0.80

    return MockConfig()


@pytest.fixture
def sample_market():
    """Sample market data for testing"""
    from Models.Polymarket.polymarket_client import Market

    return Market(
        condition_id='test_market_001',
        question='Will Bitcoin exceed $100k by end of 2025?',
        outcomes=['Yes', 'No'],
        outcome_prices=[0.45, 0.55],
        token_ids=['token_yes', 'token_no'],
        active=True,
        volume=50000.0,
        liquidity=10000.0
    )


@pytest.fixture
def sample_trade():
    """Sample trade data for testing"""
    return {
        'trade_id': 'test_trade_001',
        'timestamp': datetime.now().isoformat(),
        'platform': 'polymarket',
        'strategy': 'arbitrage',
        'symbol': 'BTC_100K_2025',
        'side': 'buy',
        'entry_price': 0.45,
        'quantity': 100,
        'metadata': {'opportunity_type': 'single_market'}
    }


@pytest.fixture
def sample_position():
    """Sample position data for testing"""
    return {
        'position_id': 'test_pos_001',
        'platform': 'polymarket',
        'strategy': 'sharky',
        'symbol': 'ELECTION_2024',
        'side': 'yes',
        'entry_price': 0.95,
        'quantity': 50,
        'capital_deployed': 47.50,
        'opened_at': datetime.now().isoformat()
    }
