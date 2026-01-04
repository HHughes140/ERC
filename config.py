"""
Configuration for ERC Trading System
"""
import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Setup module logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system environment variables


@dataclass
class Config:
    """Basic configuration for trading strategies"""
    # API URLs
    POLYMARKET_API_URL: str = "https://clob.polymarket.com"
    GAMMA_API_URL: str = "https://gamma-api.polymarket.com"
    KALSHI_API_URL: str = "https://api.elections.kalshi.com"
    
    # API Credentials (from environment variables)
    POLYMARKET_API_KEY: str = os.getenv("POLYMARKET_API_KEY", "")
    POLYMARKET_SECRET: str = os.getenv("POLYMARKET_SECRET", "")
    POLYMARKET_PASSPHRASE: str = os.getenv("POLYMARKET_PASSPHRASE", "")
    
    KALSHI_API_KEY: str = os.getenv("KALSHI_API_KEY", "")
    KALSHI_PRIVATE_KEY: str = os.getenv("KALSHI_PRIVATE_KEY", "")
    
    # Alpaca credentials
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
    
    # Wallet configuration
    PRIVATE_KEY: str = os.getenv("PRIVATE_KEY", "")
    WALLET_ADDRESS: str = os.getenv("WALLET_ADDRESS", "")
    POLYGON_RPC_URL: str = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")
    
    # Discord notifications
    DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    ENABLE_DISCORD_ALERTS: bool = os.getenv("ENABLE_DISCORD_ALERTS", "true").lower() == "true"
    
    # Trading Parameters
    MIN_PROFIT_PERCENT: float = 2.0  # Minimum 2% profit
    MIN_PROFIT_THRESHOLD: float = 0.01  # Minimum profit threshold for arb detection
    SLIPPAGE_TOLERANCE: float = 0.02  # 2% slippage tolerance
    VAR_CONFIDENCE: float = 0.95  # VaR confidence level
    MAX_POSITION_SIZE: float = 100.0  # Max $100 per position
    RISK_FREE_RATE: float = 0.05  # 5% risk-free rate

    # Strategy Settings
    ENABLE_DRY_RUN: bool = True
    ENABLE_DATABASE: bool = True
    KALSHI_ENABLED: bool = True
    MAX_CONCURRENT_POSITIONS: int = 10
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / "data"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    def __post_init__(self):
        """Ensure directories exist and validate config"""
        self.DATA_DIR.mkdir(exist_ok=True)
        self.LOGS_DIR.mkdir(exist_ok=True)
        
        # Validate critical credentials
        if not self.POLYMARKET_API_KEY:
            logger.warning("POLYMARKET_API_KEY not set")
        if not self.KALSHI_API_KEY:
            logger.warning("KALSHI_API_KEY not set")
        if not self.KALSHI_PRIVATE_KEY:
            logger.warning("KALSHI_PRIVATE_KEY not set")


@dataclass
class ERCConfig(Config):
    """Extended configuration for full ERC system"""
    # Database
    DB_PATH: Path = Path(__file__).parent / "data" / "erc_database.db"
    
    # Platforms
    PLATFORMS: List[str] = None
    
    # Capital Management
    TOTAL_CAPITAL: float = 1000.0
    MAX_POSITION_PCT: float = 0.1  # Max 10% per position
    
    # Risk Management
    MAX_DAILY_LOSS: float = 50.0  # Max $50 loss per day
    MAX_DRAWDOWN: float = 100.0  # Max $100 drawdown
    
    # Strategy Allocations (percentages)
    ARBITRAGE_ALLOCATION: float = 0.40
    SHARKY_ALLOCATION: float = 0.30
    WEATHER_ALLOCATION: float = 0.20
    RESERVE_ALLOCATION: float = 0.10
    
    # Execution
    ORDER_TIMEOUT: int = 30  # seconds
    MAX_RETRIES: int = 3
    
    # Monitoring
    SCAN_INTERVAL: int = 60  # seconds
    ALERT_THRESHOLD: float = 100.0  # Alert on $100+ profit opportunities
    
    def __post_init__(self):
        super().__post_init__()
        
        if self.PLATFORMS is None:
            self.PLATFORMS = ['polymarket', 'kalshi']
        
        # Ensure DB directory exists
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Show loaded credentials (masked) - use ASCII to avoid Windows encoding issues
        logger.info("Configuration loaded")
        logger.info(f"Polymarket API Key: {'[SET]' if self.POLYMARKET_API_KEY else '[MISSING]'}")
        logger.info(f"Kalshi API Key:     {'[SET]' if self.KALSHI_API_KEY else '[MISSING]'}")
        logger.info(f"Alpaca API Key:     {'[SET]' if self.ALPACA_API_KEY else '[MISSING]'}")
    
    def get_strategy_capital(self, strategy: str) -> float:
        """Get allocated capital for a strategy"""
        allocations = {
            'arbitrage': self.ARBITRAGE_ALLOCATION,
            'sharky': self.SHARKY_ALLOCATION,
            'weather': self.WEATHER_ALLOCATION
        }
        
        return self.TOTAL_CAPITAL * allocations.get(strategy.lower(), 0.0)


# Default config instance
default_config = ERCConfig()