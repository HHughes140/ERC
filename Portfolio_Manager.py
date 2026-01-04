"""
Portfolio Manager
Tracks capital allocation, positions, and P&L across all strategies and platforms
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from config import ERCConfig
from Central_DB.database import Database

logger = logging.getLogger(__name__)


class Platform(Enum):
    """Trading platforms"""
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"
    ALPACA = "alpaca"


@dataclass
class Position:
    """Represents an open position"""
    position_id: str
    platform: str
    strategy: str
    symbol: str
    side: str
    entry_price: float
    quantity: float
    capital_deployed: float
    opened_at: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def update_price(self, new_price: float):
        """Update current price and calculate unrealized P&L"""
        self.current_price = new_price
        if self.side.lower() in ['buy', 'yes', 'long']:
            self.unrealized_pnl = (new_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - new_price) * self.quantity

    def to_dict(self) -> Dict:
        return {
            'position_id': self.position_id,
            'platform': self.platform,
            'strategy': self.strategy,
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'capital_deployed': self.capital_deployed,
            'opened_at': self.opened_at.isoformat(),
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'metadata': self.metadata
        }


@dataclass
class StrategyAllocation:
    """Capital allocation for a strategy"""
    name: str
    allocation_pct: float
    allocated_capital: float
    deployed_capital: float = 0.0
    available_capital: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    position_count: int = 0

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def utilization(self) -> float:
        if self.allocated_capital <= 0:
            return 0.0
        return self.deployed_capital / self.allocated_capital

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'allocation_pct': self.allocation_pct,
            'allocated_capital': self.allocated_capital,
            'deployed_capital': self.deployed_capital,
            'available_capital': self.available_capital,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.total_pnl,
            'utilization': self.utilization,
            'position_count': self.position_count
        }


@dataclass
class PlatformBalance:
    """Balance on a specific platform"""
    platform: str
    available: float
    deployed: float
    total: float
    positions: int = 0

    def to_dict(self) -> Dict:
        return {
            'platform': self.platform,
            'available': self.available,
            'deployed': self.deployed,
            'total': self.total,
            'positions': self.positions
        }


class PortfolioManager:
    """
    Central portfolio manager for tracking capital, positions, and P&L

    Features:
    - Multi-platform balance tracking (Polymarket, Kalshi, Alpaca)
    - Strategy-based capital allocation
    - Position tracking with unrealized P&L
    - Risk limit monitoring
    - Performance analytics
    """

    def __init__(self, config: ERCConfig, database: Database,
                 initial_capital: float = 1000.0):
        self.config = config
        self.db = database
        self.initial_capital = initial_capital

        # Capital tracking
        self.total_capital = initial_capital
        self.deployed_capital = 0.0
        self.available_capital = initial_capital

        # Strategy allocations
        self.strategies: Dict[str, StrategyAllocation] = {}

        # Platform balances
        self.platform_balances: Dict[str, PlatformBalance] = {}

        # Open positions
        self.positions: Dict[str, Position] = {}

        # P&L tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.daily_pnl = 0.0
        self.daily_pnl_reset_time: Optional[datetime] = None

        # Risk tracking
        self.max_drawdown = 0.0
        self.peak_capital = initial_capital

        # Initialized flag
        self._initialized = False

        logger.info(f"Portfolio Manager created with ${initial_capital:,.2f} capital")

    async def initialize(self):
        """Initialize portfolio manager"""
        logger.info("Initializing Portfolio Manager...")

        # Setup strategy allocations
        self._setup_strategies()

        # Setup platform balances
        self._setup_platforms()

        # Load existing positions from database
        await self._load_positions()

        # Reset daily P&L tracking
        self._reset_daily_pnl()

        self._initialized = True
        logger.info("Portfolio Manager initialized")

    def _setup_strategies(self):
        """Setup strategy allocations based on config"""
        strategies = [
            ('arbitrage', self.config.ARBITRAGE_ALLOCATION),
            ('sharky', self.config.SHARKY_ALLOCATION),
            ('weather', self.config.WEATHER_ALLOCATION),
            ('reserve', self.config.RESERVE_ALLOCATION),
        ]

        for name, pct in strategies:
            allocated = self.total_capital * pct
            self.strategies[name] = StrategyAllocation(
                name=name,
                allocation_pct=pct,
                allocated_capital=allocated,
                available_capital=allocated
            )

        logger.info(f"Setup {len(self.strategies)} strategy allocations")

    def _setup_platforms(self):
        """Setup platform balance tracking"""
        for platform in self.config.PLATFORMS:
            # Initial distribution (can be overridden by actual balances)
            initial_balance = self.total_capital / len(self.config.PLATFORMS)
            self.platform_balances[platform] = PlatformBalance(
                platform=platform,
                available=initial_balance,
                deployed=0.0,
                total=initial_balance
            )

        logger.info(f"Setup {len(self.platform_balances)} platform balances")

    async def _load_positions(self):
        """Load open positions from database"""
        try:
            db_positions = self.db.get_open_positions()

            for pos_data in db_positions:
                position = Position(
                    position_id=pos_data['position_id'],
                    platform=pos_data['platform'],
                    strategy=pos_data['strategy'],
                    symbol=pos_data['symbol'],
                    side=pos_data['side'],
                    entry_price=pos_data['entry_price'],
                    quantity=pos_data['quantity'],
                    capital_deployed=pos_data['capital_deployed'],
                    opened_at=datetime.fromisoformat(pos_data['opened_at']),
                    current_price=pos_data.get('current_price', pos_data['entry_price']),
                    unrealized_pnl=pos_data.get('unrealized_pnl', 0.0)
                )
                self.positions[position.position_id] = position

                # Update deployed capital
                self.deployed_capital += position.capital_deployed

            self.available_capital = self.total_capital - self.deployed_capital
            logger.info(f"Loaded {len(self.positions)} open positions")

        except Exception as e:
            logger.error(f"Error loading positions: {e}")

    def _reset_daily_pnl(self):
        """Reset daily P&L tracking"""
        self.daily_pnl = 0.0
        self.daily_pnl_reset_time = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    async def update(self):
        """Update portfolio state"""
        # Check if we need to reset daily P&L
        now = datetime.now()
        if self.daily_pnl_reset_time:
            if now.date() > self.daily_pnl_reset_time.date():
                self._reset_daily_pnl()

        # Calculate unrealized P&L
        self.unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())

        # Update drawdown
        current_value = self.total_capital + self.realized_pnl + self.unrealized_pnl
        if current_value > self.peak_capital:
            self.peak_capital = current_value
        else:
            drawdown = self.peak_capital - current_value
            self.max_drawdown = max(self.max_drawdown, drawdown)

        # Update strategy metrics
        for strategy in self.strategies.values():
            strategy.position_count = sum(
                1 for pos in self.positions.values()
                if pos.strategy == strategy.name
            )
            strategy.deployed_capital = sum(
                pos.capital_deployed for pos in self.positions.values()
                if pos.strategy == strategy.name
            )
            strategy.unrealized_pnl = sum(
                pos.unrealized_pnl for pos in self.positions.values()
                if pos.strategy == strategy.name
            )
            strategy.available_capital = strategy.allocated_capital - strategy.deployed_capital

        # Update platform metrics
        for platform in self.platform_balances.values():
            platform.positions = sum(
                1 for pos in self.positions.values()
                if pos.platform == platform.platform
            )
            platform.deployed = sum(
                pos.capital_deployed for pos in self.positions.values()
                if pos.platform == platform.platform
            )
            platform.available = platform.total - platform.deployed

    def get_available_capital(self) -> float:
        """Get total available capital"""
        return self.available_capital

    def get_strategy_capital(self, strategy: str) -> float:
        """Get available capital for a specific strategy"""
        if strategy in self.strategies:
            return self.strategies[strategy].available_capital
        return 0.0

    def can_open_position(self, capital_required: float, strategy: str = None) -> bool:
        """Check if we can open a position with the required capital"""
        # Check total available
        if capital_required > self.available_capital:
            return False

        # Check strategy allocation if specified
        if strategy and strategy in self.strategies:
            if capital_required > self.strategies[strategy].available_capital:
                return False

        # Check position limits
        if len(self.positions) >= self.config.MAX_CONCURRENT_POSITIONS:
            return False

        # Check daily loss limit
        if self.daily_pnl < -self.config.MAX_DAILY_LOSS:
            logger.warning("Daily loss limit reached")
            return False

        # Check max drawdown
        if self.max_drawdown >= self.config.MAX_DRAWDOWN:
            logger.warning("Max drawdown limit reached")
            return False

        return True

    def can_trade_on_platform(self, platform: str) -> bool:
        """Check if platform has available capital"""
        if platform in self.platform_balances:
            return self.platform_balances[platform].available > 0
        return False

    async def request_capital(self, strategy: str, amount: float,
                             reason: str = "", platform: str = None) -> float:
        """
        Request capital for a trade

        Returns:
            Approved capital amount (may be less than requested)
        """
        if not self.can_open_position(amount, strategy):
            logger.debug(f"Capital request denied: ${amount:.2f} for {strategy}")
            return 0.0

        # Apply position size limits
        max_position = self.config.TOTAL_CAPITAL * self.config.MAX_POSITION_PCT
        approved = min(amount, max_position, self.available_capital)

        # Check strategy limit
        if strategy in self.strategies:
            approved = min(approved, self.strategies[strategy].available_capital)

        # Check platform limit
        if platform and platform in self.platform_balances:
            approved = min(approved, self.platform_balances[platform].available)

        if approved > 0:
            logger.debug(f"Capital approved: ${approved:.2f} for {strategy} ({reason})")

        return approved

    def record_position_opened(self, strategy: str, position_id: str,
                               capital: float, platform: str = None,
                               symbol: str = "", side: str = "buy",
                               entry_price: float = 0.0, quantity: float = 1.0):
        """Record a new position being opened"""
        position = Position(
            position_id=position_id,
            platform=platform or 'unknown',
            strategy=strategy,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            capital_deployed=capital,
            opened_at=datetime.now()
        )

        self.positions[position_id] = position

        # Update capital tracking
        self.deployed_capital += capital
        self.available_capital -= capital

        # Update strategy
        if strategy in self.strategies:
            self.strategies[strategy].deployed_capital += capital
            self.strategies[strategy].available_capital -= capital
            self.strategies[strategy].position_count += 1

        # Update platform
        if platform and platform in self.platform_balances:
            self.platform_balances[platform].deployed += capital
            self.platform_balances[platform].available -= capital
            self.platform_balances[platform].positions += 1

        logger.info(f"Position opened: {position_id} | ${capital:.2f} | {strategy}")

    def record_position_closed(self, strategy: str, position_id: str,
                               pnl: float, platform: str = None):
        """Record a position being closed"""
        if position_id not in self.positions:
            logger.warning(f"Position {position_id} not found")
            return

        position = self.positions[position_id]
        capital = position.capital_deployed

        # Update P&L
        self.realized_pnl += pnl
        self.daily_pnl += pnl

        # Return capital
        self.deployed_capital -= capital
        self.available_capital += capital + pnl  # Include P&L in available

        # Update total capital with P&L
        self.total_capital += pnl

        # Update strategy
        if strategy in self.strategies:
            self.strategies[strategy].deployed_capital -= capital
            self.strategies[strategy].available_capital += capital + pnl
            self.strategies[strategy].realized_pnl += pnl
            self.strategies[strategy].position_count -= 1

        # Update platform
        if platform and platform in self.platform_balances:
            self.platform_balances[platform].deployed -= capital
            self.platform_balances[platform].available += capital + pnl
            self.platform_balances[platform].total += pnl
            self.platform_balances[platform].positions -= 1

        # Remove from tracking
        del self.positions[position_id]

        logger.info(f"Position closed: {position_id} | P&L: ${pnl:.2f}")

    def get_state(self) -> Dict:
        """Get current portfolio state"""
        return {
            'total_capital': self.total_capital,
            'deployed_capital': self.deployed_capital,
            'available_capital': self.available_capital,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.realized_pnl + self.unrealized_pnl,
            'daily_pnl': self.daily_pnl,
            'max_drawdown': self.max_drawdown,
            'open_positions': len(self.positions),
            'strategies': {k: v.to_dict() for k, v in self.strategies.items()},
            'platforms': {k: v.to_dict() for k, v in self.platform_balances.items()}
        }

    def print_summary(self):
        """Print portfolio summary"""
        state = self.get_state()

        print("\n" + "=" * 60)
        print("PORTFOLIO SUMMARY")
        print("=" * 60)
        print(f"Total Capital:     ${state['total_capital']:,.2f}")
        print(f"Deployed:          ${state['deployed_capital']:,.2f}")
        print(f"Available:         ${state['available_capital']:,.2f}")
        print(f"Open Positions:    {state['open_positions']}")
        print("-" * 60)
        print(f"Realized P&L:      ${state['realized_pnl']:,.2f}")
        print(f"Unrealized P&L:    ${state['unrealized_pnl']:,.2f}")
        print(f"Total P&L:         ${state['total_pnl']:,.2f}")
        print(f"Daily P&L:         ${state['daily_pnl']:,.2f}")
        print(f"Max Drawdown:      ${state['max_drawdown']:,.2f}")

        # Strategy breakdown
        print("\n" + "-" * 60)
        print("STRATEGY ALLOCATIONS")
        print("-" * 60)
        print(f"{'Strategy':<15} {'Allocated':>12} {'Deployed':>12} {'Available':>12} {'P&L':>12}")
        print("-" * 60)
        for name, strat in self.strategies.items():
            print(f"{name:<15} ${strat.allocated_capital:>10,.2f} "
                  f"${strat.deployed_capital:>10,.2f} "
                  f"${strat.available_capital:>10,.2f} "
                  f"${strat.total_pnl:>10,.2f}")

        # Platform breakdown
        if self.platform_balances:
            print("\n" + "-" * 60)
            print("PLATFORM BALANCES")
            print("-" * 60)
            print(f"{'Platform':<15} {'Total':>12} {'Available':>12} {'Deployed':>12} {'Positions':>10}")
            print("-" * 60)
            for name, plat in self.platform_balances.items():
                print(f"{name:<15} ${plat.total:>10,.2f} "
                      f"${plat.available:>10,.2f} "
                      f"${plat.deployed:>10,.2f} "
                      f"{plat.positions:>10}")

        print("=" * 60 + "\n")

    async def save_snapshot(self):
        """Save portfolio snapshot to database"""
        try:
            snapshot_data = {
                'timestamp': datetime.now().isoformat(),
                'total_capital': self.total_capital,
                'deployed_capital': self.deployed_capital,
                'available_capital': self.available_capital,
                'total_pnl': self.realized_pnl + self.unrealized_pnl,
                'daily_pnl': self.daily_pnl,
                'num_positions': len(self.positions),
                'num_trades': 0,  # Would need trade counter
                'platform_balances': {k: v.to_dict() for k, v in self.platform_balances.items()}
            }

            self.db.save_portfolio_snapshot(snapshot_data)
            logger.debug("Portfolio snapshot saved")

        except Exception as e:
            logger.error(f"Failed to save portfolio snapshot: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        # Save final snapshot
        await self.save_snapshot()
        logger.info("Portfolio Manager cleanup complete")


# For backward compatibility with typo in filename
MasterPortfolioManager = PortfolioManager
