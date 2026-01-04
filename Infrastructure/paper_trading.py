"""
Paper Trading System for ERC Trading System

Provides simulated trading environment for:
- Strategy validation before live trading
- Backtesting with realistic execution
- Performance tracking
- Risk-free experimentation

CRITICAL: Always paper trade new strategies before real money!
"""

import asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class PaperOrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class PaperOrder:
    """Simulated order"""
    order_id: str
    market_id: str
    side: str           # 'buy', 'sell'
    order_type: str     # 'market', 'limit'
    quantity: float
    limit_price: Optional[float] = None
    status: PaperOrderStatus = PaperOrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    slippage: float = 0.0
    fees: float = 0.0


@dataclass
class PaperPosition:
    """Simulated position"""
    position_id: str
    market_id: str
    side: str           # 'long', 'short', 'yes', 'no'
    entry_price: float
    quantity: float
    current_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price


@dataclass
class PaperAccount:
    """Simulated trading account"""
    account_id: str
    initial_balance: float
    current_balance: float
    equity: float
    positions: Dict[str, PaperPosition] = field(default_factory=dict)
    orders: Dict[str, PaperOrder] = field(default_factory=dict)
    order_history: List[PaperOrder] = field(default_factory=list)
    trade_history: List[Dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def total_pnl(self) -> float:
        realized = sum(t.get('pnl', 0) for t in self.trade_history)
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return realized + unrealized

    @property
    def return_pct(self) -> float:
        return (self.equity - self.initial_balance) / self.initial_balance


@dataclass
class SimulationConfig:
    """Paper trading simulation config"""
    # Slippage simulation
    base_slippage_pct: float = 0.001    # 0.1% base slippage
    size_impact_factor: float = 0.0001  # Additional slippage per $1000

    # Fees
    fee_rate: float = 0.02              # 2% transaction fee

    # Fill simulation
    fill_probability: float = 0.98      # 98% fill rate for limits
    partial_fill_probability: float = 0.05

    # Latency simulation
    execution_delay_ms: int = 100       # 100ms execution delay

    # Market simulation
    price_volatility: float = 0.01      # 1% random price movement


class PaperTradingEngine:
    """
    Paper trading engine that simulates real trading conditions.

    Usage:
        engine = PaperTradingEngine(initial_balance=10000)

        # Submit orders
        order = await engine.submit_order('market_123', 'buy', 100)

        # Get positions
        positions = engine.get_positions()

        # Get performance
        perf = engine.get_performance()
    """

    def __init__(self, initial_balance: float = 100000,
                 config: Optional[SimulationConfig] = None,
                 price_feed: Optional[Callable] = None):
        """
        Args:
            initial_balance: Starting capital
            config: Simulation configuration
            price_feed: Optional function to get real prices
        """
        self.config = config or SimulationConfig()
        self.price_feed = price_feed

        # Create account
        self.account = PaperAccount(
            account_id=f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            initial_balance=initial_balance,
            current_balance=initial_balance,
            equity=initial_balance
        )

        # Price cache
        self._prices: Dict[str, float] = {}

        # Order ID counter
        self._order_counter = 0

        # Performance tracking
        self._equity_history: List[tuple] = [(datetime.now(), initial_balance)]
        self._daily_returns: List[float] = []

    # ========== Order Management ==========

    async def submit_order(self, market_id: str, side: str, quantity: float,
                          order_type: str = 'market',
                          limit_price: Optional[float] = None) -> PaperOrder:
        """
        Submit a paper order.

        Args:
            market_id: Market identifier
            side: 'buy' or 'sell'
            quantity: Order quantity
            order_type: 'market' or 'limit'
            limit_price: Required for limit orders

        Returns:
            PaperOrder with execution details
        """
        self._order_counter += 1
        order_id = f"paper_{self._order_counter}"

        order = PaperOrder(
            order_id=order_id,
            market_id=market_id,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price
        )

        # Simulate execution delay
        await asyncio.sleep(self.config.execution_delay_ms / 1000)

        # Get current price
        current_price = await self._get_price(market_id)

        if order_type == 'market':
            order = self._execute_market_order(order, current_price)
        else:
            order = self._execute_limit_order(order, current_price)

        # Store order
        self.account.orders[order_id] = order
        if order.status in [PaperOrderStatus.FILLED, PaperOrderStatus.PARTIAL]:
            self.account.order_history.append(order)
            self._update_position(order)

        return order

    def _execute_market_order(self, order: PaperOrder,
                              current_price: float) -> PaperOrder:
        """Execute market order with slippage simulation"""
        # Calculate slippage
        base_slippage = self.config.base_slippage_pct
        size_slippage = (order.quantity * current_price / 1000) * self.config.size_impact_factor
        total_slippage = base_slippage + size_slippage

        # Apply slippage (adverse direction)
        if order.side == 'buy':
            fill_price = current_price * (1 + total_slippage)
        else:
            fill_price = current_price * (1 - total_slippage)

        # Calculate fees
        fees = order.quantity * fill_price * self.config.fee_rate

        # Check sufficient balance
        total_cost = order.quantity * fill_price + fees
        if order.side == 'buy' and total_cost > self.account.current_balance:
            order.status = PaperOrderStatus.REJECTED
            return order

        # Fill order
        order.status = PaperOrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_price = fill_price
        order.slippage = total_slippage
        order.fees = fees
        order.filled_at = datetime.now()

        # Update balance
        if order.side == 'buy':
            self.account.current_balance -= total_cost
        else:
            self.account.current_balance += (order.quantity * fill_price - fees)

        logger.info(
            f"Paper order filled: {order.side} {order.quantity} @ ${fill_price:.4f} "
            f"(slippage: {total_slippage:.2%}, fees: ${fees:.2f})"
        )

        return order

    def _execute_limit_order(self, order: PaperOrder,
                             current_price: float) -> PaperOrder:
        """Execute limit order with fill probability simulation"""
        if order.limit_price is None:
            order.status = PaperOrderStatus.REJECTED
            return order

        # Check if limit price would fill
        would_fill = (
            (order.side == 'buy' and current_price <= order.limit_price) or
            (order.side == 'sell' and current_price >= order.limit_price)
        )

        if not would_fill:
            # Order stays pending (in real system would be on book)
            order.status = PaperOrderStatus.PENDING
            return order

        # Simulate fill probability
        if np.random.random() > self.config.fill_probability:
            order.status = PaperOrderStatus.PENDING
            return order

        # Simulate partial fill
        if np.random.random() < self.config.partial_fill_probability:
            fill_ratio = np.random.uniform(0.5, 0.95)
            order.filled_quantity = order.quantity * fill_ratio
            order.status = PaperOrderStatus.PARTIAL
        else:
            order.filled_quantity = order.quantity
            order.status = PaperOrderStatus.FILLED

        order.average_price = order.limit_price
        order.fees = order.filled_quantity * order.average_price * self.config.fee_rate
        order.filled_at = datetime.now()

        # Update balance
        total_cost = order.filled_quantity * order.average_price + order.fees
        if order.side == 'buy':
            self.account.current_balance -= total_cost
        else:
            self.account.current_balance += (order.filled_quantity * order.average_price - order.fees)

        return order

    def _update_position(self, order: PaperOrder):
        """Update positions based on filled order"""
        market_id = order.market_id

        if order.side == 'buy':
            position_side = 'long'
        else:
            position_side = 'short'

        if market_id in self.account.positions:
            pos = self.account.positions[market_id]

            if pos.side == position_side:
                # Adding to position - average in
                total_qty = pos.quantity + order.filled_quantity
                pos.entry_price = (
                    (pos.entry_price * pos.quantity + order.average_price * order.filled_quantity)
                    / total_qty
                )
                pos.quantity = total_qty
            else:
                # Closing or reversing position
                if order.filled_quantity >= pos.quantity:
                    # Close position
                    pnl = (order.average_price - pos.entry_price) * pos.quantity
                    if pos.side == 'short':
                        pnl = -pnl

                    self.account.trade_history.append({
                        'market_id': market_id,
                        'side': pos.side,
                        'entry_price': pos.entry_price,
                        'exit_price': order.average_price,
                        'quantity': pos.quantity,
                        'pnl': pnl,
                        'closed_at': datetime.now().isoformat()
                    })

                    remaining = order.filled_quantity - pos.quantity
                    if remaining > 0:
                        # Open reverse position
                        self.account.positions[market_id] = PaperPosition(
                            position_id=f"pos_{market_id}_{self._order_counter}",
                            market_id=market_id,
                            side=position_side,
                            entry_price=order.average_price,
                            quantity=remaining,
                            current_price=order.average_price,
                            entry_time=datetime.now()
                        )
                    else:
                        del self.account.positions[market_id]
                else:
                    # Partial close
                    pnl = (order.average_price - pos.entry_price) * order.filled_quantity
                    if pos.side == 'short':
                        pnl = -pnl

                    pos.quantity -= order.filled_quantity
                    pos.realized_pnl += pnl
        else:
            # New position
            self.account.positions[market_id] = PaperPosition(
                position_id=f"pos_{market_id}_{self._order_counter}",
                market_id=market_id,
                side=position_side,
                entry_price=order.average_price,
                quantity=order.filled_quantity,
                current_price=order.average_price,
                entry_time=datetime.now()
            )

    async def _get_price(self, market_id: str) -> float:
        """Get current price for market"""
        if self.price_feed:
            try:
                price = await self.price_feed(market_id)
                self._prices[market_id] = price
                return price
            except:
                pass

        # Use cached or simulated price
        if market_id in self._prices:
            # Simulate small price movement
            volatility = self.config.price_volatility
            move = np.random.normal(0, volatility)
            self._prices[market_id] *= (1 + move)
            return self._prices[market_id]

        # Default price
        self._prices[market_id] = 0.50
        return 0.50

    # ========== Position & Performance ==========

    async def update_positions(self):
        """Update all positions with current prices"""
        for market_id, pos in self.account.positions.items():
            current_price = await self._get_price(market_id)
            pos.current_price = current_price

            if pos.side == 'long':
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.quantity

        # Update equity
        position_value = sum(p.market_value for p in self.account.positions.values())
        unrealized_pnl = sum(p.unrealized_pnl for p in self.account.positions.values())
        self.account.equity = self.account.current_balance + position_value

        # Record equity
        self._equity_history.append((datetime.now(), self.account.equity))

    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        return [
            {
                'market_id': p.market_id,
                'side': p.side,
                'quantity': p.quantity,
                'entry_price': p.entry_price,
                'current_price': p.current_price,
                'unrealized_pnl': p.unrealized_pnl,
                'market_value': p.market_value
            }
            for p in self.account.positions.values()
        ]

    def get_performance(self) -> Dict:
        """Get performance metrics"""
        total_trades = len(self.account.trade_history)
        winning_trades = sum(1 for t in self.account.trade_history if t.get('pnl', 0) > 0)
        losing_trades = sum(1 for t in self.account.trade_history if t.get('pnl', 0) < 0)

        total_pnl = sum(t.get('pnl', 0) for t in self.account.trade_history)
        total_fees = sum(o.fees for o in self.account.order_history)

        wins = [t['pnl'] for t in self.account.trade_history if t.get('pnl', 0) > 0]
        losses = [abs(t['pnl']) for t in self.account.trade_history if t.get('pnl', 0) < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = sum(wins) / sum(losses) if losses else float('inf')

        # Equity curve analysis
        equities = [e[1] for e in self._equity_history]
        if len(equities) > 1:
            returns = np.diff(equities) / equities[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

            peak = np.maximum.accumulate(equities)
            drawdown = (peak - equities) / peak
            max_drawdown = np.max(drawdown)
        else:
            sharpe = 0
            max_drawdown = 0

        return {
            'initial_balance': self.account.initial_balance,
            'current_balance': self.account.current_balance,
            'equity': self.account.equity,
            'total_pnl': total_pnl,
            'total_fees': total_fees,
            'return_pct': self.account.return_pct,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'open_positions': len(self.account.positions)
        }

    def save_state(self, filepath: str):
        """Save paper trading state to file"""
        state = {
            'account_id': self.account.account_id,
            'initial_balance': self.account.initial_balance,
            'current_balance': self.account.current_balance,
            'equity': self.account.equity,
            'positions': {
                k: {
                    'market_id': v.market_id,
                    'side': v.side,
                    'entry_price': v.entry_price,
                    'quantity': v.quantity,
                    'current_price': v.current_price
                }
                for k, v in self.account.positions.items()
            },
            'trade_history': self.account.trade_history,
            'equity_history': [(t.isoformat(), e) for t, e in self._equity_history[-1000:]],
            'saved_at': datetime.now().isoformat()
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Paper trading state saved to {filepath}")

    def load_state(self, filepath: str):
        """Load paper trading state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.account.account_id = state['account_id']
        self.account.initial_balance = state['initial_balance']
        self.account.current_balance = state['current_balance']
        self.account.equity = state['equity']
        self.account.trade_history = state.get('trade_history', [])

        # Restore positions
        for k, v in state.get('positions', {}).items():
            self.account.positions[k] = PaperPosition(
                position_id=f"pos_{k}",
                market_id=v['market_id'],
                side=v['side'],
                entry_price=v['entry_price'],
                quantity=v['quantity'],
                current_price=v['current_price'],
                entry_time=datetime.now()
            )

        # Restore equity history
        self._equity_history = [
            (datetime.fromisoformat(t), e) for t, e in state.get('equity_history', [])
        ]

        logger.info(f"Paper trading state loaded from {filepath}")


async def test_paper_trading():
    """Test paper trading engine"""
    print("=== Paper Trading Test ===\n")

    engine = PaperTradingEngine(initial_balance=10000)

    # Simulate some trades
    print("Executing paper trades...")

    # Buy
    order1 = await engine.submit_order('market_a', 'buy', 100)
    print(f"Order 1: {order1.side} {order1.quantity} @ ${order1.average_price:.4f}")

    # Simulate price update
    engine._prices['market_a'] = 0.55

    # Sell
    order2 = await engine.submit_order('market_a', 'sell', 50)
    print(f"Order 2: {order2.side} {order2.quantity} @ ${order2.average_price:.4f}")

    # Another market
    order3 = await engine.submit_order('market_b', 'buy', 200, limit_price=0.48)
    print(f"Order 3: {order3.status.value}")

    # Update positions
    await engine.update_positions()

    # Get positions
    print("\nOpen Positions:")
    for pos in engine.get_positions():
        print(f"  {pos['market_id']}: {pos['side']} {pos['quantity']} "
              f"@ ${pos['entry_price']:.4f}, PnL: ${pos['unrealized_pnl']:.2f}")

    # Get performance
    perf = engine.get_performance()
    print("\nPerformance:")
    print(f"  Equity: ${perf['equity']:.2f}")
    print(f"  Total PnL: ${perf['total_pnl']:.2f}")
    print(f"  Return: {perf['return_pct']:.2%}")
    print(f"  Win Rate: {perf['win_rate']:.1%}")


if __name__ == "__main__":
    asyncio.run(test_paper_trading())
