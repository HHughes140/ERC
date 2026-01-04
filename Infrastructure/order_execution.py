"""
Smart Order Execution Engine for ERC Trading System

Provides:
- Order splitting for large orders
- Limit order placement
- Time-weighted execution (TWAP)
- Slippage estimation
- Execution quality analysis

CRITICAL: Large market orders move the market against you.
Smart execution preserves edge.
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    LIMIT_IOC = "limit_ioc"  # Immediate or cancel
    TWAP = "twap"            # Time-weighted average


class OrderStatus(Enum):
    """Order statuses"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ExecutionSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Order representation"""
    order_id: str
    market_id: str
    side: ExecutionSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    time_in_force: str = "GTC"  # GTC, IOC, FOK

    # Execution constraints
    max_slippage_pct: float = 0.02
    urgency: float = 0.5  # 0 = patient, 1 = urgent

    # State
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Child orders (for split execution)
    child_orders: List[str] = field(default_factory=list)

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    @property
    def is_complete(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED,
                               OrderStatus.REJECTED, OrderStatus.EXPIRED]

    @property
    def fill_rate(self) -> float:
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0


@dataclass
class Fill:
    """Single fill/execution"""
    fill_id: str
    order_id: str
    price: float
    quantity: float
    timestamp: datetime
    fee: float = 0.0

    @property
    def notional(self) -> float:
        return self.price * self.quantity


@dataclass
class ExecutionPlan:
    """Plan for executing an order"""
    strategy: str  # 'single', 'split', 'twap', 'iceberg'
    child_orders: List[Dict]
    estimated_avg_price: float
    estimated_slippage: float
    estimated_time_seconds: float
    reason: str


@dataclass
class OrderBookSnapshot:
    """Current order book state"""
    market_id: str
    timestamp: datetime
    best_bid: float
    best_ask: float
    bid_depth: List[tuple]  # [(price, size), ...]
    ask_depth: List[tuple]

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_pct(self) -> float:
        return self.spread / self.mid_price if self.mid_price > 0 else 0


class ExecutionEngine:
    """
    Smart order execution engine.

    Determines optimal execution strategy based on:
    - Order size relative to liquidity
    - Urgency
    - Market conditions
    """

    # Thresholds
    SMALL_ORDER_THRESHOLD = 0.05  # 5% of book = small order
    LARGE_ORDER_THRESHOLD = 0.20  # 20% of book = large order

    def __init__(self,
                 submit_order_func: Optional[Callable] = None,
                 get_order_book_func: Optional[Callable] = None,
                 default_fee_rate: float = 0.02):
        """
        Args:
            submit_order_func: Function to submit orders to exchange
            get_order_book_func: Function to get current order book
            default_fee_rate: Default transaction fee rate
        """
        self.submit_order = submit_order_func
        self.get_order_book = get_order_book_func
        self.fee_rate = default_fee_rate

        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []

        # Execution stats
        self.stats = {
            'total_orders': 0,
            'total_filled': 0,
            'total_slippage': 0.0,
            'avg_slippage_pct': 0.0
        }

    def plan_execution(self, order: Order,
                       order_book: OrderBookSnapshot) -> ExecutionPlan:
        """
        Create execution plan for an order.

        Args:
            order: Order to execute
            order_book: Current order book state

        Returns:
            ExecutionPlan with strategy and child orders
        """
        # Calculate order size relative to available liquidity
        if order.side == ExecutionSide.BUY:
            available_liquidity = sum(size for _, size in order_book.ask_depth)
            target_price = order_book.best_ask
        else:
            available_liquidity = sum(size for _, size in order_book.bid_depth)
            target_price = order_book.best_bid

        if available_liquidity == 0:
            return ExecutionPlan(
                strategy='reject',
                child_orders=[],
                estimated_avg_price=0,
                estimated_slippage=1.0,
                estimated_time_seconds=0,
                reason='No liquidity available'
            )

        size_ratio = order.quantity / available_liquidity

        # Determine strategy based on size and urgency
        if size_ratio < self.SMALL_ORDER_THRESHOLD:
            # Small order - single market order is fine
            return self._plan_single_order(order, order_book, target_price)

        elif size_ratio < self.LARGE_ORDER_THRESHOLD:
            # Medium order - use limit order slightly aggressive
            return self._plan_limit_order(order, order_book, target_price)

        else:
            # Large order - split execution
            if order.urgency > 0.7:
                return self._plan_iceberg(order, order_book, target_price)
            else:
                return self._plan_twap(order, order_book, target_price)

    def _plan_single_order(self, order: Order, book: OrderBookSnapshot,
                           target_price: float) -> ExecutionPlan:
        """Plan single market order execution"""
        # Estimate fill price walking the book
        estimated_price, slippage = self._estimate_execution_price(
            order.quantity, book, order.side
        )

        return ExecutionPlan(
            strategy='single',
            child_orders=[{
                'type': 'market',
                'quantity': order.quantity,
                'price': None
            }],
            estimated_avg_price=estimated_price,
            estimated_slippage=slippage,
            estimated_time_seconds=1,
            reason='Small order - immediate execution'
        )

    def _plan_limit_order(self, order: Order, book: OrderBookSnapshot,
                          target_price: float) -> ExecutionPlan:
        """Plan limit order execution"""
        # Set limit price slightly better than market to increase fill probability
        if order.side == ExecutionSide.BUY:
            # Buy slightly above best ask
            limit_price = book.best_ask * (1 + 0.001)
        else:
            # Sell slightly below best bid
            limit_price = book.best_bid * (1 - 0.001)

        return ExecutionPlan(
            strategy='limit',
            child_orders=[{
                'type': 'limit',
                'quantity': order.quantity,
                'price': limit_price
            }],
            estimated_avg_price=limit_price,
            estimated_slippage=(limit_price - target_price) / target_price,
            estimated_time_seconds=30,
            reason='Medium order - aggressive limit'
        )

    def _plan_iceberg(self, order: Order, book: OrderBookSnapshot,
                      target_price: float) -> ExecutionPlan:
        """Plan iceberg order execution (hidden quantity)"""
        # Split into visible chunks
        chunk_size = order.quantity * 0.1  # Show 10% at a time
        num_chunks = int(np.ceil(order.quantity / chunk_size))

        child_orders = []
        for i in range(num_chunks):
            qty = min(chunk_size, order.quantity - i * chunk_size)
            child_orders.append({
                'type': 'limit',
                'quantity': qty,
                'price': target_price,
                'hidden': True
            })

        _, slippage = self._estimate_execution_price(order.quantity, book, order.side)

        return ExecutionPlan(
            strategy='iceberg',
            child_orders=child_orders,
            estimated_avg_price=target_price * (1 + slippage),
            estimated_slippage=slippage * 0.7,  # Iceberg reduces impact
            estimated_time_seconds=60 * num_chunks,
            reason='Large urgent order - iceberg to hide size'
        )

    def _plan_twap(self, order: Order, book: OrderBookSnapshot,
                   target_price: float) -> ExecutionPlan:
        """Plan TWAP execution over time"""
        # Execute over 10 intervals
        num_intervals = 10
        interval_seconds = 60  # 1 minute intervals
        chunk_size = order.quantity / num_intervals

        child_orders = []
        for i in range(num_intervals):
            child_orders.append({
                'type': 'market',
                'quantity': chunk_size,
                'delay_seconds': i * interval_seconds
            })

        _, base_slippage = self._estimate_execution_price(
            chunk_size, book, order.side
        )

        return ExecutionPlan(
            strategy='twap',
            child_orders=child_orders,
            estimated_avg_price=target_price * (1 + base_slippage),
            estimated_slippage=base_slippage,
            estimated_time_seconds=num_intervals * interval_seconds,
            reason='Large patient order - TWAP to minimize impact'
        )

    def _estimate_execution_price(self, quantity: float,
                                   book: OrderBookSnapshot,
                                   side: ExecutionSide) -> tuple:
        """
        Estimate execution price by walking the order book.

        Returns:
            (estimated_avg_price, slippage_pct)
        """
        if side == ExecutionSide.BUY:
            levels = book.ask_depth
            best_price = book.best_ask
        else:
            levels = book.bid_depth
            best_price = book.best_bid

        if not levels:
            return best_price, 0.0

        remaining = quantity
        total_cost = 0.0

        for price, size in levels:
            fill_qty = min(remaining, size)
            total_cost += fill_qty * price
            remaining -= fill_qty

            if remaining <= 0:
                break

        if remaining > 0:
            # Order exceeds book depth
            total_cost += remaining * levels[-1][0] * 1.05  # Assume 5% worse

        avg_price = total_cost / quantity
        slippage = (avg_price - best_price) / best_price

        return avg_price, abs(slippage)

    async def execute(self, order: Order) -> Order:
        """
        Execute an order using the planned strategy.

        Args:
            order: Order to execute

        Returns:
            Updated order with fill information
        """
        if not self.submit_order:
            logger.warning("No submit_order function configured - simulating")
            return self._simulate_execution(order)

        # Get current order book
        if self.get_order_book:
            book = await self.get_order_book(order.market_id)
        else:
            # Create dummy book for planning
            book = OrderBookSnapshot(
                market_id=order.market_id,
                timestamp=datetime.now(),
                best_bid=0.49 if order.limit_price else 0.50,
                best_ask=0.51 if order.limit_price else 0.50,
                bid_depth=[(0.49, 1000), (0.48, 2000)],
                ask_depth=[(0.51, 1000), (0.52, 2000)]
            )

        # Create execution plan
        plan = self.plan_execution(order, book)

        if plan.strategy == 'reject':
            order.status = OrderStatus.REJECTED
            return order

        logger.info(
            f"Executing order {order.order_id}: "
            f"{plan.strategy} strategy, "
            f"est. slippage {plan.estimated_slippage:.2%}"
        )

        # Execute based on strategy
        if plan.strategy == 'single':
            order = await self._execute_single(order, plan)
        elif plan.strategy == 'limit':
            order = await self._execute_limit(order, plan)
        elif plan.strategy == 'twap':
            order = await self._execute_twap(order, plan)
        elif plan.strategy == 'iceberg':
            order = await self._execute_iceberg(order, plan)

        # Update stats
        self._update_stats(order)

        return order

    async def _execute_single(self, order: Order, plan: ExecutionPlan) -> Order:
        """Execute single market order"""
        child = plan.child_orders[0]

        result = await self.submit_order(
            market_id=order.market_id,
            side=order.side.value,
            order_type='market',
            quantity=child['quantity']
        )

        if result.get('success'):
            order.status = OrderStatus.FILLED
            order.filled_quantity = result.get('filled_qty', order.quantity)
            order.average_price = result.get('avg_price', plan.estimated_avg_price)
        else:
            order.status = OrderStatus.REJECTED

        return order

    async def _execute_limit(self, order: Order, plan: ExecutionPlan) -> Order:
        """Execute limit order"""
        child = plan.child_orders[0]

        result = await self.submit_order(
            market_id=order.market_id,
            side=order.side.value,
            order_type='limit',
            quantity=child['quantity'],
            price=child['price']
        )

        if result.get('success'):
            order.status = OrderStatus.SUBMITTED
            # Would need to poll for fills in production
            order.filled_quantity = result.get('filled_qty', 0)
            order.average_price = result.get('avg_price', child['price'])
        else:
            order.status = OrderStatus.REJECTED

        return order

    async def _execute_twap(self, order: Order, plan: ExecutionPlan) -> Order:
        """Execute TWAP over time"""
        total_filled = 0.0
        total_cost = 0.0

        for i, child in enumerate(plan.child_orders):
            if i > 0:
                await asyncio.sleep(child.get('delay_seconds', 60))

            result = await self.submit_order(
                market_id=order.market_id,
                side=order.side.value,
                order_type='market',
                quantity=child['quantity']
            )

            if result.get('success'):
                filled = result.get('filled_qty', child['quantity'])
                price = result.get('avg_price', plan.estimated_avg_price)
                total_filled += filled
                total_cost += filled * price

        order.filled_quantity = total_filled
        order.average_price = total_cost / total_filled if total_filled > 0 else 0
        order.status = OrderStatus.FILLED if total_filled > 0 else OrderStatus.REJECTED

        return order

    async def _execute_iceberg(self, order: Order, plan: ExecutionPlan) -> Order:
        """Execute iceberg order"""
        # Similar to TWAP but with limit orders
        total_filled = 0.0
        total_cost = 0.0

        for child in plan.child_orders:
            result = await self.submit_order(
                market_id=order.market_id,
                side=order.side.value,
                order_type='limit',
                quantity=child['quantity'],
                price=child['price']
            )

            if result.get('success'):
                filled = result.get('filled_qty', child['quantity'])
                price = result.get('avg_price', child['price'])
                total_filled += filled
                total_cost += filled * price

            # Small delay between chunks
            await asyncio.sleep(5)

        order.filled_quantity = total_filled
        order.average_price = total_cost / total_filled if total_filled > 0 else 0
        order.status = OrderStatus.FILLED if total_filled >= order.quantity * 0.95 else OrderStatus.PARTIAL

        return order

    def _simulate_execution(self, order: Order) -> Order:
        """Simulate order execution for testing"""
        # Assume immediate fill with small slippage
        slippage = np.random.uniform(0, order.max_slippage_pct)

        if order.limit_price:
            fill_price = order.limit_price
        else:
            base_price = 0.50  # Dummy
            if order.side == ExecutionSide.BUY:
                fill_price = base_price * (1 + slippage)
            else:
                fill_price = base_price * (1 - slippage)

        order.filled_quantity = order.quantity
        order.average_price = fill_price
        order.status = OrderStatus.FILLED

        return order

    def _update_stats(self, order: Order):
        """Update execution statistics"""
        self.stats['total_orders'] += 1

        if order.status == OrderStatus.FILLED:
            self.stats['total_filled'] += 1

            if order.limit_price:
                slippage = (order.average_price - order.limit_price) / order.limit_price
                self.stats['total_slippage'] += abs(slippage)
                self.stats['avg_slippage_pct'] = (
                    self.stats['total_slippage'] / self.stats['total_filled']
                )

    def get_execution_quality(self, order: Order,
                              benchmark_price: float) -> Dict:
        """
        Analyze execution quality.

        Args:
            order: Executed order
            benchmark_price: Price at order creation (e.g., mid price)

        Returns:
            Quality metrics
        """
        if order.average_price == 0:
            return {'error': 'Order not filled'}

        slippage = (order.average_price - benchmark_price) / benchmark_price

        # Implementation shortfall
        expected_cost = benchmark_price * order.quantity
        actual_cost = order.average_price * order.filled_quantity
        shortfall = (actual_cost - expected_cost) / expected_cost

        # Fee impact
        fee_impact = self.fee_rate * order.filled_quantity * order.average_price

        return {
            'slippage_pct': slippage,
            'implementation_shortfall': shortfall,
            'fee_impact': fee_impact,
            'total_cost': actual_cost + fee_impact,
            'fill_rate': order.fill_rate,
            'time_to_fill': (order.updated_at - order.created_at).total_seconds()
        }


def test_execution_engine():
    """Test the execution engine"""
    print("=== Smart Order Execution Test ===\n")

    engine = ExecutionEngine()

    # Create test order book
    book = OrderBookSnapshot(
        market_id="test_market",
        timestamp=datetime.now(),
        best_bid=0.48,
        best_ask=0.52,
        bid_depth=[(0.48, 100), (0.47, 200), (0.46, 500)],
        ask_depth=[(0.52, 100), (0.53, 200), (0.54, 500)]
    )

    print(f"Order Book: bid={book.best_bid}, ask={book.best_ask}, spread={book.spread_pct:.2%}")

    # Test different order sizes
    test_cases = [
        ("Small order", 30),
        ("Medium order", 150),
        ("Large order", 600),
    ]

    for name, qty in test_cases:
        order = Order(
            order_id=f"test_{qty}",
            market_id="test_market",
            side=ExecutionSide.BUY,
            order_type=OrderType.MARKET,
            quantity=qty,
            urgency=0.5
        )

        plan = engine.plan_execution(order, book)

        print(f"\n{name} ({qty} shares):")
        print(f"  Strategy: {plan.strategy}")
        print(f"  Est. avg price: ${plan.estimated_avg_price:.4f}")
        print(f"  Est. slippage: {plan.estimated_slippage:.2%}")
        print(f"  Est. time: {plan.estimated_time_seconds}s")
        print(f"  Reason: {plan.reason}")

        if plan.child_orders:
            print(f"  Child orders: {len(plan.child_orders)}")

    # Test execution simulation
    print("\n--- Simulated Execution ---")
    order = Order(
        order_id="sim_test",
        market_id="test_market",
        side=ExecutionSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100,
        max_slippage_pct=0.02
    )

    filled_order = engine._simulate_execution(order)
    print(f"Order filled: {filled_order.filled_quantity} @ ${filled_order.average_price:.4f}")

    quality = engine.get_execution_quality(filled_order, 0.50)
    print(f"Execution quality: slippage={quality['slippage_pct']:.2%}")


if __name__ == "__main__":
    test_execution_engine()
