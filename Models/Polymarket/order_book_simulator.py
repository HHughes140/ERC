"""
Order Book Simulator for Arbitrage Validation

Validates that arbitrage opportunities are actually executable before trading.

Key Functions:
1. Simulate execution across order book depth
2. Calculate slippage and actual fill prices
3. Validate that profit remains positive after execution
4. Determine optimal position size given liquidity

CRITICAL: Arbitrage detection without order book validation is DANGEROUS.
A "5% arbitrage" that can only fill 10 shares is worthless.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OrderBookLevel:
    """Single level in an order book"""
    price: float
    size: float  # Number of shares available at this price


@dataclass
class OrderBook:
    """Order book with bids and asks"""
    bids: List[OrderBookLevel]  # Buy orders (sorted high to low)
    asks: List[OrderBookLevel]  # Sell orders (sorted low to high)

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 1.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_pct(self) -> float:
        return self.spread / self.mid_price if self.mid_price > 0 else 0

    @property
    def total_bid_liquidity(self) -> float:
        return sum(level.size for level in self.bids)

    @property
    def total_ask_liquidity(self) -> float:
        return sum(level.size for level in self.asks)


@dataclass
class ExecutionResult:
    """Result of simulated order execution"""
    filled_shares: float
    average_price: float
    total_cost: float
    slippage: float  # Price impact vs best price
    slippage_pct: float
    fully_filled: bool
    levels_consumed: int
    execution_details: List[Dict]


@dataclass
class ArbitrageValidation:
    """Complete validation of an arbitrage opportunity"""
    is_valid: bool
    is_executable: bool
    optimal_size: float
    expected_profit: float
    expected_profit_pct: float
    yes_execution: ExecutionResult
    no_execution: ExecutionResult
    combined_slippage_pct: float
    reason: str


class OrderBookSimulator:
    """
    Simulates order execution and validates arbitrage opportunities.

    Key insight: Arbitrage only works if you can execute BOTH sides
    at prices that still yield profit after slippage.
    """

    def __init__(self, max_slippage_pct: float = 0.02, min_fill_ratio: float = 0.95):
        """
        Args:
            max_slippage_pct: Maximum acceptable slippage (default 2%)
            min_fill_ratio: Minimum portion of order that must fill (default 95%)
        """
        self.max_slippage_pct = max_slippage_pct
        self.min_fill_ratio = min_fill_ratio

    def simulate_market_buy(self, order_book: OrderBook,
                            target_shares: float) -> ExecutionResult:
        """
        Simulate buying shares by walking up the ask side of order book.

        This simulates a MARKET BUY order that fills at best available prices.
        """
        if not order_book.asks or target_shares <= 0:
            return ExecutionResult(
                filled_shares=0,
                average_price=0,
                total_cost=0,
                slippage=0,
                slippage_pct=0,
                fully_filled=False,
                levels_consumed=0,
                execution_details=[]
            )

        filled = 0.0
        total_cost = 0.0
        levels_consumed = 0
        details = []
        best_price = order_book.asks[0].price

        for level in order_book.asks:
            if filled >= target_shares:
                break

            available = level.size
            needed = target_shares - filled
            fill_amount = min(available, needed)

            cost = fill_amount * level.price
            total_cost += cost
            filled += fill_amount
            levels_consumed += 1

            details.append({
                'price': level.price,
                'size': fill_amount,
                'cost': cost
            })

        avg_price = total_cost / filled if filled > 0 else 0
        slippage = avg_price - best_price
        slippage_pct = slippage / best_price if best_price > 0 else 0

        return ExecutionResult(
            filled_shares=filled,
            average_price=avg_price,
            total_cost=total_cost,
            slippage=slippage,
            slippage_pct=slippage_pct,
            fully_filled=filled >= target_shares * self.min_fill_ratio,
            levels_consumed=levels_consumed,
            execution_details=details
        )

    def simulate_market_sell(self, order_book: OrderBook,
                             target_shares: float) -> ExecutionResult:
        """
        Simulate selling shares by walking down the bid side.
        """
        if not order_book.bids or target_shares <= 0:
            return ExecutionResult(
                filled_shares=0,
                average_price=0,
                total_cost=0,
                slippage=0,
                slippage_pct=0,
                fully_filled=False,
                levels_consumed=0,
                execution_details=[]
            )

        filled = 0.0
        total_revenue = 0.0
        levels_consumed = 0
        details = []
        best_price = order_book.bids[0].price

        for level in order_book.bids:
            if filled >= target_shares:
                break

            available = level.size
            needed = target_shares - filled
            fill_amount = min(available, needed)

            revenue = fill_amount * level.price
            total_revenue += revenue
            filled += fill_amount
            levels_consumed += 1

            details.append({
                'price': level.price,
                'size': fill_amount,
                'revenue': revenue
            })

        avg_price = total_revenue / filled if filled > 0 else 0
        slippage = best_price - avg_price  # Negative slippage = we got worse price
        slippage_pct = slippage / best_price if best_price > 0 else 0

        return ExecutionResult(
            filled_shares=filled,
            average_price=avg_price,
            total_cost=total_revenue,  # Revenue in this case
            slippage=slippage,
            slippage_pct=slippage_pct,
            fully_filled=filled >= target_shares * self.min_fill_ratio,
            levels_consumed=levels_consumed,
            execution_details=details
        )

    def validate_arbitrage(self, yes_book: OrderBook, no_book: OrderBook,
                          target_shares: float,
                          transaction_fee: float = 0.0) -> ArbitrageValidation:
        """
        Validate that an arbitrage opportunity is actually executable.

        For arbitrage to be valid:
        1. Must be able to fill BOTH sides at sufficient size
        2. Combined cost after slippage must still be < $1.00
        3. Net profit must be positive after fees

        Args:
            yes_book: Order book for YES outcome
            no_book: Order book for NO outcome
            target_shares: Desired number of arbitrage sets
            transaction_fee: Fee per transaction (fraction)

        Returns:
            ArbitrageValidation with complete analysis
        """
        # Simulate buying YES shares (we buy from ask side)
        yes_exec = self.simulate_market_buy(yes_book, target_shares)

        # Simulate buying NO shares
        no_exec = self.simulate_market_buy(no_book, target_shares)

        # Check if both sides can fill
        if not yes_exec.fully_filled or not no_exec.fully_filled:
            return ArbitrageValidation(
                is_valid=False,
                is_executable=False,
                optimal_size=0,
                expected_profit=0,
                expected_profit_pct=0,
                yes_execution=yes_exec,
                no_execution=no_exec,
                combined_slippage_pct=yes_exec.slippage_pct + no_exec.slippage_pct,
                reason=f"Insufficient liquidity: YES filled {yes_exec.filled_shares:.0f}, NO filled {no_exec.filled_shares:.0f}"
            )

        # Calculate actual filled amounts (use minimum for hedge)
        actual_shares = min(yes_exec.filled_shares, no_exec.filled_shares)

        # Calculate costs
        yes_cost_per_share = yes_exec.average_price
        no_cost_per_share = no_exec.average_price
        total_cost_per_set = yes_cost_per_share + no_cost_per_share

        # Apply transaction fees
        total_fees = total_cost_per_set * transaction_fee * 2  # Fees on both sides

        # Calculate profit
        payout_per_set = 1.0  # Guaranteed $1 per set
        profit_per_set = payout_per_set - total_cost_per_set - total_fees
        profit_pct = profit_per_set / total_cost_per_set if total_cost_per_set > 0 else 0

        # Check if still profitable after slippage
        is_profitable = profit_per_set > 0

        # Check combined slippage
        combined_slippage = yes_exec.slippage_pct + no_exec.slippage_pct
        acceptable_slippage = combined_slippage <= self.max_slippage_pct

        is_executable = is_profitable and acceptable_slippage

        if not is_profitable:
            reason = f"No profit after slippage: cost ${total_cost_per_set:.4f} >= $1.00"
        elif not acceptable_slippage:
            reason = f"Slippage too high: {combined_slippage:.2%} > {self.max_slippage_pct:.2%}"
        else:
            reason = f"Valid arbitrage: {profit_pct:.2%} profit per set"

        return ArbitrageValidation(
            is_valid=is_profitable,
            is_executable=is_executable,
            optimal_size=actual_shares,
            expected_profit=profit_per_set * actual_shares,
            expected_profit_pct=profit_pct,
            yes_execution=yes_exec,
            no_execution=no_exec,
            combined_slippage_pct=combined_slippage,
            reason=reason
        )

    def find_optimal_size(self, yes_book: OrderBook, no_book: OrderBook,
                          min_profit_pct: float = 0.005,
                          transaction_fee: float = 0.0) -> Dict:
        """
        Find the optimal position size that maximizes profit while
        maintaining minimum profit threshold.

        Uses binary search to find the largest size where profit % >= threshold.

        Args:
            yes_book: Order book for YES outcome
            no_book: Order book for NO outcome
            min_profit_pct: Minimum acceptable profit percentage
            transaction_fee: Fee per transaction

        Returns:
            Dict with optimal size and expected profit
        """
        # Determine maximum possible size from liquidity
        max_yes = yes_book.total_ask_liquidity
        max_no = no_book.total_ask_liquidity
        max_possible = min(max_yes, max_no)

        if max_possible <= 0:
            return {
                'optimal_size': 0,
                'expected_profit': 0,
                'profit_pct': 0,
                'reason': 'No liquidity'
            }

        # Binary search for optimal size
        low = 1.0
        high = max_possible
        best_size = 0
        best_profit = 0
        best_pct = 0

        while high - low > 1:
            mid = (low + high) / 2
            validation = self.validate_arbitrage(yes_book, no_book, mid, transaction_fee)

            if validation.is_executable and validation.expected_profit_pct >= min_profit_pct:
                best_size = mid
                best_profit = validation.expected_profit
                best_pct = validation.expected_profit_pct
                low = mid  # Try larger
            else:
                high = mid  # Try smaller

        # Final validation at best size
        if best_size > 0:
            final = self.validate_arbitrage(yes_book, no_book, best_size, transaction_fee)
            return {
                'optimal_size': best_size,
                'expected_profit': final.expected_profit,
                'profit_pct': final.expected_profit_pct,
                'yes_avg_price': final.yes_execution.average_price,
                'no_avg_price': final.no_execution.average_price,
                'combined_slippage': final.combined_slippage_pct,
                'reason': final.reason
            }

        return {
            'optimal_size': 0,
            'expected_profit': 0,
            'profit_pct': 0,
            'reason': 'No profitable size found'
        }

    def estimate_market_impact(self, order_book: OrderBook,
                               size: float) -> Dict:
        """
        Estimate the market impact of a trade.

        Returns:
            Dict with impact metrics
        """
        exec_result = self.simulate_market_buy(order_book, size)

        # Estimate how much our trade would move the market
        total_liquidity = order_book.total_ask_liquidity
        our_pct_of_book = size / total_liquidity if total_liquidity > 0 else 1.0

        # Price impact (how much worse than best price)
        price_impact = exec_result.slippage_pct

        # Levels consumed (market depth used)
        depth_consumed = exec_result.levels_consumed

        return {
            'size': size,
            'avg_price': exec_result.average_price,
            'best_price': order_book.best_ask,
            'slippage_pct': price_impact,
            'pct_of_book': our_pct_of_book,
            'levels_consumed': depth_consumed,
            'fully_fillable': exec_result.fully_filled,
            'market_impact_score': our_pct_of_book * 0.5 + price_impact * 0.5
        }


def create_order_book_from_clob(clob_data: Dict) -> Tuple[OrderBook, OrderBook]:
    """
    Create OrderBook objects from Polymarket CLOB API response.

    Args:
        clob_data: Response from Polymarket order book endpoint

    Returns:
        Tuple of (yes_book, no_book)
    """
    def parse_levels(levels: List) -> List[OrderBookLevel]:
        return [OrderBookLevel(price=float(l['price']), size=float(l['size']))
                for l in levels if float(l['size']) > 0]

    yes_bids = parse_levels(clob_data.get('yes_bids', []))
    yes_asks = parse_levels(clob_data.get('yes_asks', []))
    no_bids = parse_levels(clob_data.get('no_bids', []))
    no_asks = parse_levels(clob_data.get('no_asks', []))

    # Sort appropriately
    yes_bids.sort(key=lambda x: x.price, reverse=True)
    yes_asks.sort(key=lambda x: x.price)
    no_bids.sort(key=lambda x: x.price, reverse=True)
    no_asks.sort(key=lambda x: x.price)

    yes_book = OrderBook(bids=yes_bids, asks=yes_asks)
    no_book = OrderBook(bids=no_bids, asks=no_asks)

    return yes_book, no_book


def test_order_book_simulator():
    """Test the order book simulator"""
    # Create sample order books
    yes_book = OrderBook(
        bids=[
            OrderBookLevel(price=0.48, size=100),
            OrderBookLevel(price=0.47, size=200),
        ],
        asks=[
            OrderBookLevel(price=0.50, size=100),
            OrderBookLevel(price=0.51, size=150),
            OrderBookLevel(price=0.52, size=200),
        ]
    )

    no_book = OrderBook(
        bids=[
            OrderBookLevel(price=0.48, size=100),
            OrderBookLevel(price=0.47, size=200),
        ],
        asks=[
            OrderBookLevel(price=0.48, size=100),  # Total = 0.98, potential arb
            OrderBookLevel(price=0.49, size=150),
            OrderBookLevel(price=0.50, size=200),
        ]
    )

    simulator = OrderBookSimulator(max_slippage_pct=0.05)

    print("=== Order Book Simulator Test ===\n")

    # Test with small size
    validation = simulator.validate_arbitrage(yes_book, no_book, 50)
    print(f"50 shares: Valid={validation.is_valid}, Profit={validation.expected_profit:.2f}")
    print(f"  Reason: {validation.reason}")

    # Test with larger size (should have more slippage)
    validation = simulator.validate_arbitrage(yes_book, no_book, 200)
    print(f"\n200 shares: Valid={validation.is_valid}, Profit={validation.expected_profit:.2f}")
    print(f"  Reason: {validation.reason}")

    # Find optimal size
    optimal = simulator.find_optimal_size(yes_book, no_book, min_profit_pct=0.005)
    print(f"\nOptimal size: {optimal['optimal_size']:.0f} shares")
    print(f"  Expected profit: ${optimal['expected_profit']:.2f}")
    print(f"  Profit %: {optimal['profit_pct']:.2%}")


if __name__ == "__main__":
    test_order_book_simulator()
