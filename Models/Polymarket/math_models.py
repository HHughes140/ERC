"""
Mathematical Models for Arbitrage Detection and Optimization
Advanced arbitrage detection with proper mathematical foundations
"""
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ArbitrageDetector:
    """Advanced mathematical arbitrage detection"""
    
    def __init__(self, transaction_cost: float = 0.005):
        self.transaction_cost = transaction_cost
        
    def detect_single_market(self, yes_price: float, no_price: float) -> Optional[Dict]:
        """
        Single-market arbitrage detection
        Returns arbitrage if YES + NO < 1 - transaction_costs
        """
        total_cost = yes_price + no_price
        threshold = 1.0 - (2 * self.transaction_cost)
        
        if total_cost < threshold:
            profit = 1.0 - total_cost
            return {
                "type": "single_market",
                "yes_price": yes_price,
                "no_price": no_price,
                "cost": total_cost,
                "guaranteed_profit": profit,
                "profit_percentage": (profit / total_cost) * 100,
                "optimal_allocation": self._optimize_single_market(yes_price, no_price)
            }
        return None
    
    def _optimize_single_market(self, yes_price: float, no_price: float) -> Dict:
        """
        Optimize position sizing for single market arbitrage
        CRITICAL: Must buy EQUAL shares of YES and NO for perfect hedge
        
        For arbitrage: Shares_YES == Shares_NO
        Cost = Shares * (Yes_Price + No_Price)
        Payout = Shares * $1.00 (guaranteed)
        """
        total_unit_cost = yes_price + no_price
        
        # Calculate ROI for this arbitrage
        roi = (1.0 - total_unit_cost) / total_unit_cost if total_unit_cost > 0 else 0
        
        # Conservative Kelly fraction (for liquidity/risk management)
        kelly_fraction = min(roi * 0.25, 0.5)
        
        # Example bankroll (in production, pass this as parameter)
        max_bankroll = 1000.0
        bankroll_to_deploy = max_bankroll * kelly_fraction
        
        # Calculate equal shares for perfect hedge
        # Shares = Bankroll / (Yes_Price + No_Price)
        num_shares = bankroll_to_deploy / total_unit_cost if total_unit_cost > 0 else 0
        
        return {
            "yes_shares": num_shares,      # EQUAL shares
            "no_shares": num_shares,       # EQUAL shares (CRITICAL for hedge)
            "total_cost": num_shares * total_unit_cost,
            "guaranteed_payout": num_shares * 1.0,
            "net_profit": num_shares * (1.0 - total_unit_cost),
            "kelly_fraction": kelly_fraction,
            "roi_percent": roi * 100
        }
    
    def detect_multi_outcome(self, prices: List[float]) -> Optional[Dict]:
        """
        Multi-outcome arbitrage detection
        Sum of all outcome prices < 1 - transaction_costs
        """
        n_outcomes = len(prices)
        total_cost = sum(prices)
        threshold = 1.0 - (n_outcomes * self.transaction_cost)
        
        if total_cost < threshold:
            profit = 1.0 - total_cost
            optimal = self._optimize_multi_outcome(prices)
            
            return {
                "type": "multi_outcome",
                "prices": prices,
                "n_outcomes": n_outcomes,
                "total_cost": total_cost,
                "guaranteed_profit": profit,
                "profit_percentage": (profit / total_cost) * 100,
                "optimal_allocation": optimal
            }
        return None
    
    def _optimize_multi_outcome(self, prices: List[float]) -> Dict:
        """
        Optimize multi-outcome arbitrage
        
        For mutually exclusive outcomes, buy EQUAL shares of EVERY outcome
        to guarantee payout regardless of which outcome wins.
        """
        n = len(prices)
        total_unit_cost = sum(prices)
        
        # Check for arbitrage
        if total_unit_cost >= 1.0:
            return {
                "shares": [0] * n,
                "expected_cost": total_unit_cost,
                "expected_profit": 0,
                "note": "No arbitrage - sum of prices >= 1.0"
            }
        
        # Calculate equal shares for perfect hedge
        max_bankroll = 1000.0
        profit_per_set = 1.0 - total_unit_cost
        roi = profit_per_set / total_unit_cost if total_unit_cost > 0 else 0
        kelly_fraction = min(roi * 0.25, 0.5)
        
        bankroll_to_deploy = max_bankroll * kelly_fraction
        num_shares = bankroll_to_deploy / total_unit_cost if total_unit_cost > 0 else 0
        
        # Equal shares for all outcomes (perfect hedge)
        shares = [num_shares] * n
        
        return {
            "shares": shares,
            "num_shares_per_outcome": num_shares,
            "total_cost": num_shares * total_unit_cost,
            "guaranteed_payout": num_shares * 1.0,
            "expected_profit": num_shares * profit_per_set,
            "roi_percent": roi * 100
        }
    
    def detect_cross_platform(self, platform1_yes: float, 
                             platform2_no: float) -> Optional[Dict]:
        """
        Cross-platform arbitrage
        YES on platform1 + NO on platform2 < 1
        """
        total_cost = platform1_yes + platform2_no
        threshold = 1.0 - (2 * self.transaction_cost) - 0.01  # Extra buffer
        
        if total_cost < threshold:
            profit = 1.0 - total_cost
            return {
                "type": "cross_platform",
                "platform1_yes": platform1_yes,
                "platform2_no": platform2_no,
                "total_cost": total_cost,
                "guaranteed_profit": profit,
                "profit_percentage": (profit / total_cost) * 100
            }
        return None


class PositionSizer:
    """
    CORRECTED Position Sizing for Prediction Markets

    CRITICAL DISTINCTION:
    1. ARBITRAGE positions are DETERMINISTIC - use liquidity-constrained sizing
    2. DIRECTIONAL positions are UNCERTAIN - use Kelly Criterion

    Kelly Criterion DOES NOT APPLY to risk-free arbitrage because:
    - Kelly formula: f* = edge / variance
    - For arbitrage: variance = 0, so f* = undefined (division by zero)
    - Arbitrage size should be MAX(liquidity_allows) not Kelly-optimized

    This class properly handles both cases.
    """

    def __init__(self, fractional_kelly: float = 0.25, max_position_pct: float = 0.50):
        self.fractional_kelly = fractional_kelly  # Conservative Kelly (quarter-Kelly)
        self.max_position_pct = max_position_pct  # Max % of bankroll per position

    def size_arbitrage_position(self, cost_per_set: float,
                                 profit_per_set: float,
                                 bankroll: float,
                                 liquidity_limit: Optional[float] = None,
                                 execution_confidence: float = 1.0) -> Dict:
        """
        Size a DETERMINISTIC arbitrage position.

        For true arbitrage (guaranteed profit), position size is constrained by:
        1. Available bankroll
        2. Order book liquidity
        3. Execution confidence (probability of successful execution)
        4. Platform position limits

        NOT by Kelly Criterion (which is for uncertain bets).

        Args:
            cost_per_set: Cost of one complete arbitrage set (YES + NO)
            profit_per_set: Guaranteed profit per set
            bankroll: Total available capital
            liquidity_limit: Maximum position size allowed by order book
            execution_confidence: Probability of successful execution (0-1)

        Returns:
            Dict with position sizing details
        """
        if cost_per_set <= 0 or profit_per_set <= 0:
            return {
                'num_sets': 0,
                'total_cost': 0,
                'expected_profit': 0,
                'sizing_method': 'arbitrage',
                'is_valid': False,
                'reason': 'Invalid cost or profit'
            }

        # Calculate maximum possible sets from bankroll
        max_sets_from_bankroll = bankroll / cost_per_set

        # Apply position limit (don't put more than X% in one trade)
        max_sets_from_limit = (bankroll * self.max_position_pct) / cost_per_set

        # Apply liquidity constraint if provided
        if liquidity_limit:
            max_sets_from_liquidity = liquidity_limit
        else:
            max_sets_from_liquidity = float('inf')

        # Take the minimum of all constraints
        num_sets = min(max_sets_from_bankroll, max_sets_from_limit, max_sets_from_liquidity)

        # Apply execution confidence (reduce size if execution is uncertain)
        num_sets *= execution_confidence

        # Round down to avoid fractional shares issues
        num_sets = int(num_sets)

        total_cost = num_sets * cost_per_set
        expected_profit = num_sets * profit_per_set
        roi_pct = (profit_per_set / cost_per_set) * 100 if cost_per_set > 0 else 0

        return {
            'num_sets': num_sets,
            'total_cost': total_cost,
            'expected_profit': expected_profit,
            'roi_pct': roi_pct,
            'sizing_method': 'arbitrage_liquidity_constrained',
            'is_valid': num_sets > 0,
            'constraints': {
                'bankroll_limit': max_sets_from_bankroll,
                'position_limit': max_sets_from_limit,
                'liquidity_limit': max_sets_from_liquidity if liquidity_limit else 'unlimited',
                'execution_confidence': execution_confidence
            }
        }

    def size_directional_position(self, entry_price: float,
                                   win_probability: float,
                                   bankroll: float,
                                   platform_fees: float = 0.02) -> Dict:
        """
        Size a DIRECTIONAL (uncertain) position using proper Kelly Criterion.

        For prediction markets:
        - Win: Receive $1.00 per share (profit = 1 - entry_price - fees)
        - Lose: Receive $0.00 per share (loss = entry_price)

        Kelly formula for binary bets:
        f* = (p * b - q) / b

        Where:
        - p = probability of winning
        - q = 1 - p = probability of losing
        - b = win_payout / loss_amount (odds)

        Args:
            entry_price: Price per share (0-1)
            win_probability: Estimated probability of winning (0-1)
            bankroll: Total available capital
            platform_fees: Estimated fees as fraction (default 2%)

        Returns:
            Dict with Kelly-optimized position sizing
        """
        # Validate inputs
        if not (0 < entry_price < 1):
            return {'num_shares': 0, 'is_valid': False, 'reason': 'Invalid entry price'}
        if not (0 < win_probability < 1):
            return {'num_shares': 0, 'is_valid': False, 'reason': 'Invalid probability'}

        # Calculate payoffs
        win_payout = (1.0 - entry_price) * (1 - platform_fees)  # Net win after fees
        loss_amount = entry_price  # Lose entire stake

        # Calculate odds (b in Kelly formula)
        b = win_payout / loss_amount if loss_amount > 0 else 0

        p = win_probability
        q = 1 - p

        # Kelly formula: f* = (p*b - q) / b
        # Simplified: f* = p - q/b
        edge = p * win_payout - q * loss_amount

        if edge <= 0:
            return {
                'num_shares': 0,
                'kelly_fraction': 0,
                'is_valid': False,
                'reason': 'Negative expected value',
                'expected_value': edge
            }

        # Calculate Kelly fraction
        kelly_fraction_raw = (p * b - q) / b if b > 0 else 0

        # Apply fractional Kelly (more conservative)
        kelly_fraction_adj = kelly_fraction_raw * self.fractional_kelly

        # Apply maximum position constraint
        kelly_fraction_final = min(kelly_fraction_adj, self.max_position_pct)

        # Calculate position size
        position_dollars = bankroll * kelly_fraction_final
        num_shares = position_dollars / entry_price

        return {
            'num_shares': num_shares,
            'position_dollars': position_dollars,
            'entry_price': entry_price,
            'kelly_fraction_raw': kelly_fraction_raw,
            'kelly_fraction_adjusted': kelly_fraction_adj,
            'kelly_fraction_final': kelly_fraction_final,
            'expected_value': edge * num_shares,
            'expected_value_pct': (edge / entry_price) * 100,
            'max_loss': position_dollars,
            'potential_profit': num_shares * win_payout,
            'sizing_method': 'kelly_criterion',
            'is_valid': num_shares > 0,
            'win_probability': p,
            'odds': b
        }

    def size_uncertain_arbitrage(self, cost_per_set: float,
                                  profit_per_set: float,
                                  execution_success_prob: float,
                                  execution_failure_loss: float,
                                  bankroll: float) -> Dict:
        """
        Size arbitrage with UNCERTAIN execution (e.g., cross-platform).

        When execution is not guaranteed (MEV risk, latency, etc.),
        arbitrage becomes an uncertain bet and Kelly DOES apply.

        Args:
            cost_per_set: Cost of arbitrage set
            profit_per_set: Profit if execution succeeds
            execution_success_prob: Probability of successful execution
            execution_failure_loss: Amount lost if execution fails
            bankroll: Available capital

        Returns:
            Kelly-optimized position for uncertain arbitrage
        """
        p = execution_success_prob
        q = 1 - p

        # Payoffs
        win_payout = profit_per_set
        loss_amount = execution_failure_loss

        # Edge
        edge = p * win_payout - q * loss_amount

        if edge <= 0:
            return {
                'num_sets': 0,
                'is_valid': False,
                'reason': 'Negative EV due to execution risk',
                'expected_value': edge
            }

        # Kelly for uncertain arbitrage
        b = win_payout / loss_amount if loss_amount > 0 else 0
        kelly_fraction = (p * b - q) / b if b > 0 else 0

        # Apply fractional Kelly
        kelly_fraction = min(kelly_fraction * self.fractional_kelly, self.max_position_pct)

        position_dollars = bankroll * kelly_fraction
        num_sets = position_dollars / cost_per_set if cost_per_set > 0 else 0

        return {
            'num_sets': num_sets,
            'position_dollars': position_dollars,
            'kelly_fraction': kelly_fraction,
            'expected_value': edge * num_sets,
            'sizing_method': 'kelly_uncertain_arbitrage',
            'is_valid': num_sets > 0,
            'execution_probability': p,
            'risk_adjusted_roi': (edge / cost_per_set) * 100
        }


class KellyOptimizer:
    """
    CORRECTED Kelly Criterion Optimizer

    Kelly Criterion for binary bets:
    f* = (p * b - q) / b

    Where:
    - p = probability of winning
    - q = 1 - p = probability of losing
    - b = net odds (win amount / lose amount)

    IMPORTANT: Kelly does NOT apply to risk-free arbitrage.
    Use PositionSizer.size_arbitrage_position() for arbitrage.
    """

    def __init__(self, fraction: float = 0.25):
        self.kelly_fraction = fraction

    def calculate_kelly_fraction(self, win_prob: float, odds: float) -> float:
        """
        Calculate Kelly fraction for a binary bet.

        Args:
            win_prob: Probability of winning (0-1)
            odds: Net odds (profit if win / loss if lose)

        Returns:
            Optimal fraction of bankroll to bet
        """
        p = win_prob
        q = 1 - p
        b = odds

        if b <= 0:
            return 0

        # Kelly formula
        kelly = (p * b - q) / b

        # Never bet more than Kelly suggests
        if kelly <= 0:
            return 0

        # Apply fractional Kelly for safety
        return min(kelly * self.kelly_fraction, 0.5)

    def calculate_for_prediction_market(self, entry_price: float,
                                         win_prob: float,
                                         fees: float = 0.02) -> Dict:
        """
        Calculate Kelly for prediction market position.

        Args:
            entry_price: Price paid per share (e.g., 0.70 for 70 cents)
            win_prob: Estimated win probability
            fees: Platform fees as fraction

        Returns:
            Kelly analysis with recommended fraction
        """
        win_payout = (1.0 - entry_price) * (1 - fees)
        loss_amount = entry_price
        odds = win_payout / loss_amount if loss_amount > 0 else 0

        kelly = self.calculate_kelly_fraction(win_prob, odds)

        # Calculate expected value
        ev = win_prob * win_payout - (1 - win_prob) * loss_amount

        return {
            'kelly_fraction': kelly,
            'expected_value_per_dollar': ev,
            'odds': odds,
            'is_positive_ev': ev > 0,
            'recommendation': 'BET' if kelly > 0 else 'SKIP'
        }

    def multi_asset_kelly(self, returns: np.ndarray,
                          covariance: np.ndarray) -> np.ndarray:
        """
        Multi-asset Kelly criterion for portfolio optimization.

        f* = Σ^(-1) * μ

        Note: This is for diversified betting, not arbitrage.
        """
        try:
            # Check for singular matrix
            if np.linalg.cond(covariance) > 1e10:
                logger.warning("Covariance matrix near-singular, using equal weights")
                return np.ones(len(returns)) / len(returns)

            inv_cov = np.linalg.inv(covariance)
            kelly_weights = inv_cov @ returns

            # Apply fractional Kelly
            kelly_weights *= self.kelly_fraction

            # Clip negative weights (no shorting)
            kelly_weights = np.maximum(kelly_weights, 0)

            # Normalize
            total = np.sum(kelly_weights)
            if total > 0:
                kelly_weights = kelly_weights / total
            else:
                kelly_weights = np.ones(len(returns)) / len(returns)

            return kelly_weights

        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix singular, using equal weights")
            return np.ones(len(returns)) / len(returns)


class RiskManager:
    """Risk management for prediction markets (Binary/Bernoulli distribution)"""
    
    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence
    
    def calculate_binary_var(self, position_size: float, win_prob: float) -> float:
        """
        Calculate Value at Risk for binary outcome (prediction market)
        
        Prediction markets are Bernoulli, NOT Gaussian
        - Win: Get $1.00 per share
        - Lose: Get $0.00 per share
        
        For ARBITRAGE: VaR = 0 (excluding smart contract risk)
        For DIRECTIONAL: VaR = position_size * (1 - win_prob)
        """
        loss_probability = 1 - win_prob
        
        # VaR is the potential loss at confidence level
        if loss_probability > (1 - self.confidence):
            var = position_size  # Could lose entire position
        else:
            var = 0  # Below confidence threshold
        
        return var
    
    def calculate_arbitrage_risk(self, yes_shares: float, no_shares: float,
                                yes_price: float, no_price: float) -> Dict:
        """
        Calculate risk for arbitrage position
        
        For PERFECT HEDGE (yes_shares == no_shares):
        - Market Risk: 0
        - Smart Contract Risk: Position dependent
        - Execution Risk: Slippage dependent
        """
        total_cost = yes_shares * yes_price + no_shares * no_price
        guaranteed_payout = min(yes_shares, no_shares) * 1.0
        
        # Check if properly hedged
        is_hedged = abs(yes_shares - no_shares) < 0.01
        
        if is_hedged:
            market_risk = 0.0
            max_loss = 0.0  # Arbitrage is risk-free
        else:
            # Unhedged position
            imbalance = abs(yes_shares - no_shares)
            market_risk = imbalance * max(yes_price, no_price)
            max_loss = market_risk
        
        return {
            "is_hedged": is_hedged,
            "market_risk": market_risk,
            "max_loss": max_loss,
            "expected_profit": guaranteed_payout - total_cost,
            "var_95": max_loss,
            "cvar_95": max_loss
        }


class ExecutionOptimizer:
    """Execution optimizer for atomic blockchain transactions"""
    
    def __init__(self, max_slippage: float = 0.01):
        self.max_slippage = max_slippage
    
    def calculate_execution_price(self, order_book: List[Tuple[float, float]], 
                                  target_size: float) -> Dict:
        """
        Calculate weighted average execution price from order book
        
        On blockchain (Polymarket CLOB), execution is ATOMIC
        - Either transaction succeeds at current prices
        - Or it fails (front-run by MEV bot)
        """
        if not order_book or target_size <= 0:
            return {
                "average_price": 0,
                "total_cost": 0,
                "filled": 0,
                "slippage": 0,
                "executable": False
            }
        
        # Sort order book by price (best first)
        sorted_book = sorted(order_book, key=lambda x: x[0])
        
        filled = 0
        total_cost = 0
        
        for price, size in sorted_book:
            if filled >= target_size:
                break
            
            fill_amount = min(size, target_size - filled)
            total_cost += fill_amount * price
            filled += fill_amount
        
        if filled > 0:
            avg_price = total_cost / filled
            best_price = sorted_book[0][0]
            slippage = (avg_price - best_price) / best_price if best_price > 0 else 0
            executable = slippage <= self.max_slippage and filled >= target_size
        else:
            avg_price = 0
            slippage = 0
            executable = False
        
        return {
            "average_price": avg_price,
            "total_cost": total_cost,
            "filled": filled,
            "requested": target_size,
            "slippage": slippage,
            "slippage_percent": slippage * 100,
            "executable": executable,
            "sufficient_liquidity": filled >= target_size
        }
    
    def check_arbitrage_executable(self, yes_book: List[Tuple[float, float]],
                                   no_book: List[Tuple[float, float]],
                                   target_shares: float) -> Dict:
        """
        Check if arbitrage is executable given order book depth
        
        Must fill BOTH sides atomically (single transaction)
        """
        yes_execution = self.calculate_execution_price(yes_book, target_shares)
        no_execution = self.calculate_execution_price(no_book, target_shares)
        
        # Both sides must be executable
        executable = (yes_execution["executable"] and 
                     no_execution["executable"])
        
        if executable:
            total_cost = yes_execution["total_cost"] + no_execution["total_cost"]
            guaranteed_payout = target_shares * 1.0
            net_profit = guaranteed_payout - total_cost
            
            return {
                "executable": True,
                "yes_avg_price": yes_execution["average_price"],
                "no_avg_price": no_execution["average_price"],
                "total_cost": total_cost,
                "guaranteed_payout": guaranteed_payout,
                "net_profit": net_profit,
                "combined_slippage": yes_execution["slippage"] + no_execution["slippage"]
            }
        
        return {
            "executable": False,
            "reason": "Insufficient liquidity or excessive slippage",
            "yes_executable": yes_execution["executable"],
            "no_executable": no_execution["executable"]
        }


class PolymarketArbitrage:
    """
    Mathematically Correct Arbitrage for Polymarket
    
    Key Principles:
    1. Equal shares for perfect hedge (YES shares == NO shares)
    2. Binary outcomes (Bernoulli, not Gaussian)
    3. Atomic execution (no time-based splitting)
    """
    
    def __init__(self, transaction_fee: float = 0.0):
        self.fee = transaction_fee
    
    def detect_and_size(self, yes_price: float, no_price: float, 
                       max_bankroll: float = 1000.0) -> Optional[Dict]:
        """
        Detect arbitrage and calculate exact position sizes
        
        Returns None if no arbitrage exists
        """
        # 1. Check for arbitrage opportunity
        cost_of_one_set = yes_price + no_price
        breakeven = 1.0 - (2 * self.fee)  # Fees on both sides
        
        if cost_of_one_set >= breakeven:
            return None  # No arbitrage
        
        # 2. Calculate profit per set
        # Buy 1 YES + 1 NO = guaranteed $1.00 payout
        profit_per_set = 1.0 - cost_of_one_set
        roi = profit_per_set / cost_of_one_set
        
        # 3. Size position (liquidity constrained)
        num_sets = max_bankroll / cost_of_one_set if cost_of_one_set > 0 else 0
        
        return {
            "action": "execute",
            "buy_yes_qty": num_sets,      # CRITICAL: Equal quantities
            "buy_no_qty": num_sets,       # CRITICAL: Equal quantities
            "yes_price": yes_price,
            "no_price": no_price,
            "total_cost": num_sets * cost_of_one_set,
            "guaranteed_revenue": num_sets * 1.0,
            "net_profit": num_sets * profit_per_set,
            "roi_percent": roi * 100,
            "profit_per_dollar": profit_per_set / cost_of_one_set
        }
    
    def validate_hedge(self, yes_shares: float, no_shares: float) -> Dict:
        """
        Validate that position is properly hedged
        
        For pure arbitrage: YES shares MUST equal NO shares
        """
        is_hedged = abs(yes_shares - no_shares) < 0.001
        imbalance = abs(yes_shares - no_shares)
        
        if is_hedged:
            risk_level = "ZERO"
            message = "Perfect hedge - guaranteed profit"
        elif imbalance < yes_shares * 0.05:  # <5% imbalance
            risk_level = "LOW"
            message = f"Minor imbalance: {imbalance:.2f} shares exposed"
        else:
            risk_level = "HIGH"
            message = f"DANGER: {imbalance:.2f} shares unhedged"
        
        return {
            "is_hedged": is_hedged,
            "imbalance": imbalance,
            "risk_level": risk_level,
            "message": message
        }


def test_arbitrage_detection():
    """Test arbitrage detection functions"""
    detector = ArbitrageDetector(transaction_cost=0.005)
    
    # Test single market
    print("=== Single Market Test ===")
    arb = detector.detect_single_market(yes_price=0.45, no_price=0.50)
    if arb:
        print(f"Arbitrage found! Profit: {arb['guaranteed_profit']:.4f}")
        print(f"Allocation: {arb['optimal_allocation']}")
    
    # Test multi-outcome
    print("\n=== Multi-Outcome Test ===")
    arb = detector.detect_multi_outcome(prices=[0.30, 0.35, 0.25])
    if arb:
        print(f"Arbitrage found! Profit: {arb['guaranteed_profit']:.4f}")
        print(f"Optimal shares: {arb['optimal_allocation']['shares']}")


if __name__ == "__main__":
    test_arbitrage_detection()