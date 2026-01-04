"""
Correlation Module
Tracks correlations between markets and positions for risk management
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class CorrelationPair:
    """Correlation between two markets/assets"""
    market_a: str
    market_b: str
    correlation: float  # -1 to 1
    sample_size: int
    last_updated: datetime

    def to_dict(self) -> Dict:
        return {
            'market_a': self.market_a,
            'market_b': self.market_b,
            'correlation': self.correlation,
            'sample_size': self.sample_size,
            'last_updated': self.last_updated.isoformat()
        }


class CorrelationTracker:
    """
    Tracks correlations between prediction markets

    Features:
    - Price change correlation tracking
    - Category-based correlation assumptions
    - Portfolio correlation analysis
    - Diversification scoring
    """

    def __init__(self):
        # Correlation matrix (market_id -> market_id -> correlation)
        self.correlations: Dict[str, Dict[str, float]] = {}

        # Price history for correlation calculation
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}

        # Category correlations (assumed correlations by category)
        self.category_correlations: Dict[str, Dict[str, float]] = {
            'crypto': {'crypto': 0.7, 'politics': 0.1, 'sports': 0.0, 'weather': 0.0},
            'politics': {'crypto': 0.1, 'politics': 0.5, 'sports': 0.0, 'weather': 0.0},
            'sports': {'crypto': 0.0, 'politics': 0.0, 'sports': 0.3, 'weather': 0.0},
            'weather': {'crypto': 0.0, 'politics': 0.0, 'sports': 0.0, 'weather': 0.2},
        }

        # Minimum samples for reliable correlation
        self.min_samples = 20

        logger.info("Correlation Tracker initialized")

    def record_price(self, market_id: str, price: float, timestamp: datetime = None):
        """Record a price observation for a market"""
        if timestamp is None:
            timestamp = datetime.now()

        if market_id not in self.price_history:
            self.price_history[market_id] = []

        self.price_history[market_id].append((timestamp, price))

        # Keep limited history (last 100 observations)
        if len(self.price_history[market_id]) > 100:
            self.price_history[market_id] = self.price_history[market_id][-100:]

    def calculate_correlation(self, market_a: str, market_b: str) -> Optional[float]:
        """
        Calculate correlation between two markets

        Returns:
            Correlation coefficient (-1 to 1) or None if insufficient data
        """
        if market_a not in self.price_history or market_b not in self.price_history:
            return None

        prices_a = self.price_history[market_a]
        prices_b = self.price_history[market_b]

        # Align time series (simple approach: use most recent shared period)
        min_len = min(len(prices_a), len(prices_b))
        if min_len < self.min_samples:
            return None

        # Calculate price changes
        changes_a = []
        changes_b = []

        for i in range(1, min_len):
            changes_a.append(prices_a[-min_len + i][1] - prices_a[-min_len + i - 1][1])
            changes_b.append(prices_b[-min_len + i][1] - prices_b[-min_len + i - 1][1])

        if not changes_a or not changes_b:
            return None

        # Calculate correlation
        n = len(changes_a)
        mean_a = sum(changes_a) / n
        mean_b = sum(changes_b) / n

        numerator = sum((changes_a[i] - mean_a) * (changes_b[i] - mean_b) for i in range(n))

        var_a = sum((x - mean_a) ** 2 for x in changes_a)
        var_b = sum((x - mean_b) ** 2 for x in changes_b)

        denominator = math.sqrt(var_a * var_b)

        if denominator == 0:
            return 0.0

        correlation = numerator / denominator

        # Store in matrix
        if market_a not in self.correlations:
            self.correlations[market_a] = {}
        self.correlations[market_a][market_b] = correlation

        return correlation

    def get_correlation(self, market_a: str, market_b: str) -> float:
        """Get correlation between two markets (from cache or calculate)"""
        # Same market
        if market_a == market_b:
            return 1.0

        # Check cache
        if market_a in self.correlations and market_b in self.correlations[market_a]:
            return self.correlations[market_a][market_b]

        # Try reverse lookup
        if market_b in self.correlations and market_a in self.correlations[market_b]:
            return self.correlations[market_b][market_a]

        # Calculate if possible
        calc_corr = self.calculate_correlation(market_a, market_b)
        if calc_corr is not None:
            return calc_corr

        # Fall back to category-based assumption
        return self._get_category_correlation(market_a, market_b)

    def _get_category_correlation(self, market_a: str, market_b: str) -> float:
        """Get assumed correlation based on market categories"""
        cat_a = self._categorize_market(market_a)
        cat_b = self._categorize_market(market_b)

        if cat_a in self.category_correlations:
            return self.category_correlations[cat_a].get(cat_b, 0.0)

        return 0.0

    def _categorize_market(self, market_id: str) -> str:
        """Categorize a market based on its ID/name"""
        market_lower = market_id.lower()

        if any(kw in market_lower for kw in ['bitcoin', 'btc', 'eth', 'crypto', 'sol']):
            return 'crypto'
        elif any(kw in market_lower for kw in ['election', 'president', 'vote', 'trump', 'biden']):
            return 'politics'
        elif any(kw in market_lower for kw in ['nfl', 'nba', 'game', 'match', 'team', 'win']):
            return 'sports'
        elif any(kw in market_lower for kw in ['weather', 'temperature', 'rain', 'snow']):
            return 'weather'

        return 'other'

    def calculate_portfolio_correlation(self, positions: List[Dict]) -> float:
        """
        Calculate overall portfolio correlation

        Higher values indicate more concentrated/correlated portfolio

        Args:
            positions: List of position dicts with 'market_id' and 'capital'

        Returns:
            Portfolio correlation (0 to 1)
        """
        if len(positions) <= 1:
            return 1.0

        total_capital = sum(p.get('capital', 0) for p in positions)
        if total_capital == 0:
            return 0.0

        # Calculate weighted average pairwise correlation
        total_weighted_corr = 0.0
        total_weight = 0.0

        for i, pos_i in enumerate(positions):
            for j, pos_j in enumerate(positions):
                if i >= j:
                    continue

                market_i = pos_i.get('market_id', pos_i.get('symbol', ''))
                market_j = pos_j.get('market_id', pos_j.get('symbol', ''))

                correlation = self.get_correlation(market_i, market_j)

                # Weight by position sizes
                weight_i = pos_i.get('capital', 0) / total_capital
                weight_j = pos_j.get('capital', 0) / total_capital
                weight = weight_i * weight_j

                total_weighted_corr += abs(correlation) * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_weighted_corr / total_weight

    def get_diversification_score(self, positions: List[Dict]) -> float:
        """
        Get diversification score (0 to 1)
        Higher = better diversified
        """
        portfolio_corr = self.calculate_portfolio_correlation(positions)

        # Invert correlation to get diversification
        # 0 correlation = 1 diversification, 1 correlation = 0 diversification
        return 1.0 - portfolio_corr

    def find_uncorrelated_opportunities(self, current_positions: List[Dict],
                                        opportunities: List[Any],
                                        max_correlation: float = 0.3) -> List[Any]:
        """
        Find opportunities that are uncorrelated with current positions

        Args:
            current_positions: Current portfolio positions
            opportunities: Available opportunities
            max_correlation: Maximum correlation threshold

        Returns:
            Filtered list of uncorrelated opportunities
        """
        if not current_positions:
            return opportunities

        uncorrelated = []

        for opp in opportunities:
            opp_market = getattr(opp, 'market_id', '')
            if not opp_market and hasattr(opp, 'market'):
                opp_market = getattr(opp.market, 'condition_id', '')

            # Check correlation with each position
            is_uncorrelated = True

            for pos in current_positions:
                pos_market = pos.get('market_id', pos.get('symbol', ''))

                correlation = self.get_correlation(opp_market, pos_market)

                if abs(correlation) > max_correlation:
                    is_uncorrelated = False
                    break

            if is_uncorrelated:
                uncorrelated.append(opp)

        return uncorrelated

    def get_correlation_matrix(self, market_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """Get correlation matrix for specified markets"""
        matrix = {}

        for market_a in market_ids:
            matrix[market_a] = {}
            for market_b in market_ids:
                matrix[market_a][market_b] = self.get_correlation(market_a, market_b)

        return matrix

    def print_report(self, positions: List[Dict] = None):
        """Print correlation report"""
        print("\n" + "=" * 50)
        print("CORRELATION REPORT")
        print("=" * 50)
        print(f"Tracked Markets: {len(self.price_history)}")
        print(f"Correlation Pairs: {sum(len(v) for v in self.correlations.values())}")

        if positions:
            div_score = self.get_diversification_score(positions)
            port_corr = self.calculate_portfolio_correlation(positions)

            print(f"\nPortfolio Analysis:")
            print(f"  Positions:           {len(positions)}")
            print(f"  Portfolio Correlation: {port_corr:.2f}")
            print(f"  Diversification Score: {div_score:.2f}")

        print("=" * 50 + "\n")
