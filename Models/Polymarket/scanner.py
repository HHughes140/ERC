"""
Arbitrage Scanner (v2.0)
Detects arbitrage opportunities across Polymarket and Kalshi

ENHANCEMENTS:
- Order book simulation for realistic execution modeling
- Bayesian probability estimation with confidence intervals
- Semantic matching for cross-platform market alignment
- Infrastructure integration (caching, logging, error handling)
"""
import asyncio
from typing import List, Dict, Optional, Set
from datetime import datetime
import logging

from .polymarket_client import PolymarketClient, GammaMarketsAPI, Market
from .kalshi_client import KalshiClient
from .math_models import ArbitrageDetector
from .arbitrage import ArbitrageOpportunity

# Import enhanced modules
try:
    from .order_book_simulator import OrderBookSimulator
    ORDER_BOOK_AVAILABLE = True
except ImportError:
    ORDER_BOOK_AVAILABLE = False

try:
    from .probability_models import ArbitrageProbabilityAnalyzer as ProbabilityEstimator
    PROBABILITY_AVAILABLE = True
except ImportError:
    PROBABILITY_AVAILABLE = False

try:
    from .semantic_matcher import SemanticMatcher
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

# Import infrastructure modules
try:
    from Infrastructure.cache import CacheManager
    from Infrastructure.logging_config import get_logger
    from Infrastructure.error_handling import with_retry, RateLimiter
    INFRASTRUCTURE_AVAILABLE = True
    logger = get_logger('arbitrage_scanner')
except ImportError:
    INFRASTRUCTURE_AVAILABLE = False
    logger = logging.getLogger(__name__)


class ArbitrageScanner:
    """
    Scans markets for arbitrage opportunities (v2.0)

    Enhanced with:
    - Order book simulation for realistic slippage estimation
    - Bayesian probability estimation for confidence intervals
    - Semantic matching for accurate cross-platform market pairing
    - Caching, logging, and error handling infrastructure
    """

    def __init__(self, config):
        self.config = config

        # Initialize detector
        self.detector = ArbitrageDetector(
            transaction_cost=config.MIN_PROFIT_THRESHOLD
        )

        # Initialize enhanced modules
        self.order_book_sim = OrderBookSimulator() if ORDER_BOOK_AVAILABLE else None
        self.prob_estimator = ProbabilityEstimator() if PROBABILITY_AVAILABLE else None
        self.semantic_matcher = SemanticMatcher() if SEMANTIC_AVAILABLE else None

        # Initialize infrastructure
        self.cache = CacheManager() if INFRASTRUCTURE_AVAILABLE else None
        self.rate_limiter = RateLimiter(rate=5, per=1.0) if INFRASTRUCTURE_AVAILABLE else None

        # Clients
        self.poly_client: Optional[PolymarketClient] = None
        self.gamma_client: Optional[GammaMarketsAPI] = None
        self.kalshi_client: Optional[KalshiClient] = None

        # State
        self.opportunities: List[ArbitrageOpportunity] = []
        self.market_cache: Dict[str, Market] = {}

        # Log module availability
        logger.info(f"Scanner initialized - OrderBook: {ORDER_BOOK_AVAILABLE}, "
                   f"Probability: {PROBABILITY_AVAILABLE}, "
                   f"Semantic: {SEMANTIC_AVAILABLE}, "
                   f"Infrastructure: {INFRASTRUCTURE_AVAILABLE}")
        
    async def initialize(self):
        """Initialize API clients"""
        # Polymarket
        self.poly_client = PolymarketClient(
            api_url=self.config.POLYMARKET_API_URL,
            gamma_url=self.config.GAMMA_API_URL,
            api_key=self.config.POLYMARKET_API_KEY,
            api_secret=self.config.POLYMARKET_SECRET,
            passphrase=self.config.POLYMARKET_PASSPHRASE
        )
        await self.poly_client.__aenter__()
        
        # Gamma API
        self.gamma_client = GammaMarketsAPI(
            base_url=self.config.GAMMA_API_URL
        )
        await self.gamma_client.__aenter__()
        
        # Kalshi (if enabled)
        if hasattr(self.config, 'KALSHI_ENABLED') and self.config.KALSHI_ENABLED:
            try:
                self.kalshi_client = KalshiClient(
                    api_url=self.config.KALSHI_API_URL,
                    api_key=self.config.KALSHI_API_KEY,
                    private_key_str=self.config.KALSHI_PRIVATE_KEY
                )
                await self.kalshi_client.__aenter__()
                logger.info("Kalshi client initialized for cross-platform arbitrage")
            except Exception as e:
                logger.error(f"Failed to initialize Kalshi: {e}")
                self.kalshi_client = None
        
        logger.info("Scanner initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.poly_client:
            await self.poly_client.__aexit__(None, None, None)
        if self.gamma_client:
            await self.gamma_client.__aexit__(None, None, None)
        if self.kalshi_client:
            await self.kalshi_client.__aexit__(None, None, None)
    
    async def scan_all(self) -> List[ArbitrageOpportunity]:
        """Scan all market types for arbitrage"""
        opportunities = []
        
        # Fetch Polymarket markets
        try:
            markets = await self.poly_client.get_markets(active_only=True, limit=500)
            logger.info(f"Fetched {len(markets)} Polymarket markets")
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []
        
        # Update cache
        for market in markets:
            self.market_cache[market.condition_id] = market
        
        # Scan different types
        tasks = [
            self.scan_single_market(markets),
            self.scan_multi_outcome(markets),
        ]
        
        # Add cross-platform if Kalshi available
        if self.kalshi_client:
            tasks.append(self.scan_cross_platform(markets))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                opportunities.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Scan error: {result}")
        
        logger.info(f"Found {len(opportunities)} raw opportunities")
        
        # Filter and rank
        opportunities = self._filter_opportunities(opportunities)
        opportunities = sorted(opportunities, key=lambda x: x.profit_pct, reverse=True)
        
        logger.info(f"After filtering: {len(opportunities)} opportunities")
        
        self.opportunities = opportunities
        return opportunities
    
    async def scan_single_market(self, markets: List[Market]) -> List[ArbitrageOpportunity]:
        """Scan for single-market arbitrage"""
        opportunities = []
        
        for market in markets:
            if len(market.outcomes) != 2:
                continue
            
            if not market.outcome_prices or len(market.outcome_prices) < 2:
                continue
            
            yes_price = market.outcome_prices[0]
            no_price = market.outcome_prices[1]
            
            # Validate prices
            if yes_price <= 0 or yes_price >= 1 or no_price <= 0 or no_price >= 1:
                continue
            
            try:
                arb = self.detector.detect_single_market(yes_price, no_price)
                
                if arb and arb['guaranteed_profit'] > self.config.MIN_PROFIT_THRESHOLD:
                    # Calculate risk score
                    risk_score = self._calculate_risk_score(market)
                    
                    # Calculate capital requirements
                    optimal = arb['optimal_allocation']
                    capital_required = optimal['total_cost']
                    
                    opp = ArbitrageOpportunity(
                        opportunity_id=f"single_{market.condition_id}_{int(datetime.now().timestamp())}",
                        arb_type='single_market',
                        platform='polymarket',
                        market_id=market.condition_id,
                        question=market.question,
                        prices={'yes': yes_price, 'no': no_price},
                        total_cost=yes_price + no_price,
                        guaranteed_profit=arb['guaranteed_profit'],
                        profit_pct=arb['profit_percentage'],
                        roi=optimal['roi_percent'],
                        recommended_shares={
                            'yes': optimal['yes_shares'],
                            'no': optimal['no_shares']
                        },
                        capital_required=capital_required,
                        risk_score=risk_score,
                        is_executable=risk_score < 0.5,
                        detected_at=datetime.now()
                    )
                    opportunities.append(opp)
                    
            except Exception as e:
                logger.debug(f"Error scanning market: {e}")
                continue
        
        return opportunities
    
    async def scan_multi_outcome(self, markets: List[Market]) -> List[ArbitrageOpportunity]:
        """Scan for multi-outcome arbitrage"""
        opportunities = []
        
        for market in markets:
            if len(market.outcomes) < 3:
                continue
            
            if not market.outcome_prices or len(market.outcome_prices) < 3:
                continue
            
            # Validate prices
            if any(p <= 0 or p >= 1 for p in market.outcome_prices):
                continue
            
            try:
                arb = self.detector.detect_multi_outcome(market.outcome_prices)
                
                if arb and arb['guaranteed_profit'] > self.config.MIN_PROFIT_THRESHOLD:
                    risk_score = self._calculate_risk_score(market)
                    
                    optimal = arb['optimal_allocation']
                    capital_required = optimal['total_cost']
                    
                    # Build recommended shares dict
                    recommended_shares = {
                        market.outcomes[i]: optimal['shares'][i]
                        for i in range(len(market.outcomes))
                    }
                    
                    opp = ArbitrageOpportunity(
                        opportunity_id=f"multi_{market.condition_id}_{int(datetime.now().timestamp())}",
                        arb_type='multi_outcome',
                        platform='polymarket',
                        market_id=market.condition_id,
                        question=market.question,
                        prices={market.outcomes[i]: market.outcome_prices[i] for i in range(len(market.outcomes))},
                        total_cost=sum(market.outcome_prices),
                        guaranteed_profit=arb['guaranteed_profit'],
                        profit_pct=arb['profit_percentage'],
                        roi=optimal['roi_percent'],
                        recommended_shares=recommended_shares,
                        capital_required=capital_required,
                        risk_score=risk_score,
                        is_executable=risk_score < 0.5,
                        detected_at=datetime.now()
                    )
                    opportunities.append(opp)
                    
            except Exception as e:
                logger.debug(f"Error scanning multi-outcome market: {e}")
                continue
        
        return opportunities
    
    async def scan_cross_platform(self, poly_markets: List[Market]) -> List[ArbitrageOpportunity]:
        """
        Scan for cross-platform arbitrage with Kalshi (v2.0)

        Enhanced with semantic matching for more accurate market pairing.
        """
        opportunities = []

        if not self.kalshi_client:
            return opportunities

        try:
            # Fetch Kalshi markets (with caching if available)
            cache_key = "kalshi_markets"
            kalshi_markets = None

            if self.cache:
                kalshi_markets = self.cache.get(cache_key, namespace='market_data')

            if kalshi_markets is None:
                kalshi_markets = await self.kalshi_client.get_markets()
                if self.cache:
                    self.cache.set(cache_key, kalshi_markets, namespace='market_data', ttl=30)

            logger.info(f"Fetched {len(kalshi_markets)} Kalshi markets")

            for pm_market in poly_markets:
                if len(pm_market.outcomes) != 2:
                    continue

                if len(pm_market.outcome_prices) < 2:
                    continue

                pm_yes = pm_market.outcome_prices[0]
                pm_no = pm_market.outcome_prices[1]

                # Validate Polymarket prices
                if pm_yes <= 0 or pm_yes >= 1 or pm_no <= 0 or pm_no >= 1:
                    continue

                # Find matching Kalshi markets using semantic matching (v2.0)
                for k_market in kalshi_markets:
                    # Validate Kalshi prices
                    if k_market.yes_price <= 0 or k_market.yes_price >= 1:
                        continue
                    if k_market.no_price <= 0 or k_market.no_price >= 1:
                        continue

                    # Use semantic matcher if available (v2.0 enhancement)
                    if self.semantic_matcher:
                        match_result = self.semantic_matcher.match_markets(
                            pm_market.question,
                            k_market.question
                        )
                        similarity = match_result['similarity']
                        is_inverted = match_result.get('is_inverted', False)

                        # Require 80% semantic similarity (up from 30% word overlap)
                        if similarity < 0.80:
                            continue

                        logger.debug(f"Semantic match ({similarity:.2f}): "
                                   f"PM: {pm_market.question[:40]} | "
                                   f"K: {k_market.question[:40]}")
                    else:
                        # Fallback to simple word overlap
                        pm_question = pm_market.question.lower()
                        k_question = k_market.question.lower()

                        pm_words = set(pm_question.split()) - {'will', 'the', 'be', 'a', 'an', 'in', 'on'}
                        k_words = set(k_question.split()) - {'will', 'the', 'be', 'a', 'an', 'in', 'on'}

                        if not pm_words or not k_words:
                            continue

                        overlap = len(pm_words & k_words)
                        similarity = overlap / max(len(pm_words), len(k_words))
                        is_inverted = False

                        if similarity < 0.3:  # At least 30% word overlap
                            continue
                        # Strategy 1: Buy PM YES + Kalshi NO
                        cost1 = pm_yes + k_market.no_price
                        if cost1 < 0.98:
                            profit = 1.0 - cost1
                            
                            opp = ArbitrageOpportunity(
                                opportunity_id=f"cross_{pm_market.condition_id}_{k_market.ticker}_{int(datetime.now().timestamp())}",
                                arb_type='cross_platform',
                                platform='polymarket+kalshi',
                                market_id=f"{pm_market.condition_id}|{k_market.ticker}",
                                question=f"PM: {pm_market.question} | Kalshi: {k_market.title}",
                                prices={
                                    'pm_yes': pm_yes,
                                    'kalshi_no': k_market.no_price
                                },
                                total_cost=cost1,
                                guaranteed_profit=profit,
                                profit_pct=(profit / cost1) * 100,
                                roi=(profit / cost1) * 100,
                                recommended_shares={
                                    'pm_yes': 100,  # Example
                                    'kalshi_no': 100
                                },
                                capital_required=cost1 * 100,
                                risk_score=0.4,  # Higher risk for cross-platform
                                is_executable=True,
                                detected_at=datetime.now()
                            )
                            opportunities.append(opp)
                            logger.info(f"Cross-platform arbitrage: {profit*100:.2f}% profit")
                        
                        # Strategy 2: Buy Kalshi YES + PM NO
                        cost2 = k_market.yes_price + pm_no
                        if cost2 < 0.98:
                            profit = 1.0 - cost2
                            
                            opp = ArbitrageOpportunity(
                                opportunity_id=f"cross_{pm_market.condition_id}_{k_market.ticker}_{int(datetime.now().timestamp())}_rev",
                                arb_type='cross_platform',
                                platform='polymarket+kalshi',
                                market_id=f"{pm_market.condition_id}|{k_market.ticker}",
                                question=f"PM: {pm_market.question} | Kalshi: {k_market.title}",
                                prices={
                                    'kalshi_yes': k_market.yes_price,
                                    'pm_no': pm_no
                                },
                                total_cost=cost2,
                                guaranteed_profit=profit,
                                profit_pct=(profit / cost2) * 100,
                                roi=(profit / cost2) * 100,
                                recommended_shares={
                                    'kalshi_yes': 100,
                                    'pm_no': 100
                                },
                                capital_required=cost2 * 100,
                                risk_score=0.4,
                                is_executable=True,
                                detected_at=datetime.now()
                            )
                            opportunities.append(opp)
        
        except Exception as e:
            logger.error(f"Cross-platform scan error: {e}")
        
        return opportunities
    
    def _calculate_risk_score(self, market: Market) -> float:
        """
        Calculate risk score for a market (v2.0)

        Enhanced with probability estimation if available.
        """
        # Liquidity risk
        liquidity_risk = 1.0 / (1.0 + market.liquidity / 1000)

        # Volume risk
        volume_risk = 1.0 / (1.0 + market.volume / 5000)

        # Probability confidence risk (v2.0 enhancement)
        prob_risk = 0.0
        if self.prob_estimator and len(market.outcome_prices) >= 2:
            yes_price = market.outcome_prices[0]
            # Get probability distribution
            prob_dist = self.prob_estimator.estimate_implied_probability(
                price=yes_price,
                volume=market.volume,
                liquidity=market.liquidity
            )
            # Higher variance = higher risk
            if prob_dist:
                variance = prob_dist.get('variance', 0)
                prob_risk = min(variance * 2, 0.3)  # Cap at 0.3

        # Combined risk (adjusted weights for v2.0)
        risk = (
            liquidity_risk * 0.4 +
            volume_risk * 0.3 +
            prob_risk * 0.3
        )

        return min(risk, 1.0)

    def _filter_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """
        Filter opportunities by quality (v2.0)

        Enhanced with order book simulation for execution validation.
        """
        filtered = []

        for opp in opportunities:
            # Must meet minimum profit
            if opp.profit_pct < self.config.MIN_PROFIT_THRESHOLD * 100:
                continue

            # Must have reasonable risk
            if opp.risk_score > 0.8:
                continue

            # Order book validation (v2.0 enhancement)
            if self.order_book_sim and opp.arb_type == 'single_market':
                # Simulate execution to check if profit survives slippage
                market = self.market_cache.get(opp.market_id)
                if market:
                    execution_result = self.order_book_sim.estimate_execution(
                        prices=opp.prices,
                        target_shares=opp.recommended_shares.get('yes', 100),
                        liquidity=market.liquidity
                    )

                    # Adjust profit for slippage
                    if execution_result:
                        slippage = execution_result.get('estimated_slippage', 0)
                        adjusted_profit = opp.profit_pct - (slippage * 100)

                        if adjusted_profit < self.config.MIN_PROFIT_THRESHOLD * 100:
                            logger.debug(f"Filtered out {opp.opportunity_id}: "
                                       f"profit {opp.profit_pct:.2f}% -> {adjusted_profit:.2f}% after slippage")
                            continue

                        # Store adjusted values
                        opp.metadata = opp.metadata or {}
                        opp.metadata['slippage_estimate'] = slippage
                        opp.metadata['adjusted_profit_pct'] = adjusted_profit

            # Probability confidence check (v2.0 enhancement)
            if self.prob_estimator and opp.arb_type != 'cross_platform':
                # Calculate confidence interval
                prices = opp.prices
                if 'yes' in prices and 'no' in prices:
                    arb_confidence = self.prob_estimator.calculate_arbitrage_confidence(
                        yes_price=prices['yes'],
                        no_price=prices['no']
                    )

                    # Filter out low-confidence opportunities
                    if arb_confidence and arb_confidence.get('confidence', 0) < 0.7:
                        logger.debug(f"Filtered out {opp.opportunity_id}: "
                                   f"low arbitrage confidence {arb_confidence.get('confidence', 0):.2f}")
                        continue

                    opp.metadata = opp.metadata or {}
                    opp.metadata['arb_confidence'] = arb_confidence

            filtered.append(opp)

        return filtered
    
    def get_best_opportunities(self, limit: int = 10) -> List[ArbitrageOpportunity]:
        """Get best opportunities"""
        return sorted(self.opportunities, key=lambda x: x.profit_pct, reverse=True)[:limit]