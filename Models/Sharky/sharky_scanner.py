"""
Sharky Near-Certainty Scalping Strategy - MULTI-PLATFORM (v2.1)
Works on both Polymarket AND Kalshi

Strategy:
1. Near-certainty scalping (80% capital): 97%+ certain outcomes at 97-99¢
2. Anti-extreme directional (20% capital): Bet against unlikely price targets

IMPROVEMENTS (v2.0):
- Historical calibration instead of circular certainty logic
- Fee-adjusted profit calculations
- Multi-factor certainty estimation
- Volatility penalty for bid-ask spread

IMPROVEMENTS (v2.1):
- Exit engine integration for dynamic position management
- Infrastructure modules (caching, logging, error handling)
- Portfolio risk manager integration
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.Polymarket.polymarket_client import PolymarketClient, GammaMarketsAPI, Market
from Models.Polymarket.kalshi_client import KalshiClient
from config import Config

# Import calibration module
try:
    from Factors.calibration import (
        HistoricalCalibrator, FeeAdjustedProfitCalculator,
        get_calibrator, record_trade_outcome
    )
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    logging.warning("Calibration module not available, using fallback logic")

# Import exit engine (v2.1)
try:
    from Models.Sharky.exit_engine import DynamicExitEngine as ExitEngine, ExitSignal
    EXIT_ENGINE_AVAILABLE = True
except ImportError:
    EXIT_ENGINE_AVAILABLE = False

# Import infrastructure modules (v2.1)
try:
    from Infrastructure.cache import CacheManager
    from Infrastructure.logging_config import get_logger
    from Infrastructure.error_handling import with_retry, RateLimiter
    from Infrastructure.portfolio_risk import PortfolioRiskManager
    INFRASTRUCTURE_AVAILABLE = True
    logger = get_logger('sharky_scanner')
except ImportError:
    INFRASTRUCTURE_AVAILABLE = False
    logger = logging.getLogger(__name__)


class SharkyOpportunity:
    """Represents a Sharky-style trading opportunity"""
    
    def __init__(self, market: Market, opportunity_type: str, 
                 certainty: float, entry_price: float, profit_potential: float,
                 platform: str = 'polymarket'):  # NEW: Track platform
        self.market = market
        self.opportunity_type = opportunity_type  # 'scalp' or 'directional'
        self.certainty = certainty
        self.entry_price = entry_price
        self.profit_potential = profit_potential
        self.detected_at = datetime.now()
        self.position_side = None  # 'Yes' or 'No'
        self.platform = platform  # 'polymarket' or 'kalshi'
        
    def __repr__(self):
        return (f"SharkyOpportunity({self.platform.upper()}, {self.opportunity_type}, "
                f"{self.market.question[:50]}, "
                f"certainty={self.certainty:.2%}, "
                f"profit={self.profit_potential:.2%})")


class CertaintyAnalyzer:
    """
    ENHANCED Certainty Analyzer (v2.0)

    Replaces circular logic (certainty = market_price) with multi-factor estimation:
    1. Historical calibration - what % of markets at this price actually win?
    2. Volatility penalty - high bid-ask spread = more uncertainty
    3. Market type adjustment - some types are more predictable
    4. Time decay factor - closer to resolution = more reliable

    CRITICAL FIX: No longer uses market price directly as certainty!
    """

    def __init__(self):
        self.min_certainty = 0.92  # Lowered since we now adjust for calibration
        self.calibrator = get_calibrator() if CALIBRATION_AVAILABLE else None
        self.fee_calculators = {
            'polymarket': FeeAdjustedProfitCalculator('polymarket') if CALIBRATION_AVAILABLE else None,
            'kalshi': FeeAdjustedProfitCalculator('kalshi') if CALIBRATION_AVAILABLE else None,
        }

    def analyze_market(self, market: Market, platform: str = 'polymarket') -> Optional[Dict]:
        """
        Analyze a market using multi-factor certainty estimation.

        Returns: {
            'certainty': float (0-1) - CALIBRATED, not raw price
            'raw_certainty': float - original market price
            'side': 'Yes' or 'No',
            'reasoning': str,
            'confidence': float - how confident in this estimate
            'net_profit_if_win': float - after fees
            'is_positive_ev': bool
        }
        """
        # Determine market type
        market_type = self._classify_market(market)
        if market_type is None:
            return None

        # Get raw market signal
        raw_analysis = self._get_raw_signal(market, market_type)
        if raw_analysis is None:
            return None

        raw_certainty = raw_analysis['raw_certainty']
        side = raw_analysis['side']

        # Factor 1: Historical calibration adjustment
        if self.calibrator:
            calibration = self.calibrator.get_calibrated_certainty(
                raw_certainty, market_type, platform
            )
            calibrated_certainty = calibration['calibrated_probability']
            calibration_confidence = calibration['confidence']
        else:
            # Fallback: apply conservative haircut
            calibrated_certainty = raw_certainty * 0.95  # 5% haircut
            calibration_confidence = 0.3

        # Factor 2: Volatility/spread penalty
        volatility_penalty = self._get_volatility_penalty(market)
        adjusted_certainty = calibrated_certainty - volatility_penalty

        # Factor 3: Market type reliability adjustment
        type_adjustment = self._get_type_adjustment(market_type)
        adjusted_certainty *= type_adjustment

        # Ensure certainty is in valid range
        final_certainty = max(0.5, min(0.99, adjusted_certainty))

        # Factor 4: Fee-adjusted profit check
        if side == 'Yes':
            entry_price = market.outcome_prices[0] if market.outcome_prices else 0
        else:
            entry_price = market.outcome_prices[1] if len(market.outcome_prices) > 1 else 0

        fee_calc = self.fee_calculators.get(platform)
        if fee_calc:
            profit_result = fee_calc.calculate_net_profit(entry_price, 1.0, True)
            ev_result = fee_calc.expected_value(entry_price, final_certainty)
            net_profit = profit_result['net_profit_pct']
            is_positive_ev = ev_result['is_positive_ev']
        else:
            # Fallback: simple calculation
            net_profit = (1.0 - entry_price) - 0.02  # Assume 2% fees
            is_positive_ev = final_certainty * net_profit > (1 - final_certainty) * entry_price

        # Skip if not positive EV
        if not is_positive_ev:
            logger.debug(f"Skipping {market.question[:40]}: negative EV")
            return None

        # Skip if certainty below threshold
        if final_certainty < self.min_certainty:
            return None

        return {
            'certainty': final_certainty,
            'raw_certainty': raw_certainty,
            'calibrated_certainty': calibrated_certainty,
            'side': side,
            'reasoning': raw_analysis['reasoning'],
            'market_type': market_type,
            'confidence': calibration_confidence * type_adjustment,
            'volatility_penalty': volatility_penalty,
            'net_profit_if_win': net_profit,
            'is_positive_ev': is_positive_ev,
            'entry_price': entry_price
        }

    def _classify_market(self, market: Market) -> Optional[str]:
        """Classify market type"""
        if self._is_directional_market(market):
            return 'directional'
        elif self._is_range_market(market):
            return 'range'
        elif self._is_sports_market(market):
            return 'sports'
        elif self._is_extreme_target(market):
            return 'extreme'
        return None

    def _get_raw_signal(self, market: Market, market_type: str) -> Optional[Dict]:
        """Get raw market signal (which side favored and by how much)"""
        if len(market.outcomes) != 2 or len(market.outcome_prices) < 2:
            return None

        yes_price = market.outcome_prices[0]
        no_price = market.outcome_prices[1]

        # Determine which side is favored
        if yes_price > no_price and yes_price >= 0.90:
            return {
                'raw_certainty': yes_price,
                'side': 'Yes',
                'reasoning': f'{market_type.capitalize()} market favors Yes'
            }
        elif no_price > yes_price and no_price >= 0.90:
            return {
                'raw_certainty': no_price,
                'side': 'No',
                'reasoning': f'{market_type.capitalize()} market favors No'
            }

        # Special case: extreme targets - look for No at 85-97%
        if market_type == 'extreme' and 0.85 <= no_price <= 0.97:
            return {
                'raw_certainty': no_price,
                'side': 'No',
                'reasoning': f'Extreme target unlikely (implied {1-no_price:.1%} chance)'
            }

        return None

    def _get_volatility_penalty(self, market: Market) -> float:
        """
        Calculate volatility penalty based on bid-ask spread.

        High spread = more uncertainty = lower certainty
        """
        if len(market.outcome_prices) >= 2:
            yes_price = market.outcome_prices[0]
            no_price = market.outcome_prices[1]

            # In efficient market: yes + no = 1.0 (minus fees)
            # Deviation from this indicates spread/inefficiency
            spread = abs(1.0 - yes_price - no_price)

            # Penalty increases with spread:
            # 2% spread -> 0.01 penalty
            # 10% spread -> 0.05 penalty
            return min(spread * 0.5, 0.10)

        return 0.02  # Default penalty

    def _get_type_adjustment(self, market_type: str) -> float:
        """
        Adjustment factor by market type based on historical reliability.

        Some market types are more predictable than others.
        """
        type_multipliers = {
            'directional': 0.98,  # Price direction fairly predictable
            'range': 0.95,        # Range markets slightly less reliable
            'sports': 0.92,       # Sports can have upsets
            'extreme': 0.90,      # Extreme targets hard to predict
        }
        return type_multipliers.get(market_type, 0.95)

    def _is_directional_market(self, market: Market) -> bool:
        """Check if market is a directional 'Up or Down' market"""
        keywords = ['up or down', 'higher or lower', 'increase or decrease',
                   'close above', 'close below', 'settle above', 'settle below']
        title_lower = market.question.lower()
        return any(kw in title_lower for kw in keywords)

    def _is_range_market(self, market: Market) -> bool:
        """Check if market is a price range market"""
        keywords = ['between', 'in the range', 'from', 'to']
        title_lower = market.question.lower()
        return any(kw in title_lower for kw in keywords)

    def _is_sports_market(self, market: Market) -> bool:
        """Check if market is a sports outcome market"""
        keywords = ['will', 'win', 'fc', 'team', 'match', 'game', 'nfl', 'nba', 'nhl']
        title_lower = market.question.lower()
        has_sports_terms = sum(1 for kw in keywords if kw in title_lower) >= 2
        has_team_indicator = 'fc' in title_lower or 'united' in title_lower
        return has_sports_terms or has_team_indicator

    def _is_extreme_target(self, market: Market) -> bool:
        """Check if market involves extreme price targets"""
        keywords = ['reach', 'hit', 'above', 'below', '$']
        title_lower = market.question.lower()
        return sum(1 for kw in keywords if kw in title_lower) >= 2


class SharkyScanner:
    """
    Scans markets for Sharky-style opportunities on BOTH Polymarket and Kalshi (v2.1)

    Enhanced with:
    - Exit engine for dynamic position management
    - Infrastructure modules for caching, logging, error handling
    - Portfolio risk manager integration
    """

    def __init__(self, config: Config):
        self.config = config
        self.analyzer = CertaintyAnalyzer()

        # Polymarket clients
        self.client: Optional[PolymarketClient] = None
        self.gamma_client: Optional[GammaMarketsAPI] = None

        # Kalshi client
        self.kalshi_client: Optional[KalshiClient] = None

        # Strategy parameters
        self.min_scalp_certainty = 0.97  # 97%+ for scalping
        self.min_directional_certainty = 0.90  # 90%+ for directional
        self.max_scalp_price = 0.999  # Don't buy above 99.9¢
        self.min_directional_price = 0.80  # For medium-term positions

        # Initialize enhanced modules (v2.1)
        self.exit_engine = ExitEngine() if EXIT_ENGINE_AVAILABLE else None
        self.cache = CacheManager() if INFRASTRUCTURE_AVAILABLE else None
        self.rate_limiter = RateLimiter(rate=5, per=1.0) if INFRASTRUCTURE_AVAILABLE else None
        self.risk_manager = PortfolioRiskManager() if INFRASTRUCTURE_AVAILABLE else None

        # Track active positions for exit monitoring
        self.active_positions: Dict[str, Dict] = {}

        # Log module availability
        logger.info(f"SharkyScanner initialized - Calibration: {CALIBRATION_AVAILABLE}, "
                   f"ExitEngine: {EXIT_ENGINE_AVAILABLE}, "
                   f"Infrastructure: {INFRASTRUCTURE_AVAILABLE}")
        
    async def initialize(self):
        """Initialize API clients for both platforms"""
        # Initialize Polymarket
        self.client = PolymarketClient(
            api_url=self.config.POLYMARKET_API_URL,
            api_key=self.config.POLYMARKET_API_KEY,
            api_secret=self.config.POLYMARKET_SECRET,
            passphrase=self.config.POLYMARKET_PASSPHRASE
        )
        await self.client.__aenter__()
        
        self.gamma_client = GammaMarketsAPI(
            base_url=self.config.GAMMA_API_URL
        )
        await self.gamma_client.__aenter__()
        
        # Initialize Kalshi (uses context manager)
        self.kalshi_client = KalshiClient(
            api_url=self.config.KALSHI_API_URL,
            api_key=self.config.KALSHI_API_KEY,
            private_key_str=self.config.KALSHI_PRIVATE_KEY
        )
        await self.kalshi_client.__aenter__()  # Use __aenter__ instead of initialize()

        # Warn if private key not loaded - prevents authentication and live orders
        if not self.kalshi_client.private_key:
            logger.warning("Kalshi private key not loaded; Kalshi live execution will be disabled until configured")
        else:
            logger.info("Kalshi private key loaded successfully")

        logger.info("Sharky scanner initialized (Polymarket + Kalshi)")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            await self.client.__aexit__(None, None, None)
        if self.gamma_client:
            await self.gamma_client.__aexit__(None, None, None)
        if self.kalshi_client:
            await self.kalshi_client.__aexit__(None, None, None)  # Use __aexit__ instead of cleanup()
    
    async def scan_for_opportunities(self) -> List[SharkyOpportunity]:
        """
        Scan ALL markets (Polymarket AND Kalshi) for Sharky-style opportunities
        
        Returns:
            List of opportunities from both platforms
        """
        all_opportunities = []
        
        # Scan Polymarket
        try:
            polymarket_opps = await self._scan_polymarket()
            all_opportunities.extend(polymarket_opps)
            logger.info(f"Found {len(polymarket_opps)} Polymarket Sharky opportunities")
        except Exception as e:
            logger.error(f"Error scanning Polymarket: {e}")
        
        # Scan Kalshi (NEW)
        try:
            kalshi_opps = await self._scan_kalshi()
            all_opportunities.extend(kalshi_opps)
            logger.info(f"Found {len(kalshi_opps)} Kalshi Sharky opportunities")
        except Exception as e:
            logger.error(f"Error scanning Kalshi: {e}")
        
        # Sort by profit potential
        all_opportunities = sorted(
            all_opportunities, 
            key=lambda x: x.certainty * x.profit_potential, 
            reverse=True
        )
        
        logger.info(f"Found {len(all_opportunities)} total Sharky opportunities")
        return all_opportunities
    
    async def _scan_polymarket(self) -> List[SharkyOpportunity]:
        """Scan Polymarket for opportunities"""
        opportunities = []
        
        # Fetch all Polymarket markets
        try:
            markets = await self.client.get_markets(active_only=True)
            if not markets:
                gamma_markets = await self.gamma_client.get_all_markets()
                markets = self._convert_gamma_markets(gamma_markets)
        except Exception as e:
            logger.error(f"Error fetching Polymarket markets: {e}")
            return []
        
        logger.info(f"Scanning {len(markets)} Polymarket markets")
        
        # Analyze each market
        for market in markets:
            try:
                opp = await self._analyze_market_for_opportunity(market, platform='polymarket')
                if opp:
                    opportunities.append(opp)
            except Exception as e:
                logger.debug(f"Error analyzing Polymarket market: {e}")
                continue
        
        return opportunities
    
    async def _scan_kalshi(self) -> List[SharkyOpportunity]:
        """Scan Kalshi for opportunities (NEW)"""
        opportunities = []
        
        if not self.kalshi_client:
            return []
        
        # Fetch all Kalshi markets
        try:
            kalshi_markets = await self.kalshi_client.get_markets(
                status='open',
                limit=200
            )
        except Exception as e:
            logger.error(f"Error fetching Kalshi markets: {e}")
            return []
        
        logger.info(f"Scanning {len(kalshi_markets)} Kalshi markets")
        
        # Convert KalshiMarket objects to unified Market objects
        markets = []
        for km in kalshi_markets:
            try:
                # Use the built-in conversion method
                market = km.to_polymarket_format()
                markets.append(market)
            except Exception as e:
                logger.debug(f"Error converting Kalshi market: {e}")
                continue
        
        # Analyze each market
        for market in markets:
            try:
                opp = await self._analyze_market_for_opportunity(market, platform='kalshi')
                if opp:
                    opportunities.append(opp)
            except Exception as e:
                logger.debug(f"Error analyzing Kalshi market: {e}")
                continue
        
        return opportunities
    
    async def _analyze_market_for_opportunity(self, market: Market,
                                             platform: str = 'polymarket') -> Optional[SharkyOpportunity]:
        """
        Analyze a single market for trading opportunity.

        ENHANCED (v2.0):
        - Uses calibrated certainty (not raw market price)
        - Checks fee-adjusted profitability
        - Requires positive expected value
        """
        # Skip markets with insufficient liquidity
        if market.liquidity < 100:
            return None

        # Analyze with multi-factor certainty estimation
        analysis = self.analyzer.analyze_market(market, platform)
        if not analysis:
            return None

        # Extract analysis results
        certainty = analysis['certainty']  # This is CALIBRATED, not raw
        raw_certainty = analysis.get('raw_certainty', certainty)
        side = analysis['side']
        entry_price = analysis.get('entry_price', 0)
        is_positive_ev = analysis.get('is_positive_ev', True)
        net_profit = analysis.get('net_profit_if_win', 0)

        # CRITICAL: Skip if not positive expected value (after fees)
        if not is_positive_ev:
            logger.debug(f"[{platform.upper()}] Skipping {market.question[:40]}: negative EV after fees")
            return None

        # Calculate fee-adjusted profit potential
        if net_profit > 0:
            profit_potential = net_profit
        else:
            profit_potential = 1.0 - entry_price - 0.02  # Fallback with 2% fee estimate

        # Skip if profit potential is too low after fees
        min_profit = 0.005  # Minimum 0.5% profit after fees
        if profit_potential < min_profit:
            logger.debug(f"[{platform.upper()}] Skipping {market.question[:40]}: profit too low after fees")
            return None

        # Determine opportunity type based on CALIBRATED certainty
        opportunity_type = None

        # Type 1: Near-certainty scalping (95%+ calibrated, trading at 95-99.9¢)
        # Note: Thresholds adjusted for calibrated certainty
        if certainty >= 0.95 and entry_price <= self.max_scalp_price:
            opportunity_type = 'scalp'

        # Type 2: Directional (92%+ calibrated, trading at 80-97¢)
        elif certainty >= 0.92 and self.min_directional_price <= entry_price <= 0.97:
            opportunity_type = 'directional'

        if not opportunity_type:
            return None

        # Create opportunity
        opp = SharkyOpportunity(
            market=market,
            opportunity_type=opportunity_type,
            certainty=certainty,
            entry_price=entry_price,
            profit_potential=profit_potential,
            platform=platform
        )
        opp.position_side = side

        logger.info(f"[{platform.upper()}] Found {opportunity_type}: {market.question[:50]} | "
                   f"{side} @ {entry_price:.3f} | "
                   f"Raw: {raw_certainty:.1%} -> Calibrated: {certainty:.1%} | "
                   f"Net Profit: {profit_potential:.2%}")

        return opp

    # ========================================
    # EXIT ENGINE INTEGRATION (v2.1)
    # ========================================

    def register_position(self, position_id: str, market: Market,
                         entry_price: float, certainty: float,
                         side: str, platform: str):
        """Register an active position for exit monitoring"""
        self.active_positions[position_id] = {
            'market': market,
            'entry_price': entry_price,
            'entry_certainty': certainty,
            'side': side,
            'platform': platform,
            'entry_time': datetime.now(),
        }
        logger.info(f"Registered position {position_id} for exit monitoring")

    def unregister_position(self, position_id: str):
        """Remove a position from exit monitoring"""
        if position_id in self.active_positions:
            del self.active_positions[position_id]
            logger.info(f"Unregistered position {position_id}")

    async def check_exit_signals(self) -> List[Dict]:
        """
        Check all active positions for exit signals (v2.1)

        Returns list of positions that should be exited with reasons.
        """
        exit_signals = []

        if not self.exit_engine or not self.active_positions:
            return exit_signals

        for position_id, position in self.active_positions.items():
            try:
                market = position['market']
                platform = position['platform']

                # Get current market prices
                if platform == 'polymarket':
                    current_prices = await self._get_current_polymarket_price(market.condition_id)
                else:
                    current_prices = await self._get_current_kalshi_price(market.condition_id)

                if not current_prices:
                    continue

                # Get current price for our side
                side = position['side']
                current_price = current_prices.get(side.lower(), position['entry_price'])

                # Calculate current PnL
                entry_price = position['entry_price']
                pnl_pct = (current_price - entry_price) / entry_price

                # Re-analyze market for current certainty
                analysis = self.analyzer.analyze_market(market, platform)
                current_certainty = analysis['certainty'] if analysis else position['entry_certainty']

                # Check exit signal
                signal = self.exit_engine.check_exit(
                    position_id=position_id,
                    entry_price=entry_price,
                    current_price=current_price,
                    entry_certainty=position['entry_certainty'],
                    current_certainty=current_certainty,
                    entry_time=position['entry_time']
                )

                if signal and signal.should_exit:
                    exit_signals.append({
                        'position_id': position_id,
                        'market_id': market.condition_id,
                        'platform': platform,
                        'side': side,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'pnl_pct': pnl_pct,
                        'exit_reason': signal.reason,
                        'exit_urgency': signal.urgency,
                        'signal': signal
                    })

                    logger.info(f"Exit signal for {position_id}: {signal.reason} "
                               f"(PnL: {pnl_pct:.2%}, Urgency: {signal.urgency})")

            except Exception as e:
                logger.error(f"Error checking exit for {position_id}: {e}")
                continue

        return exit_signals

    async def _get_current_polymarket_price(self, market_id: str) -> Optional[Dict[str, float]]:
        """Get current prices from Polymarket"""
        try:
            if self.cache:
                cached = self.cache.get(f"pm_price_{market_id}", namespace='market_data')
                if cached:
                    return cached

            # Fetch fresh price (would need actual API call)
            # For now, return None - in production, fetch from client
            return None
        except Exception as e:
            logger.debug(f"Error fetching Polymarket price: {e}")
            return None

    async def _get_current_kalshi_price(self, market_id: str) -> Optional[Dict[str, float]]:
        """Get current prices from Kalshi"""
        try:
            if self.cache:
                cached = self.cache.get(f"kalshi_price_{market_id}", namespace='market_data')
                if cached:
                    return cached

            # Fetch fresh price (would need actual API call)
            # For now, return None - in production, fetch from client
            return None
        except Exception as e:
            logger.debug(f"Error fetching Kalshi price: {e}")
            return None

    # ========================================
    # RISK MANAGEMENT INTEGRATION (v2.1)
    # ========================================

    def check_position_risk(self, opportunity: 'SharkyOpportunity',
                           proposed_size: float) -> Dict:
        """
        Check if a position meets risk requirements (v2.1)

        Returns approval status and any adjustments needed.
        """
        if not self.risk_manager:
            return {'approved': True, 'adjusted_size': proposed_size}

        # Check portfolio-level risk
        check_result = self.risk_manager.check_position(
            model='sharky',
            platform=opportunity.platform,
            market_id=opportunity.market.condition_id,
            proposed_size=proposed_size
        )

        if not check_result['approved']:
            logger.warning(f"Position rejected by risk manager: {check_result.get('reason')}")

        return check_result

    def _convert_gamma_markets(self, gamma_markets: List[Dict]) -> List[Market]:
        """Convert Gamma API markets to Market objects"""
        markets = []
        for gm in gamma_markets:
            if not isinstance(gm, dict):
                continue
            
            try:
                outcomes = gm.get("outcomes", ["Yes", "No"])
                outcome_prices = gm.get("outcomePrices", ["0.5", "0.5"])
                
                market = Market(
                    condition_id=gm.get("conditionId", gm.get("id", "")),
                    question=gm.get("question", gm.get("title", "")),
                    outcomes=outcomes,
                    outcome_prices=[float(p) for p in outcome_prices],
                    token_ids=gm.get("tokens", []),
                    active=gm.get("active", True),
                    volume=float(gm.get("volume", 0)),
                    liquidity=float(gm.get("liquidity", 0))
                )
                markets.append(market)
            except Exception as e:
                logger.debug(f"Error converting Gamma market: {e}")
                continue
        
        return markets


async def test_sharky_scanner():
    """Test the Sharky scanner on both platforms"""
    from config import Config
    
    config = Config()
    scanner = SharkyScanner(config)
    
    try:
        await scanner.initialize()
        
        print("=" * 80)
        print("SHARKY NEAR-CERTAINTY SCANNER (POLYMARKET + KALSHI)")
        print("=" * 80)
        
        opportunities = await scanner.scan_for_opportunities()
        
        print(f"\nFound {len(opportunities)} total opportunities:\n")
        
        # Group by platform
        polymarket_opps = [o for o in opportunities if o.platform == 'polymarket']
        kalshi_opps = [o for o in opportunities if o.platform == 'kalshi']
        
        print(f"📊 POLYMARKET OPPORTUNITIES ({len(polymarket_opps)}):")
        print("-" * 80)
        for i, opp in enumerate(polymarket_opps[:5], 1):
            print(f"{i}. {opp.market.question[:60]}")
            print(f"   Type: {opp.opportunity_type} | Side: {opp.position_side} @ {opp.entry_price:.3f}")
            print(f"   Certainty: {opp.certainty:.1%} | Profit: {opp.profit_potential:.2%}")
            print()
        
        print(f"\n📈 KALSHI OPPORTUNITIES ({len(kalshi_opps)}):")
        print("-" * 80)
        for i, opp in enumerate(kalshi_opps[:5], 1):
            print(f"{i}. {opp.market.question[:60]}")
            print(f"   Type: {opp.opportunity_type} | Side: {opp.position_side} @ {opp.entry_price:.3f}")
            print(f"   Certainty: {opp.certainty:.1%} | Profit: {opp.profit_potential:.2%}")
            print()
        
    finally:
        await scanner.cleanup()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(test_sharky_scanner())