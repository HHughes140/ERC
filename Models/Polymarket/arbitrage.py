"""
Arbitrage Model
Implements arbitrage detection and execution strategies
"""
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .math_models import (
    ArbitrageDetector,
    PolymarketArbitrage,
    ExecutionOptimizer,
    RiskManager
)

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity structure"""
    opportunity_id: str
    arb_type: str  # 'single_market', 'multi_outcome', 'cross_platform'
    platform: str
    
    # Market info
    market_id: str
    question: str
    
    # Prices
    prices: Dict[str, float]  # {'yes': 0.45, 'no': 0.50}
    
    # Profitability
    total_cost: float
    guaranteed_profit: float
    profit_pct: float
    roi: float
    
    # Sizing
    recommended_shares: Dict[str, float]
    capital_required: float
    
    # Risk
    risk_score: float
    is_executable: bool
    
    # Metadata
    detected_at: datetime
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            'opportunity_id': self.opportunity_id,
            'arb_type': self.arb_type,
            'platform': self.platform,
            'market_id': self.market_id,
            'question': self.question,
            'prices': self.prices,
            'total_cost': self.total_cost,
            'guaranteed_profit': self.guaranteed_profit,
            'profit_pct': self.profit_pct,
            'roi': self.roi,
            'recommended_shares': self.recommended_shares,
            'capital_required': self.capital_required,
            'risk_score': self.risk_score,
            'is_executable': self.is_executable,
            'detected_at': self.detected_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }


class ArbitrageModel:
    """
    Main arbitrage detection and execution model
    Uses mathematical models for detection and sizing
    """
    
    def __init__(self, config):
        self.config = config
        
        # Initialize mathematical models
        self.detector = ArbitrageDetector(
            transaction_cost=config.MIN_PROFIT_THRESHOLD
        )
        self.poly_arb = PolymarketArbitrage(transaction_fee=0.0)
        self.optimizer = ExecutionOptimizer(
            max_slippage=config.SLIPPAGE_TOLERANCE
        )
        self.risk_mgr = RiskManager(confidence=config.VAR_CONFIDENCE)
        
        self.opportunities: List[ArbitrageOpportunity] = []
    
    async def scan_single_market(self, market: Any) -> Optional[ArbitrageOpportunity]:
        """
        Scan a single market for arbitrage opportunities
        
        Args:
            market: Market object with prices and metadata
        """
        try:
            # Extract prices
            if not hasattr(market, 'outcome_prices') or len(market.outcome_prices) < 2:
                return None
            
            yes_price = market.outcome_prices[0]
            no_price = market.outcome_prices[1]
            
            # Validate prices
            if yes_price <= 0 or no_price <= 0 or yes_price >= 1 or no_price >= 1:
                return None
            
            # Detect arbitrage
            arb = self.detector.detect_single_market(yes_price, no_price)
            
            if not arb:
                return None
            
            # Calculate position sizing
            sizing = self.poly_arb.detect_and_size(
                yes_price=yes_price,
                no_price=no_price,
                max_bankroll=self.config.MAX_POSITION_SIZE
            )
            
            if not sizing:
                return None
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(
                yes_price=yes_price,
                no_price=no_price,
                liquidity=getattr(market, 'liquidity', 0),
                volume=getattr(market, 'volume', 0)
            )
            
            # Create opportunity
            opportunity = ArbitrageOpportunity(
                opportunity_id=f"arb_{int(datetime.now().timestamp())}_{market.condition_id}",
                arb_type='single_market',
                platform='polymarket',
                market_id=market.condition_id,
                question=market.question,
                prices={'yes': yes_price, 'no': no_price},
                total_cost=sizing['total_cost'],
                guaranteed_profit=sizing['net_profit'],
                profit_pct=arb['profit_percentage'],
                roi=sizing['roi_percent'],
                recommended_shares={
                    'yes': sizing['buy_yes_qty'],
                    'no': sizing['buy_no_qty']
                },
                capital_required=sizing['total_cost'],
                risk_score=risk_score,
                is_executable=risk_score < 0.5,  # Executable if risk is low
                detected_at=datetime.now(),
                expires_at=getattr(market, 'end_date', None)
            )
            
            self.opportunities.append(opportunity)
            
            logger.info(f"[ARB FOUND] {market.question[:50]} | "
                       f"Profit: {opportunity.profit_pct:.2f}% | "
                       f"Capital: ${opportunity.capital_required:.2f}")
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error scanning market: {e}")
            return None
    
    async def scan_markets(self, markets: List[Any]) -> List[ArbitrageOpportunity]:
        """
        Scan multiple markets for arbitrage
        
        Args:
            markets: List of market objects
            
        Returns:
            List of arbitrage opportunities
        """
        logger.info(f"Scanning {len(markets)} markets for arbitrage...")
        
        opportunities = []
        
        for market in markets:
            opp = await self.scan_single_market(market)
            if opp:
                opportunities.append(opp)
            
            # Rate limiting
            await asyncio.sleep(0.01)
        
        logger.info(f"Found {len(opportunities)} arbitrage opportunities")
        
        return opportunities
    
    def _calculate_risk_score(self, yes_price: float, no_price: float,
                             liquidity: float, volume: float) -> float:
        """
        Calculate risk score for arbitrage opportunity
        
        Risk factors:
        - Price deviation from 0.5 (higher is riskier)
        - Liquidity (lower is riskier)
        - Volume (lower is riskier)
        
        Returns: Score between 0 (low risk) and 1 (high risk)
        """
        # Price risk: How far from efficient pricing
        total_price = yes_price + no_price
        price_risk = abs(0.99 - total_price)  # Deviation from efficient
        
        # Liquidity risk
        liquidity_risk = 1.0 if liquidity < 100 else (1000 / max(liquidity, 1000))
        
        # Volume risk
        volume_risk = 1.0 if volume < 100 else (1000 / max(volume, 1000))
        
        # Combined risk (weighted average)
        risk_score = (
            price_risk * 0.3 +
            liquidity_risk * 0.4 +
            volume_risk * 0.3
        )
        
        return min(risk_score, 1.0)
    
    def get_best_opportunities(self, max_risk: float = 0.5,
                              min_profit: float = 0.01,
                              limit: int = 10) -> List[ArbitrageOpportunity]:
        """
        Get best arbitrage opportunities filtered by risk and profit
        
        Args:
            max_risk: Maximum acceptable risk score
            min_profit: Minimum profit percentage
            limit: Maximum number of opportunities to return
        """
        # Filter by risk and profit
        filtered = [
            opp for opp in self.opportunities
            if opp.risk_score <= max_risk and opp.profit_pct >= min_profit
        ]
        
        # Sort by profit percentage (descending)
        sorted_opps = sorted(filtered, key=lambda x: x.profit_pct, reverse=True)
        
        return sorted_opps[:limit]
    
    def clear_expired_opportunities(self):
        """Remove expired opportunities"""
        now = datetime.now()
        self.opportunities = [
            opp for opp in self.opportunities
            if not opp.expires_at or opp.expires_at > now
        ]
    
    def get_statistics(self) -> Dict:
        """Get arbitrage statistics"""
        if not self.opportunities:
            return {
                'total_opportunities': 0,
                'avg_profit_pct': 0.0,
                'avg_risk_score': 0.0,
                'total_capital_required': 0.0
            }
        
        return {
            'total_opportunities': len(self.opportunities),
            'avg_profit_pct': sum(o.profit_pct for o in self.opportunities) / len(self.opportunities),
            'avg_risk_score': sum(o.risk_score for o in self.opportunities) / len(self.opportunities),
            'total_capital_required': sum(o.capital_required for o in self.opportunities),
            'best_opportunity': max(self.opportunities, key=lambda x: x.profit_pct).to_dict()
        }
    
    def print_summary(self):
        """Print arbitrage opportunities summary"""
        if not self.opportunities:
            print("\nNo arbitrage opportunities found")
            return
        
        print("\n" + "="*80)
        print("ARBITRAGE OPPORTUNITIES SUMMARY")
        print("="*80)
        
        stats = self.get_statistics()
        print(f"\nTotal Opportunities: {stats['total_opportunities']}")
        print(f"Average Profit: {stats['avg_profit_pct']:.2f}%")
        print(f"Average Risk Score: {stats['avg_risk_score']:.2f}")
        print(f"Total Capital Required: ${stats['total_capital_required']:,.2f}")
        
        print(f"\n{'='*80}")
        print("TOP OPPORTUNITIES")
        print(f"{'='*80}")
        print(f"{'Question':<40} {'Profit%':<10} {'Capital':<15} {'Risk':<8}")
        print("-"*80)
        
        for opp in self.get_best_opportunities(limit=10):
            print(f"{opp.question[:38]:<40} "
                  f"{opp.profit_pct:>9.2f}% "
                  f"${opp.capital_required:>13,.2f} "
                  f"{opp.risk_score:>7.2f}")
        
        print("="*80 + "\n")