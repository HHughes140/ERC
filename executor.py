"""
Unified Trade Executor
Executes trades across all platforms (Polymarket, Kalshi, Alpaca)
Coordinates with portfolio manager and database
"""
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import logging

from config import ERCConfig
from Models.Sharky.sharky_scanner import SharkyOpportunity
from Models.Sharky.allocation_engine import AllocationDecision
from Models.Polymarket.polymarket_client import PolymarketClient, Order
from Models.Polymarket.kalshi_client import KalshiClient
from Models.Polymarket.arbitrage import ArbitrageOpportunity
from Central_DB.database import Database, TradeRecord, Position

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of trade execution"""
    success: bool
    platform: str
    position_id: str = ""
    capital: float = 0.0
    executed_at: datetime = None
    error_message: Optional[str] = None
    order_ids: List[str] = None
    orders: List[Dict] = None
    profit: float = 0.0
    cost: float = 0.0
    
    def __post_init__(self):
        if self.executed_at is None:
            self.executed_at = datetime.now()
        if self.order_ids is None:
            self.order_ids = []
        if self.orders is None:
            self.orders = []
        if self.cost == 0.0:
            self.cost = self.capital
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "platform": self.platform,
            "position_id": self.position_id,
            "capital": self.capital,
            "executed_at": self.executed_at.isoformat(),
            "order_ids": self.order_ids,
            "orders": self.orders,
            "profit": self.profit,
            "cost": self.cost,
            "error": self.error_message
        }


class UnifiedExecutor:
    """
    Unified executor for all trading platforms
    Handles order execution, position tracking, and coordination with database
    """
    
    def __init__(self, config: ERCConfig, database: Database, dry_run: bool = True):
        self.config = config
        self.db = database
        self.dry_run = dry_run
        
        # Platform clients
        self.poly_client: Optional[PolymarketClient] = None
        self.kalshi_client: Optional[KalshiClient] = None
        
        # Execution state
        self.execution_history: List[ExecutionResult] = []
        self.open_positions: Dict[str, Dict] = {}
        
        logger.info(f"Executor initialized in {'DRY RUN' if dry_run else 'LIVE'} mode")
    
    async def initialize(self):
        """Initialize platform clients"""
        # Polymarket
        if 'polymarket' in self.config.PLATFORMS:
            self.poly_client = PolymarketClient(
                api_url=self.config.POLYMARKET_API_URL,
                gamma_url=self.config.GAMMA_API_URL,
                api_key=self.config.POLYMARKET_API_KEY,
                api_secret=self.config.POLYMARKET_SECRET,
                passphrase=self.config.POLYMARKET_PASSPHRASE
            )
            await self.poly_client.__aenter__()
            logger.info("Polymarket client initialized")
        
        # Kalshi
        if 'kalshi' in self.config.PLATFORMS:
            self.kalshi_client = KalshiClient(
                api_url=self.config.KALSHI_API_URL,
                api_key=self.config.KALSHI_API_KEY,
                private_key_str=self.config.KALSHI_PRIVATE_KEY
            )
            await self.kalshi_client.__aenter__()
            logger.info("Kalshi client initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.poly_client:
            await self.poly_client.__aexit__(None, None, None)
        if self.kalshi_client:
            await self.kalshi_client.__aexit__(None, None, None)
    
    # ========================================
    # ARBITRAGE EXECUTION
    # ========================================
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity,
                               capital: float) -> ExecutionResult:
        """
        Execute arbitrage opportunity
        
        Args:
            opportunity: Arbitrage opportunity to execute
            capital: Capital to deploy
        """
        if opportunity.arb_type == 'cross_platform':
            return await self._execute_cross_platform_arb(opportunity, capital)
        else:
            return await self._execute_single_platform_arb(opportunity, capital)
    
    async def _execute_single_platform_arb(self, opportunity: ArbitrageOpportunity,
                                          capital: float) -> ExecutionResult:
        """Execute single-platform arbitrage (buy YES + NO on same market)"""
        if self.dry_run:
            return self._simulate_arbitrage(opportunity, capital)
        
        if opportunity.platform != 'polymarket':
            result = ExecutionResult(False, opportunity.platform, capital=capital)
            result.error_message = f"Platform {opportunity.platform} not supported for arbitrage"
            return result
        
        if not self.poly_client:
            result = ExecutionResult(False, 'polymarket', capital=capital)
            result.error_message = "Polymarket client not initialized"
            return result
        
        try:
            # Get recommended shares
            recommended = opportunity.recommended_shares
            yes_shares = recommended.get('yes', 0)
            no_shares = recommended.get('no', 0)
            
            # Get prices
            yes_price = opportunity.prices.get('yes', 0.5)
            no_price = opportunity.prices.get('no', 0.5)
            
            logger.info(f"[ARBITRAGE EXECUTION] {opportunity.market_id}")
            logger.info(f"  YES: {yes_shares:.2f} shares @ ${yes_price:.4f}")
            logger.info(f"  NO: {no_shares:.2f} shares @ ${no_price:.4f}")
            logger.info(f"  Total cost: ${yes_shares * yes_price + no_shares * no_price:.2f}")
            
            # Place YES order
            yes_order = Order(
                market_id=opportunity.market_id,
                side='BUY',
                price=yes_price,
                size=yes_shares,
                outcome='YES'
            )
            
            yes_order_id = await self.poly_client.place_order(yes_order, dry_run=self.dry_run)
            
            # Place NO order
            no_order = Order(
                market_id=opportunity.market_id,
                side='BUY',
                price=no_price,
                size=no_shares,
                outcome='NO'
            )
            
            no_order_id = await self.poly_client.place_order(no_order, dry_run=self.dry_run)
            
            # Check success
            order_ids = []
            orders = []
            
            if yes_order_id:
                order_ids.append(yes_order_id)
                orders.append({
                    'platform': 'polymarket',
                    'order_id': yes_order_id,
                    'outcome': 'YES',
                    'size': yes_shares,
                    'price': yes_price
                })
                logger.info(f"  YES order placed: {yes_order_id}")
            
            if no_order_id:
                order_ids.append(no_order_id)
                orders.append({
                    'platform': 'polymarket',
                    'order_id': no_order_id,
                    'outcome': 'NO',
                    'size': no_shares,
                    'price': no_price
                })
                logger.info(f"  NO order placed: {no_order_id}")
            
            if len(order_ids) == 2:
                # Both orders successful
                position_id = f"arb_{opportunity.market_id}_{int(datetime.now().timestamp())}"
                
                result = ExecutionResult(
                    success=True,
                    platform='polymarket',
                    position_id=position_id,
                    capital=capital
                )
                result.order_ids = order_ids
                result.orders = orders
                result.cost = yes_shares * yes_price + no_shares * no_price
                result.profit = opportunity.guaranteed_profit
                
                # Save to database
                self._save_arbitrage_position(opportunity, result)
                
                # Track position
                self.open_positions[position_id] = {
                    'platform': 'polymarket',
                    'market_id': opportunity.market_id,
                    'type': 'arbitrage',
                    'order_ids': order_ids,
                    'capital': capital,
                    'yes_shares': yes_shares,
                    'no_shares': no_shares,
                    'opened_at': datetime.now()
                }
                
                logger.info(f"[SUCCESS] Arbitrage executed: {position_id}")
                
                return result
            else:
                # Partial execution - need to handle cleanup
                result = ExecutionResult(False, 'polymarket', capital=capital)
                result.error_message = "Partial execution - only one leg filled"
                result.order_ids = order_ids
                
                logger.error(f"[FAILED] Partial execution - need to cancel orders")
                
                return result
        
        except Exception as e:
            logger.error(f"[ERROR] Arbitrage execution failed: {e}")
            result = ExecutionResult(False, 'polymarket', capital=capital)
            result.error_message = str(e)
            return result
    
    async def _execute_cross_platform_arb(self, opportunity: ArbitrageOpportunity,
                                         capital: float) -> ExecutionResult:
        """Execute cross-platform arbitrage (Polymarket + Kalshi)"""
        if self.dry_run:
            return self._simulate_arbitrage(opportunity, capital)
        
        # Extract platforms and sides
        prices = opportunity.prices
        
        # Determine which platform for each side
        poly_side = None
        kalshi_side = None
        
        if 'pm_yes' in prices:
            poly_side = 'YES'
            poly_price = prices['pm_yes']
        elif 'pm_no' in prices:
            poly_side = 'NO'
            poly_price = prices['pm_no']
        
        if 'kalshi_yes' in prices:
            kalshi_side = 'yes'
            kalshi_price = prices['kalshi_yes']
        elif 'kalshi_no' in prices:
            kalshi_side = 'no'
            kalshi_price = prices['kalshi_no']
        
        if not poly_side or not kalshi_side:
            result = ExecutionResult(False, 'multi', capital=capital)
            result.error_message = "Could not determine cross-platform sides"
            return result
        
        logger.info(f"[CROSS-PLATFORM ARB]")
        logger.info(f"  Polymarket {poly_side} @ ${poly_price:.4f}")
        logger.info(f"  Kalshi {kalshi_side} @ ${kalshi_price:.4f}")
        
        # Calculate shares
        shares = capital / (poly_price + kalshi_price) if (poly_price + kalshi_price) > 0 else 0
        
        try:
            # Place Polymarket order
            market_ids = opportunity.market_id.split('|')
            poly_market_id = market_ids[0] if len(market_ids) > 0 else opportunity.market_id
            
            poly_order = Order(
                market_id=poly_market_id,
                side='BUY',
                price=poly_price,
                size=shares,
                outcome=poly_side
            )
            
            poly_order_id = await self.poly_client.place_order(poly_order, dry_run=self.dry_run)
            
            # Place Kalshi order
            kalshi_ticker = market_ids[1] if len(market_ids) > 1 else opportunity.market_id
            kalshi_quantity = max(1, int(shares))
            
            kalshi_order_id = await self.kalshi_client.place_order(
                ticker=kalshi_ticker,
                side=kalshi_side,
                price=kalshi_price,
                quantity=kalshi_quantity,
                dry_run=self.dry_run
            )
            
            # Check success
            if poly_order_id and kalshi_order_id:
                position_id = f"cross_arb_{int(datetime.now().timestamp())}"
                
                result = ExecutionResult(
                    success=True,
                    platform='polymarket+kalshi',
                    position_id=position_id,
                    capital=capital
                )
                result.order_ids = [poly_order_id, kalshi_order_id]
                result.orders = [
                    {'platform': 'polymarket', 'order_id': poly_order_id, 'side': poly_side, 'size': shares},
                    {'platform': 'kalshi', 'order_id': kalshi_order_id, 'side': kalshi_side, 'quantity': kalshi_quantity}
                ]
                result.cost = shares * poly_price + kalshi_quantity * kalshi_price
                result.profit = opportunity.guaranteed_profit
                
                self.open_positions[position_id] = {
                    'platform': 'multi',
                    'type': 'cross_platform_arbitrage',
                    'order_ids': [poly_order_id, kalshi_order_id],
                    'capital': capital,
                    'opened_at': datetime.now()
                }
                
                logger.info(f"[SUCCESS] Cross-platform arbitrage executed")
                
                return result
            else:
                result = ExecutionResult(False, 'multi', capital=capital)
                result.error_message = "Failed to execute both sides"
                return result
        
        except Exception as e:
            logger.error(f"[ERROR] Cross-platform execution failed: {e}")
            result = ExecutionResult(False, 'multi', capital=capital)
            result.error_message = str(e)
            return result
    
    def _simulate_arbitrage(self, opportunity: ArbitrageOpportunity,
                          capital: float) -> ExecutionResult:
        """Simulate arbitrage execution"""
        position_id = f"sim_arb_{int(datetime.now().timestamp())}"
        
        logger.info(f"[DRY RUN] Arbitrage simulation")
        logger.info(f"  Type: {opportunity.arb_type}")
        logger.info(f"  Market: {opportunity.question[:60]}")
        logger.info(f"  Profit: {opportunity.profit_pct:.2f}%")
        logger.info(f"  Capital: ${capital:.2f}")
        
        result = ExecutionResult(
            success=True,
            platform='simulation',
            position_id=position_id,
            capital=capital
        )
        result.profit = opportunity.guaranteed_profit
        result.cost = capital
        
        return result
    
    # ========================================
    # DATABASE INTEGRATION
    # ========================================
    
    def _save_arbitrage_position(self, opportunity: ArbitrageOpportunity,
                                result: ExecutionResult):
        """Save arbitrage position to database"""
        try:
            # Save trade record
            trade_data = {
                'trade_id': result.position_id,
                'timestamp': result.executed_at.isoformat(),
                'platform': result.platform,
                'strategy': opportunity.arb_type,
                'symbol': opportunity.market_id,
                'side': 'arbitrage',
                'entry_price': result.cost / max(opportunity.capital_required, 1),
                'quantity': 1,
                'metadata': {
                    'opportunity_id': opportunity.opportunity_id,
                    'profit_pct': opportunity.profit_pct,
                    'order_ids': result.order_ids
                }
            }
            
            self.db.insert_trade(trade_data)
            
            # Save position
            position_data = {
                'position_id': result.position_id,
                'platform': result.platform,
                'strategy': opportunity.arb_type,
                'symbol': opportunity.market_id,
                'side': 'arbitrage',
                'entry_price': result.cost,
                'quantity': 1,
                'capital_deployed': result.capital,
                'opened_at': result.executed_at.isoformat(),
                'metadata': {
                    'opportunity': opportunity.to_dict()
                }
            }
            
            self.db.insert_position(position_data)
            
            logger.info(f"Position saved to database: {result.position_id}")
            
        except Exception as e:
            logger.error(f"Failed to save position to database: {e}")
    
    # ========================================
    # POSITION MANAGEMENT
    # ========================================
    
    async def close_position(self, position_id: str, reason: str = "manual") -> Optional[float]:
        """Close an open position"""
        if position_id not in self.open_positions:
            logger.warning(f"Position {position_id} not found")
            return None
        
        position = self.open_positions[position_id]
        platform = position['platform']
        
        logger.info(f"[CLOSING] Position {position_id} | Reason: {reason}")
        
        if self.dry_run:
            # Simulate close
            pnl = position['capital'] * 0.02  # Assume 2% profit
            logger.info(f"[DRY RUN] Position closed | P&L: ${pnl:.2f}")
            del self.open_positions[position_id]
            return pnl
        
        # Real closing logic would go here
        # For now, mark as closed
        pnl = position.get('profit', 0)
        
        # Update database
        self.db.close_position(position_id)
        
        # Remove from tracking
        del self.open_positions[position_id]
        
        logger.info(f"[CLOSED] Position {position_id} | P&L: ${pnl:.2f}")
        
        return pnl
    
    def get_open_positions_count(self) -> int:
        """Get number of open positions"""
        return len(self.open_positions)
    
    def get_deployed_capital(self) -> float:
        """Get total deployed capital"""
        return sum(pos['capital'] for pos in self.open_positions.values())
    
    def get_statistics(self) -> Dict:
        """Get execution statistics"""
        successful = sum(1 for r in self.execution_history if r.success)
        total = len(self.execution_history)
        
        total_profit = sum(r.profit for r in self.execution_history if r.success)
        
        return {
            "total_executions": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0,
            "total_profit": total_profit,
            "open_positions": len(self.open_positions),
            "deployed_capital": self.get_deployed_capital()
        }
    
    def print_statistics(self):
        """Print execution statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("EXECUTOR STATISTICS")
        print("="*70)
        print(f"Total Executions:    {stats['total_executions']}")
        print(f"Successful:          {stats['successful']}")
        print(f"Failed:              {stats['failed']}")
        print(f"Success Rate:        {stats['success_rate']:.1%}")
        print(f"Total Profit:        ${stats['total_profit']:,.2f}")
        print(f"Open Positions:      {stats['open_positions']}")
        print(f"Deployed Capital:    ${stats['deployed_capital']:,.2f}")
        print("="*70 + "\n")


async def test_executor():
    """Test executor"""
    from config import ERCConfig
    
    config = ERCConfig()
    db = Database(str(config.DB_PATH))
    
    executor = UnifiedExecutor(config, db, dry_run=True)
    await executor.initialize()
    
    print("Executor initialized successfully")
    executor.print_statistics()
    
    await executor.cleanup()


if __name__ == "__main__":
    asyncio.run(test_executor())