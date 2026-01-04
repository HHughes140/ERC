"""
Master Trading Engine - FULLY INTEGRATED VERSION
Combines: Arbitrage + Sharky + Weather + Alpaca ML

All strategies working together with Portfolio Manager
"""
import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import signal

# Force UTF-8 for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Path setup
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Core imports
from config import ERCConfig

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('master_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MasterEngine')


class EngineState(Enum):
    """Engine operational states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class EngineMetrics:
    """Real-time engine metrics"""
    state: EngineState = EngineState.STOPPED
    uptime_seconds: float = 0.0
    total_scans: int = 0
    opportunities_found: int = 0
    trades_executed: int = 0
    total_pnl: float = 0.0
    current_positions: int = 0
    deployed_capital: float = 0.0
    last_scan_time: Optional[datetime] = None
    errors_count: int = 0

    def to_dict(self) -> Dict:
        return {
            'state': self.state.value,
            'uptime_seconds': self.uptime_seconds,
            'total_scans': self.total_scans,
            'opportunities_found': self.opportunities_found,
            'trades_executed': self.trades_executed,
            'total_pnl': self.total_pnl,
            'current_positions': self.current_positions,
            'deployed_capital': self.deployed_capital,
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'errors_count': self.errors_count
        }


class MasterEngine:
    """
    Master Trading Engine coordinating ALL strategies

    Strategies:
    1. Arbitrage Scanner (Polymarket + Kalshi)
    2. Sharky Scanner (Near-certainty scalping)
    3. Weather Bot (Kalshi weather markets)
    4. Alpaca ML (Stock trading with ML)
    
    All managed by Portfolio Manager for capital allocation
    """

    def __init__(self, config: ERCConfig = None, initial_capital: float = 1000.0,
                 simulation_mode: bool = True):
        self.config = config or ERCConfig()
        self.initial_capital = initial_capital
        self.simulation_mode = simulation_mode

        # State
        self.state = EngineState.STOPPED
        self.start_time: Optional[datetime] = None
        self.running = False
        self.scan_count = 0

        # Core components
        self.database = None
        self.executor = None
        self.portfolio_manager = None
        self.notification_manager = None

        # Scanners/Strategies
        self.arb_scanner = None
        self.sharky_scanner = None
        self.weather_bot = None
        self.alpaca_strategy = None

        # Metrics
        self.metrics = EngineMetrics()

        logger.info(f"Master Engine created (simulation={simulation_mode})")

    async def initialize(self):
        """Initialize ALL components"""
        self.state = EngineState.STARTING
        self.start_time = datetime.now()

        logger.info("=" * 80)
        logger.info("MASTER TRADING ENGINE - FULL INITIALIZATION")
        logger.info("=" * 80)

        try:
            # 1. Database
            logger.info("\n[1/8] Initializing Database...")
            from Central_DB.database import Database
            self.database = Database(str(self.config.DB_PATH))
            logger.info("  + Database initialized")

            # 2. Portfolio Manager
            logger.info("[2/8] Initializing Portfolio Manager...")
            from Portfolio_Manager import PortfolioManager
            self.portfolio_manager = PortfolioManager(
                self.config,
                self.database,
                initial_capital=self.initial_capital
            )
            await self.portfolio_manager.initialize()
            logger.info("  + Portfolio Manager initialized")

            # 3. Arbitrage Scanner
            logger.info("[3/8] Initializing Arbitrage Scanner...")
            try:
                from Models.Polymarket.scanner import ArbitrageScanner
                self.arb_scanner = ArbitrageScanner(self.config)
                await self.arb_scanner.initialize()
                logger.info("  + Arbitrage Scanner initialized")
            except Exception as e:
                logger.warning(f"  - Arbitrage Scanner unavailable: {e}")

            # 4. Sharky Scanner
            logger.info("[4/8] Initializing Sharky Scanner...")
            try:
                from Models.Sharky.sharky_scanner import SharkyScanner
                self.sharky_scanner = SharkyScanner(self.config)
                await self.sharky_scanner.initialize()
                logger.info("  + Sharky Scanner initialized")
            except Exception as e:
                logger.warning(f"  - Sharky Scanner unavailable: {e}")

            # 5. Weather Bot
            logger.info("[5/8] Initializing Weather Bot...")
            try:
                from Models.Weather.weather_bot import SimpleWeatherBot
                self.weather_bot = SimpleWeatherBot(self.config)
                await self.weather_bot.initialize()
                logger.info("  + Weather Bot initialized")
            except Exception as e:
                logger.warning(f"  - Weather Bot unavailable: {e}")

            # 6. Alpaca ML Strategy
            logger.info("[6/8] Initializing Alpaca ML Strategy...")
            try:
                # Import the engine from your start.py
                import sys
                sys.path.insert(0, os.path.join(root_dir, 'Models', 'Alpaca'))
                from Models.C.start import MultiTimeframeEngine
                
                # Create watchlist for Alpaca
                alpaca_symbols = ['NVDA', 'QQQ', 'MSTR', 'SMCI', 'AMD', 'AVGO']
                self.alpaca_strategy = MultiTimeframeEngine(alpaca_symbols)
                logger.info("  + Alpaca ML Strategy initialized")
            except Exception as e:
                logger.warning(f"  - Alpaca ML Strategy unavailable: {e}")

            # 7. Executor
            logger.info("[7/8] Initializing Executor...")
            try:
                from executor import UnifiedExecutor
                self.executor = UnifiedExecutor(
                    self.config,
                    self.database,
                    dry_run=self.simulation_mode
                )
                await self.executor.initialize()
                logger.info(f"  + Executor initialized (mode: {'SIMULATION' if self.simulation_mode else 'LIVE'})")
            except Exception as e:
                logger.warning(f"  - Executor unavailable: {e}")

            # 8. Notifications
            logger.info("[8/8] Initializing Notification Manager...")
            try:
                from Notifications import NotificationManager
                self.notification_manager = NotificationManager(self.config)
                logger.info("  + Notification Manager initialized")
            except Exception as e:
                logger.warning(f"  - Notifications unavailable: {e}")

            # Update state
            self.state = EngineState.RUNNING
            self.metrics.state = EngineState.RUNNING

            logger.info("\n" + "=" * 80)
            logger.info("INITIALIZATION COMPLETE")
            logger.info("=" * 80)

            # Show active strategies
            logger.info(f"\nMode: {'SIMULATION' if self.simulation_mode else 'LIVE TRADING'}")
            logger.info(f"Capital: ${self.initial_capital:,.2f}")
            logger.info("\nActive Strategies:")
            if self.arb_scanner:
                logger.info("  + Arbitrage Scanner (Polymarket + Kalshi)")
            if self.sharky_scanner:
                logger.info("  + Sharky Scanner (Near-Certainty Scalping)")
            if self.weather_bot:
                logger.info("  + Weather Bot (Kalshi Weather Markets)")
            if self.alpaca_strategy:
                logger.info("  + Alpaca ML (Multi-Timeframe Stock Trading)")
            if self.executor:
                logger.info("  + Trade Executor")
            else:
                logger.warning("  ! No executor - SCAN-ONLY MODE")
            if self.notification_manager:
                logger.info("  + Discord Notifications")

            # Send startup notification
            if self.notification_manager:
                await self.notification_manager.send_alert(
                    level="info",
                    title="🚀 ERC Master Engine Started",
                    message=f"All systems online in {'SIMULATION' if self.simulation_mode else 'LIVE'} mode"
                )

        except Exception as e:
            self.state = EngineState.ERROR
            self.metrics.state = EngineState.ERROR
            self.metrics.errors_count += 1
            logger.error(f"Initialization failed: {e}", exc_info=True)
            raise

    async def cleanup(self):
        """Cleanup all resources"""
        logger.info("\nCleaning up...")
        self.state = EngineState.STOPPING

        try:
            if self.arb_scanner:
                await self.arb_scanner.cleanup()
            if self.sharky_scanner:
                await self.sharky_scanner.cleanup()
            if self.weather_bot:
                await self.weather_bot.cleanup()
            if self.executor:
                await self.executor.cleanup()
            if self.database:
                self.database.close()

            self.state = EngineState.STOPPED
            self.metrics.state = EngineState.STOPPED
            logger.info("Cleanup complete")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def scan_cycle(self):
        """Execute a single scan cycle across ALL strategies"""
        self.scan_count += 1
        scan_start = datetime.now()

        logger.info("\n" + "=" * 80)
        logger.info(f"SCAN CYCLE #{self.scan_count} - {scan_start.strftime('%H:%M:%S')}")
        logger.info("=" * 80)

        try:
            # Update portfolio state
            if self.portfolio_manager:
                await self.portfolio_manager.update()

            # Gather opportunities from ALL scanners
            all_opportunities = {
                'arbitrage': [],
                'sharky_scalp': [],
                'sharky_directional': [],
                'weather': [],
                'alpaca_ml': []
            }

            # 1. Scan Arbitrage
            if self.arb_scanner:
                try:
                    arb_opps = await self.arb_scanner.scan_all()
                    all_opportunities['arbitrage'] = arb_opps or []
                    logger.info(f"  Arbitrage: {len(all_opportunities['arbitrage'])} opportunities")
                except Exception as e:
                    logger.error(f"  Arbitrage scan error: {e}")

            # 2. Scan Sharky
            if self.sharky_scanner:
                try:
                    sharky_opps = await self.sharky_scanner.scan_for_opportunities()

                    # Separate by type
                    for opp in (sharky_opps or []):
                        if getattr(opp, 'opportunity_type', '') == 'scalp':
                            all_opportunities['sharky_scalp'].append(opp)
                        else:
                            all_opportunities['sharky_directional'].append(opp)

                    logger.info(f"  Sharky Scalp: {len(all_opportunities['sharky_scalp'])} opportunities")
                    logger.info(f"  Sharky Directional: {len(all_opportunities['sharky_directional'])} opportunities")
                except Exception as e:
                    logger.error(f"  Sharky scan error: {e}")

            # 3. Scan Weather
            if self.weather_bot:
                try:
                    weather_markets = await self.weather_bot.fetch_weather_markets()
                    tradeable = [m for m in weather_markets if m.is_tradeable()]
                    all_opportunities['weather'] = tradeable
                    logger.info(f"  Weather: {len(tradeable)} tradeable markets")
                except Exception as e:
                    logger.error(f"  Weather scan error: {e}")

            # 4. Alpaca ML (runs asynchronously, just log status)
            if self.alpaca_strategy:
                logger.info(f"  Alpaca ML: Running (separate thread)")

            # Update metrics
            total_opps = sum(len(v) for v in all_opportunities.values() if isinstance(v, list))
            self.metrics.opportunities_found += total_opps
            self.metrics.total_scans += 1
            self.metrics.last_scan_time = datetime.now()

            # Calculate allocations
            allocations = self._calculate_allocations(all_opportunities)

            # Execute opportunities
            if total_opps > 0:
                if self.executor:
                    await self._execute_opportunities(all_opportunities, allocations)
                else:
                    logger.warning("\n⚠️  Opportunities found but no executor available")
                    self._log_opportunities(all_opportunities)
            else:
                logger.info("\nNo opportunities found")

            # Update metrics
            scan_duration = (datetime.now() - scan_start).total_seconds()
            logger.info(f"\nScan completed in {scan_duration:.2f}s")

            # Print summary periodically
            if self.scan_count % 5 == 0:
                self.print_status()

        except Exception as e:
            logger.error(f"Error in scan cycle: {e}", exc_info=True)
            self.metrics.errors_count += 1

    def _log_opportunities(self, opportunities: Dict[str, List]):
        """Log opportunities when executor is not available"""
        logger.info("\n" + "=" * 80)
        logger.info("OPPORTUNITIES FOUND (SCAN-ONLY MODE)")
        logger.info("=" * 80)

        for strategy, opps in opportunities.items():
            if opps:
                logger.info(f"\n[{strategy.upper()}] - {len(opps)} opportunities")
                for i, opp in enumerate(opps[:3], 1):
                    if hasattr(opp, 'question'):
                        logger.info(f"  {i}. {opp.question[:60]}")
                    elif hasattr(opp, 'market'):
                        logger.info(f"  {i}. {opp.market.question[:60]}")
                    elif hasattr(opp, 'title'):
                        logger.info(f"  {i}. {opp.title[:60]}")

    def _calculate_allocations(self, opportunities: Dict[str, List]) -> Dict[str, float]:
        """Calculate capital allocation for EACH strategy"""
        allocations = {}

        if self.portfolio_manager:
            available = self.portfolio_manager.get_available_capital()
        else:
            available = self.initial_capital

        # Strategy allocation percentages (adjusted for 5 strategies)
        strategy_pcts = {
            'arbitrage': 0.30,        # 30% to arbitrage
            'sharky_scalp': 0.20,     # 20% to sharky scalping
            'sharky_directional': 0.10,  # 10% to directional
            'weather': 0.20,          # 20% to weather
            'alpaca_ml': 0.20,        # 20% to Alpaca ML
        }

        for strategy, opps in opportunities.items():
            if opps or strategy == 'alpaca_ml':  # Alpaca ML always gets allocation
                pct = strategy_pcts.get(strategy, 0.1)
                allocations[strategy] = available * pct
            else:
                allocations[strategy] = 0.0

        return allocations

    async def _execute_opportunities(self, opportunities: Dict[str, List],
                                     allocations: Dict[str, float]):
        """Execute ALL opportunities"""

        logger.info("\n" + "=" * 80)
        logger.info("EXECUTING OPPORTUNITIES")
        logger.info("=" * 80)

        # 1. Arbitrage (highest priority)
        arb_opps = opportunities.get('arbitrage', [])
        if arb_opps and allocations.get('arbitrage', 0) > 0:
            logger.info(f"\n[ARBITRAGE] {len(arb_opps)} opportunities | Allocation: ${allocations['arbitrage']:,.2f}")
            for i, opp in enumerate(arb_opps[:5], 1):
                await self._execute_arbitrage(opp, allocations['arbitrage'] / min(5, len(arb_opps)), i)

        # 2. Sharky Scalping
        scalp_opps = opportunities.get('sharky_scalp', [])
        if scalp_opps and allocations.get('sharky_scalp', 0) > 0:
            logger.info(f"\n[SHARKY SCALP] {len(scalp_opps)} opportunities | Allocation: ${allocations['sharky_scalp']:,.2f}")
            for i, opp in enumerate(scalp_opps[:10], 1):
                await self._execute_sharky(opp, allocations['sharky_scalp'] / min(10, len(scalp_opps)), 'scalp', i)

        # 3. Sharky Directional
        dir_opps = opportunities.get('sharky_directional', [])
        if dir_opps and allocations.get('sharky_directional', 0) > 0:
            logger.info(f"\n[SHARKY DIRECTIONAL] {len(dir_opps)} opportunities | Allocation: ${allocations['sharky_directional']:,.2f}")
            for i, opp in enumerate(dir_opps[:5], 1):
                await self._execute_sharky(opp, allocations['sharky_directional'] / min(5, len(dir_opps)), 'directional', i)

        # 4. Weather Bot
        weather_opps = opportunities.get('weather', [])
        if weather_opps and allocations.get('weather', 0) > 0:
            logger.info(f"\n[WEATHER] {len(weather_opps)} opportunities | Allocation: ${allocations['weather']:,.2f}")
            for i, market in enumerate(weather_opps[:3], 1):
                await self._execute_weather(market, i)

        # 5. Alpaca ML (runs separately, just log)
        if allocations.get('alpaca_ml', 0) > 0:
            logger.info(f"\n[ALPACA ML] Allocation: ${allocations['alpaca_ml']:,.2f} | Running continuously")

    async def _execute_arbitrage(self, opp, capital: float, index: int):
        """Execute arbitrage opportunity"""
        try:
            market_desc = getattr(opp, 'question', 'Unknown')[:50]
            profit_pct = getattr(opp, 'profit_pct', 0.0)
            
            # Extract platform info from the opportunity
            platform1 = getattr(opp, 'platform1', 'polymarket')
            platform2 = getattr(opp, 'platform2', 'kalshi')

            logger.info(f"\n{index}. [ARB] {market_desc}")
            logger.info(f"   Profit: {profit_pct:.2f}% | Capital: ${capital:,.2f}")

            if self.executor:
                result = await self.executor.execute_arbitrage(opp, capital)

                if result.success:
                    self.metrics.trades_executed += 1
                    self.metrics.total_pnl += result.profit
                    logger.info(f"   SUCCESS: Position {result.position_id}")

                    # ALWAYS notify with platform info
                    if self.notification_manager:
                        await self.notification_manager.send_trade_alert(
                            trade_type="arbitrage",
                            symbol=market_desc,
                            profit=result.profit,
                            platform=f"{platform1}/{platform2}",  # ← FIXED: Added platform
                            simulation=self.simulation_mode
                        )
                else:
                    logger.error(f"   FAILED: {result.error_message}")

        except Exception as e:
            logger.error(f"   Error executing arbitrage: {e}")

    async def _execute_sharky(self, opp, capital: float, strategy_type: str, index: int):
        """Execute Sharky opportunity"""
        try:
            market = getattr(opp, 'market', None)
            market_desc = market.question[:50] if market else 'Unknown'
            platform = getattr(opp, 'platform', 'polymarket')
            side = getattr(opp, 'position_side', 'Unknown')
            entry_price = getattr(opp, 'entry_price', 0.0)
            profit_potential = getattr(opp, 'profit_potential', 0.0)

            logger.info(f"\n{index}. [{platform.upper()}] {market_desc}")
            logger.info(f"   {side} @ {entry_price:.3f} | Profit: {profit_potential:.2%}")
            logger.info(f"   Capital: ${capital:,.2f}")

            # Always execute (simulation or live)
            position_id = f"sim_{strategy_type}_{int(datetime.now().timestamp())}"
            simulated_pnl = capital * profit_potential * 0.5

            self.metrics.trades_executed += 1
            self.metrics.total_pnl += simulated_pnl

            logger.info(f"   {'[SIMULATED]' if self.simulation_mode else '[LIVE]'} Position: {position_id} | P&L: ${simulated_pnl:.2f}")

            # ALWAYS send notification with platform
            if self.notification_manager:
                await self.notification_manager.send_trade_alert(
                    trade_type=f"sharky_{strategy_type}",
                    symbol=market_desc,
                    profit=simulated_pnl,
                    side=side,
                    price=entry_price,
                    platform=platform,  # ← FIXED: Added platform
                    simulation=self.simulation_mode
                )

        except Exception as e:
            logger.error(f"   Error executing Sharky: {e}")

    async def _execute_weather(self, market, index: int):
        """Execute Weather Bot opportunity"""
        try:
            logger.info(f"\n{index}. [WEATHER] {market.title}")
            
            top1, top2 = market.get_two_highest_probabilities()
            if not top1 or not top2:
                return

            total_cost = market.calculate_total_cost()
            potential_profit = 1.00 - total_cost

            logger.info(f"   Top 2: {top1.label} + {top2.label}")
            logger.info(f"   Cost: ${total_cost:.2f} | Profit: ${potential_profit:.2f}")

            # Execute through weather bot
            success = await self.weather_bot.execute_simple_trade(market, dry_run=self.simulation_mode)

            if success:
                self.metrics.trades_executed += 1
                logger.info(f"   {'[SIMULATED]' if self.simulation_mode else '[LIVE]'} Trade executed")

                # ALWAYS send notification with platform
                if self.notification_manager:
                    await self.notification_manager.send_trade_alert(
                        trade_type="weather",
                        symbol=market.city,
                        profit=potential_profit,
                        platform="kalshi",  # ← FIXED: Added platform
                        price=total_cost,
                        simulation=self.simulation_mode
                    )

        except Exception as e:
            logger.error(f"   Error executing Weather trade: {e}")

    def pause(self):
        """Pause the engine"""
        if self.state == EngineState.RUNNING:
            self.state = EngineState.PAUSED
            self.metrics.state = EngineState.PAUSED
            logger.info("Engine paused")

    def resume(self):
        """Resume the engine"""
        if self.state == EngineState.PAUSED:
            self.state = EngineState.RUNNING
            self.metrics.state = EngineState.RUNNING
            logger.info("Engine resumed")

    async def run_continuous(self, scan_interval: int = 60):
        """Run continuous scanning"""
        self.running = True

        logger.info(f"\nStarting continuous operation (Interval: {scan_interval}s)")
        logger.info("Press Ctrl+C to stop\n")

        def stop_handler(sig, frame):
            logger.info("\nShutdown signal received")
            self.running = False

        try:
            signal.signal(signal.SIGINT, stop_handler)
            signal.signal(signal.SIGTERM, stop_handler)
        except (ValueError, OSError):
            pass

        try:
            while self.running:
                if self.state == EngineState.RUNNING:
                    await self.scan_cycle()
                elif self.state == EngineState.PAUSED:
                    logger.debug("Engine paused, waiting...")

                if self.running:
                    # Update uptime
                    if self.start_time:
                        self.metrics.uptime_seconds = (datetime.now() - self.start_time).total_seconds()

                    logger.info(f"\nSleeping {scan_interval}s...")
                    await asyncio.sleep(scan_interval)

        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            self.metrics.errors_count += 1

        finally:
            logger.info("\nEngine stopping...")
            await self.cleanup()

    async def run_once(self):
        """Run a single scan cycle"""
        await self.scan_cycle()
        self.print_status()
        await self.cleanup()

    def get_metrics(self) -> Dict:
        """Get current engine metrics"""
        return self.metrics.to_dict()

    def print_status(self):
        """Print engine status"""
        metrics = self.get_metrics()

        print("\n" + "=" * 60)
        print("ERC MASTER ENGINE STATUS")
        print("=" * 60)
        print(f"State:             {metrics['state']}")
        print(f"Mode:              {'SIMULATION' if self.simulation_mode else 'LIVE TRADING'}")
        print(f"Uptime:            {metrics['uptime_seconds']:.0f} seconds")
        print(f"Total Scans:       {metrics['total_scans']}")
        print(f"Opportunities:     {metrics['opportunities_found']}")
        print(f"Trades Executed:   {metrics['trades_executed']}")
        print(f"Total P&L:         ${metrics['total_pnl']:,.2f}")
        print(f"Open Positions:    {metrics['current_positions']}")
        print(f"Deployed Capital:  ${metrics['deployed_capital']:,.2f}")
        print(f"Errors:            {metrics['errors_count']}")
        print("=" * 60 + "\n")

        if self.portfolio_manager:
            self.portfolio_manager.print_summary()


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='ERC Master Trading Engine - ALL STRATEGIES')
    parser.add_argument('--capital', type=float, default=1000,
                       help='Total capital (default: $1000)')
    parser.add_argument('--simulate', action='store_true', default=True,
                       help='Run in SIMULATION mode (default)')
    parser.add_argument('--live', action='store_true',
                       help='Run in LIVE mode (requires --confirm)')
    parser.add_argument('--confirm', action='store_true',
                       help='Confirm live trading')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit')
    parser.add_argument('--interval', type=int, default=60,
                       help='Scan interval in seconds (default: 60)')

    args = parser.parse_args()

    # Safety check for live trading
    simulation_mode = True
    if args.live:
        if not args.confirm:
            print("ERROR: Live trading requires --confirm flag")
            print("Usage: python Master_Engine.py --live --confirm")
            sys.exit(1)
        simulation_mode = False
        print("\n" + "!" * 60)
        print("WARNING: LIVE TRADING MODE - REAL MONEY AT RISK")
        print("!" * 60 + "\n")

    # Create and run engine
    config = ERCConfig()
    engine = MasterEngine(
        config=config,
        initial_capital=args.capital,
        simulation_mode=simulation_mode
    )

    await engine.initialize()

    if args.once:
        await engine.run_once()
    else:
        await engine.run_continuous(scan_interval=args.interval)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)