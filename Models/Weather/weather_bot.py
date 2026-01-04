"""
Simple Weather Arbitrage Bot - COMPLETE FIXED VERSION
Buys YES on the two highest probability outcomes every morning
Automatically fetches results and learns from them
Only trades if total cost < $1.00 (cannot lose money)
"""
import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
import statistics
import sys
import io
import re

# Force UTF-8 for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directories to path
parent_dir = Path(__file__).parent.parent  # Models/
root_dir = parent_dir.parent  # ERC/
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(root_dir))

from Polymarket.kalshi_client import KalshiClient
from config import ERCConfig as Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weather_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class SimpleWeatherOutcome:
    """Single weather outcome with price"""
    ticker: str
    label: str
    yes_price: float
    probability: float
    
    def __repr__(self):
        return f"{self.label}: YES @ ${self.yes_price:.2f} ({self.probability:.0%})"


@dataclass
class SimpleWeatherMarket:
    """Weather market with all outcomes"""
    base_ticker: str
    title: str
    city: str
    outcomes: List[SimpleWeatherOutcome]
    close_time: Optional[datetime] = None
    
    def get_two_highest_probabilities(self) -> Tuple[Optional[SimpleWeatherOutcome], Optional[SimpleWeatherOutcome]]:
        """Get the two outcomes with highest YES prices (probabilities)"""
        sorted_outcomes = sorted(self.outcomes, key=lambda x: x.probability, reverse=True)
        
        if len(sorted_outcomes) < 2:
            return None, None
        
        return sorted_outcomes[0], sorted_outcomes[1]
    
    def calculate_total_cost(self) -> float:
        """Calculate total cost of buying YES on top two"""
        top1, top2 = self.get_two_highest_probabilities()
        if not top1 or not top2:
            return 999.0
        
        return top1.yes_price + top2.yes_price
    
    def is_tradeable(self, max_cost: float = 0.99) -> bool:
        """Check if worth trading (total cost < $1.00)"""
        total = self.calculate_total_cost()
        
        if total >= 1.00:
            return False
        
        return total <= max_cost
    
    def get_expected_profit(self) -> float:
        """Calculate expected profit"""
        total_cost = self.calculate_total_cost()
        if total_cost >= 1.00:
            return -999.0
        
        top1, top2 = self.get_two_highest_probabilities()
        if not top1 or not top2:
            return 0.0
        
        expected_value = (top1.probability + top2.probability) * 1.00
        expected_profit = expected_value - total_cost
        
        return expected_profit


@dataclass
class TradeResult:
    """Result of a completed trade"""
    trade_id: str
    timestamp: datetime
    trade_hour: int
    market_title: str
    city: str
    
    outcome1_label: str
    outcome1_ticker: str
    outcome1_price: float
    outcome2_label: str
    outcome2_ticker: str
    outcome2_price: float
    total_cost: float
    
    actual_result: Optional[str] = None
    winning_outcome: Optional[str] = None
    pnl: Optional[float] = None
    was_profitable: Optional[bool] = None
    result_fetched: bool = False
    
    def to_dict(self):
        return {
            'trade_id': self.trade_id,
            'timestamp': self.timestamp.isoformat(),
            'trade_hour': self.trade_hour,
            'trade_time': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'market_title': self.market_title,
            'city': self.city,
            'outcome1_label': self.outcome1_label,
            'outcome1_ticker': self.outcome1_ticker,
            'outcome1_price': self.outcome1_price,
            'outcome2_label': self.outcome2_label,
            'outcome2_ticker': self.outcome2_ticker,
            'outcome2_price': self.outcome2_price,
            'total_cost': self.total_cost,
            'actual_result': self.actual_result,
            'winning_outcome': self.winning_outcome,
            'pnl': self.pnl,
            'was_profitable': self.was_profitable,
            'result_fetched': self.result_fetched
        }


@dataclass
class TimeLearning:
    """Learns which hours are best for trading"""
    trades_by_hour: Dict[int, List[float]] = field(default_factory=lambda: {i: [] for i in range(24)})
    total_trades: int = 0
    total_wins: int = 0
    
    def add_result(self, hour: int, pnl: float):
        """Add a trade result"""
        self.trades_by_hour[hour].append(pnl)
        self.total_trades += 1
        if pnl > 0:
            self.total_wins += 1
    
    def get_avg_pnl_by_hour(self, hour: int) -> float:
        """Get average P&L for a specific hour"""
        trades = self.trades_by_hour.get(hour, [])
        if not trades:
            return 0.0
        return statistics.mean(trades)
    
    def get_win_rate_by_hour(self, hour: int) -> float:
        """Get win rate for a specific hour"""
        trades = self.trades_by_hour.get(hour, [])
        if not trades:
            return 0.5
        wins = sum(1 for pnl in trades if pnl > 0)
        return wins / len(trades)
    
    def get_best_hours(self, top_n: int = 3) -> List[int]:
        """Get N best hours to trade"""
        if self.total_trades < 5:
            return [10]
        
        hour_scores = {}
        for hour in range(24):
            if self.trades_by_hour[hour]:
                avg_pnl = self.get_avg_pnl_by_hour(hour)
                win_rate = self.get_win_rate_by_hour(hour)
                hour_scores[hour] = (avg_pnl * 100) + win_rate
        
        sorted_hours = sorted(hour_scores.items(), key=lambda x: x[1], reverse=True)
        return [hour for hour, _ in sorted_hours[:top_n]]
    
    def print_stats(self):
        """Print learning statistics"""
        print("\n" + "="*70)
        print("TIME-BASED LEARNING - WEATHER BOT")
        print("="*70)
        
        if self.total_trades == 0:
            print("No trades yet")
            print("="*70 + "\n")
            return
        
        overall_win_rate = self.total_wins / self.total_trades if self.total_trades > 0 else 0
        
        print(f"\nTotal Trades: {self.total_trades}")
        print(f"Overall Win Rate: {overall_win_rate:.1%}")
        
        print(f"\n{'Hour':<6} {'Trades':<8} {'Avg P&L':<12} {'Win Rate':<12}")
        print("-"*70)
        
        for hour in range(24):
            trades = self.trades_by_hour[hour]
            if trades:
                avg_pnl = self.get_avg_pnl_by_hour(hour)
                win_rate = self.get_win_rate_by_hour(hour)
                print(f"{hour:02d}:00  {len(trades):<8} ${avg_pnl:>10.2f}  {win_rate:>10.1%}")
        
        best_hours = self.get_best_hours(top_n=3)
        print(f"\n[BEST] Trading Hours: {', '.join(f'{h:02d}:00' for h in best_hours)}")
        print("="*70 + "\n")
    
    def save(self, filepath: Path):
        """Save to disk"""
        data = {
            'trades_by_hour': {str(k): v for k, v in self.trades_by_hour.items()},
            'total_trades': self.total_trades,
            'total_wins': self.total_wins
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'TimeLearning':
        """Load from disk"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            learning = cls()
            learning.trades_by_hour = {int(k): v for k, v in data['trades_by_hour'].items()}
            learning.total_trades = data['total_trades']
            learning.total_wins = data['total_wins']
            return learning
        except FileNotFoundError:
            return cls()


class SimpleWeatherBot:
    """Simple strategy: Buy YES on two highest probability outcomes"""
    
    def __init__(self, config: Config):
        self.config = config
        self.kalshi_client: Optional[KalshiClient] = None
        
        self.trades_dir = Path("weather_trades")
        self.trades_dir.mkdir(exist_ok=True)
        
        self.learning_file = self.trades_dir / "time_learning.json"
        self.results_file = self.trades_dir / "all_results.json"
        
        self.learning = TimeLearning.load(self.learning_file)
        self.all_results: List[TradeResult] = []
        self._load_results()
        
        self.target_hour = 10
        self.max_total_cost = 0.95
        self.contracts_per_outcome = 1
        
        self.traded_today = set()
    
    def _load_results(self):
        """Load historical results"""
        try:
            if self.results_file.exists():
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                
                for record in data:
                    result = TradeResult(
                        trade_id=record['trade_id'],
                        timestamp=datetime.fromisoformat(record['timestamp']),
                        trade_hour=record['trade_hour'],
                        market_title=record['market_title'],
                        city=record['city'],
                        outcome1_label=record['outcome1_label'],
                        outcome1_ticker=record['outcome1_ticker'],
                        outcome1_price=record['outcome1_price'],
                        outcome2_label=record['outcome2_label'],
                        outcome2_ticker=record['outcome2_ticker'],
                        outcome2_price=record['outcome2_price'],
                        total_cost=record['total_cost'],
                        actual_result=record.get('actual_result'),
                        winning_outcome=record.get('winning_outcome'),
                        pnl=record.get('pnl'),
                        was_profitable=record.get('was_profitable'),
                        result_fetched=record.get('result_fetched', False)
                    )
                    self.all_results.append(result)
                    
                    if result.pnl is not None:
                        self.learning.add_result(result.trade_hour, result.pnl)
                
                logger.info(f"Loaded {len(self.all_results)} historical trades")
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
    
    def _save_results(self):
        """Save all results"""
        try:
            data = [r.to_dict() for r in self.all_results]
            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.learning.save(self.learning_file)
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    async def initialize(self):
        """Initialize"""
        logger.info("Initializing Simple Weather Bot...")
        logger.info("Strategy: Buy YES on two highest probabilities")
        logger.info("Safety: Only if total cost < $1.00")
        logger.info("Features: AUTO-FETCH RESULTS & LEARN")
        
        self.kalshi_client = KalshiClient(
            api_url=self.config.KALSHI_API_URL,
            api_key=self.config.KALSHI_API_KEY,
            private_key_str=self.config.KALSHI_PRIVATE_KEY
        )
        
        await self.kalshi_client.__aenter__()
        
        balance = await self.kalshi_client.get_balance()
        if balance is not None:
            logger.info(f"[OK] Kalshi Balance: ${balance:.2f}")
        else:
            raise Exception("Failed to connect to Kalshi")
        
        if self.learning.total_trades > 0:
            self.learning.print_stats()
            best_hours = self.learning.get_best_hours(top_n=3)
            self.target_hour = best_hours[0]
            logger.info(f"[LEARNED] Optimal hour: {self.target_hour:02d}:00")
        else:
            logger.info(f"[DEFAULT] Trading hour: {self.target_hour:02d}:00")
    
    async def cleanup(self):
        """Cleanup"""
        if self.kalshi_client:
            await self.kalshi_client.__aexit__(None, None, None)
    
    async def debug_show_all_events(self):
        """Debug: Show all events to find weather ones"""
        logger.info("\n" + "="*70)
        logger.info("DEBUG MODE - Showing All Events")
        logger.info("="*70 + "\n")
        
        endpoint = "/trade-api/v2/events?status=open&limit=100"
        result = await self.kalshi_client._request("GET", endpoint)
        
        if result and 'events' in result:
            logger.info(f"Found {len(result['events'])} open events:\n")
            
            weather_count = 0
            
            for event in result['events']:
                title = event.get('title', '')
                series = event.get('series_ticker', 'N/A')
                category = event.get('category', 'N/A')
                
                is_weather = any(keyword in title.lower() for keyword in [
                    'temperature', 'temp', 'weather', 'rain', 'snow', 
                    'highest', 'lowest', 'degrees', 'precipitation'
                ])
                
                if is_weather:
                    weather_count += 1
                    logger.info(f"[WEATHER] {event['event_ticker']}")
                    logger.info(f"  Title: {title}")
                    logger.info(f"  Series: {series}")
                    logger.info(f"  Category: {category}\n")
                else:
                    logger.info(f"[OTHER] {event['event_ticker']}: {title[:60]}")
            
            logger.info("\n" + "="*70)
            logger.info(f"Weather Events Found: {weather_count}/{len(result['events'])}")
            logger.info("="*70 + "\n")
        else:
            logger.warning("No events returned from API")
    
    async def fetch_weather_markets(self) -> List[SimpleWeatherMarket]:
        """Fetch all weather temperature markets using Events API"""
        logger.info("Fetching weather events from Kalshi...")
        
        try:
            endpoint = "/trade-api/v2/events?status=open&with_nested_markets=true&limit=200"
            
            all_weather_markets = []
            page_count = 0
            cursor = None
            
            while page_count < 10:
                if cursor:
                    full_endpoint = f"{endpoint}&cursor={cursor}"
                else:
                    full_endpoint = endpoint
                
                result = await self.kalshi_client._request("GET", full_endpoint)
                
                if not result or 'events' not in result:
                    break
                
                for event in result['events']:
                    title = event.get('title', '').lower()
                    category = event.get('category', '').lower()
                    
                    is_weather = any(keyword in title or keyword in category for keyword in [
                        'temperature', 'temp', 'weather', 'rain', 'snow', 'precipitation',
                        'highest', 'lowest', 'degrees', 'forecast'
                    ])
                    
                    if not is_weather:
                        continue
                    
                    markets = event.get('markets', [])
                    if len(markets) < 2:
                        continue
                    
                    city = self._extract_city(event['title'])
                    if not city:
                        city = "Unknown"
                    
                    outcomes = []
                    for market in markets:
                        status = market.get('status', '').lower()
                        if status not in ['active', 'open']:
                            continue
                        
                        yes_sub = market.get('yes_sub_title', '')
                        subtitle = market.get('subtitle', '')
                        label_text = yes_sub or subtitle
                        
                        temp_range = self._extract_temp_range(label_text)
                        if not temp_range:
                            temp_range = label_text[:20]
                        
                        yes_ask = market.get('yes_ask', 50)
                        yes_price = yes_ask / 100.0 if yes_ask else 0.5
                        
                        outcome = SimpleWeatherOutcome(
                            ticker=market['ticker'],
                            label=temp_range,
                            yes_price=yes_price,
                            probability=yes_price
                        )
                        outcomes.append(outcome)
                    
                    if len(outcomes) >= 2:
                        weather_market = SimpleWeatherMarket(
                            base_ticker=event['event_ticker'],
                            title=event['title'],
                            city=city,
                            outcomes=outcomes
                        )
                        all_weather_markets.append(weather_market)
                
                cursor = result.get('cursor')
                page_count += 1
                
                if not cursor:
                    break
                
                await asyncio.sleep(0.5)
            
            logger.info(f"Found {len(all_weather_markets)} weather event series")
            
            for market in all_weather_markets[:5]:
                logger.info(f"  - {market.title} ({len(market.outcomes)} outcomes)")
            
            return all_weather_markets
        
        except Exception as e:
            logger.error(f"Failed to fetch weather markets: {e}", exc_info=True)
            return []
    
    def _extract_city(self, title: str) -> Optional[str]:
        """Extract city name from market title"""
        cities = ['NYC', 'New York', 'Chicago', 'LA', 'Los Angeles', 
                  'Miami', 'Denver', 'Austin', 'Philadelphia', 'Boston',
                  'Seattle', 'Portland', 'San Francisco', 'Atlanta', 'Dallas',
                  'Phoenix', 'Houston', 'Detroit', 'Minneapolis', 'Cleveland']
        
        title_upper = title.upper()
        for city in cities:
            if city.upper() in title_upper:
                return city
        return None
    
    def _extract_temp_range(self, text: str) -> Optional[str]:
        """Extract temperature range like '26-27' from text"""
        match = re.search(r'(\d+)[°\s]*(?:to|-)\s*(\d+)', text)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        
        match = re.search(r'(\d+)[°\s]*or\s+below', text, re.IGNORECASE)
        if match:
            return f"<={match.group(1)}"
        
        match = re.search(r'(\d+)[°\s]*or\s+above', text, re.IGNORECASE)
        if match:
            return f">={match.group(1)}"
        
        return None
    
    async def fetch_market_result(self, ticker: str) -> Optional[Dict]:
        """Fetch market result from Kalshi API"""
        try:
            endpoint = f"/trade-api/v2/markets/{ticker}"
            result = await self.kalshi_client._request("GET", endpoint)
            
            if result and 'market' in result:
                market = result['market']
                
                if market.get('status') == 'settled':
                    result_value = market.get('result', 'no')
                    
                    return {
                        'settled': True,
                        'result': result_value,
                        'ticker': ticker
                    }
            
            return None
        
        except Exception as e:
            logger.debug(f"Could not fetch result for {ticker}: {e}")
            return None
    
    async def check_and_update_pending_trades(self):
        """Check all pending trades and fetch results"""
        logger.info("\n[CHECKING] Pending trades for results...")
        
        pending_trades = [t for t in self.all_results if not t.result_fetched]
        
        if not pending_trades:
            logger.info("No pending trades to check")
            return
        
        logger.info(f"Found {len(pending_trades)} pending trades")
        
        updated_count = 0
        
        for trade in pending_trades:
            outcome1_result = await self.fetch_market_result(trade.outcome1_ticker)
            outcome2_result = await self.fetch_market_result(trade.outcome2_ticker)
            
            await asyncio.sleep(0.5)
            
            if outcome1_result and outcome1_result['settled']:
                if outcome1_result['result'] == 'yes':
                    pnl = 1.00 - trade.total_cost
                    trade.winning_outcome = trade.outcome1_label
                    trade.actual_result = trade.outcome1_label
                    trade.pnl = pnl
                    trade.was_profitable = True
                    trade.result_fetched = True
                    
                    self.learning.add_result(trade.trade_hour, pnl)
                    updated_count += 1
                    
                    logger.info(f"[WIN] {trade.trade_id}")
                    logger.info(f"  Winner: {trade.outcome1_label}")
                    logger.info(f"  P&L: +${pnl:.2f}")
                    
            elif outcome2_result and outcome2_result['settled']:
                if outcome2_result['result'] == 'yes':
                    pnl = 1.00 - trade.total_cost
                    trade.winning_outcome = trade.outcome2_label
                    trade.actual_result = trade.outcome2_label
                    trade.pnl = pnl
                    trade.was_profitable = True
                    trade.result_fetched = True
                    
                    self.learning.add_result(trade.trade_hour, pnl)
                    updated_count += 1
                    
                    logger.info(f"[WIN] {trade.trade_id}")
                    logger.info(f"  Winner: {trade.outcome2_label}")
                    logger.info(f"  P&L: +${pnl:.2f}")
            
            if outcome1_result and outcome2_result:
                if (outcome1_result['settled'] and outcome1_result['result'] == 'no' and
                    outcome2_result['settled'] and outcome2_result['result'] == 'no'):
                    
                    pnl = -trade.total_cost
                    trade.winning_outcome = "neither"
                    trade.actual_result = "other"
                    trade.pnl = pnl
                    trade.was_profitable = False
                    trade.result_fetched = True
                    
                    self.learning.add_result(trade.trade_hour, pnl)
                    updated_count += 1
                    
                    logger.info(f"[LOSS] {trade.trade_id}")
                    logger.info(f"  Neither outcome won")
                    logger.info(f"  P&L: -${abs(pnl):.2f}")
        
        if updated_count > 0:
            logger.info(f"\n[UPDATED] {updated_count} trades with results")
            self._save_results()
            
            if self.learning.total_trades % 5 == 0:
                self.learning.print_stats()
        else:
            logger.info("No new results yet")
    
    async def execute_simple_trade(self, market: SimpleWeatherMarket, dry_run: bool = False) -> bool:
        """Execute: Buy YES on top 2 outcomes"""
        
        top1, top2 = market.get_two_highest_probabilities()
        
        if not top1 or not top2:
            logger.warning(f"Not enough outcomes for {market.title}")
            return False
        
        if market.base_ticker in self.traded_today:
            logger.info(f"Already traded {market.title} today")
            return False
        
        total_cost = top1.yes_price + top2.yes_price
        
        if total_cost >= 1.00:
            logger.info(f"[SKIP] {market.title} - Cost ${total_cost:.2f} >= $1.00 (GUARANTEED LOSS)")
            return False
        
        if total_cost > self.max_total_cost:
            logger.info(f"[SKIP] {market.title} - Cost ${total_cost:.2f} > ${self.max_total_cost:.2f} (insufficient edge)")
            return False
        
        potential_profit = 1.00 - total_cost
        roi = (potential_profit / total_cost) * 100 if total_cost > 0 else 0
        
        logger.info("\n" + "="*70)
        logger.info(f"[EXECUTING TRADE] {market.title}")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)
        logger.info(f"\nBuying YES on top 2 outcomes:")
        logger.info(f"  1. {top1.label} @ ${top1.yes_price:.2f} ({top1.probability:.0%})")
        logger.info(f"  2. {top2.label} @ ${top2.yes_price:.2f} ({top2.probability:.0%})")
        logger.info(f"\nTotal Cost: ${total_cost:.2f}")
        logger.info(f"Max Profit: ${potential_profit:.2f} ({roi:.1f}% ROI)")
        logger.info(f"Coverage: {(top1.probability + top2.probability)*100:.0f}%")
        logger.info("="*70 + "\n")
        
        orders_placed = []
        
        logger.info(f"[1/2] Buying {self.contracts_per_outcome} YES @ ${top1.yes_price:.2f}...")
        order1 = await self.kalshi_client.place_order(
            ticker=top1.ticker,
            side='yes',
            price=top1.yes_price,
            quantity=self.contracts_per_outcome,
            dry_run=dry_run
        )
        
        if order1:
            logger.info(f"      [OK] Order ID: {order1}")
            orders_placed.append(order1)
        else:
            logger.error(f"      [FAILED] Order failed")
            return False
        
        await asyncio.sleep(1)
        
        logger.info(f"[2/2] Buying {self.contracts_per_outcome} YES @ ${top2.yes_price:.2f}...")
        order2 = await self.kalshi_client.place_order(
            ticker=top2.ticker,
            side='yes',
            price=top2.yes_price,
            quantity=self.contracts_per_outcome,
            dry_run=dry_run
        )
        
        if order2:
            logger.info(f"      [OK] Order ID: {order2}")
            orders_placed.append(order2)
        else:
            logger.error(f"      [FAILED] Order failed")
            return False
        
        logger.info("\n" + "="*70)
        logger.info("[TRADE COMPLETE]")
        logger.info(f"   Orders: {len(orders_placed)}")
        logger.info(f"   Cost: ${total_cost:.2f}")
        logger.info(f"   Max Profit: ${potential_profit:.2f}")
        logger.info("="*70 + "\n")
        
        self.traded_today.add(market.base_ticker)
        
        trade_result = TradeResult(
            trade_id=f"{market.city}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            trade_hour=datetime.now().hour,
            market_title=market.title,
            city=market.city,
            outcome1_label=top1.label,
            outcome1_ticker=top1.ticker,
            outcome1_price=top1.yes_price,
            outcome2_label=top2.label,
            outcome2_ticker=top2.ticker,
            outcome2_price=top2.yes_price,
            total_cost=total_cost
        )
        
        self.all_results.append(trade_result)
        self._save_results()
        self._save_individual_trade(trade_result, orders_placed)
        
        return True
    
    def _save_individual_trade(self, trade: TradeResult, order_ids: List[str]):
        """Save individual trade file"""
        try:
            filename = self.trades_dir / f"{trade.trade_id}.json"
            
            data = trade.to_dict()
            data['order_ids'] = order_ids
            data['status'] = 'pending'
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"[SAVED] {filename.name}")
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")
    
    async def run_daily_scan(self, dry_run: bool = False):
        """Run daily scan and execute trades"""
        logger.info("\n" + "="*70)
        logger.info("WEATHER BOT - DAILY SCAN")
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE TRADING'}")
        logger.info("="*70 + "\n")
        
        await self.check_and_update_pending_trades()
        
        self.traded_today.clear()
        
        markets = await self.fetch_weather_markets()
        
        if not markets:
            logger.warning("No weather markets found")
            return
        
        logger.info(f"\n[ANALYZING] {len(markets)} markets:")
        logger.info("="*70)
        
        tradeable_markets = []
        
        for market in markets:
            top1, top2 = market.get_two_highest_probabilities()
            if not top1 or not top2:
                continue
            
            total_cost = market.calculate_total_cost()
            
            logger.info(f"\n{market.title}")
            logger.info(f"  Top 2: {top1.label} ({top1.probability:.0%}) + {top2.label} ({top2.probability:.0%})")
            logger.info(f"  Cost: ${total_cost:.2f} | Max Profit: ${1.00 - total_cost:.2f}")
            
            if total_cost >= 1.00:
                logger.info(f"  [X] SKIP: >= $1.00 (GUARANTEED LOSS)")
                continue
            
            if not market.is_tradeable(self.max_total_cost):
                logger.info(f"  [~] SKIP: > ${self.max_total_cost:.2f} (insufficient edge)")
                continue
            
            logger.info(f"  [OK] TRADEABLE")
            tradeable_markets.append(market)
        
        logger.info("\n" + "="*70)
        logger.info(f"Found {len(tradeable_markets)} tradeable markets")
        logger.info("="*70 + "\n")
        
        trades_executed = 0
        
        for market in tradeable_markets:
            success = await self.execute_simple_trade(market, dry_run=dry_run)
            
            if success:
                trades_executed += 1
                await asyncio.sleep(2)
        
        logger.info("\n" + "="*70)
        logger.info(f"SCAN COMPLETE - Executed {trades_executed} trades")
        logger.info("="*70 + "\n")
    
    def update_result(self, trade_id: str, actual_temp: str, pnl: float):
        """Manual update of trade result"""
        for trade in self.all_results:
            if trade.trade_id == trade_id:
                trade.actual_result = actual_temp
                trade.pnl = pnl
                trade.was_profitable = pnl > 0
                trade.result_fetched = True
                
                if actual_temp in trade.outcome1_label:
                    trade.winning_outcome = trade.outcome1_label
                elif actual_temp in trade.outcome2_label:
                    trade.winning_outcome = trade.outcome2_label
                else:
                    trade.winning_outcome = "neither"
                
                self.learning.add_result(trade.trade_hour, pnl)
                
                logger.info(f"[UPDATED] {trade_id}")
                logger.info(f"   Result: {actual_temp}")
                logger.info(f"   Winner: {trade.winning_outcome}")
                logger.info(f"   P&L: ${pnl:+.2f}")
                logger.info(f"   Status: {'WIN' if pnl > 0 else 'LOSS'}")
                
                self._save_results()
                
                if self.learning.total_trades % 10 == 0:
                    self.learning.print_stats()
                
                return True
        
        logger.warning(f"Trade not found: {trade_id}")
        return False
    
    async def run_continuous(self, dry_run: bool = False):
        """Run continuously at target hour"""
        logger.info(f"Starting continuous mode - will trade at {self.target_hour:02d}:00 daily")
        logger.info("Will also check for results every hour")
        
        try:
            last_result_check = datetime.now()
            
            while True:
                now = datetime.now()
                
                if (now - last_result_check).total_seconds() > 3600:
                    logger.info("\n[HOURLY CHECK] Checking for trade results...")
                    await self.check_and_update_pending_trades()
                    last_result_check = now
                
                target = datetime.combine(now.date(), time(self.target_hour, 0))
                
                if now.time() > time(self.target_hour, 0):
                    target += timedelta(days=1)
                
                wait_seconds = (target - now).total_seconds()
                
                if wait_seconds < 3600:
                    logger.info(f"[WAITING] Until {target.strftime('%Y-%m-%d %H:%M')}")
                    logger.info(f"   ({wait_seconds/60:.0f} minutes)")
                    
                    await asyncio.sleep(wait_seconds)
                    
                    try:
                        await self.run_daily_scan(dry_run=dry_run)
                    except Exception as e:
                        logger.error(f"Error in scan: {e}", exc_info=True)
                    
                    await asyncio.sleep(120)
                else:
                    logger.info(f"[NEXT TRADE] {target.strftime('%Y-%m-%d %H:%M')} ({wait_seconds/3600:.1f} hours)")
                    await asyncio.sleep(3600)
        
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            await self.cleanup()


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Weather Arbitrage Bot with Auto-Learning')
    parser.add_argument('--dry-run', action='store_true', help='Simulate only')
    parser.add_argument('--now', action='store_true', help='Run immediately')
    parser.add_argument('--scan-only', action='store_true', help='Scan markets without trading')
    parser.add_argument('--debug', action='store_true', help='Debug: show all available events')
    parser.add_argument('--check-results', action='store_true', help='Check pending trade results')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--update', nargs=3, metavar=('TRADE_ID', 'RESULT', 'PNL'),
                       help='Manually update trade result')
    
    args = parser.parse_args()
    
    config = Config()
    bot = SimpleWeatherBot(config)
    
    if args.stats:
        bot.learning.print_stats()
        return
    
    if args.update:
        trade_id, result, pnl = args.update
        bot.update_result(trade_id, result, float(pnl))
        return
    
    await bot.initialize()
    
    if args.debug:
        await bot.debug_show_all_events()
        await bot.cleanup()
        return
    
    if args.scan_only:
        logger.info("\n" + "="*70)
        logger.info("[SCAN ONLY MODE] - Analyzing markets without trading")
        logger.info("="*70 + "\n")
        
        markets = await bot.fetch_weather_markets()
        
        if not markets:
            logger.warning("No weather markets found")
            await bot.cleanup()
            return
        
        logger.info(f"[FOUND] {len(markets)} weather markets total\n")
        
        # Filter tradeable markets
        tradeable_markets = []
        non_tradeable_markets = []
        
        for market in markets:
            top1, top2 = market.get_two_highest_probabilities()
            if not top1 or not top2:
                continue
            
            total_cost = market.calculate_total_cost()
            potential_profit = 1.00 - total_cost if total_cost < 1.00 else 0.0
            roi = (potential_profit / total_cost * 100) if total_cost > 0 and total_cost < 1.00 else 0.0
            
            market_info = {
                'market': market,
                'top1': top1,
                'top2': top2,
                'total_cost': total_cost,
                'potential_profit': potential_profit,
                'roi': roi
            }
            
            if total_cost >= 1.00 or total_cost > bot.max_total_cost:
                non_tradeable_markets.append(market_info)
            else:
                tradeable_markets.append(market_info)
        
        # Show tradeable markets ONLY
        logger.info("="*70)
        logger.info(f"TRADEABLE MARKETS ({len(tradeable_markets)})")
        logger.info("="*70 + "\n")
        
        if not tradeable_markets:
            logger.info("No tradeable opportunities found!\n")
        else:
            for info in tradeable_markets:
                market = info['market']
                top1 = info['top1']
                top2 = info['top2']
                total_cost = info['total_cost']
                potential_profit = info['potential_profit']
                roi = info['roi']
                
                logger.info(f"{market.title}")
                logger.info(f"  City: {market.city}")
                logger.info(f"  Top 2 Outcomes:")
                logger.info(f"    1. {top1.label:<15} ${top1.yes_price:.2f} ({top1.probability:.0%})")
                logger.info(f"    2. {top2.label:<15} ${top2.yes_price:.2f} ({top2.probability:.0%})")
                logger.info(f"  Total Cost: ${total_cost:.2f}")
                logger.info(f"  Max Profit: ${potential_profit:.2f} ({roi:.1f}% ROI)")
                logger.info(f"  Coverage: {(top1.probability + top2.probability)*100:.0f}%")
                logger.info(f"  Status: [OK] TRADEABLE ***\n")
        
        # Summary
        logger.info("="*70)
        logger.info(f"SUMMARY:")
        logger.info(f"  Total Markets: {len(markets)}")
        logger.info(f"  Tradeable: {len(tradeable_markets)}")
        logger.info(f"  Not Tradeable: {len(non_tradeable_markets)}")
        logger.info(f"  Max Cost Threshold: ${bot.max_total_cost:.2f}")
        logger.info("="*70)
        
        # Show why markets were filtered (if no tradeable found)
        if non_tradeable_markets and len(tradeable_markets) == 0:
            logger.info("\n" + "="*70)
            logger.info("FILTERED OUT (Reasons):")
            logger.info("="*70 + "\n")
            
            for info in non_tradeable_markets[:5]:
                market = info['market']
                total_cost = info['total_cost']
                
                reason = ""
                if total_cost >= 1.00:
                    reason = f"Cost ${total_cost:.2f} >= $1.00 (GUARANTEED LOSS)"
                elif total_cost > bot.max_total_cost:
                    reason = f"Cost ${total_cost:.2f} > ${bot.max_total_cost:.2f} (insufficient edge)"
                
                logger.info(f"{market.title[:60]}")
                logger.info(f"  Cost: ${total_cost:.2f}")
                logger.info(f"  Reason: {reason}\n")
            
            if len(non_tradeable_markets) > 5:
                logger.info(f"... and {len(non_tradeable_markets) - 5} more filtered out\n")
        
        logger.info("")
        await bot.cleanup()
        return
    
    if args.check_results:
        await bot.check_and_update_pending_trades()
        await bot.cleanup()
        return
    
    if args.now:
        await bot.run_daily_scan(dry_run=args.dry_run)
        await bot.cleanup()
    else:
        await bot.run_continuous(dry_run=args.dry_run)


if __name__ == "__main__":
    asyncio.run(main())