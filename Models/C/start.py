#start.py - Multi-Timeframe Trading Engine with Data Collection
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
import time
import sys
import os
import logging
import requests
import json
from pathlib import Path
from flask import Flask, request, jsonify
from alpaca_trade_api.rest import REST, TimeFrame
import threading
from scipy import stats
from collections import defaultdict

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / '.env')
except ImportError:
    pass  # dotenv not installed, rely on system env vars

warnings.filterwarnings('ignore')

# ==========================
# ====  API CREDENTIALS ====
# ==========================
# Load from environment variables (NEVER hard-code credentials!)
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')

# ==========================
# ====  CONFIGURATION   ====
# ==========================
ANALYSIS_INTERVAL = 60  # 1 minute - CONTINUOUS
TRADE_CAPITAL_PER_POSITION = 100

# Create data directories
DATA_DIR = Path("trading_data")
DATA_DIR.mkdir(exist_ok=True)
TRADES_DIR = DATA_DIR / "trades"
TRADES_DIR.mkdir(exist_ok=True)
SIGNALS_DIR = DATA_DIR / "signals"
SIGNALS_DIR.mkdir(exist_ok=True)
PERFORMANCE_DIR = DATA_DIR / "performance"
PERFORMANCE_DIR.mkdir(exist_ok=True)

# Multiple timeframes for analysis
TIMEFRAMES = {
    '5Min': {'bars': 500, 'name': '5m'},
    '15Min': {'bars': 800, 'name': '15m'},
    '1Hour': {'bars': 1000, 'name': '1h'},
    '1Day': {'bars': 1000, 'name': '1d'},
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("metric_space_engine.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Fix console encoding for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

app = Flask(__name__)
trading_client = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

is_trading = True
OPEN_POSITIONS: Dict[str, Dict] = {}
TRADE_SYMBOLS = set()

# Cache for data to avoid excessive API calls
DATA_CACHE: Dict[str, Dict] = defaultdict(dict)
CACHE_DURATION = 60  # Cache data for 60 seconds

# ==========================
# ==== DATA COLLECTOR   ====
# ==========================
class TradingDataCollector:
    """Collect and store all trading data for future learning"""
    
    def __init__(self):
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.trades_file = TRADES_DIR / f"trades_{self.session_id}.jsonl"
        self.signals_file = SIGNALS_DIR / f"signals_{self.session_id}.jsonl"
        self.performance_file = PERFORMANCE_DIR / f"performance_{self.session_id}.json"
        
        self.session_stats = {
            'start_time': datetime.now().isoformat(),
            'total_signals': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'symbols_traded': set(),
            'timeframes_used': defaultdict(int)
        }
    
    def log_signal(self, symbol: str, timeframe: str, signal_data: Dict):
        """Log every signal generated"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'prediction': float(signal_data.get('prediction', 0)),
            'confidence': float(signal_data.get('confidence', 0)),
            'win_prob': float(signal_data.get('win_prob', 0)),
            'hurst': float(signal_data.get('hurst', 0)),
            'sharpe': float(signal_data.get('sharpe', 0)),
            'is_long': bool(signal_data.get('is_long', False)),
            'is_short': bool(signal_data.get('is_short', False)),
            'price': float(signal_data.get('price', 0))
        }
        
        # Append to JSONL file (one JSON per line)
        with open(self.signals_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
        
        self.session_stats['total_signals'] += 1
        self.session_stats['timeframes_used'][timeframe] += 1
    
    def log_trade_entry(self, symbol: str, entry_data: Dict):
        """Log trade entry"""
        record = {
            'trade_id': f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'type': 'ENTRY',
            'symbol': symbol,
            'entry_price': float(entry_data['entry']),
            'stop_loss': float(entry_data['stop']),
            'target': float(entry_data['target']),
            'prediction': float(entry_data['prediction']),
            'confidence': float(entry_data['confidence']),
            'win_prob': float(entry_data['win_prob']),
            'hurst': float(entry_data['hurst']),
            'sharpe': float(entry_data['sharpe']),
            'timeframe': str(entry_data['timeframe']),
            'capital': float(entry_data.get('capital', TRADE_CAPITAL_PER_POSITION))
        }
        
        with open(self.trades_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
        
        self.session_stats['total_trades'] += 1
        self.session_stats['symbols_traded'].add(symbol)
    
    def log_trade_exit(self, symbol: str, exit_data: Dict):
        """Log trade exit and calculate actual performance"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'type': 'EXIT',
            'symbol': symbol,
            'entry_price': float(exit_data['entry_price']),
            'exit_price': float(exit_data['exit_price']),
            'pnl': float(exit_data['pnl']),
            'pnl_percent': float(exit_data['pnl_percent']),
            'hold_time_hours': float(exit_data.get('hold_time_hours', 0))
        }
        
        with open(self.trades_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
        
        # Update session stats
        if exit_data['pnl'] > 0:
            self.session_stats['winning_trades'] += 1
        else:
            self.session_stats['losing_trades'] += 1
        
        self.session_stats['total_pnl'] += exit_data['pnl']
    
    def save_session_performance(self):
        """Save overall session performance"""
        stats = self.session_stats.copy()
        stats['end_time'] = datetime.now().isoformat()
        stats['symbols_traded'] = list(stats['symbols_traded'])
        stats['timeframes_used'] = dict(stats['timeframes_used'])
        
        if stats['total_trades'] > 0:
            stats['win_rate'] = stats['winning_trades'] / stats['total_trades']
        else:
            stats['win_rate'] = 0.0
        
        with open(self.performance_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logging.info(f"[DATA] Session performance saved to {self.performance_file}")
    
    def analyze_historical_performance(self) -> Dict:
        """Analyze all historical trades to improve model"""
        all_trades = []
        
        # Load all trade files
        for trade_file in TRADES_DIR.glob("trades_*.jsonl"):
            with open(trade_file, 'r') as f:
                for line in f:
                    all_trades.append(json.loads(line))
        
        if not all_trades:
            return {}
        
        # Separate entries and exits
        entries = [t for t in all_trades if t.get('type') == 'ENTRY']
        exits = [t for t in all_trades if t.get('type') == 'EXIT']
        
        # Calculate statistics
        analysis = {
            'total_trades': len(entries),
            'total_exits': len(exits),
            'avg_win_prob': np.mean([t['win_prob'] for t in entries if 'win_prob' in t]) if entries else 0,
            'avg_confidence': np.mean([t['confidence'] for t in entries if 'confidence' in t]) if entries else 0,
            'avg_hurst': np.mean([t['hurst'] for t in entries if 'hurst' in t]) if entries else 0,
            'avg_pnl': np.mean([t['pnl'] for t in exits if 'pnl' in t]) if exits else 0,
            'win_rate': len([t for t in exits if t.get('pnl', 0) > 0]) / len(exits) if exits else 0,
            'best_timeframes': self._get_best_timeframes(entries, exits),
            'best_symbols': self._get_best_symbols(entries, exits)
        }
        
        return analysis
    
    def _get_best_timeframes(self, entries, exits) -> Dict:
        """Find which timeframes perform best"""
        tf_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0})
        
        # Match entries with exits by symbol and timestamp
        for entry in entries:
            symbol = entry['symbol']
            tf = entry.get('timeframe', 'unknown')
            
            # Find corresponding exit
            matching_exits = [e for e in exits if e['symbol'] == symbol 
                            and e['timestamp'] > entry['timestamp']]
            
            if matching_exits:
                exit_trade = matching_exits[0]
                pnl = exit_trade.get('pnl', 0)
                
                if pnl > 0:
                    tf_performance[tf]['wins'] += 1
                else:
                    tf_performance[tf]['losses'] += 1
                
                tf_performance[tf]['total_pnl'] += pnl
        
        return dict(tf_performance)
    
    def _get_best_symbols(self, entries, exits) -> Dict:
        """Find which symbols perform best"""
        symbol_performance = defaultdict(lambda: {'trades': 0, 'wins': 0, 'total_pnl': 0})
        
        for entry in entries:
            symbol = entry['symbol']
            matching_exits = [e for e in exits if e['symbol'] == symbol 
                            and e['timestamp'] > entry['timestamp']]
            
            if matching_exits:
                exit_trade = matching_exits[0]
                pnl = exit_trade.get('pnl', 0)
                
                symbol_performance[symbol]['trades'] += 1
                if pnl > 0:
                    symbol_performance[symbol]['wins'] += 1
                symbol_performance[symbol]['total_pnl'] += pnl
        
        # Sort by total PnL
        sorted_symbols = sorted(symbol_performance.items(), 
                               key=lambda x: x[1]['total_pnl'], 
                               reverse=True)
        
        return dict(sorted_symbols[:10])  # Top 10 symbols

# Initialize global data collector
data_collector = TradingDataCollector()

# ==========================
# ==== DISCORD NOTIFIER ====
# ==========================
def send_discord_embed(title: str, color: int, fields: List[Dict], symbol: str = None):
    """Universal Discord notification system"""
    try:
        embed = {
            "title": title,
            "color": color,
            "fields": fields,
            "footer": {"text": "Metric Space Engine • Multi-Timeframe ML"},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        payload = {
            "username": "Metric Space Engine",
            "embeds": [embed]
        }
        
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status()
        logging.info(f"Discord notification sent: {title}")
        
    except Exception as e:
        logging.error(f"Discord notification failed: {e}")

def notify_long_entry(symbol: str, entry: float, stop: float, target: float, 
                     confidence: float, prediction: float, hurst: float, sharpe: float,
                     win_prob: float, timeframe: str, capital: float = 100):
    """Send LONG entry notification AND LOG DATA"""
    
    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr_ratio = reward / risk if risk > 0 else 0
    position = capital / entry
    expected_pnl = rr_ratio * confidence * capital
    
    regime = "TRENDING" if hurst > 0.55 else "RANGING" if hurst < 0.45 else "NEUTRAL"
    
    # Store position
    OPEN_POSITIONS[symbol] = {
        "entry": entry,
        "confidence": confidence,
        "timestamp": datetime.utcnow(),
        "prediction": prediction,
        "hurst": hurst,
        "win_prob": win_prob,
        "timeframe": timeframe
    }
    
    # LOG THE TRADE ENTRY
    data_collector.log_trade_entry(symbol, {
        'entry': entry,
        'stop': stop,
        'target': target,
        'prediction': prediction,
        'confidence': confidence,
        'win_prob': win_prob,
        'hurst': hurst,
        'sharpe': sharpe,
        'timeframe': timeframe,
        'capital': capital
    })
    
    fields = [
        {"name": "Timeframe", "value": timeframe, "inline": True},
        {"name": "Entry Price", "value": f"${entry:.2f}", "inline": True},
        {"name": "Stop Loss", "value": f"${stop:.2f}", "inline": True},
        {"name": "Target", "value": f"${target:.2f}", "inline": True},
        {"name": "R/R Ratio", "value": f"{rr_ratio:.2f}", "inline": True},
        {"name": "Position Size", "value": f"{position:.4f}", "inline": True},
        {"name": "Expected PnL", "value": f"${expected_pnl:.2f}", "inline": True},
        {"name": "Confidence", "value": f"{confidence:.1%}", "inline": True},
        {"name": "Win Probability", "value": f"{win_prob:.1%}", "inline": True},
        {"name": "Sharpe Ratio", "value": f"{sharpe:.2f}", "inline": True},
        {"name": "ML Prediction", "value": f"{prediction:.2f}", "inline": True},
        {"name": "Market Regime", "value": f"{regime} (H={hurst:.2f})", "inline": False}
    ]
    
    send_discord_embed(f"LONG {symbol}", 3066993, fields, symbol)

def notify_exit(symbol: str, exit_price: float, entry_price: float, 
               pnl: float, pnl_percent: float):
    """Send EXIT notification AND LOG DATA"""
    
    if symbol not in OPEN_POSITIONS:
        return
    
    position_data = OPEN_POSITIONS[symbol]
    hold_time = datetime.utcnow() - position_data['timestamp']
    hours_held = hold_time.total_seconds() / 3600
    
    # LOG THE TRADE EXIT
    data_collector.log_trade_exit(symbol, {
        'entry_price': entry_price,
        'exit_price': exit_price,
        'pnl': pnl,
        'pnl_percent': pnl_percent,
        'hold_time_hours': hours_held
    })
    
    color = 3066993 if pnl > 0 else 15158332  # Green if profit, red if loss
    
    fields = [
        {"name": "Timeframe", "value": position_data.get('timeframe', 'N/A'), "inline": True},
        {"name": "Exit Price", "value": f"${exit_price:.2f}", "inline": True},
        {"name": "PnL", "value": f"${pnl:.2f}", "inline": True},
        {"name": "Return %", "value": f"{pnl_percent:.2f}%", "inline": True},
        {"name": "Entry Price", "value": f"${entry_price:.2f}", "inline": True},
        {"name": "Hold Time", "value": f"{hours_held:.1f}h", "inline": True},
        {"name": "Entry Date", "value": position_data['timestamp'].strftime("%Y-%m-%d %H:%M"), "inline": True}
    ]
    
    send_discord_embed(f"EXIT {symbol}", color, fields, symbol)
    
    # Remove from open positions
    del OPEN_POSITIONS[symbol]

# ==========================
# ==== ALPACA DATA      ====
# ==========================
def get_alpaca_bars(symbol: str, timeframe_key: str, limit: int = 1000) -> Tuple[pd.DataFrame, str]:
    """Fetch data from Alpaca with caching - FIXED DATE FORMAT"""
    
    # Check cache
    cache_key = f"{symbol}_{timeframe_key}"
    if cache_key in DATA_CACHE:
        cached_data, cache_time = DATA_CACHE[cache_key]
        if (datetime.now() - cache_time).total_seconds() < CACHE_DURATION:
            return cached_data, "Alpaca (Cached)"
    
    try:
        tf_config = TIMEFRAMES[timeframe_key]
        
        # Calculate start date based on timeframe
        if timeframe_key == '5Min':
            start = datetime.now() - timedelta(days=10)
        elif timeframe_key == '15Min':
            start = datetime.now() - timedelta(days=30)
        elif timeframe_key == '1Hour':
            start = datetime.now() - timedelta(days=90)
        else:  # 1Day
            start = datetime.now() - timedelta(days=730)
        
        # FIX: Format as YYYY-MM-DD only (Alpaca doesn't want time component)
        start_str = start.strftime('%Y-%m-%d')
        
        logging.info(f"[{timeframe_key}] Fetching {symbol} from {start_str}...")
        
        # Fetch bars using the correct TimeFrame
        if timeframe_key == '5Min':
            bars = trading_client.get_bars(
                symbol, 
                TimeFrame.Minute,
                start=start_str,
                limit=limit * 5  # Get 5x to ensure we have enough 5-min bars
            ).df
            # Resample to 5-minute bars
            if not bars.empty:
                bars = bars.resample('5Min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
        elif timeframe_key == '15Min':
            bars = trading_client.get_bars(
                symbol, 
                TimeFrame.Minute,
                start=start_str,
                limit=limit * 15  # Get more data since we'll resample
            ).df
            # Resample to 15-minute bars
            if not bars.empty:
                bars = bars.resample('15Min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
        elif timeframe_key == '1Hour':
            bars = trading_client.get_bars(
                symbol, 
                TimeFrame.Hour,
                start=start_str,
                limit=limit
            ).df
            
        else:  # 1Day
            bars = trading_client.get_bars(
                symbol, 
                TimeFrame.Day,
                start=start_str,
                limit=limit
            ).df
        
        if bars.empty:
            logging.warning(f"[{timeframe_key}] No data for {symbol}")
            return None, "Alpaca (Empty)"
        
        # Standardize column names
        bars.columns = [col.lower() for col in bars.columns]
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in bars.columns for col in required):
            logging.warning(f"[{timeframe_key}] Missing columns for {symbol}")
            return None, "Alpaca (Missing Columns)"
        
        # Cache the data
        DATA_CACHE[cache_key] = (bars, datetime.now())
        
        logging.info(f"[{timeframe_key}] Got {len(bars)} bars for {symbol}")
        return bars, "Alpaca"
        
    except Exception as e:
        logging.error(f"[{timeframe_key}] Alpaca error for {symbol}: {e}")
        return None, f"Alpaca Error: {str(e)[:50]}"

def get_live_price(symbol: str) -> Optional[float]:
    """Get current price from Alpaca"""
    try:
        latest_trade = trading_client.get_latest_trade(symbol)
        return float(latest_trade.p)
    except Exception as e:
        logging.error(f"Error getting price for {symbol}: {e}")
        return None

def has_open_position(symbol: str) -> bool:
    """Check if we have an open position"""
    try:
        positions = trading_client.list_positions()
        return any(position.symbol == symbol for position in positions)
    except Exception as e:
        logging.error(f"Error checking position: {e}")
        return False

def execute_buy(symbol: str, entry: float, stop: float, target: float,
               prediction: float, confidence: float, hurst: float, 
               sharpe: float, win_prob: float, timeframe: str) -> bool:
    """Execute buy order and send notification"""
    if not is_trading:
        logging.info(f"[PAPER] Would buy {symbol}")
        return False
    
    try:
        account = trading_client.get_account()
        balance = float(account.cash)
        
        if balance < TRADE_CAPITAL_PER_POSITION:
            logging.error(f"Insufficient balance: ${balance:.2f}")
            return False
        
        quantity = TRADE_CAPITAL_PER_POSITION / entry
        
        if quantity <= 0 or quantity < 0.000001:
            logging.error(f"Invalid quantity: {quantity}")
            return False
        
        order = trading_client.submit_order(
            symbol=symbol,
            qty=round(quantity, 6),
            side='buy',
            type='market',
            time_in_force='day'
        )
        
        logging.info(
            f"[BUY] {quantity:.6f} {symbol} @ ${entry:.2f} | "
            f"TF: {timeframe} | WinProb: {win_prob:.1%}"
        )
        
        # Send Discord notification
        notify_long_entry(
            symbol=symbol,
            entry=entry,
            stop=stop,
            target=target,
            confidence=confidence,
            prediction=prediction,
            hurst=hurst,
            sharpe=sharpe,
            win_prob=win_prob,
            timeframe=timeframe,
            capital=TRADE_CAPITAL_PER_POSITION
        )
        
        TRADE_SYMBOLS.add(symbol)
        return True
        
    except Exception as e:
        logging.error(f"Error buying {symbol}: {e}")
        return False

def execute_sell(symbol: str) -> bool:
    """Execute sell order and send notification"""
    try:
        position = trading_client.get_position(symbol)
        qty = float(position.qty) * 0.99
        entry_price = float(position.avg_entry_price)
        current_price = float(position.current_price)
        
        pnl = (current_price - entry_price) * qty
        pnl_percent = ((current_price / entry_price) - 1) * 100
        
        order = trading_client.submit_order(
            symbol=symbol,
            qty=round(qty, 6),
            side='sell',
            type='market',
            time_in_force='day'
        )
        
        logging.info(f"[SELL] {qty:.6f} {symbol} @ ${current_price:.2f} | PnL: ${pnl:.2f} ({pnl_percent:.2f}%)")
        
        # Send Discord notification
        notify_exit(
            symbol=symbol,
            exit_price=current_price,
            entry_price=entry_price,
            pnl=pnl,
            pnl_percent=pnl_percent
        )
        
        return True
        
    except Exception as e:
        logging.error(f"Error selling {symbol}: {e}")
        return False

# ==========================
# ==== DATACLASSES      ====
# ==========================
@dataclass
class Settings:
    neighbors_count: int = 8
    max_bars_back: int = 400
    feature_count: int = 5
    use_dynamic_exits: bool = True
    use_adaptive_k: bool = True
    volatility_scaling: bool = True

@dataclass
class FilterSettings:
    use_volatility_filter: bool = True
    use_regime_filter: bool = True
    regime_threshold: float = 0.5
    min_trend_strength: float = 0.35
    use_parkinson_volatility: bool = True

@dataclass
class KernelSettings:
    lookback_window: int = 8
    relative_weighting: float = 8.0
    regression_level: int = 25
    lag: int = 2

@dataclass
class FeatureConfig:
    feature_type: str
    param_a: int
    param_b: int
    weight: float = 1.0

# ==========================
# ==== MATH UTILS       ====
# ==========================
class MathUtils:
    @staticmethod
    def rolling_z_score(series: pd.Series, window: int = 100) -> pd.Series:
        roll = series.rolling(window=window, min_periods=1)
        return (series - roll.mean()) / (roll.std() + 1e-10)

    @staticmethod
    def soft_max(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum() + 1e-10)
    
    @staticmethod
    def hurst_exponent(prices: np.ndarray, max_lag: int = 20) -> float:
        if len(prices) < max_lag * 2:
            return 0.5
        
        lags = range(2, min(max_lag, len(prices) // 2))
        tau = []
        for lag in lags:
            std = np.std(np.subtract(prices[lag:], prices[:-lag]))
            if std > 0:
                tau.append(np.sqrt(std))
        
        if len(tau) < 2:
            return 0.5
        
        try:
            poly = np.polyfit(np.log(list(lags)[:len(tau)]), np.log(tau), 1)
            return max(0, min(1, poly[0] * 2.0))
        except:
            return 0.5
    
    @staticmethod
    def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
        """Parkinson volatility - more sensitive than close-based"""
        hl_ratio = np.log(high / low)
        parkinson_var = (1 / (4 * np.log(2))) * (hl_ratio ** 2)
        return np.sqrt(parkinson_var.rolling(window=window).mean())

# ==========================
# ==== INDICATORS       ====
# ==========================
class AdvancedTechnicalIndicators:
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def n_wt(hlc3: pd.Series, param_a: int, param_b: int) -> pd.Series:
        esa = hlc3.ewm(span=param_a, adjust=False).mean()
        d = (hlc3 - esa).abs().ewm(span=param_a, adjust=False).mean()
        ci = (hlc3 - esa) / (0.015 * d + 1e-10)
        wt = ci.ewm(span=param_b, adjust=False).mean()
        return wt
    
    @staticmethod
    def n_cci(close: pd.Series, param_a: int, param_b: int) -> pd.Series:
        tp = close
        sma = tp.rolling(window=param_a).mean()
        mad = (tp - sma).abs().rolling(window=param_a).mean()
        cci = (tp - sma) / (0.015 * mad + 1e-10)
        if param_b > 1:
            cci = cci.ewm(span=param_b, adjust=False).mean()
        return cci
    
    @staticmethod
    def adx(high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10))
        dx = (abs(plus_di - minus_di) / (abs(plus_di + minus_di) + 1e-10)) * 100
        return dx.ewm(alpha=1/period, adjust=False).mean()
    
    @staticmethod
    def trend_strength(close: pd.Series, period: int = 20) -> pd.Series:
        def calc_r_squared(y):
            if len(y) < 2:
                return 0
            x = np.arange(len(y))
            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            return r_value ** 2
        return close.rolling(window=period).apply(calc_r_squared, raw=True)

# ==========================
# ==== CLASSIFIER       ====
# ==========================
class MetricSpaceClassifier:
    """Optimized Lorentzian classifier"""
    
    def __init__(self, settings, filter_settings, kernel_settings, feature_configs):
        self.settings = settings
        self.filter_settings = filter_settings
        self.kernel_settings = kernel_settings
        self.feature_configs = feature_configs
    
    def nadaraya_watson_kernel(self, source: pd.Series, h: int, r: float, x: int, lag: int = 2) -> pd.Series:
        """
        Causal Nadaraya-Watson kernel with explicit lag to prevent lookahead bias.

        Uses only data from [i - window_size - lag, i - lag] - never touches current or future bars.
        The lag parameter ensures we don't use the most recent data which may not be
        finalized in live trading.
        """
        src_vals = source.values
        n = len(src_vals)
        output = np.zeros(n)
        window_size = min(x * 2, 500)

        # Precompute weights for efficiency
        indices = np.arange(window_size)
        dists = indices[::-1]  # Distance from most recent bar in window
        weights = np.power(1 + (dists**2) / (2 * r * h**2), -r)

        # Start later to account for lag
        for i in range(window_size + lag, n):
            # CRITICAL: Use data from [i - window_size - lag, i - lag]
            # This ensures we never use current bar or future data
            window_end = i - lag
            window_start = window_end - window_size

            if window_start < 0:
                continue

            window = src_vals[window_start:window_end]
            if len(window) != window_size:
                continue

            weighted_sum = np.dot(window, weights)
            sum_weights = np.sum(weights)
            output[i] = weighted_sum / (sum_weights + 1e-10)

        result = pd.Series(output, index=source.index)
        # Only forward-fill - NEVER backward-fill as it introduces lookahead
        return result.replace(0, np.nan).ffill()
    
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df['close']
        high = df['high']
        low = df['low']
        hlc3 = (high + low + close) / 3
        
        results = df.copy()
        
        # Features
        feature_matrix_list = []
        for config in self.feature_configs[:self.settings.feature_count]:
            if config.feature_type == "RSI":
                raw = AdvancedTechnicalIndicators.rsi(close, config.param_a)
            elif config.feature_type == "WT":
                raw = AdvancedTechnicalIndicators.n_wt(hlc3, config.param_a, config.param_b)
            elif config.feature_type == "CCI":
                raw = AdvancedTechnicalIndicators.n_cci(close, config.param_a, config.param_b)
            elif config.feature_type == "ADX":
                raw = AdvancedTechnicalIndicators.adx(high, low, close, config.param_a)
            else:
                raw = pd.Series(0, index=df.index)
            
            norm = MathUtils.rolling_z_score(raw, window=100).fillna(0) * config.weight
            feature_matrix_list.append(norm.values)
        
        X = np.column_stack(feature_matrix_list)

        # Targets for TRAINING ONLY (historical data where outcome is known)
        # IMPORTANT: These targets use future returns which are ONLY available
        # for historical bars. For the most recent bars, we don't know the
        # actual outcome yet - they are marked as 0 (unknown).
        #
        # The kNN classifier uses ONLY historical bars where targets are known
        # to make predictions for the current bar.
        forward_periods = 4
        future_returns = close.pct_change(forward_periods).shift(-forward_periods)
        Y = np.where(future_returns > 0, 1, -1)
        # Mark recent bars as unknown - these targets don't exist yet in live trading
        Y[-forward_periods:] = 0
        # Also mark any NaN as 0
        Y = np.where(np.isnan(future_returns.values), 0, Y)
        
        # Market Regime
        hurst_values = []
        for i in range(100, len(close)):
            window = close.iloc[max(0, i-100):i].values
            h = MathUtils.hurst_exponent(window, max_lag=20)
            hurst_values.append(h)
        
        hurst_series = pd.Series([0.5] * 100 + hurst_values, index=df.index)
        results['hurst'] = hurst_series
        is_trending = hurst_series > 0.55
        
        # Parkinson Volatility
        if self.filter_settings.use_parkinson_volatility:
            park_vol = MathUtils.parkinson_volatility(high, low, window=20)
            extreme_volatility = park_vol > park_vol.rolling(100).quantile(0.90)
            results['parkinson_vol'] = park_vol
        else:
            extreme_volatility = pd.Series(False, index=df.index)
        
        # Kernel
        yhat1 = self.nadaraya_watson_kernel(
            close,
            self.kernel_settings.lookback_window,
            self.kernel_settings.relative_weighting,
            self.kernel_settings.regression_level
        )
        results['yhat1'] = yhat1
        
        # Predictions
        predictions = np.zeros(len(df))
        confidences = np.zeros(len(df))
        
        available_bars = len(df)
        max_lookback = self.settings.max_bars_back
        min_bars_needed = max(200, min(max_lookback + 50, available_bars - 50))
        effective_max_bars = min(max_lookback, available_bars - 50)
        
        k = self.settings.neighbors_count
        returns = close.pct_change()
        volatility = returns.rolling(window=20).std().fillna(0.02).values
        trend_strength_series = AdvancedTechnicalIndicators.trend_strength(close, 20).fillna(0.5)
        
        for i in range(min_bars_needed, len(df)):
            current_vec = X[i]
            start_search = max(0, i - effective_max_bars)

            # Search only historical bars, excluding the most recent ones
            # where we don't yet know the outcome (forward_periods bars)
            search_end = max(start_search, i - 4 - 1)  # Exclude last 4 bars + current
            search_indices = np.arange(start_search, search_end, 4)

            if len(search_indices) == 0:
                continue

            history_matrix = X[search_indices]
            history_labels = Y[search_indices]

            # Filter out any bars with unknown targets (Y=0)
            valid_mask = history_labels != 0
            if np.sum(valid_mask) < k:
                continue

            history_matrix = history_matrix[valid_mask]
            history_labels = history_labels[valid_mask]

            if len(history_matrix) < k:
                continue
            
            abs_diff = np.abs(history_matrix - current_vec)
            log_diff = np.log(1 + abs_diff)
            distances = np.sum(log_diff, axis=1)
            
            current_vol = volatility[i]
            adaptive_k = int(k * (1 + np.clip(current_vol * 0.5, 0, 0.5)))
            adaptive_k = np.clip(adaptive_k, k, k * 2)
            
            if len(distances) < adaptive_k:
                continue

            nearest_indices = np.argpartition(distances, min(adaptive_k, len(distances) - 1))[:adaptive_k]
            nearest_dists = distances[nearest_indices]
            nearest_labels = history_labels[nearest_indices]
            
            weights = MathUtils.soft_max(-nearest_dists)
            pred = np.sum(nearest_labels * weights)
            predictions[i] = pred
            
            label_std = np.std(nearest_labels)
            agreement_conf = 1.0 - (label_std / (1.0 + 1e-10))
            mean_dist = np.mean(nearest_dists)
            dist_conf = 1.0 - np.clip(mean_dist / 10.0, 0, 1)
            confidences[i] = (agreement_conf * 0.7 + dist_conf * 0.3)
        
        results['prediction'] = predictions
        results['confidence'] = confidences
        
        smooth_pred = pd.Series(predictions, index=df.index).rolling(5, min_periods=1).mean()
        results['smooth_pred'] = smooth_pred
        
        pred_threshold = smooth_pred.abs().rolling(252, min_periods=20).quantile(0.75).fillna(0.35)
        conf_threshold = 0.45
        
        kernel_bullish = yhat1 > yhat1.shift(1)
        kernel_bearish = yhat1 < yhat1.shift(1)
        
        long_cond = pd.Series(
            (smooth_pred > pred_threshold) & (confidences > conf_threshold), 
            index=df.index
        )
        short_cond = pd.Series(
            (smooth_pred < -pred_threshold) & (confidences > conf_threshold), 
            index=df.index
        )
        
        if self.filter_settings.min_trend_strength > 0:
            strong_trend = trend_strength_series > self.filter_settings.min_trend_strength
        else:
            strong_trend = pd.Series(True, index=df.index)
        
        trade_allowed = is_trending & ~extreme_volatility
        
        results['start_long_trade'] = long_cond & kernel_bullish.fillna(False) & strong_trend & trade_allowed
        results['start_short_trade'] = short_cond & kernel_bearish.fillna(False) & trade_allowed
        results['end_long_trade'] = kernel_bearish.fillna(False)
        results['end_short_trade'] = kernel_bullish.fillna(False)
        
        # Sharpe
        if len(returns) > 20:
            sharpe = (returns.rolling(window=20).mean() / returns.rolling(window=20).std()) * np.sqrt(252)
            results['sharpe_ratio'] = sharpe
        else:
            results['sharpe_ratio'] = 0
        
        # Win Probability
        win_probabilities = []
        for i in range(len(df)):
            pred = predictions[i]
            conf = confidences[i]
            trend = trend_strength_series.iloc[i]
            
            normalized_pred = np.tanh(pred / 8.0)
            strength = abs(normalized_pred) * conf * trend
            win_prob = 1 / (1 + np.exp(-5 * strength))
            
            if conf < 0.3:
                win_prob = 0.5 + (win_prob - 0.5) * conf / 0.3
            
            win_probabilities.append(np.clip(win_prob, 0.3, 0.9))
        
        results['win_probability'] = win_probabilities
        
        return results

def create_metric_space_classifier():
    settings = Settings(
        neighbors_count=8,
        max_bars_back=400,
        feature_count=5,
        use_adaptive_k=True,
        volatility_scaling=True
    )
    
    filter_settings = FilterSettings(
        use_volatility_filter=True,
        use_regime_filter=True,
        regime_threshold=0.5,
        min_trend_strength=0.35,
        use_parkinson_volatility=True
    )
    
    kernel_settings = KernelSettings(
        lookback_window=8,
        relative_weighting=8.0,
        regression_level=25,
        lag=2
    )
    
    feature_configs = [
        FeatureConfig("RSI", 14, 1, weight=1.0),
        FeatureConfig("WT", 10, 11, weight=1.0),
        FeatureConfig("CCI", 20, 1, weight=1.0),
        FeatureConfig("ADX", 14, 1, weight=1.0),
        FeatureConfig("RSI", 9, 1, weight=0.8)
    ]
    
    return MetricSpaceClassifier(settings, filter_settings, kernel_settings, feature_configs)

# ==========================
# ==== MULTI-TF ENGINE  ====
# ==========================
class MultiTimeframeEngine:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.classifier = create_metric_space_classifier()
        self.last_signals = defaultdict(dict)
        
    def calculate_stop_and_target(self, entry_price: float, atr: float, direction: str = 'long'):
        stop_distance = 2.0 * atr
        target_distance = 3.0 * atr
        
        if direction == 'long':
            stop_loss = entry_price - stop_distance
            target = entry_price + target_distance
        else:
            stop_loss = entry_price + stop_distance
            target = entry_price - target_distance
        
        return stop_loss, target
    
    def evaluate_symbol_on_timeframe(self, symbol: str, timeframe_key: str) -> Optional[Dict]:
        """Evaluate a single symbol on a single timeframe AND LOG SIGNALS"""
        try:
            df, source = get_alpaca_bars(symbol, timeframe_key)
            
            if df is None or len(df) < 200:
                return None
            
            results = self.classifier.fit_predict(df)
            latest = results.iloc[-1]
            
            price = get_live_price(symbol)
            if not price:
                return None
            
            atr = results['high'].iloc[-20:].subtract(results['low'].iloc[-20:]).mean()
            
            signal_data = {
                'price': price,
                'prediction': latest['prediction'],
                'confidence': latest['confidence'],
                'win_prob': latest['win_probability'],
                'is_long': latest['start_long_trade'],
                'is_short': latest['start_short_trade'],
                'sharpe': latest.get('sharpe_ratio', 0),
                'hurst': latest['hurst'],
                'atr': atr,
                'timeframe': TIMEFRAMES[timeframe_key]['name']
            }
            
            # LOG THE SIGNAL
            data_collector.log_signal(symbol, TIMEFRAMES[timeframe_key]['name'], signal_data)
            
            return signal_data
            
        except Exception as e:
            logging.error(f"[{timeframe_key}] Error evaluating {symbol}: {e}")
            return None
    
    def evaluate_symbol_multi_timeframe(self, symbol: str):
        """Evaluate symbol across all timeframes and make trading decisions"""
        
        # Collect signals from all timeframes
        timeframe_signals = {}
        for tf_key in TIMEFRAMES.keys():
            signal = self.evaluate_symbol_on_timeframe(symbol, tf_key)
            if signal:
                timeframe_signals[tf_key] = signal
        
        if not timeframe_signals:
            return
        
        # Log analysis for all timeframes
        for tf_key, signal in timeframe_signals.items():
            regime = "TRENDING" if signal['hurst'] > 0.55 else "RANGING"
            logging.info(
                f"[{symbol}] [{signal['timeframe']}]: "
                f"Pred={signal['prediction']:.1f} | Conf={signal['confidence']:.2%} | "
                f"WinProb={signal['win_prob']:.2%} | Hurst={signal['hurst']:.2f} ({regime})"
            )
        
        # MULTI-TIMEFRAME CONFIRMATION STRATEGY
        # Require at least 2 timeframes to agree for entry
        long_signals = [tf for tf, sig in timeframe_signals.items() 
                       if sig['is_long'] and sig['win_prob'] > 0.6 and sig['hurst'] > 0.55]
        
        short_signals = [tf for tf, sig in timeframe_signals.items() 
                        if sig['is_short']]
        
        # ENTRY: At least 2 timeframes confirm
        if len(long_signals) >= 2 and not has_open_position(symbol):
            # Use the highest timeframe signal for entry
            best_tf = long_signals[-1]  # Highest timeframe
            signal = timeframe_signals[best_tf]
            
            stop, target = self.calculate_stop_and_target(signal['price'], signal['atr'], 'long')
            
            logging.info(
                f"[MULTI-TF LONG] {symbol} | "
                f"Confirming TFs: {', '.join([TIMEFRAMES[tf]['name'] for tf in long_signals])} | "
                f"Entry=${signal['price']:.2f} | WinProb={signal['win_prob']:.1%}"
            )
            
            execute_buy(
                symbol=symbol,
                entry=signal['price'],
                stop=stop,
                target=target,
                prediction=signal['prediction'],
                confidence=signal['confidence'],
                hurst=signal['hurst'],
                sharpe=signal['sharpe'],
                win_prob=signal['win_prob'],
                timeframe=f"{', '.join([TIMEFRAMES[tf]['name'] for tf in long_signals])}"
            )
        
        # EXIT: Any timeframe gives exit signal
        elif len(short_signals) > 0 and has_open_position(symbol):
            logging.info(f"[MULTI-TF EXIT] {symbol} | Signal from {TIMEFRAMES[short_signals[0]]['name']}")
            execute_sell(symbol)
    
    def run_continuous(self):
        """CONTINUOUS analysis loop - runs every minute"""
        logging.info("="*80)
        logging.info("Multi-Timeframe Metric Space Engine LIVE")
        logging.info(f"Symbols: {', '.join(self.symbols)}")
        logging.info(f"Timeframes: {', '.join([tf['name'] for tf in TIMEFRAMES.values()])}")
        logging.info(f"Analysis interval: {ANALYSIS_INTERVAL} seconds (CONTINUOUS)")
        logging.info(f"Data collection: ENABLED -> {DATA_DIR}/")
        logging.info("="*80)
        
        while True:
            try:
                logging.info("")
                logging.info("="*80)
                logging.info(f"[SCAN] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logging.info("="*80)
                
                for symbol in self.symbols:
                    self.evaluate_symbol_multi_timeframe(symbol)
                
                logging.info("="*80)
                logging.info(f"[PAUSE] Next scan in {ANALYSIS_INTERVAL} seconds...")
                logging.info("")
                
                time.sleep(ANALYSIS_INTERVAL)
                
            except KeyboardInterrupt:
                logging.info("[STOP] Stopping Metric Space Engine...")
                break
            except Exception as e:
                logging.error(f"[ERROR] Main loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(60)

# ==========================
# ==== CLEANUP HANDLER  ====
# ==========================
def cleanup_on_exit():
    """Save all data before exiting"""
    logging.info("[CLEANUP] Saving session data...")
    data_collector.save_session_performance()
    
    # Analyze and log historical performance
    analysis = data_collector.analyze_historical_performance()
    if analysis:
        logging.info("[ANALYSIS] Historical Performance:")
        logging.info(f"  Total Trades: {analysis.get('total_trades', 0)}")
        logging.info(f"  Win Rate: {analysis.get('win_rate', 0):.2%}")
        logging.info(f"  Avg PnL: ${analysis.get('avg_pnl', 0):.2f}")
        logging.info(f"  Best Timeframes: {analysis.get('best_timeframes', {})}")
        logging.info(f"  Best Symbols: {list(analysis.get('best_symbols', {}).keys())[:5]}")

# ==========================
# ==== FLASK ROUTES     ====
# ==========================
@app.route('/status')
def status():
    try:
        positions = trading_client.list_positions()
        account = trading_client.get_account()
        
        # Get historical analysis
        analysis = data_collector.analyze_historical_performance()
        
        return jsonify({
            'status': 'running',
            'version': 'Multi-Timeframe Engine v4.1 (Data Collection)',
            'features': [
                'Multi-Timeframe Analysis (5m, 15m, 1h, 1d)',
                'Continuous 1-Minute Scanning',
                'Alpaca-Only Data',
                'Timeframe Confirmation (2+ TFs)',
                'Hurst Regime Detection',
                'Parkinson Volatility Filter',
                'Win Probability 60%+',
                'Discord Notifications',
                'Data Collection & Learning'
            ],
            'trading_enabled': is_trading,
            'balance': float(account.cash),
            'equity': float(account.equity),
            'positions': len(positions),
            'active_symbols': list(TRADE_SYMBOLS),
            'open_positions': list(OPEN_POSITIONS.keys()),
            'timeframes': [tf['name'] for tf in TIMEFRAMES.values()],
            'cache_size': len(DATA_CACHE),
            'session_stats': {
                'signals': data_collector.session_stats['total_signals'],
                'trades': data_collector.session_stats['total_trades'],
                'wins': data_collector.session_stats['winning_trades'],
                'losses': data_collector.session_stats['losing_trades'],
                'pnl': data_collector.session_stats['total_pnl']
            },
            'historical_performance': analysis
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/')
def index():
    return """
    <h1>Multi-Timeframe Metric Space Engine</h1>
    <h2>Production Features:</h2>
    <ul>
        <li>Multi-Timeframe Analysis (5m, 15m, 1h, 1d)</li>
        <li>Continuous 1-Minute Scanning</li>
        <li>Alpaca-Only Data (No Yahoo Finance)</li>
        <li>Timeframe Confirmation Strategy (2+ TFs)</li>
        <li>Smart Caching (60s)</li>
        <li>Hurst Regime Detection</li>
        <li>Parkinson Volatility Filter</li>
        <li>High Win Probability (60%+)</li>
        <li>Discord Real-Time Notifications</li>
        <li><strong>DATA COLLECTION & LEARNING</strong></li>
    </ul>
    <h2>Data Storage:</h2>
    <ul>
        <li>Signals: trading_data/signals/</li>
        <li>Trades: trading_data/trades/</li>
        <li>Performance: trading_data/performance/</li>
    </ul>
    <p>Status: <a href="/status">Check System Status</a></p>
    """

# ==========================
# ====  MAIN EXECUTION  ====
# ==========================
if __name__ == '__main__':
    WATCHLIST = [
        'NVDA', 'QQQ', 'MSTR', 'SMCI', 'CELH', 'LLY', 'AMD', 'AVGO', 
        'MU', 'ARM', 'COIN', 'MARA', 'RIOT', 'TQQQ', 'SOXL', 'TECL',
    ]
    
    engine = MultiTimeframeEngine(WATCHLIST)
    
    def run_flask():
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    logging.info("[FLASK] Server started on port 5001")
    logging.info("[WEB] Visit http://your-server:5001 for system info")
    logging.info(f"[DATA] Collecting to: {DATA_DIR.absolute()}")
    time.sleep(2)
    
    try:
        # Run continuous multi-timeframe engine
        engine.run_continuous()
    except KeyboardInterrupt:
        logging.info("[STOP] Shutting down...")
    finally:
        cleanup_on_exit()