"""
SQLite Database Handler
Single source of truth for all trading data
"""
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ========================================
# DATACLASSES
# ========================================

@dataclass
class TradeRecord:
    """Record of a completed trade"""
    trade_id: str
    timestamp: str
    platform: str
    strategy: str
    symbol: str
    side: str
    entry_price: float
    quantity: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[str] = None
    pnl: Optional[float] = None
    status: str = 'open'
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'trade_id': self.trade_id,
            'timestamp': self.timestamp,
            'platform': self.platform,
            'strategy': self.strategy,
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'exit_price': self.exit_price,
            'exit_timestamp': self.exit_timestamp,
            'pnl': self.pnl,
            'status': self.status,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeRecord':
        """Create from dictionary"""
        return cls(
            trade_id=data['trade_id'],
            timestamp=data['timestamp'],
            platform=data['platform'],
            strategy=data['strategy'],
            symbol=data['symbol'],
            side=data['side'],
            entry_price=data['entry_price'],
            quantity=data['quantity'],
            exit_price=data.get('exit_price'),
            exit_timestamp=data.get('exit_timestamp'),
            pnl=data.get('pnl'),
            status=data.get('status', 'open'),
            metadata=data.get('metadata')
        )


@dataclass
class Position:
    """Record of an open position"""
    position_id: str
    platform: str
    strategy: str
    symbol: str
    side: str
    entry_price: float
    quantity: float
    capital_deployed: float
    opened_at: str
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    status: str = 'open'
    updated_at: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'position_id': self.position_id,
            'platform': self.platform,
            'strategy': self.strategy,
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'quantity': self.quantity,
            'capital_deployed': self.capital_deployed,
            'unrealized_pnl': self.unrealized_pnl,
            'status': self.status,
            'opened_at': self.opened_at,
            'updated_at': self.updated_at,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Position':
        """Create from dictionary"""
        return cls(
            position_id=data['position_id'],
            platform=data['platform'],
            strategy=data['strategy'],
            symbol=data['symbol'],
            side=data['side'],
            entry_price=data['entry_price'],
            quantity=data['quantity'],
            capital_deployed=data['capital_deployed'],
            opened_at=data['opened_at'],
            current_price=data.get('current_price'),
            unrealized_pnl=data.get('unrealized_pnl', 0.0),
            status=data.get('status', 'open'),
            updated_at=data.get('updated_at'),
            metadata=data.get('metadata')
        )


# ========================================
# DATABASE CLASS
# ========================================

class Database:
    """Central SQLite database for all trading data"""
    
    def __init__(self, db_path: str = "ERC/data/erc.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None
        self._initialize_schema()
    
    def connect(self):
        """Connect to database"""
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def _initialize_schema(self):
        """Create all tables"""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                platform TEXT NOT NULL,
                strategy TEXT NOT NULL,
                
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                
                exit_price REAL,
                exit_timestamp TEXT,
                pnl REAL,
                
                status TEXT DEFAULT 'open',
                metadata TEXT,
                
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                position_id TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                strategy TEXT NOT NULL,
                
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL,
                quantity REAL NOT NULL,
                
                unrealized_pnl REAL DEFAULT 0,
                capital_deployed REAL NOT NULL,
                
                status TEXT DEFAULT 'open',
                opened_at TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                
                metadata TEXT
            )
        """)
        
        # Portfolio snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                
                total_capital REAL NOT NULL,
                deployed_capital REAL NOT NULL,
                available_capital REAL NOT NULL,
                
                total_pnl REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                
                num_positions INTEGER NOT NULL,
                num_trades INTEGER NOT NULL,
                
                platform_balances TEXT,
                
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Strategy performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_performance (
                strategy TEXT PRIMARY KEY,
                
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                
                total_pnl REAL DEFAULT 0,
                avg_win REAL DEFAULT 0,
                avg_loss REAL DEFAULT 0,
                
                win_rate REAL DEFAULT 0,
                profit_factor REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Risk metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,

                var_95 REAL,
                max_drawdown REAL,
                current_drawdown REAL,

                portfolio_beta REAL,
                portfolio_volatility REAL,

                correlation_matrix TEXT,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Market outcomes table (for historical calibration)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_outcomes (
                outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                market_type TEXT,

                entry_price REAL NOT NULL,
                resolution_price REAL,

                won BOOLEAN,
                pnl REAL,

                entry_timestamp TEXT NOT NULL,
                resolution_timestamp TEXT,

                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index for calibration lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_outcomes_price_bucket
            ON market_outcomes (platform, market_type, entry_price)
        """)

        # Model states table (for ML model persistence)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_states (
                model_id TEXT PRIMARY KEY,
                model_type TEXT NOT NULL,

                parameters TEXT,
                weights TEXT,
                feature_names TEXT,

                training_samples INTEGER,
                last_trained TEXT,
                performance_metrics TEXT,

                version INTEGER DEFAULT 1,
                is_active BOOLEAN DEFAULT 1,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # API cache table (optional persistent cache)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_cache (
                cache_key TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,

                value TEXT NOT NULL,
                expires_at TEXT NOT NULL,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index for cache expiration cleanup
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_expiry
            ON api_cache (expires_at)
        """)

        # Sentiment signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_signals (
                signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT,

                source TEXT NOT NULL,
                text TEXT,
                sentiment_score REAL,
                confidence REAL,
                engagement INTEGER,

                keywords TEXT,
                timestamp TEXT NOT NULL,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # A/B test results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_test_results (
                test_id TEXT PRIMARY KEY,
                variant_a TEXT NOT NULL,
                variant_b TEXT NOT NULL,

                metric_name TEXT NOT NULL,

                samples_a INTEGER,
                samples_b INTEGER,
                mean_a REAL,
                mean_b REAL,

                p_value REAL,
                effect_size REAL,
                winner TEXT,
                is_significant BOOLEAN,

                started_at TEXT,
                completed_at TEXT,

                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        logger.info(f"Database initialized at {self.db_path}")
    
    # ========================================
    # TRADE OPERATIONS
    # ========================================
    
    def insert_trade(self, trade_data: Dict) -> bool:
        """Insert new trade"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades (
                    trade_id, timestamp, platform, strategy,
                    symbol, side, entry_price, quantity, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data['trade_id'],
                trade_data['timestamp'],
                trade_data['platform'],
                trade_data['strategy'],
                trade_data['symbol'],
                trade_data['side'],
                trade_data['entry_price'],
                trade_data['quantity'],
                json.dumps(trade_data.get('metadata', {}))
            ))
            
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to insert trade: {e}")
            return False
    
    def update_trade_exit(self, trade_id: str, exit_price: float, pnl: float) -> bool:
        """Update trade with exit information"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE trades
                SET exit_price = ?, exit_timestamp = ?, pnl = ?, status = 'closed'
                WHERE trade_id = ?
            """, (exit_price, datetime.now().isoformat(), pnl, trade_id))
            
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update trade exit: {e}")
            return False
    
    def get_open_trades(self, platform: Optional[str] = None) -> List[Dict]:
        """Get all open trades"""
        conn = self.connect()
        cursor = conn.cursor()
        
        if platform:
            cursor.execute("""
                SELECT * FROM trades 
                WHERE status = 'open' AND platform = ?
                ORDER BY timestamp DESC
            """, (platform,))
        else:
            cursor.execute("""
                SELECT * FROM trades 
                WHERE status = 'open'
                ORDER BY timestamp DESC
            """)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_trade_history(self, limit: int = 100, strategy: Optional[str] = None) -> List[Dict]:
        """Get trade history"""
        conn = self.connect()
        cursor = conn.cursor()
        
        if strategy:
            cursor.execute("""
                SELECT * FROM trades
                WHERE strategy = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (strategy, limit))
        else:
            cursor.execute("""
                SELECT * FROM trades
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    # ========================================
    # POSITION OPERATIONS
    # ========================================
    
    def insert_position(self, position_data: Dict) -> bool:
        """Insert new position"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO positions (
                    position_id, platform, strategy, symbol, side,
                    entry_price, current_price, quantity, capital_deployed,
                    opened_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position_data['position_id'],
                position_data['platform'],
                position_data['strategy'],
                position_data['symbol'],
                position_data['side'],
                position_data['entry_price'],
                position_data['entry_price'],  # Current price starts at entry
                position_data['quantity'],
                position_data['capital_deployed'],
                position_data['opened_at'],
                json.dumps(position_data.get('metadata', {}))
            ))
            
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to insert position: {e}")
            return False
    
    def update_position(self, position_id: str, current_price: float, unrealized_pnl: float) -> bool:
        """Update position with current market data"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE positions
                SET current_price = ?, unrealized_pnl = ?, updated_at = ?
                WHERE position_id = ?
            """, (current_price, unrealized_pnl, datetime.now().isoformat(), position_id))
            
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update position: {e}")
            return False
    
    def close_position(self, position_id: str) -> bool:
        """Close a position"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE positions
                SET status = 'closed', updated_at = ?
                WHERE position_id = ?
            """, (datetime.now().isoformat(), position_id))
            
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False
    
    def get_open_positions(self, platform: Optional[str] = None) -> List[Dict]:
        """Get all open positions"""
        conn = self.connect()
        cursor = conn.cursor()
        
        if platform:
            cursor.execute("""
                SELECT * FROM positions
                WHERE status = 'open' AND platform = ?
                ORDER BY opened_at DESC
            """, (platform,))
        else:
            cursor.execute("""
                SELECT * FROM positions
                WHERE status = 'open'
                ORDER BY opened_at DESC
            """)
        
        return [dict(row) for row in cursor.fetchall()]
    
    # ========================================
    # PORTFOLIO SNAPSHOTS
    # ========================================
    
    def save_portfolio_snapshot(self, snapshot_data: Dict) -> bool:
        """Save portfolio snapshot"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO portfolio_snapshots (
                    timestamp, total_capital, deployed_capital, available_capital,
                    total_pnl, daily_pnl, num_positions, num_trades, platform_balances
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot_data['timestamp'],
                snapshot_data['total_capital'],
                snapshot_data['deployed_capital'],
                snapshot_data['available_capital'],
                snapshot_data['total_pnl'],
                snapshot_data['daily_pnl'],
                snapshot_data['num_positions'],
                snapshot_data['num_trades'],
                json.dumps(snapshot_data.get('platform_balances', {}))
            ))
            
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save portfolio snapshot: {e}")
            return False
    
    def get_latest_snapshot(self) -> Optional[Dict]:
        """Get most recent portfolio snapshot"""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM portfolio_snapshots
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        return dict(row) if row else None
    
    # ========================================
    # STRATEGY PERFORMANCE
    # ========================================
    
    def update_strategy_performance(self, strategy: str, performance_data: Dict) -> bool:
        """Update strategy performance metrics"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO strategy_performance (
                    strategy, total_trades, winning_trades, losing_trades,
                    total_pnl, avg_win, avg_loss, win_rate, profit_factor, sharpe_ratio,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy,
                performance_data['total_trades'],
                performance_data['winning_trades'],
                performance_data['losing_trades'],
                performance_data['total_pnl'],
                performance_data['avg_win'],
                performance_data['avg_loss'],
                performance_data['win_rate'],
                performance_data['profit_factor'],
                performance_data['sharpe_ratio'],
                datetime.now().isoformat()
            ))
            
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update strategy performance: {e}")
            return False
    
    def get_strategy_performance(self, strategy: str) -> Optional[Dict]:
        """Get strategy performance"""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM strategy_performance
            WHERE strategy = ?
        """, (strategy,))
        
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_all_strategy_performance(self) -> List[Dict]:
        """Get all strategy performance"""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM strategy_performance")
        return [dict(row) for row in cursor.fetchall()]

    # ========================================
    # MARKET OUTCOMES (Calibration)
    # ========================================

    def record_market_outcome(self, outcome_data: Dict) -> bool:
        """Record a market outcome for calibration"""
        try:
            conn = self.connect()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO market_outcomes (
                    market_id, platform, market_type, entry_price,
                    resolution_price, won, pnl, entry_timestamp,
                    resolution_timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                outcome_data['market_id'],
                outcome_data['platform'],
                outcome_data.get('market_type'),
                outcome_data['entry_price'],
                outcome_data.get('resolution_price'),
                outcome_data.get('won'),
                outcome_data.get('pnl'),
                outcome_data['entry_timestamp'],
                outcome_data.get('resolution_timestamp'),
                json.dumps(outcome_data.get('metadata', {}))
            ))

            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to record market outcome: {e}")
            return False

    def get_calibration_data(self, platform: str, market_type: Optional[str] = None,
                             price_min: float = 0.0, price_max: float = 1.0) -> List[Dict]:
        """Get calibration data for a price bucket"""
        conn = self.connect()
        cursor = conn.cursor()

        if market_type:
            cursor.execute("""
                SELECT * FROM market_outcomes
                WHERE platform = ? AND market_type = ?
                AND entry_price >= ? AND entry_price < ?
                AND won IS NOT NULL
            """, (platform, market_type, price_min, price_max))
        else:
            cursor.execute("""
                SELECT * FROM market_outcomes
                WHERE platform = ?
                AND entry_price >= ? AND entry_price < ?
                AND won IS NOT NULL
            """, (platform, price_min, price_max))

        return [dict(row) for row in cursor.fetchall()]

    def get_win_rate_by_bucket(self, platform: str, bucket_size: float = 0.05) -> Dict[str, Dict]:
        """Get win rate statistics by price bucket"""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                CAST(entry_price / ? AS INTEGER) * ? AS bucket_start,
                COUNT(*) as total,
                SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) as wins
            FROM market_outcomes
            WHERE platform = ? AND won IS NOT NULL
            GROUP BY bucket_start
            ORDER BY bucket_start
        """, (bucket_size, bucket_size, platform))

        results = {}
        for row in cursor.fetchall():
            bucket = f"{row['bucket_start']:.2f}-{row['bucket_start'] + bucket_size:.2f}"
            results[bucket] = {
                'total': row['total'],
                'wins': row['wins'],
                'win_rate': row['wins'] / row['total'] if row['total'] > 0 else 0
            }

        return results

    # ========================================
    # MODEL STATES (ML Persistence)
    # ========================================

    def save_model_state(self, model_data: Dict) -> bool:
        """Save or update ML model state"""
        try:
            conn = self.connect()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO model_states (
                    model_id, model_type, parameters, weights, feature_names,
                    training_samples, last_trained, performance_metrics,
                    version, is_active, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_data['model_id'],
                model_data['model_type'],
                json.dumps(model_data.get('parameters', {})),
                json.dumps(model_data.get('weights', [])),
                json.dumps(model_data.get('feature_names', [])),
                model_data.get('training_samples', 0),
                model_data.get('last_trained', datetime.now().isoformat()),
                json.dumps(model_data.get('performance_metrics', {})),
                model_data.get('version', 1),
                model_data.get('is_active', True),
                datetime.now().isoformat()
            ))

            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save model state: {e}")
            return False

    def get_model_state(self, model_id: str) -> Optional[Dict]:
        """Get model state by ID"""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM model_states WHERE model_id = ?
        """, (model_id,))

        row = cursor.fetchone()
        if row:
            result = dict(row)
            result['parameters'] = json.loads(result['parameters']) if result['parameters'] else {}
            result['weights'] = json.loads(result['weights']) if result['weights'] else []
            result['feature_names'] = json.loads(result['feature_names']) if result['feature_names'] else []
            result['performance_metrics'] = json.loads(result['performance_metrics']) if result['performance_metrics'] else {}
            return result
        return None

    def get_active_models(self, model_type: Optional[str] = None) -> List[Dict]:
        """Get all active models"""
        conn = self.connect()
        cursor = conn.cursor()

        if model_type:
            cursor.execute("""
                SELECT * FROM model_states
                WHERE is_active = 1 AND model_type = ?
            """, (model_type,))
        else:
            cursor.execute("""
                SELECT * FROM model_states WHERE is_active = 1
            """)

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            result['parameters'] = json.loads(result['parameters']) if result['parameters'] else {}
            result['performance_metrics'] = json.loads(result['performance_metrics']) if result['performance_metrics'] else {}
            results.append(result)

        return results

    # ========================================
    # API CACHE
    # ========================================

    def cache_set(self, key: str, value: Any, namespace: str, ttl_seconds: int) -> bool:
        """Set cache value"""
        try:
            conn = self.connect()
            cursor = conn.cursor()

            expires_at = (datetime.now() + timedelta(seconds=ttl_seconds)).isoformat()

            cursor.execute("""
                INSERT OR REPLACE INTO api_cache (cache_key, namespace, value, expires_at)
                VALUES (?, ?, ?, ?)
            """, (key, namespace, json.dumps(value), expires_at))

            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to set cache: {e}")
            return False

    def cache_get(self, key: str) -> Optional[Any]:
        """Get cache value (returns None if expired)"""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT value, expires_at FROM api_cache WHERE cache_key = ?
        """, (key,))

        row = cursor.fetchone()
        if row:
            expires_at = datetime.fromisoformat(row['expires_at'])
            if datetime.now() < expires_at:
                return json.loads(row['value'])
            else:
                # Clean up expired entry
                cursor.execute("DELETE FROM api_cache WHERE cache_key = ?", (key,))
                conn.commit()

        return None

    def cache_cleanup(self) -> int:
        """Remove expired cache entries"""
        try:
            conn = self.connect()
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM api_cache WHERE expires_at < ?
            """, (datetime.now().isoformat(),))

            deleted = cursor.rowcount
            conn.commit()
            return deleted
        except Exception as e:
            logger.error(f"Failed to cleanup cache: {e}")
            return 0

    # ========================================
    # SENTIMENT SIGNALS
    # ========================================

    def save_sentiment_signal(self, signal_data: Dict) -> bool:
        """Save sentiment signal"""
        try:
            conn = self.connect()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO sentiment_signals (
                    market_id, source, text, sentiment_score,
                    confidence, engagement, keywords, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_data.get('market_id'),
                signal_data['source'],
                signal_data.get('text'),
                signal_data['sentiment_score'],
                signal_data.get('confidence', 0.5),
                signal_data.get('engagement', 0),
                json.dumps(signal_data.get('keywords', [])),
                signal_data.get('timestamp', datetime.now().isoformat())
            ))

            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save sentiment signal: {e}")
            return False

    def get_recent_sentiment(self, market_id: str, hours: int = 24) -> List[Dict]:
        """Get recent sentiment signals for a market"""
        conn = self.connect()
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        cursor.execute("""
            SELECT * FROM sentiment_signals
            WHERE market_id = ? AND timestamp > ?
            ORDER BY timestamp DESC
        """, (market_id, cutoff))

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            result['keywords'] = json.loads(result['keywords']) if result['keywords'] else []
            results.append(result)

        return results

    # ========================================
    # A/B TEST RESULTS
    # ========================================

    def save_ab_test_result(self, test_data: Dict) -> bool:
        """Save A/B test result"""
        try:
            conn = self.connect()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO ab_test_results (
                    test_id, variant_a, variant_b, metric_name,
                    samples_a, samples_b, mean_a, mean_b,
                    p_value, effect_size, winner, is_significant,
                    started_at, completed_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_data['test_id'],
                test_data['variant_a'],
                test_data['variant_b'],
                test_data['metric_name'],
                test_data.get('samples_a', 0),
                test_data.get('samples_b', 0),
                test_data.get('mean_a'),
                test_data.get('mean_b'),
                test_data.get('p_value'),
                test_data.get('effect_size'),
                test_data.get('winner'),
                test_data.get('is_significant', False),
                test_data.get('started_at'),
                test_data.get('completed_at'),
                json.dumps(test_data.get('metadata', {}))
            ))

            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save A/B test result: {e}")
            return False

    def get_ab_test_result(self, test_id: str) -> Optional[Dict]:
        """Get A/B test result"""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM ab_test_results WHERE test_id = ?
        """, (test_id,))

        row = cursor.fetchone()
        if row:
            result = dict(row)
            result['metadata'] = json.loads(result['metadata']) if result['metadata'] else {}
            return result
        return None