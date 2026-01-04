"""
Trade Records Module
Handles all trade-related operations
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any
import uuid


@dataclass
class TradeRecord:
    """Complete trade record from entry to exit"""
    trade_id: str
    timestamp: str
    platform: str
    strategy: str
    
    # Entry
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    quantity: float
    
    # Exit (filled when closed)
    exit_price: Optional[float] = None
    exit_timestamp: Optional[str] = None
    pnl: Optional[float] = None
    
    # Status
    status: str = 'open'  # 'open', 'closed', 'cancelled'
    
    # Metadata
    metadata: Dict[str, Any] = None
    
    @classmethod
    def create_new(cls, platform: str, strategy: str, symbol: str, 
                   side: str, entry_price: float, quantity: float,
                   metadata: Optional[Dict] = None) -> 'TradeRecord':
        """Create new trade record"""
        return cls(
            trade_id=f"{platform}_{symbol}_{int(datetime.now().timestamp())}",
            timestamp=datetime.now().isoformat(),
            platform=platform,
            strategy=strategy,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            metadata=metadata or {}
        )
    
    def close_trade(self, exit_price: float) -> float:
        """Close trade and calculate P&L"""
        self.exit_price = exit_price
        self.exit_timestamp = datetime.now().isoformat()
        self.status = 'closed'
        
        # Calculate P&L
        if self.side.lower() == 'buy':
            self.pnl = (exit_price - self.entry_price) * self.quantity
        else:  # sell/short
            self.pnl = (self.entry_price - exit_price) * self.quantity
        
        return self.pnl
    
    def get_capital_deployed(self) -> float:
        """Get amount of capital deployed"""
        return self.entry_price * self.quantity
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        if self.status != 'open':
            return self.pnl or 0.0
        
        if self.side.lower() == 'buy':
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeRecord':
        """Create from dictionary"""
        return cls(**data)
    
    def __repr__(self) -> str:
        status_str = f"[{self.status.upper()}]"
        pnl_str = f"P&L: ${self.pnl:,.2f}" if self.pnl is not None else "OPEN"
        return f"Trade {status_str} {self.symbol} {self.side} {self.quantity}@${self.entry_price:.2f} - {pnl_str}"


class TradeManager:
    """Manage trades in database"""
    
    def __init__(self, database):
        self.db = database
    
    def create_trade(self, platform: str, strategy: str, symbol: str,
                    side: str, entry_price: float, quantity: float,
                    metadata: Optional[Dict] = None) -> TradeRecord:
        """Create and save new trade"""
        trade = TradeRecord.create_new(
            platform=platform,
            strategy=strategy,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            metadata=metadata
        )
        
        # Save to database
        self.db.insert_trade(trade.to_dict())
        
        return trade
    
    def close_trade(self, trade_id: str, exit_price: float) -> Optional[float]:
        """Close trade and update database"""
        # Get trade from database
        trades = self.db.get_open_trades()
        trade_data = next((t for t in trades if t['trade_id'] == trade_id), None)
        
        if not trade_data:
            return None
        
        # Create TradeRecord object
        trade = TradeRecord.from_dict(trade_data)
        
        # Close trade
        pnl = trade.close_trade(exit_price)
        
        # Update database
        self.db.update_trade_exit(trade_id, exit_price, pnl)
        
        return pnl
    
    def get_open_trades(self, platform: Optional[str] = None) -> list[TradeRecord]:
        """Get all open trades"""
        trades_data = self.db.get_open_trades(platform)
        return [TradeRecord.from_dict(t) for t in trades_data]
    
    def get_trade_history(self, limit: int = 100, strategy: Optional[str] = None) -> list[TradeRecord]:
        """Get trade history"""
        trades_data = self.db.get_trade_history(limit, strategy)
        return [TradeRecord.from_dict(t) for t in trades_data]
    
    def get_trade_stats(self, strategy: Optional[str] = None) -> Dict:
        """Get trade statistics"""
        trades = self.get_trade_history(limit=1000, strategy=strategy)
        
        closed_trades = [t for t in trades if t.status == 'closed' and t.pnl is not None]
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        
        return {
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(closed_trades) if closed_trades else 0.0,
            'total_pnl': sum(t.pnl for t in closed_trades),
            'avg_win': total_wins / len(winning_trades) if winning_trades else 0.0,
            'avg_loss': total_losses / len(losing_trades) if losing_trades else 0.0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else 0.0
        }