"""
Position Tracking Module
Active positions with real-time P&L
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List


@dataclass
class Position:
    """Active trading position"""
    position_id: str
    platform: str
    strategy: str
    
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    current_price: float
    quantity: float
    
    unrealized_pnl: float = 0.0
    capital_deployed: float = 0.0
    
    status: str = 'open'  # 'open', 'closed'
    opened_at: str = None
    updated_at: str = None
    
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.opened_at is None:
            self.opened_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}
        
        self.capital_deployed = self.entry_price * self.quantity
        self.update_pnl(self.current_price)
    
    @classmethod
    def create_new(cls, platform: str, strategy: str, symbol: str,
                   side: str, entry_price: float, quantity: float,
                   metadata: Optional[Dict] = None) -> 'Position':
        """Create new position"""
        return cls(
            position_id=f"{platform}_{symbol}_{int(datetime.now().timestamp())}",
            platform=platform,
            strategy=strategy,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_price=entry_price,
            quantity=quantity,
            metadata=metadata or {}
        )
    
    def update_pnl(self, current_price: float):
        """Update unrealized P&L"""
        self.current_price = current_price
        self.updated_at = datetime.now().isoformat()
        
        if self.side.lower() in ['long', 'buy']:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # short/sell
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
    
    def get_return_pct(self) -> float:
        """Get return percentage"""
        if self.capital_deployed == 0:
            return 0.0
        return (self.unrealized_pnl / self.capital_deployed) * 100
    
    def close_position(self) -> float:
        """Close position and return final P&L"""
        self.status = 'closed'
        self.updated_at = datetime.now().isoformat()
        return self.unrealized_pnl
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Position':
        """Create from dictionary"""
        return cls(**data)
    
    def __repr__(self) -> str:
        return (f"Position[{self.status.upper()}] {self.symbol} {self.side} "
                f"{self.quantity}@${self.entry_price:.2f} | "
                f"Current: ${self.current_price:.2f} | "
                f"P&L: ${self.unrealized_pnl:+,.2f} ({self.get_return_pct():+.2f}%)")


class PositionManager:
    """Manage positions in database"""
    
    def __init__(self, database):
        self.db = database
    
    def open_position(self, platform: str, strategy: str, symbol: str,
                     side: str, entry_price: float, quantity: float,
                     metadata: Optional[Dict] = None) -> Position:
        """Open new position"""
        position = Position.create_new(
            platform=platform,
            strategy=strategy,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            metadata=metadata
        )
        
        # Save to database
        self.db.insert_position(position.to_dict())
        
        return position
    
    def update_position(self, position_id: str, current_price: float) -> Optional[Position]:
        """Update position with current price"""
        # Get position from database
        positions = self.db.get_open_positions()
        pos_data = next((p for p in positions if p['position_id'] == position_id), None)
        
        if not pos_data:
            return None
        
        # Create Position object
        position = Position.from_dict(pos_data)
        
        # Update P&L
        position.update_pnl(current_price)
        
        # Save to database
        self.db.update_position(position_id, current_price, position.unrealized_pnl)
        
        return position
    
    def close_position(self, position_id: str, exit_price: float) -> Optional[float]:
        """Close position and return final P&L"""
        # Get position
        positions = self.db.get_open_positions()
        pos_data = next((p for p in positions if p['position_id'] == position_id), None)
        
        if not pos_data:
            return None
        
        position = Position.from_dict(pos_data)
        
        # Update with exit price
        position.update_pnl(exit_price)
        final_pnl = position.close_position()
        
        # Update database
        self.db.close_position(position_id)
        
        return final_pnl
    
    def get_open_positions(self, platform: Optional[str] = None) -> List[Position]:
        """Get all open positions"""
        positions_data = self.db.get_open_positions(platform)
        return [Position.from_dict(p) for p in positions_data]
    
    def get_total_deployed_capital(self, platform: Optional[str] = None) -> float:
        """Get total capital deployed in positions"""
        positions = self.get_open_positions(platform)
        return sum(p.capital_deployed for p in positions)
    
    def get_total_unrealized_pnl(self, platform: Optional[str] = None) -> float:
        """Get total unrealized P&L"""
        positions = self.get_open_positions(platform)
        return sum(p.unrealized_pnl for p in positions)
    
    def update_all_positions(self, price_updates: Dict[str, float]):
        """
        Update multiple positions at once
        
        Args:
            price_updates: Dict of {symbol: current_price}
        """
        positions = self.get_open_positions()
        
        for position in positions:
            if position.symbol in price_updates:
                self.update_position(position.position_id, price_updates[position.symbol])
    
    def get_position_summary(self) -> Dict:
        """Get summary of all positions"""
        positions = self.get_open_positions()
        
        return {
            'total_positions': len(positions),
            'total_deployed': sum(p.capital_deployed for p in positions),
            'total_unrealized_pnl': sum(p.unrealized_pnl for p in positions),
            'positions_by_platform': self._group_by_platform(positions),
            'positions_by_strategy': self._group_by_strategy(positions)
        }
    
    def _group_by_platform(self, positions: List[Position]) -> Dict:
        """Group positions by platform"""
        grouped = {}
        for pos in positions:
            if pos.platform not in grouped:
                grouped[pos.platform] = {
                    'count': 0,
                    'deployed': 0.0,
                    'unrealized_pnl': 0.0
                }
            grouped[pos.platform]['count'] += 1
            grouped[pos.platform]['deployed'] += pos.capital_deployed
            grouped[pos.platform]['unrealized_pnl'] += pos.unrealized_pnl
        return grouped
    
    def _group_by_strategy(self, positions: List[Position]) -> Dict:
        """Group positions by strategy"""
        grouped = {}
        for pos in positions:
            if pos.strategy not in grouped:
                grouped[pos.strategy] = {
                    'count': 0,
                    'deployed': 0.0,
                    'unrealized_pnl': 0.0
                }
            grouped[pos.strategy]['count'] += 1
            grouped[pos.strategy]['deployed'] += pos.capital_deployed
            grouped[pos.strategy]['unrealized_pnl'] += pos.unrealized_pnl
        return grouped