"""
Polymarket API Client - FIXED VERSION
Handles all interactions with Polymarket CLOB API and Gamma Markets API
"""
import aiohttp
import asyncio
import json
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class Market:
    """Market data structure"""
    condition_id: str
    question: str
    outcomes: List[str]
    outcome_prices: List[float]
    token_ids: List[str]
    clob_token_ids: List[str]  # CLOB token IDs for order book
    active: bool
    end_date: Optional[datetime] = None
    volume: float = 0.0
    liquidity: float = 0.0
    slug: Optional[str] = None
    event_slug: Optional[str] = None

@dataclass
class OrderBook:
    """Order book data"""
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]
    timestamp: float

@dataclass
class Order:
    """Order structure"""
    market_id: str
    side: str  # BUY or SELL
    price: float
    size: float
    outcome: str
    order_type: str = "GTC"  # Good till cancelled


class PolymarketClient:
    """Async client for Polymarket CLOB API"""
    
    def __init__(self, api_url: str = "https://clob.polymarket.com",
                 gamma_url: str = "https://gamma-api.polymarket.com",
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 passphrase: Optional[str] = None):
        self.api_url = api_url  # CLOB API for trading
        self.gamma_url = gamma_url  # Gamma API for market data
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, timestamp: str, method: str, 
                           path: str, body: str = "") -> str:
        """Generate HMAC signature for authenticated requests"""
        if not self.api_secret:
            return ""
        
        message = f"{timestamp}{method}{path}{body}"
        
        import base64
        secret_bytes = base64.b64decode(self.api_secret)
        
        signature = hmac.new(
            secret_bytes,
            message.encode(),
            hashlib.sha256
        ).digest()
        
        signature_b64 = base64.b64encode(signature).decode()
        return signature_b64
    
    async def _clob_request(self, method: str, endpoint: str, 
                           data: Optional[Dict] = None,
                           authenticated: bool = False) -> Any:
        """Make HTTP request to CLOB API (for trading)"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async with.")
        
        url = f"{self.api_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        if authenticated and self.api_key:
            timestamp = str(int(time.time() * 1000))
            body = json.dumps(data) if data else ""
            signature = self._generate_signature(timestamp, method, endpoint, body)
            
            headers.update({
                "POLY-ADDRESS": self.api_key,
                "POLY-SIGNATURE": signature,
                "POLY-TIMESTAMP": timestamp,
                "POLY-PASSPHRASE": self.passphrase or ""
            })
        
        try:
            async with self.session.request(
                method, url, json=data, headers=headers
            ) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"CLOB API request failed: {e}")
            return None
    
    async def _gamma_request(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Make HTTP request to Gamma API (for market data)"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async with.")
        
        url = f"{self.gamma_url}{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Gamma API request failed: {e}")
            return None
    
    async def get_markets(self, active_only: bool = True, limit: int = 100) -> List[Market]:
        """Fetch all available markets from Gamma API"""
        endpoint = "/markets"
        params = {"limit": limit}
        
        if active_only:
            params["active"] = "true"
        
        data = await self._gamma_request(endpoint, params)
        
        if not data:
            logger.warning("No data returned from Gamma markets API")
            return []
        
        markets = []
        
        for market_data in data:
            if not isinstance(market_data, dict):
                continue
            
            try:
                # Extract CLOB token IDs
                clob_token_ids = market_data.get("clob_token_ids", [])
                if isinstance(clob_token_ids, str):
                    clob_token_ids = [clob_token_ids]
                
                outcomes = market_data.get("outcomes", ["Yes", "No"])
                outcome_prices = market_data.get("outcome_prices", [0.5] * len(outcomes))
                
                # Parse end date
                end_date_str = market_data.get("end_date_iso")
                end_date = None
                if end_date_str:
                    try:
                        end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                    except:
                        pass
                
                market = Market(
                    condition_id=market_data.get("condition_id", ""),
                    question=market_data.get("question", "Unknown"),
                    outcomes=outcomes,
                    outcome_prices=[float(p) for p in outcome_prices[:len(outcomes)]],
                    token_ids=market_data.get("tokens", []),
                    clob_token_ids=clob_token_ids,
                    active=market_data.get("active", True),
                    end_date=end_date,
                    volume=float(market_data.get("volume", 0)),
                    liquidity=float(market_data.get("liquidity", 0)),
                    slug=market_data.get("market_slug"),
                    event_slug=market_data.get("event_slug")
                )
                markets.append(market)
                
            except Exception as e:
                logger.debug(f"Error parsing market: {e}")
                continue
        
        logger.info(f"Fetched {len(markets)} markets from Gamma API")
        return markets
    
    async def get_order_book(self, token_id: str) -> Optional[OrderBook]:
        """Fetch order book for a specific CLOB token"""
        endpoint = f"/book?token_id={token_id}"
        
        data = await self._clob_request("GET", endpoint)
        
        if not data:
            return None
        
        try:
            bids = [
                (float(bid["price"]), float(bid["size"]))
                for bid in data.get("bids", [])
            ]
            asks = [
                (float(ask["price"]), float(ask["size"]))
                for ask in data.get("asks", [])
            ]
            
            return OrderBook(
                bids=bids,
                asks=asks,
                timestamp=time.time()
            )
        except Exception as e:
            logger.error(f"Error parsing order book: {e}")
            return None
    
    async def place_order(self, order: Order, dry_run: bool = False) -> Optional[str]:
        """Place an order (requires authentication)"""
        if dry_run:
            logger.info(f"[DRY RUN] Would place order: {order.side} {order.size} @ ${order.price}")
            return f"dry_run_{int(time.time())}"
        
        endpoint = "/order"
        
        order_data = {
            "market": order.market_id,
            "side": order.side,
            "price": str(order.price),
            "size": str(order.size),
            "outcome": order.outcome,
            "type": order.order_type
        }
        
        response = await self._clob_request(
            "POST", endpoint, data=order_data, authenticated=True
        )
        
        if response and "order_id" in response:
            return response["order_id"]
        return None
    
    async def get_open_orders(self) -> List[Dict]:
        """Get all open orders for authenticated user"""
        endpoint = "/orders"
        response = await self._clob_request("GET", endpoint, authenticated=True)
        return response if response else []
    
    async def verify_credentials(self) -> Dict:
        """Verify Polymarket credentials"""
        res = {"authenticated": False, "open_orders": None}

        if not self.api_key:
            logger.warning("Polymarket API key not configured")
            return res

        orders = await self.get_open_orders()
        if orders is None:
            logger.error("Polymarket authentication failed")
            return res

        res['authenticated'] = True
        res['open_orders'] = orders
        logger.info("Polymarket authentication successful")
        
        return res


class GammaMarketsAPI:
    """Client for Gamma Markets API (market data only)"""
    
    def __init__(self, base_url: str = "https://gamma-api.polymarket.com"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_all_markets(self, limit: int = 100) -> List[Dict]:
        """Get list of all markets"""
        if not self.session:
            return []
        
        url = f"{self.base_url}/markets?limit={limit}"
        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                return data if data else []
        except aiohttp.ClientError as e:
            logger.error(f"Gamma API request failed: {e}")
            return []