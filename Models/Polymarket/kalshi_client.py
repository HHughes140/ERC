"""
Kalshi API Client - V2 API with RSA Header Auth
"""
import aiohttp
import asyncio
import json
import time
import base64
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


@dataclass
class KalshiMarket:
    """Kalshi market data structure"""
    ticker: str
    title: str
    question: str
    yes_price: float
    no_price: float
    volume: float
    open_interest: float
    close_time: Optional[datetime] = None
    status: str = "active"
    category: str = ""
    
    def to_polymarket_format(self):
        """Convert to Polymarket Market format for compatibility"""
        from .polymarket_client import Market
        return Market(
            condition_id=self.ticker,
            question=self.title,
            outcomes=["Yes", "No"],
            outcome_prices=[self.yes_price, self.no_price],
            token_ids=[f"{self.ticker}_yes", f"{self.ticker}_no"],
            clob_token_ids=[],
            active=self.status == "active",
            volume=self.volume,
            liquidity=self.open_interest
        )


class KalshiClient:
    """Async client for Kalshi API V2 - Header-based auth"""
    
    def __init__(self, api_url: str = "https://api.elections.kalshi.com",
                 api_key: Optional[str] = None,
                 private_key_str: Optional[str] = None):
        self.api_url = api_url
        self.api_key = api_key
        self.private_key_str = private_key_str
        self.session: Optional[aiohttp.ClientSession] = None
        self.token: Optional[str] = "header_auth"
        self.private_key = None
        
        # Load private key
        if private_key_str:
            try:
                clean_key = private_key_str.strip()

                # Remove surrounding quotes if present
                if clean_key.startswith('"') and clean_key.endswith('"'):
                    clean_key = clean_key[1:-1]
                if clean_key.startswith("'") and clean_key.endswith("'"):
                    clean_key = clean_key[1:-1]

                # Handle escaped newlines from .env file (literal backslash + n)
                # Use explicit character codes: chr(92) = backslash, chr(10) = newline
                literal_backslash_n = chr(92) + 'n'  # \n as two characters
                actual_newline = chr(10)  # Real newline

                if literal_backslash_n in clean_key:
                    clean_key = clean_key.replace(literal_backslash_n, actual_newline)
                    logger.debug(f"Converted {clean_key.count(actual_newline)} escaped newlines in private key")

                # Validate PEM format
                if not clean_key.startswith('-----BEGIN'):
                    logger.error("Private key does not start with PEM header")
                    logger.debug(f"Key starts with: {clean_key[:50]}...")
                else:
                    # Count expected newlines in PEM (header, ~25 lines of base64, footer)
                    newline_count = clean_key.count('\n')
                    if newline_count < 5:
                        logger.warning(f"Private key has only {newline_count} newlines - may be malformed")
                    else:
                        logger.debug(f"Private key has {newline_count} newlines")

                self.private_key = serialization.load_pem_private_key(
                    clean_key.encode(),
                    password=None,
                    backend=default_backend()
                )
                logger.info("Kalshi private key loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load private key: {e}")
                # Log diagnostic info
                key_preview = private_key_str[:80] if private_key_str else "None"
                logger.debug(f"Key preview: {key_preview}...")
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _sign_message(self, timestamp: str, method: str, path: str) -> str:
        """Sign message with RSA-PSS"""
        if not self.private_key:
            return ""
        
        try:
            message = f"{timestamp}{method}{path}"
            
            signature = self.private_key.sign(
                message.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH
                ),
                hashes.SHA256()
            )
            
            return base64.b64encode(signature).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to sign: {e}")
            return ""
    
    def _get_auth_headers(self, method: str, path: str) -> Dict:
        """Generate auth headers"""
        if not self.api_key or not self.private_key:
            return {}
        
        timestamp = str(int(time.time() * 1000))
        signature = self._sign_message(timestamp, method, path)
        
        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    async def _request(self, method: str, endpoint: str, 
                       data: Optional[Dict] = None, retries: int = 2) -> Any:
        """Make HTTP request with header auth"""
        if not self.session:
            raise RuntimeError("Client not initialized")
        
        url = f"{self.api_url}{endpoint}"
        attempt = 0
        backoff = 0.5
        
        while attempt <= retries:
            headers = self._get_auth_headers(method, endpoint)
            
            if not headers:
                logger.error("Cannot generate auth headers")
                return None
            
            try:
                await asyncio.sleep(0.1)  # Rate limiting
                
                async with self.session.request(method, url, json=data, headers=headers) as response:
                    text = await response.text()
                    status = response.status
                    
                    if status == 200 or status == 201:
                        try:
                            return json.loads(text)
                        except:
                            return text
                    
                    if status in (429, 500, 502, 503, 504):
                        logger.warning(f"Kalshi {status}")
                        if attempt < retries:
                            await asyncio.sleep(backoff)
                            backoff *= 2
                            attempt += 1
                            continue
                    
                    logger.error(f"Kalshi {status} for {endpoint}: {text}")
                    return None
                    
            except Exception as e:
                logger.error(f"Kalshi request error: {e}")
                if attempt < retries:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    attempt += 1
                    continue
                return None
            
            finally:
                attempt += 1
        
        return None
    
    async def get_markets(self, status: str = "open", limit: int = 200) -> List[KalshiMarket]:
        """Fetch markets"""
        all_markets = []
        cursor = None
        page_count = 0
        
        while True:
            if page_count > 0:
                await asyncio.sleep(0.5)
            
            endpoint = f"/trade-api/v2/markets?status={status}&limit={limit}"
            if cursor:
                endpoint += f"&cursor={cursor}"
            
            data = await self._request("GET", endpoint)
            
            if not data or "markets" not in data:
                break
            
            for market_data in data["markets"]:
                try:
                    yes_price = market_data.get("yes_bid", 0.5)
                    no_price = market_data.get("no_bid", 0.5)
                    
                    if "last_price" in market_data:
                        yes_price = market_data["last_price"] / 100.0
                        no_price = 1.0 - yes_price
                    
                    if "yes_ask" in market_data and "no_ask" in market_data:
                        yes_price = market_data["yes_ask"] / 100.0
                        no_price = market_data["no_ask"] / 100.0
                    
                    market = KalshiMarket(
                        ticker=market_data.get("ticker", ""),
                        title=market_data.get("title", ""),
                        question=market_data.get("subtitle", market_data.get("title", "")),
                        yes_price=yes_price,
                        no_price=no_price,
                        volume=float(market_data.get("volume", 0)),
                        open_interest=float(market_data.get("open_interest", 0)),
                        status=market_data.get("status", "active"),
                        category=market_data.get("category", "")
                    )
                    all_markets.append(market)
                    
                except Exception as e:
                    logger.debug(f"Error parsing market: {e}")
                    continue
            
            page_count += 1
            cursor = data.get("cursor")
            
            if not cursor or page_count >= 15:
                break
        
        logger.info(f"Fetched {len(all_markets)} Kalshi markets in {page_count} pages")
        return all_markets
    
    async def place_order(self, ticker: str, side: str, price: float, 
                         quantity: int, dry_run: bool = False) -> Optional[str]:
        """Place order on Kalshi V2"""
        client_order_id = f"order_{int(time.time() * 1000)}"
        
        if dry_run:
            logger.info(f"[DRY RUN] Kalshi: {ticker} {side} @ {price} x {quantity}")
            return f"dryrun_{client_order_id}"
        
        if not self.api_key or not self.private_key:
            logger.error("Cannot place order: no credentials")
            return None
        
        endpoint = "/trade-api/v2/portfolio/orders"
        price_cents = int(price * 100)
        
        order_data = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": side.lower(),
            "action": "buy",
            "count": quantity,
            "type": "limit",
            "yes_price": price_cents if side.lower() == "yes" else None,
            "no_price": price_cents if side.lower() == "no" else None
        }
        
        logger.info(f"[KALSHI] Placing order: {ticker} {side} @ ${price} x{quantity}")
        
        for attempt in range(3):
            result = await self._request("POST", endpoint, data=order_data)
            
            if result and "order" in result:
                order_id = result["order"].get("order_id")
                logger.info(f"[KALSHI] Order placed: {order_id}")
                return order_id
            
            if attempt < 2:
                logger.info(f"Retry {attempt+1}/3...")
                await asyncio.sleep(0.5 * (2 ** attempt))
        
        logger.error("Kalshi order failed after retries")
        return None
    
    async def get_balance(self) -> Optional[float]:
        """Get account balance"""
        endpoint = "/trade-api/v2/portfolio/balance"
        data = await self._request("GET", endpoint)
        
        if data and "balance" in data:
            return float(data["balance"]) / 100.0
        return None
    
    async def verify_credentials(self) -> Dict[str, Any]:
        """Verify credentials"""
        result = {"authenticated": False, "balance": None}
        
        balance = await self.get_balance()
        if balance is not None:
            result["authenticated"] = True
            result["balance"] = balance
            logger.info(f"Kalshi verified: ${balance:.2f}")
        
        return result