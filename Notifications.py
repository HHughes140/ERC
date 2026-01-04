"""
Notification Manager - Discord Alerts
Sends notifications for all trades (with deduplication)
"""
import aiohttp
import logging
from typing import Optional, Dict, Set
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class NotificationManager:
    """Sends Discord notifications for trading events"""
    
    def __init__(self, config):
        self.webhook_url = config.DISCORD_WEBHOOK_URL
        self.enabled = bool(self.webhook_url)
        
        # Track sent notifications to avoid duplicates
        self.sent_notifications: Set[str] = set()
        
        if self.enabled:
            logger.info("Notification Manager initialized")
        else:
            logger.warning("Discord webhook not configured - notifications disabled")
    
    def _generate_notification_id(self, trade_type: str, symbol: str, 
                                  side: str = "", platform: str = "") -> str:
        """Generate unique ID for notification to prevent duplicates"""
        # Create hash from trade details
        key = f"{trade_type}_{symbol}_{side}_{platform}_{datetime.now().strftime('%Y%m%d%H')}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _is_duplicate(self, notification_id: str) -> bool:
        """Check if notification was already sent"""
        if notification_id in self.sent_notifications:
            return True
        self.sent_notifications.add(notification_id)
        
        # Clean old notifications (keep last 1000)
        if len(self.sent_notifications) > 1000:
            self.sent_notifications = set(list(self.sent_notifications)[-500:])
        
        return False
    
    async def send_alert(self, level: str, title: str, message: str):
        """Send general alert"""
        if not self.enabled:
            return
        
        # Color based on level
        colors = {
            'info': 3447003,      # Blue
            'success': 3066993,   # Green
            'warning': 16776960,  # Yellow
            'error': 15158332     # Red
        }
        
        color = colors.get(level, 3447003)
        
        embed = {
            'title': title,
            'description': message,
            'color': color,
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {'text': 'ERC Master Engine'}
        }
        
        await self._send_webhook({'embeds': [embed]})
    
    async def send_trade_alert(self, trade_type: str, symbol: str, 
                              profit: float, side: str = "LONG",
                              platform: str = "Unknown Platform",
                              price: float = 0.0,
                              simulation: bool = False):
        """Send trade notification with full details"""
        if not self.enabled:
            return
        
        # Check for duplicate
        notification_id = self._generate_notification_id(trade_type, symbol, side, platform)
        if self._is_duplicate(notification_id):
            logger.debug(f"Skipping duplicate notification: {trade_type} - {symbol}")
            return
        
        # Color: Green for profit, Red for loss, Blue for entries
        if profit > 0:
            color = 3066993  # Green
        elif profit < 0:
            color = 15158332  # Red
        else:
            color = 3447003  # Blue for entries
        
        # Emoji based on trade type
        emojis = {
            'arbitrage': '⚡',
            'sharky_scalp': '🦈',
            'sharky_directional': '📈',
            'weather': '🌤️',
            'alpaca_ml': '🤖'
        }
        emoji = emojis.get(trade_type, '📊')
        
        # Platform emoji
        platform_emojis = {
            'polymarket': '🎲',
            'kalshi': '📊',
            'alpaca': '📈',
            'unknown platform': '🔹'
        }
        platform_emoji = platform_emojis.get(platform.lower(), '🔹')
        
        # Build fields - Platform is ALWAYS first and visible
        fields = [
            {'name': '📍 Platform', 'value': f"{platform_emoji} {platform.title()}", 'inline': True},
            {'name': '📊 Symbol', 'value': symbol[:50], 'inline': True},
            {'name': '📈 Side', 'value': side.upper(), 'inline': True}
        ]
        
        if price > 0:
            fields.append({'name': '💰 Price', 'value': f"${price:.3f}", 'inline': True})
        
        if profit != 0:
            fields.append({'name': '💵 Profit', 'value': f"${profit:.2f}", 'inline': True})
        
        # Add simulation badge if applicable
        title_suffix = " [SIMULATION]" if simulation else ""
        
        embed = {
            'title': f"{emoji} {trade_type.upper().replace('_', ' ')} TRADE{title_suffix}",
            'color': color,
            'fields': fields,
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {'text': 'ERC Master Engine'}
        }
        
        await self._send_webhook({'embeds': [embed]})
        logger.info(f"✅ Sent trade alert: {trade_type} - {platform} - {symbol} - {side}")
    
    async def send_arbitrage_alert(self, market1: str, market2: str, 
                                  platform1: str, platform2: str,
                                  side1: str, side2: str,
                                  price1: float, price2: float,
                                  profit_pct: float, capital: float,
                                  simulation: bool = False):
        """Send arbitrage-specific alert"""
        if not self.enabled:
            return
        
        # Check for duplicate
        notification_id = self._generate_notification_id('arbitrage', f"{market1}_{market2}", 
                                                        f"{side1}_{side2}", f"{platform1}_{platform2}")
        if self._is_duplicate(notification_id):
            return
        
        expected_profit = capital * (profit_pct / 100)
        
        # Platform emojis
        p1_emoji = '🎲' if platform1.lower() == 'polymarket' else '📊'
        p2_emoji = '🎲' if platform2.lower() == 'polymarket' else '📊'
        
        title_suffix = " [SIMULATION]" if simulation else ""
        
        embed = {
            'title': f"⚡ ARBITRAGE OPPORTUNITY{title_suffix}",
            'color': 3066993,
            'fields': [
                {'name': f'{p1_emoji} {platform1.title()} - {side1.upper()}', 
                 'value': f"{market1[:50]}\nPrice: ${price1:.3f}", 
                 'inline': False},
                {'name': f'{p2_emoji} {platform2.title()} - {side2.upper()}', 
                 'value': f"{market2[:50]}\nPrice: ${price2:.3f}", 
                 'inline': False},
                {'name': '💰 Profit %', 'value': f"{profit_pct:.2f}%", 'inline': True},
                {'name': '💵 Capital', 'value': f"${capital:.2f}", 'inline': True},
                {'name': '✨ Expected Profit', 'value': f"${expected_profit:.2f}", 'inline': True}
            ],
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {'text': 'ERC Master Engine'}
        }
        
        await self._send_webhook({'embeds': [embed]})
        logger.info(f"✅ Sent arbitrage alert: {platform1}/{platform2} - {market1}")
    
    async def send_sharky_alert(self, market: str, platform: str, 
                               side: str, price: float,
                               certainty: float, profit_potential: float,
                               capital: float, simulation: bool = False):
        """Send Sharky-specific alert"""
        if not self.enabled:
            return
        
        # Check for duplicate
        notification_id = self._generate_notification_id('sharky', market, side, platform)
        if self._is_duplicate(notification_id):
            return
        
        platform_emoji = '🎲' if platform.lower() == 'polymarket' else '📊'
        expected_profit = capital * profit_potential
        
        title_suffix = " [SIMULATION]" if simulation else ""
        
        embed = {
            'title': f"🦈 SHARKY SCALP TRADE{title_suffix}",
            'color': 3447003,
            'fields': [
                {'name': '📍 Platform', 'value': f"{platform_emoji} {platform.title()}", 'inline': True},
                {'name': '📊 Market', 'value': market[:100], 'inline': False},
                {'name': '📈 Side', 'value': side.upper(), 'inline': True},
                {'name': '💰 Entry Price', 'value': f"${price:.3f}", 'inline': True},
                {'name': '🎯 Certainty', 'value': f"{certainty:.1%}", 'inline': True},
                {'name': '💵 Capital', 'value': f"${capital:.2f}", 'inline': True},
                {'name': '✨ Profit Potential', 'value': f"${expected_profit:.2f} ({profit_potential:.1%})", 'inline': True}
            ],
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {'text': 'ERC Master Engine'}
        }
        
        await self._send_webhook({'embeds': [embed]})
        logger.info(f"✅ Sent Sharky alert: {platform} - {market} - {side}")
    
    async def send_weather_alert(self, city: str, outcome1: str, outcome2: str,
                                price1: float, price2: float,
                                cost: float, profit: float,
                                simulation: bool = False):
        """Send weather bot alert"""
        if not self.enabled:
            return
        
        # Check for duplicate
        notification_id = self._generate_notification_id('weather', city, 
                                                        f"{outcome1}_{outcome2}", 'kalshi')
        if self._is_duplicate(notification_id):
            return
        
        title_suffix = " [SIMULATION]" if simulation else ""
        
        embed = {
            'title': f"🌤️ WEATHER TRADE - {city}{title_suffix}",
            'color': 3447003,
            'fields': [
                {'name': '📍 Platform', 'value': '📊 Kalshi', 'inline': True},
                {'name': '🌍 City', 'value': city, 'inline': True},
                {'name': '📈 Outcome 1', 'value': f"{outcome1}\nYES @ ${price1:.2f}", 'inline': True},
                {'name': '📈 Outcome 2', 'value': f"{outcome2}\nYES @ ${price2:.2f}", 'inline': True},
                {'name': '💵 Total Cost', 'value': f"${cost:.2f}", 'inline': True},
                {'name': '✨ Max Profit', 'value': f"${profit:.2f}", 'inline': True}
            ],
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {'text': 'ERC Master Engine'}
        }
        
        await self._send_webhook({'embeds': [embed]})
        logger.info(f"✅ Sent weather alert: Kalshi - {city}")
    
    async def send_alpaca_alert(self, symbol: str, side: str, 
                               price: float, prediction: float,
                               confidence: float, win_prob: float,
                               timeframe: str, capital: float,
                               simulation: bool = False):
        """Send Alpaca ML alert"""
        if not self.enabled:
            return
        
        # Check for duplicate
        notification_id = self._generate_notification_id('alpaca', symbol, side, timeframe)
        if self._is_duplicate(notification_id):
            return
        
        title_suffix = " [SIMULATION]" if simulation else ""
        
        embed = {
            'title': f"🤖 ALPACA ML TRADE - {symbol}{title_suffix}",
            'color': 3447003,
            'fields': [
                {'name': '📍 Platform', 'value': '📈 Alpaca', 'inline': True},
                {'name': '📊 Symbol', 'value': symbol, 'inline': True},
                {'name': '📈 Side', 'value': side.upper(), 'inline': True},
                {'name': '💰 Entry Price', 'value': f"${price:.2f}", 'inline': True},
                {'name': '⏰ Timeframe', 'value': timeframe, 'inline': True},
                {'name': '💵 Capital', 'value': f"${capital:.2f}", 'inline': True},
                {'name': '🎯 ML Prediction', 'value': f"{prediction:.2f}", 'inline': True},
                {'name': '📊 Confidence', 'value': f"{confidence:.1%}", 'inline': True},
                {'name': '🎲 Win Probability', 'value': f"{win_prob:.1%}", 'inline': True}
            ],
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {'text': 'ERC Master Engine'}
        }
        
        await self._send_webhook({'embeds': [embed]})
        logger.info(f"✅ Sent Alpaca ML alert: {symbol} - {side}")
    
    async def send_portfolio_update(self, total_capital: float, 
                                   deployed: float, pnl: float, 
                                   positions: int):
        """Send portfolio status update"""
        if not self.enabled:
            return
        
        available = total_capital - deployed
        color = 3066993 if pnl >= 0 else 15158332
        
        embed = {
            'title': '💼 PORTFOLIO UPDATE',
            'color': color,
            'fields': [
                {'name': 'Total Capital', 'value': f"${total_capital:,.2f}", 'inline': True},
                {'name': 'Deployed', 'value': f"${deployed:,.2f}", 'inline': True},
                {'name': 'Available', 'value': f"${available:,.2f}", 'inline': True},
                {'name': 'Total P&L', 'value': f"${pnl:,.2f}", 'inline': True},
                {'name': 'Open Positions', 'value': str(positions), 'inline': True}
            ],
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {'text': 'ERC Master Engine'}
        }
        
        await self._send_webhook({'embeds': [embed]})
        logger.info(f"✅ Sent portfolio update: ${total_capital:,.2f} total, {positions} positions")
    
    async def _send_webhook(self, payload: Dict):
        """Send webhook to Discord"""
        if not self.enabled:
            return
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 204:
                        logger.error(f"Discord webhook failed: {response.status}")
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")