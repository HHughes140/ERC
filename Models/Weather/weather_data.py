"""
Weather Data Provider for Weather Bot

Integrates with weather APIs to get temperature forecasts and
calculate probabilities for prediction market ranges.

Key Features:
1. Multi-source weather data (Open-Meteo, NWS backup)
2. Probability calculations using forecast distributions
3. Historical accuracy tracking for model calibration
4. City coordinate management

CRITICAL: Weather markets are time-sensitive. Stale forecasts = bad predictions.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
import logging
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class WeatherForecast:
    """Weather forecast for a specific time and location"""
    city: str
    target_time: datetime
    forecast_time: datetime  # When this forecast was made
    temperature_mean: float  # Expected temperature (Fahrenheit)
    temperature_std: float   # Uncertainty (standard deviation)
    temperature_min: float   # Forecast low
    temperature_max: float   # Forecast high
    precipitation_prob: float  # 0-1
    confidence: float        # Model confidence 0-1
    source: str             # API source

    @property
    def hours_ahead(self) -> float:
        """Hours between forecast and target time"""
        return (self.target_time - self.forecast_time).total_seconds() / 3600

    @property
    def age_minutes(self) -> float:
        """How old this forecast is"""
        return (datetime.now() - self.forecast_time).total_seconds() / 60


@dataclass
class CityCoordinates:
    """City location for weather queries"""
    name: str
    lat: float
    lon: float
    timezone: str
    aliases: List[str]


class WeatherDataProvider:
    """
    Fetches weather data and calculates prediction market probabilities.

    Uses multiple APIs for redundancy:
    1. Open-Meteo (free, no API key)
    2. NWS (US only, free)
    """

    # Major cities with coordinates
    CITIES = {
        'new_york': CityCoordinates('New York', 40.7128, -74.0060, 'America/New_York',
                                    ['nyc', 'manhattan', 'new york city']),
        'los_angeles': CityCoordinates('Los Angeles', 34.0522, -118.2437, 'America/Los_Angeles',
                                       ['la', 'los angeles']),
        'chicago': CityCoordinates('Chicago', 41.8781, -87.6298, 'America/Chicago',
                                   ['chi']),
        'houston': CityCoordinates('Houston', 29.7604, -95.3698, 'America/Chicago',
                                   ['htx']),
        'phoenix': CityCoordinates('Phoenix', 33.4484, -112.0740, 'America/Phoenix',
                                   ['phx']),
        'philadelphia': CityCoordinates('Philadelphia', 39.9526, -75.1652, 'America/New_York',
                                        ['philly']),
        'san_antonio': CityCoordinates('San Antonio', 29.4241, -98.4936, 'America/Chicago', []),
        'san_diego': CityCoordinates('San Diego', 32.7157, -117.1611, 'America/Los_Angeles', []),
        'dallas': CityCoordinates('Dallas', 32.7767, -96.7970, 'America/Chicago', ['dfw']),
        'austin': CityCoordinates('Austin', 30.2672, -97.7431, 'America/Chicago', ['atx']),
        'miami': CityCoordinates('Miami', 25.7617, -80.1918, 'America/New_York', ['mia']),
        'denver': CityCoordinates('Denver', 39.7392, -104.9903, 'America/Denver', ['den']),
        'seattle': CityCoordinates('Seattle', 47.6062, -122.3321, 'America/Los_Angeles', ['sea']),
        'boston': CityCoordinates('Boston', 42.3601, -71.0589, 'America/New_York', ['bos']),
        'atlanta': CityCoordinates('Atlanta', 33.7490, -84.3880, 'America/New_York', ['atl']),
        'las_vegas': CityCoordinates('Las Vegas', 36.1699, -115.1398, 'America/Los_Angeles',
                                      ['vegas']),
        'detroit': CityCoordinates('Detroit', 42.3314, -83.0458, 'America/Detroit', ['det']),
        'minneapolis': CityCoordinates('Minneapolis', 44.9778, -93.2650, 'America/Chicago',
                                       ['msp']),
        'san_francisco': CityCoordinates('San Francisco', 37.7749, -122.4194, 'America/Los_Angeles',
                                         ['sf', 'bay area']),
    }

    def __init__(self, cache_ttl_minutes: int = 15):
        """
        Args:
            cache_ttl_minutes: How long to cache forecasts
        """
        self.cache_ttl = cache_ttl_minutes
        self._cache: Dict[str, WeatherForecast] = {}

        # Forecast uncertainty by hours ahead (based on typical NWP skill)
        # Standard deviation increases with forecast horizon
        self.uncertainty_model = {
            0: 1.0,    # 0 hours: ±1°F
            6: 1.5,    # 6 hours: ±1.5°F
            12: 2.0,   # 12 hours: ±2°F
            24: 3.0,   # 24 hours: ±3°F
            48: 4.0,   # 48 hours: ±4°F
            72: 5.0,   # 72 hours: ±5°F
            168: 7.0,  # 1 week: ±7°F
        }

    def get_city(self, name: str) -> Optional[CityCoordinates]:
        """Look up city by name or alias"""
        name_lower = name.lower().replace(' ', '_')

        # Direct match
        if name_lower in self.CITIES:
            return self.CITIES[name_lower]

        # Alias search
        for city in self.CITIES.values():
            if name_lower in city.aliases or name_lower == city.name.lower():
                return city

        return None

    def _get_uncertainty(self, hours_ahead: float) -> float:
        """
        Get forecast uncertainty (std dev) for given hours ahead.

        Uses linear interpolation between known points.
        """
        hours = sorted(self.uncertainty_model.keys())

        if hours_ahead <= hours[0]:
            return self.uncertainty_model[hours[0]]
        if hours_ahead >= hours[-1]:
            return self.uncertainty_model[hours[-1]]

        # Interpolate
        for i in range(len(hours) - 1):
            if hours[i] <= hours_ahead < hours[i + 1]:
                t = (hours_ahead - hours[i]) / (hours[i + 1] - hours[i])
                return (1 - t) * self.uncertainty_model[hours[i]] + \
                       t * self.uncertainty_model[hours[i + 1]]

        return 3.0  # Default

    async def fetch_forecast_open_meteo(self, city: CityCoordinates,
                                        target_time: datetime) -> Optional[WeatherForecast]:
        """
        Fetch forecast from Open-Meteo API (free, no key needed).

        Note: This is an async method - requires aiohttp.
        For sync usage, use fetch_forecast_sync().
        """
        try:
            import aiohttp

            # Build API URL
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={city.lat}&longitude={city.lon}"
                f"&hourly=temperature_2m,precipitation_probability"
                f"&temperature_unit=fahrenheit"
                f"&timezone={city.timezone}"
                f"&forecast_days=7"
            )

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Open-Meteo API error: {response.status}")
                        return None

                    data = await response.json()

            # Parse response
            hourly = data.get('hourly', {})
            times = hourly.get('time', [])
            temps = hourly.get('temperature_2m', [])
            precip_probs = hourly.get('precipitation_probability', [])

            # Find closest time to target
            target_str = target_time.strftime('%Y-%m-%dT%H:00')
            if target_str in times:
                idx = times.index(target_str)
            else:
                # Find nearest
                idx = min(range(len(times)),
                         key=lambda i: abs(datetime.fromisoformat(times[i]) - target_time))

            temp = temps[idx]
            precip = precip_probs[idx] / 100 if precip_probs else 0

            # Calculate uncertainty based on forecast horizon
            hours_ahead = (target_time - datetime.now()).total_seconds() / 3600
            uncertainty = self._get_uncertainty(hours_ahead)

            return WeatherForecast(
                city=city.name,
                target_time=target_time,
                forecast_time=datetime.now(),
                temperature_mean=temp,
                temperature_std=uncertainty,
                temperature_min=temp - uncertainty * 2,
                temperature_max=temp + uncertainty * 2,
                precipitation_prob=precip,
                confidence=max(0.5, 1.0 - hours_ahead / 168),  # Confidence decreases over time
                source='open-meteo'
            )

        except ImportError:
            logger.error("aiohttp not installed - use fetch_forecast_sync()")
            return None
        except Exception as e:
            logger.error(f"Open-Meteo fetch error: {e}")
            return None

    def fetch_forecast_sync(self, city: CityCoordinates,
                            target_time: datetime) -> Optional[WeatherForecast]:
        """
        Synchronous forecast fetch using requests library.

        Use this for non-async code paths.
        """
        try:
            import requests

            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={city.lat}&longitude={city.lon}"
                f"&hourly=temperature_2m,precipitation_probability"
                f"&temperature_unit=fahrenheit"
                f"&timezone={city.timezone}"
                f"&forecast_days=7"
            )

            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                logger.error(f"Open-Meteo API error: {response.status_code}")
                return None

            data = response.json()

            # Parse response
            hourly = data.get('hourly', {})
            times = hourly.get('time', [])
            temps = hourly.get('temperature_2m', [])
            precip_probs = hourly.get('precipitation_probability', [])

            if not times or not temps:
                logger.error("No forecast data in response")
                return None

            # Find closest time to target
            target_str = target_time.strftime('%Y-%m-%dT%H:00')
            if target_str in times:
                idx = times.index(target_str)
            else:
                idx = min(range(len(times)),
                         key=lambda i: abs(datetime.fromisoformat(times[i]) - target_time))

            temp = temps[idx]
            precip = precip_probs[idx] / 100 if precip_probs else 0

            hours_ahead = max(0, (target_time - datetime.now()).total_seconds() / 3600)
            uncertainty = self._get_uncertainty(hours_ahead)

            return WeatherForecast(
                city=city.name,
                target_time=target_time,
                forecast_time=datetime.now(),
                temperature_mean=temp,
                temperature_std=uncertainty,
                temperature_min=temp - uncertainty * 2,
                temperature_max=temp + uncertainty * 2,
                precipitation_prob=precip,
                confidence=max(0.5, 1.0 - hours_ahead / 168),
                source='open-meteo'
            )

        except Exception as e:
            logger.error(f"Sync forecast fetch error: {e}")
            return None

    def calculate_probability_in_range(self, forecast: WeatherForecast,
                                       low: float, high: float) -> float:
        """
        Calculate probability that actual temperature falls within range.

        Uses normal distribution with forecast mean and uncertainty.

        P(low < T < high) = CDF(high) - CDF(low)

        Args:
            forecast: Weather forecast with mean and std
            low: Lower bound of temperature range
            high: Upper bound of temperature range

        Returns:
            Probability (0-1) that temperature will be in range
        """
        if forecast.temperature_std <= 0:
            # Degenerate case - point estimate
            return 1.0 if low <= forecast.temperature_mean <= high else 0.0

        # Normal distribution CDF
        prob_low = stats.norm.cdf(low, forecast.temperature_mean, forecast.temperature_std)
        prob_high = stats.norm.cdf(high, forecast.temperature_mean, forecast.temperature_std)

        probability = prob_high - prob_low

        # Adjust by forecast confidence
        # Lower confidence = pull probability toward 0.5 (less certain)
        adjusted_prob = 0.5 + (probability - 0.5) * forecast.confidence

        return np.clip(adjusted_prob, 0.01, 0.99)

    def calculate_probability_above(self, forecast: WeatherForecast,
                                    threshold: float) -> float:
        """Calculate probability temperature is above threshold"""
        prob = 1 - stats.norm.cdf(threshold, forecast.temperature_mean, forecast.temperature_std)
        return np.clip(prob * forecast.confidence + 0.5 * (1 - forecast.confidence), 0.01, 0.99)

    def calculate_probability_below(self, forecast: WeatherForecast,
                                    threshold: float) -> float:
        """Calculate probability temperature is below threshold"""
        prob = stats.norm.cdf(threshold, forecast.temperature_mean, forecast.temperature_std)
        return np.clip(prob * forecast.confidence + 0.5 * (1 - forecast.confidence), 0.01, 0.99)

    def calculate_edge(self, our_prob: float, market_price: float,
                       min_edge: float = 0.10) -> Dict:
        """
        Calculate edge vs market price.

        Args:
            our_prob: Our estimated probability (0-1)
            market_price: Market's implied probability (price)
            min_edge: Minimum edge to consider trading

        Returns:
            Dict with edge calculation and recommendation
        """
        edge = our_prob - market_price
        edge_pct = edge / market_price if market_price > 0 else 0

        # Kelly fraction (fractional Kelly)
        if our_prob > market_price and market_price < 1:
            b = (1 - market_price) / market_price  # Odds
            q = 1 - our_prob
            kelly = (our_prob * b - q) / b if b > 0 else 0
            kelly_fraction = max(0, kelly * 0.25)  # Quarter Kelly
        else:
            kelly_fraction = 0

        should_trade = edge >= min_edge and kelly_fraction > 0

        return {
            'our_probability': our_prob,
            'market_probability': market_price,
            'edge': edge,
            'edge_pct': edge_pct,
            'kelly_fraction': kelly_fraction,
            'should_trade': should_trade,
            'recommendation': 'BUY' if should_trade else 'PASS',
            'confidence': 'HIGH' if edge > 0.15 else 'MEDIUM' if edge > 0.10 else 'LOW'
        }


class WeatherMarketAnalyzer:
    """
    Analyzes weather prediction markets for trading opportunities.
    """

    def __init__(self, provider: Optional[WeatherDataProvider] = None):
        self.provider = provider or WeatherDataProvider()

    def analyze_market(self, city_name: str, target_time: datetime,
                       market_outcomes: List[Dict]) -> List[Dict]:
        """
        Analyze a weather market and find trading opportunities.

        Args:
            city_name: City for the weather market
            target_time: When the temperature will be measured
            market_outcomes: List of {low, high, price} for each outcome

        Returns:
            List of opportunities with our probability vs market price
        """
        city = self.provider.get_city(city_name)
        if not city:
            logger.error(f"Unknown city: {city_name}")
            return []

        # Fetch forecast
        forecast = self.provider.fetch_forecast_sync(city, target_time)
        if not forecast:
            logger.error(f"Failed to fetch forecast for {city_name}")
            return []

        opportunities = []

        for outcome in market_outcomes:
            low = outcome.get('low', float('-inf'))
            high = outcome.get('high', float('inf'))
            market_price = outcome['price']

            # Calculate our probability
            if low == float('-inf'):
                our_prob = self.provider.calculate_probability_below(forecast, high)
            elif high == float('inf'):
                our_prob = self.provider.calculate_probability_above(forecast, low)
            else:
                our_prob = self.provider.calculate_probability_in_range(forecast, low, high)

            # Calculate edge
            edge_calc = self.provider.calculate_edge(our_prob, market_price)

            opportunities.append({
                'outcome': f"{low}F - {high}F",
                'forecast_mean': forecast.temperature_mean,
                'forecast_std': forecast.temperature_std,
                **edge_calc
            })

        return opportunities


def test_weather_provider():
    """Test the weather data provider"""
    provider = WeatherDataProvider()

    print("=== Weather Data Provider Test ===\n")

    # Test city lookup
    city = provider.get_city('NYC')
    print(f"City lookup 'NYC': {city.name if city else 'Not found'}")

    city = provider.get_city('Chicago')
    print(f"City lookup 'Chicago': {city.name if city else 'Not found'}")

    # Test forecast (if requests is available)
    if city:
        target = datetime.now() + timedelta(hours=12)
        print(f"\nFetching forecast for {city.name} at {target}...")

        try:
            forecast = provider.fetch_forecast_sync(city, target)
            if forecast:
                print(f"  Temperature: {forecast.temperature_mean:.1f}°F ± {forecast.temperature_std:.1f}°F")
                print(f"  Range: {forecast.temperature_min:.1f}°F - {forecast.temperature_max:.1f}°F")
                print(f"  Precipitation: {forecast.precipitation_prob:.0%}")
                print(f"  Confidence: {forecast.confidence:.0%}")

                # Test probability calculations
                print(f"\n  P(T > 70°F): {provider.calculate_probability_above(forecast, 70):.1%}")
                print(f"  P(T < 50°F): {provider.calculate_probability_below(forecast, 50):.1%}")
                print(f"  P(60°F < T < 80°F): {provider.calculate_probability_in_range(forecast, 60, 80):.1%}")

                # Test edge calculation
                edge = provider.calculate_edge(0.65, 0.50)
                print(f"\n  Edge calc (our 65% vs market 50%): {edge['edge']:.1%}, Kelly: {edge['kelly_fraction']:.1%}")
            else:
                print("  Failed to fetch forecast (may need internet connection)")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    test_weather_provider()
