# ERC Trading System

Algorithmic trading system for prediction markets (Polymarket, Kalshi) and equities (Alpaca).

## Overview

ERC (Elastic Research Capital) is a multi-strategy trading system that combines:

- **Arbitrage Detection** - Cross-platform and single-market arbitrage
- **Sharky Scalping** - Near-certainty outcome scalping (97%+ probability)
- **Weather Trading** - Weather market trading with API forecasts
- **ML Trading** - Machine learning-based equity trading

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd ERC

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API credentials
# - Polymarket API keys
# - Kalshi API keys
# - Alpaca API keys
# - Discord webhook (optional)
```

### 3. Run the System

```bash
# Run main trading engine
python Master_Engine.py

# Or run individual strategies:
python -m Models.Polymarket.scanner      # Arbitrage scanner
python -m Models.Sharky.sharky_scanner   # Sharky scanner
python -m Models.Weather.weather_bot     # Weather bot
python -m Models.C.start                 # ML trading engine
```

### 4. Run Tests

```bash
pytest
```

## Project Structure

```
ERC/
├── Master_Engine.py          # Main orchestration engine
├── Portfolio_Manager.py      # Capital allocation & position tracking
├── executor.py               # Trade execution
├── config.py                 # Configuration
├── notifications.py          # Discord alerts
│
├── Models/
│   ├── Polymarket/           # Arbitrage detection
│   │   ├── scanner.py        # Main arbitrage scanner
│   │   ├── math_models.py    # Arbitrage calculations
│   │   ├── probability_models.py  # Bayesian estimation
│   │   ├── order_book_simulator.py # Execution simulation
│   │   └── semantic_matcher.py  # Cross-platform matching
│   │
│   ├── Sharky/               # Near-certainty scalping
│   │   ├── sharky_scanner.py # Main scanner
│   │   ├── allocation_engine.py # Kelly sizing
│   │   └── exit_engine.py    # Dynamic exits
│   │
│   ├── Weather/              # Weather market trading
│   │   ├── weather_bot.py    # Main bot
│   │   └── weather_data.py   # Weather API integration
│   │
│   └── C/                    # ML equity trading
│       ├── start.py          # Main ML engine
│       ├── indicators.py     # Technical indicators
│       ├── ensemble.py       # Ensemble classifier
│       └── backtester.py     # Backtesting framework
│
├── Infrastructure/           # Shared infrastructure
│   ├── cache.py             # Caching layer
│   ├── logging_config.py    # Structured logging
│   ├── error_handling.py    # Retry, rate limiting
│   ├── portfolio_risk.py    # Risk management
│   ├── order_execution.py   # Smart order execution
│   ├── paper_trading.py     # Paper trading simulation
│   ├── sentiment.py         # Sentiment analysis
│   └── ab_testing.py        # A/B testing framework
│
├── Central_DB/              # Database layer
│   └── database.py          # SQLite database
│
├── Factors/                 # Factor modules
│   ├── calibration.py       # Historical calibration
│   └── ...
│
└── tests/                   # Test suite
    ├── conftest.py          # Pytest fixtures
    ├── test_database.py
    ├── test_infrastructure.py
    ├── test_calibration.py
    └── test_models.py
```

## Strategies

### 1. Arbitrage Strategy (40% allocation)

Detects and executes arbitrage opportunities:
- **Single-market arbitrage**: YES + NO < $1.00
- **Multi-outcome arbitrage**: Sum of all outcomes < $1.00
- **Cross-platform arbitrage**: Price differences between Polymarket and Kalshi

Key features:
- Order book simulation for realistic slippage estimation
- Semantic matching for cross-platform market alignment
- Bayesian probability estimation with confidence intervals

### 2. Sharky Strategy (30% allocation)

Near-certainty scalping strategy:
- Targets markets with 97%+ implied probability
- Fee-adjusted profit calculations
- Historical calibration for accurate certainty estimation
- Dynamic exit engine with profit targets and stop-losses

### 3. Weather Strategy (20% allocation)

Weather market trading:
- Integrates with weather APIs (Open-Meteo, NWS)
- Probability distributions for temperature ranges
- Automatic market discovery for target cities

### 4. ML Strategy (10% allocation)

Machine learning-based equity trading:
- Multi-timeframe analysis (5m, 15m, 1h, 1d)
- Technical indicator suite (RSI, MACD, Bollinger Bands, etc.)
- Ensemble classifier with dynamic weight updates
- Market regime detection

## Risk Management

- **Position limits**: Max 10% of capital per position
- **Portfolio exposure**: Max 80% deployed at any time
- **Model exposure**: Max 30% per strategy
- **Platform exposure**: Max 50% per platform
- **Correlation detection**: Prevents correlated position concentration
- **Drawdown monitoring**: Automatic position reduction on drawdowns

## Paper Trading

The system runs in paper trading mode by default. To enable live trading:

1. Set up API credentials with trading permissions
2. Modify `executor.py` to enable live execution
3. Start with small position sizes

## Monitoring

- **Logs**: `master_engine.log`, `metric_space_engine.log`
- **Discord**: Real-time trade notifications (configure webhook in `.env`)
- **Database**: SQLite at `data/erc.db`

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_database.py

# Run with coverage
pytest --cov=.
```

### Code Style

```bash
# Format code
black .

# Lint
flake8 .
```

## Configuration Reference

Key configuration in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_PROFIT_THRESHOLD` | 0.02 | Minimum profit for arbitrage |
| `MAX_POSITION_SIZE` | 100 | Maximum position size ($) |
| `ARBITRAGE_ALLOCATION` | 0.40 | Capital allocation to arbitrage |
| `SHARKY_ALLOCATION` | 0.30 | Capital allocation to Sharky |
| `WEATHER_ALLOCATION` | 0.20 | Capital allocation to Weather |
| `ML_ALLOCATION` | 0.10 | Capital allocation to ML |

## Security Notes

- **Never commit `.env`** - Contains API credentials
- **Use paper trading first** - Test strategies before live trading
- **Monitor positions** - System is not fully autonomous
- **API rate limits** - Built-in rate limiting prevents bans

## License

Proprietary - All rights reserved.
