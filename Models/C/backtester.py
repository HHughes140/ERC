"""
Backtesting Framework for ML Trading Engine

Proper backtesting with realistic assumptions.

Key Features:
1. Walk-forward testing (no lookahead)
2. Realistic transaction costs
3. Slippage simulation
4. Stop loss / take profit simulation
5. Comprehensive metrics (Sharpe, drawdown, profit factor)

CRITICAL: Backtests without transaction costs and slippage are WORTHLESS.
A strategy that looks great with 0 costs often fails in live trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class PositionSide(Enum):
    """Position sides"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Trade:
    """Single trade record"""
    entry_time: datetime
    exit_time: datetime
    side: PositionSide
    entry_price: float
    exit_price: float
    size: float
    gross_pnl: float
    fees: float
    slippage: float
    net_pnl: float
    exit_reason: str


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    # Transaction costs
    commission_rate: float = 0.001      # 0.1% per trade
    slippage_rate: float = 0.001        # 0.1% slippage
    min_commission: float = 1.0         # Minimum $1 commission

    # Position sizing
    initial_capital: float = 100000.0
    position_size_pct: float = 0.02     # 2% of capital per trade
    max_position_pct: float = 0.10      # Max 10% in single position

    # Risk management
    stop_loss_pct: float = 0.02         # 2% stop loss
    take_profit_pct: float = 0.04       # 4% take profit
    trailing_stop_pct: Optional[float] = None  # Optional trailing stop

    # Retraining
    retrain_frequency: int = 100        # Retrain model every N bars
    min_train_samples: int = 200        # Minimum samples before trading


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics"""
    # Returns
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Risk
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float
    downside_volatility: float
    var_95: float

    # Trading
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    expectancy: float

    # Additional
    best_trade: float
    worst_trade: float
    avg_holding_period: float
    trades_per_year: float


@dataclass
class Position:
    """Current position state"""
    side: PositionSide = PositionSide.FLAT
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    size: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    highest_price: float = 0.0  # For trailing stop


class BacktestEngine:
    """
    Complete backtesting engine with realistic simulation.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.reset()

    def reset(self):
        """Reset backtest state"""
        self.capital = self.config.initial_capital
        self.position = Position()
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []

    def _calculate_slippage(self, price: float, side: PositionSide) -> float:
        """Calculate slippage based on order direction"""
        slippage = price * self.config.slippage_rate
        if side == PositionSide.LONG:
            return slippage  # Pay more when buying
        else:
            return -slippage  # Receive less when selling

    def _calculate_commission(self, trade_value: float) -> float:
        """Calculate commission for trade"""
        commission = trade_value * self.config.commission_rate
        return max(commission, self.config.min_commission)

    def _get_position_size(self, price: float) -> float:
        """Calculate position size based on config"""
        target_value = self.capital * self.config.position_size_pct
        max_value = self.capital * self.config.max_position_pct

        size_value = min(target_value, max_value)
        size = size_value / price

        return size

    def _open_position(self, timestamp: datetime, price: float,
                       side: PositionSide) -> None:
        """Open a new position"""
        if self.position.side != PositionSide.FLAT:
            return

        # Apply slippage
        slippage = self._calculate_slippage(price, side)
        fill_price = price + slippage

        # Calculate size
        size = self._get_position_size(fill_price)
        trade_value = size * fill_price

        # Check if we have enough capital
        commission = self._calculate_commission(trade_value)
        if trade_value + commission > self.capital:
            size = (self.capital - commission) / fill_price
            trade_value = size * fill_price

        if size <= 0:
            return

        # Set stop loss and take profit
        if side == PositionSide.LONG:
            stop_loss = fill_price * (1 - self.config.stop_loss_pct)
            take_profit = fill_price * (1 + self.config.take_profit_pct)
        else:
            stop_loss = fill_price * (1 + self.config.stop_loss_pct)
            take_profit = fill_price * (1 - self.config.take_profit_pct)

        # Open position
        self.position = Position(
            side=side,
            entry_price=fill_price,
            entry_time=timestamp,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            highest_price=fill_price
        )

        # Deduct costs
        self.capital -= commission

    def _close_position(self, timestamp: datetime, price: float,
                        reason: str) -> None:
        """Close current position"""
        if self.position.side == PositionSide.FLAT:
            return

        # Apply slippage (opposite direction)
        close_side = PositionSide.SHORT if self.position.side == PositionSide.LONG else PositionSide.LONG
        slippage = self._calculate_slippage(price, close_side)
        fill_price = price + slippage

        # Calculate PnL
        if self.position.side == PositionSide.LONG:
            gross_pnl = (fill_price - self.position.entry_price) * self.position.size
        else:
            gross_pnl = (self.position.entry_price - fill_price) * self.position.size

        # Calculate fees
        trade_value = self.position.size * fill_price
        commission = self._calculate_commission(trade_value)
        total_slippage = abs(slippage) * self.position.size

        net_pnl = gross_pnl - commission - total_slippage

        # Record trade
        trade = Trade(
            entry_time=self.position.entry_time,
            exit_time=timestamp,
            side=self.position.side,
            entry_price=self.position.entry_price,
            exit_price=fill_price,
            size=self.position.size,
            gross_pnl=gross_pnl,
            fees=commission,
            slippage=total_slippage,
            net_pnl=net_pnl,
            exit_reason=reason
        )
        self.trades.append(trade)

        # Update capital
        self.capital += net_pnl + (self.position.entry_price * self.position.size)

        # Reset position
        self.position = Position()

    def _check_stops(self, timestamp: datetime, high: float, low: float) -> bool:
        """Check and execute stop loss / take profit"""
        if self.position.side == PositionSide.FLAT:
            return False

        # Update trailing stop if enabled
        if self.config.trailing_stop_pct:
            if self.position.side == PositionSide.LONG:
                self.position.highest_price = max(self.position.highest_price, high)
                trail_stop = self.position.highest_price * (1 - self.config.trailing_stop_pct)
                self.position.stop_loss = max(self.position.stop_loss, trail_stop)
            else:
                self.position.highest_price = min(self.position.highest_price, low)
                trail_stop = self.position.highest_price * (1 + self.config.trailing_stop_pct)
                self.position.stop_loss = min(self.position.stop_loss, trail_stop)

        if self.position.side == PositionSide.LONG:
            # Check stop loss
            if low <= self.position.stop_loss:
                self._close_position(timestamp, self.position.stop_loss, "stop_loss")
                return True
            # Check take profit
            if high >= self.position.take_profit:
                self._close_position(timestamp, self.position.take_profit, "take_profit")
                return True
        else:  # SHORT
            if high >= self.position.stop_loss:
                self._close_position(timestamp, self.position.stop_loss, "stop_loss")
                return True
            if low <= self.position.take_profit:
                self._close_position(timestamp, self.position.take_profit, "take_profit")
                return True

        return False

    def _get_equity(self, current_price: float) -> float:
        """Calculate current equity including open position"""
        equity = self.capital

        if self.position.side != PositionSide.FLAT:
            if self.position.side == PositionSide.LONG:
                unrealized = (current_price - self.position.entry_price) * self.position.size
            else:
                unrealized = (self.position.entry_price - current_price) * self.position.size
            equity += unrealized + (self.position.entry_price * self.position.size)

        return equity

    def run(self, data: pd.DataFrame,
            signal_func: Callable[[pd.DataFrame, int], int],
            train_func: Optional[Callable[[pd.DataFrame], None]] = None) -> BacktestMetrics:
        """
        Run backtest.

        Args:
            data: OHLCV DataFrame with datetime index
            signal_func: Function that returns signal (-1, 0, 1) given data and index
            train_func: Optional function to train/retrain model

        Returns:
            BacktestMetrics with performance statistics
        """
        self.reset()

        if len(data) < self.config.min_train_samples:
            raise ValueError(f"Not enough data: {len(data)} < {self.config.min_train_samples}")

        timestamps = data.index if isinstance(data.index, pd.DatetimeIndex) else \
                     pd.date_range('2020-01-01', periods=len(data), freq='1h')

        # Initial training
        if train_func:
            train_func(data.iloc[:self.config.min_train_samples])

        # Main loop
        last_train = self.config.min_train_samples

        for i in range(self.config.min_train_samples, len(data)):
            row = data.iloc[i]
            timestamp = timestamps[i]
            price = row['close']
            high = row['high']
            low = row['low']

            # Check stops first
            if not self._check_stops(timestamp, high, low):
                # Get signal
                signal = signal_func(data, i)

                # Execute trades based on signal
                if signal == 1 and self.position.side != PositionSide.LONG:
                    if self.position.side == PositionSide.SHORT:
                        self._close_position(timestamp, price, "signal_reverse")
                    self._open_position(timestamp, price, PositionSide.LONG)

                elif signal == -1 and self.position.side != PositionSide.SHORT:
                    if self.position.side == PositionSide.LONG:
                        self._close_position(timestamp, price, "signal_reverse")
                    self._open_position(timestamp, price, PositionSide.SHORT)

                elif signal == 0 and self.position.side != PositionSide.FLAT:
                    self._close_position(timestamp, price, "signal_exit")

            # Record equity
            equity = self._get_equity(price)
            self.equity_curve.append(equity)
            self.timestamps.append(timestamp)

            # Periodic retraining
            if train_func and (i - last_train) >= self.config.retrain_frequency:
                train_func(data.iloc[:i])
                last_train = i

        # Close any open position at end
        if self.position.side != PositionSide.FLAT:
            self._close_position(timestamps[-1], data.iloc[-1]['close'], "end_of_data")

        return self.calculate_metrics()

    def calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive performance metrics"""
        if len(self.equity_curve) < 2:
            return self._empty_metrics()

        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Basic returns
        total_return = (equity[-1] - equity[0]) / equity[0]

        # Assuming 252 trading days, 6.5 hours per day for intraday
        periods_per_year = 252 * 6.5 if len(returns) > 252 else 252
        annual_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1

        # Risk metrics
        volatility = np.std(returns) * np.sqrt(periods_per_year)

        # Sharpe ratio (assuming 0% risk-free rate)
        if volatility > 0:
            sharpe = (np.mean(returns) * periods_per_year) / volatility
        else:
            sharpe = 0

        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_vol = np.std(negative_returns) * np.sqrt(periods_per_year)
            sortino = (np.mean(returns) * periods_per_year) / downside_vol if downside_vol > 0 else 0
        else:
            downside_vol = 0
            sortino = sharpe * 2  # No downside = very good

        # Maximum drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = np.max(drawdown)

        # Drawdown duration
        in_drawdown = drawdown > 0
        if np.any(in_drawdown):
            dd_starts = np.where(np.diff(np.concatenate([[0], in_drawdown.astype(int)])) == 1)[0]
            dd_ends = np.where(np.diff(np.concatenate([in_drawdown.astype(int), [0]])) == -1)[0]
            if len(dd_starts) > 0 and len(dd_ends) > 0:
                durations = dd_ends[:len(dd_starts)] - dd_starts[:len(dd_ends)]
                max_dd_duration = int(np.max(durations)) if len(durations) > 0 else 0
            else:
                max_dd_duration = 0
        else:
            max_dd_duration = 0

        # Calmar ratio
        calmar = annual_return / max_dd if max_dd > 0 else 0

        # VaR 95%
        var_95 = np.percentile(returns, 5)

        # Trade statistics
        if self.trades:
            pnls = [t.net_pnl for t in self.trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            total_trades = len(self.trades)
            win_rate = len(wins) / total_trades if total_trades > 0 else 0

            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else total_wins

            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            avg_trade = np.mean(pnls)
            expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

            best_trade = max(pnls)
            worst_trade = min(pnls)

            # Average holding period
            durations = [(t.exit_time - t.entry_time).total_seconds() / 3600
                        for t in self.trades if t.entry_time and t.exit_time]
            avg_holding = np.mean(durations) if durations else 0

            # Trades per year
            if len(self.timestamps) > 1:
                total_hours = (self.timestamps[-1] - self.timestamps[0]).total_seconds() / 3600
                years = total_hours / (252 * 6.5)
                trades_per_year = total_trades / years if years > 0 else 0
            else:
                trades_per_year = 0
        else:
            total_trades = 0
            win_rate = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0
            avg_trade = 0
            expectancy = 0
            best_trade = 0
            worst_trade = 0
            avg_holding = 0
            trades_per_year = 0

        return BacktestMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            volatility=volatility,
            downside_volatility=downside_vol,
            var_95=var_95,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            expectancy=expectancy,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_holding_period=avg_holding,
            trades_per_year=trades_per_year
        )

    def _empty_metrics(self) -> BacktestMetrics:
        """Return empty metrics for failed backtest"""
        return BacktestMetrics(
            total_return=0, annual_return=0, sharpe_ratio=0, sortino_ratio=0,
            calmar_ratio=0, max_drawdown=0, max_drawdown_duration=0,
            volatility=0, downside_volatility=0, var_95=0, total_trades=0,
            win_rate=0, profit_factor=0, avg_win=0, avg_loss=0, avg_trade=0,
            expectancy=0, best_trade=0, worst_trade=0, avg_holding_period=0,
            trades_per_year=0
        )

    def get_equity_dataframe(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        return pd.DataFrame({
            'equity': self.equity_curve,
            'timestamp': self.timestamps
        }).set_index('timestamp')

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades as DataFrame"""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'side': t.side.value,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'size': t.size,
                'gross_pnl': t.gross_pnl,
                'fees': t.fees,
                'slippage': t.slippage,
                'net_pnl': t.net_pnl,
                'exit_reason': t.exit_reason
            }
            for t in self.trades
        ])


def test_backtester():
    """Test the backtesting framework"""
    print("=== Backtesting Framework Test ===\n")

    # Generate synthetic trending data
    np.random.seed(42)
    n = 1000

    # Trending market with noise
    trend = np.cumsum(np.random.randn(n) * 0.01 + 0.0005)
    close = 100 * np.exp(trend)

    data = pd.DataFrame({
        'open': close * 0.999,
        'high': close * (1 + np.abs(np.random.randn(n) * 0.01)),
        'low': close * (1 - np.abs(np.random.randn(n) * 0.01)),
        'close': close,
        'volume': np.random.randint(1000, 10000, n)
    }, index=pd.date_range('2023-01-01', periods=n, freq='1h'))

    # Simple momentum signal function
    def momentum_signal(data: pd.DataFrame, i: int) -> int:
        if i < 20:
            return 0

        # 20-period momentum
        ret = data.iloc[i]['close'] / data.iloc[i-20]['close'] - 1

        if ret > 0.02:
            return 1  # Long
        elif ret < -0.02:
            return -1  # Short
        else:
            return 0  # Flat

    # Run backtest
    config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.05,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        commission_rate=0.001,
        slippage_rate=0.0005
    )

    engine = BacktestEngine(config)
    metrics = engine.run(data, momentum_signal)

    print("Backtest Results:")
    print("-" * 50)
    print(f"Total Return:     {metrics.total_return:.2%}")
    print(f"Annual Return:    {metrics.annual_return:.2%}")
    print(f"Sharpe Ratio:     {metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio:    {metrics.sortino_ratio:.2f}")
    print(f"Max Drawdown:     {metrics.max_drawdown:.2%}")
    print(f"Volatility:       {metrics.volatility:.2%}")
    print()
    print(f"Total Trades:     {metrics.total_trades}")
    print(f"Win Rate:         {metrics.win_rate:.1%}")
    print(f"Profit Factor:    {metrics.profit_factor:.2f}")
    print(f"Avg Trade:        ${metrics.avg_trade:.2f}")
    print(f"Best Trade:       ${metrics.best_trade:.2f}")
    print(f"Worst Trade:      ${metrics.worst_trade:.2f}")

    # Trade breakdown
    trades_df = engine.get_trades_dataframe()
    if not trades_df.empty:
        print(f"\nTrade Exit Reasons:")
        print(trades_df['exit_reason'].value_counts().to_string())


if __name__ == "__main__":
    test_backtester()
