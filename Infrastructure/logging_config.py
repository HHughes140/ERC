"""
Structured Logging Framework for ERC Trading System

Provides:
- Structured JSON logging for production
- Pretty console logging for development
- Automatic context injection (trade_id, market_id, etc.)
- Performance timing
- Audit trail for trades
"""

import logging
import json
import sys
import time
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
from functools import wraps
import traceback
from contextlib import contextmanager


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def __init__(self, include_stack: bool = False):
        super().__init__()
        self.include_stack = include_stack

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data['data'] = record.extra_data

        # Add exception info
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
            }
            if self.include_stack:
                log_data['exception']['traceback'] = traceback.format_exception(*record.exc_info)

        return json.dumps(log_data)


class PrettyFormatter(logging.Formatter):
    """Pretty formatter for console output"""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Timestamp
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

        # Build message
        msg = f"{color}{timestamp} [{record.levelname:^8}]{reset} "
        msg += f"\033[90m{record.name}:{record.funcName}:{record.lineno}\033[0m "
        msg += record.getMessage()

        # Add extra data if present
        if hasattr(record, 'extra_data') and record.extra_data:
            data_str = ' '.join(f"{k}={v}" for k, v in record.extra_data.items())
            msg += f" \033[90m| {data_str}\033[0m"

        # Add exception
        if record.exc_info:
            msg += f"\n{color}{self.formatException(record.exc_info)}{reset}"

        return msg


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that automatically includes context"""

    def __init__(self, logger: logging.Logger, context: Optional[Dict] = None):
        super().__init__(logger, context or {})

    def process(self, msg, kwargs):
        # Merge context with extra data
        extra = kwargs.get('extra', {})
        extra_data = {**self.extra, **extra.get('extra_data', {})}

        if extra_data:
            kwargs['extra'] = {'extra_data': extra_data}

        return msg, kwargs

    def with_context(self, **context) -> 'ContextLogger':
        """Create new logger with additional context"""
        new_context = {**self.extra, **context}
        return ContextLogger(self.logger, new_context)


class AuditLogger:
    """
    Special logger for audit trail of trades and important events.

    Writes to both file and database for compliance.
    """

    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('audit')

    def log_trade(self, trade_type: str, market_id: str, side: str,
                  price: float, quantity: float, **kwargs):
        """Log trade execution"""
        self.logger.info(
            f"TRADE: {trade_type} {side} {quantity}@{price}",
            extra={'extra_data': {
                'event': 'trade',
                'trade_type': trade_type,
                'market_id': market_id,
                'side': side,
                'price': price,
                'quantity': quantity,
                **kwargs
            }}
        )

    def log_opportunity(self, opp_type: str, market_id: str,
                        profit_pct: float, action: str, **kwargs):
        """Log opportunity detection"""
        self.logger.info(
            f"OPPORTUNITY: {opp_type} {profit_pct:.2f}% -> {action}",
            extra={'extra_data': {
                'event': 'opportunity',
                'opp_type': opp_type,
                'market_id': market_id,
                'profit_pct': profit_pct,
                'action': action,
                **kwargs
            }}
        )

    def log_risk_event(self, event_type: str, details: str, **kwargs):
        """Log risk management events"""
        self.logger.warning(
            f"RISK: {event_type} - {details}",
            extra={'extra_data': {
                'event': 'risk',
                'event_type': event_type,
                'details': details,
                **kwargs
            }}
        )


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    json_format: bool = False,
    include_stack: bool = False
) -> None:
    """
    Setup logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        json_format: Use JSON format (for production)
        include_stack: Include stack traces in JSON
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    if json_format:
        console_handler.setFormatter(JSONFormatter(include_stack))
    else:
        console_handler.setFormatter(PrettyFormatter())

    root_logger.addHandler(console_handler)

    # File handler (always JSON for parsing)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter(include_stack=True))

        root_logger.addHandler(file_handler)

    # Set levels for noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)


def get_logger(name: str, **context) -> ContextLogger:
    """
    Get a logger with optional context.

    Usage:
        logger = get_logger('arbitrage.scanner', market_id='abc123')
        logger.info("Scanning market")  # Automatically includes market_id
    """
    base_logger = logging.getLogger(name)
    return ContextLogger(base_logger, context)


@contextmanager
def log_timing(logger: logging.Logger, operation: str, **context):
    """
    Context manager to log operation timing.

    Usage:
        with log_timing(logger, "fetch_markets"):
            markets = await fetch_markets()
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.debug(
            f"{operation} completed",
            extra={'extra_data': {
                'operation': operation,
                'elapsed_ms': round(elapsed * 1000, 2),
                **context
            }}
        )


def log_function_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function calls with timing.

    Usage:
        @log_function_call()
        def process_market(market_id):
            ...
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            start = time.time()

            logger.debug(f"Calling {func_name}", extra={'extra_data': {
                'function': func_name,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }})

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start

                logger.debug(f"{func_name} completed", extra={'extra_data': {
                    'function': func_name,
                    'elapsed_ms': round(elapsed * 1000, 2),
                    'success': True
                }})

                return result

            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"{func_name} failed: {e}", extra={'extra_data': {
                    'function': func_name,
                    'elapsed_ms': round(elapsed * 1000, 2),
                    'error': str(e)
                }}, exc_info=True)
                raise

        return wrapper
    return decorator


def test_logging():
    """Test logging configuration"""
    print("=== Logging Framework Test ===\n")

    # Setup logging
    setup_logging(level='DEBUG', json_format=False)

    # Get loggers
    logger = get_logger('test.module', user='test_user')

    # Test different levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Test with extra data
    logger.info("Processing market", extra={'extra_data': {'market_id': 'abc123', 'price': 0.95}})

    # Test context inheritance
    trade_logger = logger.with_context(trade_id='T001')
    trade_logger.info("Trade executed")

    # Test timing
    with log_timing(logger.logger, "test_operation"):
        time.sleep(0.1)

    # Test decorator
    @log_function_call()
    def test_func(x, y):
        return x + y

    result = test_func(1, 2)
    print(f"\nFunction result: {result}")

    # Test JSON format
    print("\n--- JSON Format ---")
    setup_logging(level='INFO', json_format=True)
    logger = get_logger('json.test')
    logger.info("JSON formatted message", extra={'extra_data': {'key': 'value'}})


if __name__ == "__main__":
    test_logging()
