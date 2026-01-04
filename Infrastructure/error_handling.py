"""
Error Handling and Retry Logic for ERC Trading System

Provides:
- Automatic retry with exponential backoff
- Rate limiting
- Circuit breaker pattern
- Graceful degradation
"""

import time
import asyncio
import logging
from typing import Callable, Optional, List, Type, Any, Dict
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    ignore_on: List[Type[Exception]] = field(default_factory=list)


class RetryHandler:
    """
    Handles retries with exponential backoff.

    Usage:
        retry = RetryHandler(max_attempts=3, base_delay=1.0)

        @retry.wrap
        def flaky_function():
            ...
    """

    def __init__(self, config: Optional[RetryConfig] = None, **kwargs):
        if config:
            self.config = config
        else:
            self.config = RetryConfig(**kwargs)

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff"""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)

        if self.config.jitter:
            import random
            delay *= (0.5 + random.random())

        return delay

    def _should_retry(self, exception: Exception) -> bool:
        """Determine if exception should trigger retry"""
        # Check ignore list first
        for exc_type in self.config.ignore_on:
            if isinstance(exception, exc_type):
                return False

        # Check retry list
        for exc_type in self.config.retry_on:
            if isinstance(exception, exc_type):
                return True

        return False

    def wrap(self, func: Callable) -> Callable:
        """Decorator to wrap function with retry logic"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(self.config.max_attempts):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if not self._should_retry(e):
                        raise

                    if attempt < self.config.max_attempts - 1:
                        delay = self._calculate_delay(attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {self.config.max_attempts} attempts failed"
                        )

            raise last_exception

        return wrapper

    def wrap_async(self, func: Callable) -> Callable:
        """Async version of wrap"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(self.config.max_attempts):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if not self._should_retry(e):
                        raise

                    if attempt < self.config.max_attempts - 1:
                        delay = self._calculate_delay(attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {self.config.max_attempts} attempts failed"
                        )

            raise last_exception

        return wrapper


class RateLimiter:
    """
    Rate limiter using token bucket algorithm.

    Usage:
        limiter = RateLimiter(rate=10, per=1.0)  # 10 requests per second

        if limiter.acquire():
            make_request()
    """

    def __init__(self, rate: int, per: float = 1.0, burst: Optional[int] = None):
        """
        Args:
            rate: Number of requests allowed
            per: Time period in seconds
            burst: Maximum burst size (defaults to rate)
        """
        self.rate = rate
        self.per = per
        self.burst = burst or rate

        self.tokens = float(self.burst)
        self.last_update = time.time()
        self._lock = threading.Lock()

    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.burst, self.tokens + elapsed * (self.rate / self.per))
        self.last_update = now

    def acquire(self, tokens: int = 1, blocking: bool = True) -> bool:
        """
        Acquire tokens.

        Args:
            tokens: Number of tokens to acquire
            blocking: Wait if tokens not available

        Returns:
            True if tokens acquired, False otherwise
        """
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            if not blocking:
                return False

            # Calculate wait time
            wait_time = (tokens - self.tokens) * (self.per / self.rate)
            time.sleep(wait_time)

            self._refill()
            self.tokens -= tokens
            return True

    async def acquire_async(self, tokens: int = 1) -> bool:
        """Async version of acquire"""
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            wait_time = (tokens - self.tokens) * (self.per / self.rate)

        await asyncio.sleep(wait_time)

        with self._lock:
            self._refill()
            self.tokens -= tokens
            return True

    @property
    def available_tokens(self) -> float:
        """Get available tokens"""
        with self._lock:
            self._refill()
            return self.tokens


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.

    Prevents cascading failures by failing fast when
    a service is unhealthy.

    Usage:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

        @breaker.wrap
        def call_external_service():
            ...
    """

    def __init__(self, failure_threshold: int = 5,
                 recovery_timeout: float = 30.0,
                 success_threshold: int = 2):
        """
        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            success_threshold: Successes needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    def _can_attempt(self) -> bool:
        """Check if request can be attempted"""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                # Check if recovery timeout passed
                if self.last_failure_time and \
                   time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker entering HALF_OPEN state")
                    return True
                return False

            # HALF_OPEN - allow request
            return True

    def _record_success(self):
        """Record successful call"""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info("Circuit breaker CLOSED - service recovered")
            else:
                self.failure_count = 0

    def _record_failure(self):
        """Record failed call"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning("Circuit breaker OPEN - recovery failed")
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker OPEN - {self.failure_count} failures"
                )

    def wrap(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self._can_attempt():
                raise CircuitOpenError("Circuit breaker is OPEN")

            try:
                result = func(*args, **kwargs)
                self._record_success()
                return result
            except Exception as e:
                self._record_failure()
                raise

        return wrapper

    def wrap_async(self, func: Callable) -> Callable:
        """Async version of wrap"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not self._can_attempt():
                raise CircuitOpenError("Circuit breaker is OPEN")

            try:
                result = await func(*args, **kwargs)
                self._record_success()
                return result
            except Exception as e:
                self._record_failure()
                raise

        return wrapper

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


# Convenience decorator
def with_retry(max_attempts: int = 3, base_delay: float = 1.0,
               retry_on: Optional[List[Type[Exception]]] = None):
    """
    Convenience decorator for retries.

    Usage:
        @with_retry(max_attempts=3)
        def flaky_api_call():
            ...
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        retry_on=retry_on or [Exception]
    )
    handler = RetryHandler(config)

    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            return handler.wrap_async(func)
        return handler.wrap(func)

    return decorator


# API-specific rate limiters
class APIRateLimiters:
    """Collection of rate limiters for different APIs"""

    _limiters: Dict[str, RateLimiter] = {}

    @classmethod
    def get(cls, api_name: str, default_rate: int = 10) -> RateLimiter:
        """Get or create rate limiter for API"""
        if api_name not in cls._limiters:
            cls._limiters[api_name] = RateLimiter(rate=default_rate, per=1.0)
        return cls._limiters[api_name]

    @classmethod
    def configure(cls, api_name: str, rate: int, per: float = 1.0, burst: Optional[int] = None):
        """Configure rate limiter for API"""
        cls._limiters[api_name] = RateLimiter(rate=rate, per=per, burst=burst)


def test_error_handling():
    """Test error handling components"""
    print("=== Error Handling Test ===\n")

    # Test retry handler
    print("Testing RetryHandler:")
    retry = RetryHandler(max_attempts=3, base_delay=0.1)

    attempt_count = 0

    @retry.wrap
    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError(f"Attempt {attempt_count} failed")
        return "Success!"

    result = flaky_function()
    print(f"  Result after {attempt_count} attempts: {result}")

    # Test rate limiter
    print("\nTesting RateLimiter:")
    limiter = RateLimiter(rate=5, per=1.0)

    start = time.time()
    for i in range(10):
        limiter.acquire()
        print(f"  Request {i+1} at {time.time() - start:.2f}s")

    # Test circuit breaker
    print("\nTesting CircuitBreaker:")
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

    @breaker.wrap
    def failing_service():
        raise ConnectionError("Service unavailable")

    for i in range(5):
        try:
            failing_service()
        except CircuitOpenError:
            print(f"  Attempt {i+1}: Circuit OPEN - fast fail")
        except ConnectionError:
            print(f"  Attempt {i+1}: Service error (circuit {breaker.state.value})")

    print(f"\n  Waiting for recovery timeout...")
    time.sleep(1.5)

    try:
        failing_service()
    except ConnectionError:
        print(f"  Recovery attempt: Service still failing")

    print(f"  Final circuit state: {breaker.state.value}")


if __name__ == "__main__":
    test_error_handling()
