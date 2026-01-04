"""
Caching Layer for ERC Trading System

Provides in-memory and disk caching for:
- API responses
- Embeddings (semantic matching)
- Order book data
- Market data

Features:
- TTL-based expiration
- LRU eviction
- Async support
- Disk persistence for embeddings
"""

import time
import json
import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional, Dict, Callable
from functools import wraps
from collections import OrderedDict
from dataclasses import dataclass
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    value: Any
    created_at: float
    ttl: float
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


class CacheManager:
    """
    Central cache manager with multiple cache types.

    Usage:
        cache = CacheManager()

        # Store with TTL
        cache.set('market_data', data, ttl=30)

        # Retrieve
        data = cache.get('market_data')

        # With namespace
        cache.set('embedding:market_123', embedding, namespace='embeddings', ttl=3600)
    """

    # Default TTLs by cache type (seconds)
    DEFAULT_TTLS = {
        'market_data': 30,      # Market prices
        'order_book': 10,       # Order book (very fresh)
        'embeddings': 3600,     # Semantic embeddings (1 hour)
        'api_response': 60,     # General API responses
        'calibration': 300,     # Calibration data (5 min)
        'default': 60
    }

    def __init__(self, max_size: int = 10000,
                 disk_cache_dir: Optional[str] = None,
                 enable_disk_cache: bool = True):
        """
        Args:
            max_size: Maximum entries in memory cache
            disk_cache_dir: Directory for disk cache
            enable_disk_cache: Enable disk caching for large items
        """
        self.max_size = max_size
        self.enable_disk_cache = enable_disk_cache

        # In-memory cache (namespace -> OrderedDict)
        self._cache: Dict[str, OrderedDict] = {}
        self._lock = threading.RLock()

        # Disk cache
        if enable_disk_cache:
            self.disk_cache_dir = Path(disk_cache_dir or "data/cache")
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'disk_hits': 0
        }

    def _get_namespace_cache(self, namespace: str) -> OrderedDict:
        """Get or create cache for namespace"""
        if namespace not in self._cache:
            self._cache[namespace] = OrderedDict()
        return self._cache[namespace]

    def _make_key(self, key: str, namespace: str) -> str:
        """Create full cache key"""
        return f"{namespace}:{key}"

    def set(self, key: str, value: Any, ttl: Optional[float] = None,
            namespace: str = 'default') -> None:
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to store
            ttl: Time-to-live in seconds (uses default if None)
            namespace: Cache namespace
        """
        if ttl is None:
            ttl = self.DEFAULT_TTLS.get(namespace, self.DEFAULT_TTLS['default'])

        entry = CacheEntry(
            value=value,
            created_at=time.time(),
            ttl=ttl
        )

        with self._lock:
            cache = self._get_namespace_cache(namespace)

            # Evict if at capacity
            while len(cache) >= self.max_size:
                cache.popitem(last=False)  # Remove oldest
                self.stats['evictions'] += 1

            cache[key] = entry
            # Move to end (most recently used)
            cache.move_to_end(key)

    def get(self, key: str, namespace: str = 'default') -> Optional[Any]:
        """
        Retrieve value from cache.

        Args:
            key: Cache key
            namespace: Cache namespace

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            cache = self._get_namespace_cache(namespace)

            if key not in cache:
                self.stats['misses'] += 1

                # Try disk cache for embeddings
                if namespace == 'embeddings' and self.enable_disk_cache:
                    value = self._load_from_disk(key, namespace)
                    if value is not None:
                        self.stats['disk_hits'] += 1
                        # Restore to memory cache
                        self.set(key, value, namespace=namespace)
                        return value

                return None

            entry = cache[key]

            if entry.is_expired:
                del cache[key]
                self.stats['misses'] += 1
                return None

            # Update stats and move to end
            entry.hits += 1
            cache.move_to_end(key)
            self.stats['hits'] += 1

            return entry.value

    def get_or_set(self, key: str, factory: Callable[[], Any],
                   ttl: Optional[float] = None, namespace: str = 'default') -> Any:
        """
        Get value from cache or compute and store it.

        Args:
            key: Cache key
            factory: Function to compute value if not cached
            ttl: Time-to-live
            namespace: Cache namespace

        Returns:
            Cached or computed value
        """
        value = self.get(key, namespace)

        if value is None:
            value = factory()
            self.set(key, value, ttl, namespace)

        return value

    def invalidate(self, key: str, namespace: str = 'default') -> bool:
        """Remove specific key from cache"""
        with self._lock:
            cache = self._get_namespace_cache(namespace)
            if key in cache:
                del cache[key]
                return True
            return False

    def invalidate_namespace(self, namespace: str) -> int:
        """Clear entire namespace"""
        with self._lock:
            if namespace in self._cache:
                count = len(self._cache[namespace])
                self._cache[namespace] = OrderedDict()
                return count
            return 0

    def invalidate_pattern(self, pattern: str, namespace: str = 'default') -> int:
        """Invalidate all keys matching pattern (prefix match)"""
        with self._lock:
            cache = self._get_namespace_cache(namespace)
            keys_to_delete = [k for k in cache if k.startswith(pattern)]
            for key in keys_to_delete:
                del cache[key]
            return len(keys_to_delete)

    def clear_all(self) -> None:
        """Clear all caches"""
        with self._lock:
            self._cache = {}
            self.stats = {'hits': 0, 'misses': 0, 'evictions': 0, 'disk_hits': 0}

    # Disk cache operations
    def _get_disk_path(self, key: str, namespace: str) -> Path:
        """Get disk cache path for key"""
        # Hash the key for filesystem-safe name
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.disk_cache_dir / namespace / f"{key_hash}.pkl"

    def save_to_disk(self, key: str, value: Any, namespace: str = 'embeddings') -> bool:
        """Save value to disk cache"""
        if not self.enable_disk_cache:
            return False

        try:
            path = self._get_disk_path(key, namespace)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'wb') as f:
                pickle.dump({'key': key, 'value': value, 'time': time.time()}, f)

            return True
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")
            return False

    def _load_from_disk(self, key: str, namespace: str) -> Optional[Any]:
        """Load value from disk cache"""
        if not self.enable_disk_cache:
            return None

        try:
            path = self._get_disk_path(key, namespace)

            if not path.exists():
                return None

            with open(path, 'rb') as f:
                data = pickle.load(f)

            if data.get('key') == key:
                return data.get('value')

            return None
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
            return None

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            total_entries = sum(len(c) for c in self._cache.values())
            hit_rate = self.stats['hits'] / max(self.stats['hits'] + self.stats['misses'], 1)

            return {
                **self.stats,
                'total_entries': total_entries,
                'hit_rate': hit_rate,
                'namespaces': list(self._cache.keys())
            }


# Decorator for caching function results
def cached(ttl: Optional[float] = None, namespace: str = 'default',
           key_func: Optional[Callable] = None):
    """
    Decorator to cache function results.

    Usage:
        @cached(ttl=30, namespace='market_data')
        def fetch_market_data(market_id):
            return api.get_market(market_id)

    Args:
        ttl: Time-to-live in seconds
        namespace: Cache namespace
        key_func: Function to generate cache key from args
    """
    # Global cache instance
    _cache = CacheManager(max_size=5000)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: use function name and args
                key_parts = [func.__name__]
                key_parts.extend(str(a) for a in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ':'.join(key_parts)

            # Try cache
            result = _cache.get(cache_key, namespace)

            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            _cache.set(cache_key, result, ttl, namespace)

            return result

        # Attach cache methods to wrapper
        wrapper.invalidate = lambda *args, **kwargs: _cache.invalidate(
            key_func(*args, **kwargs) if key_func else ':'.join([func.__name__] + [str(a) for a in args]),
            namespace
        )
        wrapper.cache = _cache

        return wrapper

    return decorator


# Async cache decorator
def async_cached(ttl: Optional[float] = None, namespace: str = 'default',
                 key_func: Optional[Callable] = None):
    """Async version of cached decorator"""
    _cache = CacheManager(max_size=5000)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(a) for a in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ':'.join(key_parts)

            result = _cache.get(cache_key, namespace)

            if result is not None:
                return result

            result = await func(*args, **kwargs)
            _cache.set(cache_key, result, ttl, namespace)

            return result

        wrapper.cache = _cache
        return wrapper

    return decorator


def test_cache():
    """Test cache functionality"""
    print("=== Cache Manager Test ===\n")

    cache = CacheManager(max_size=100)

    # Basic set/get
    cache.set('test_key', {'data': 123}, ttl=5, namespace='test')
    result = cache.get('test_key', namespace='test')
    print(f"Basic set/get: {result}")

    # Get or set
    def expensive_computation():
        print("  Computing...")
        return {'computed': True}

    result = cache.get_or_set('computed_key', expensive_computation, namespace='test')
    print(f"First call (computed): {result}")

    result = cache.get_or_set('computed_key', expensive_computation, namespace='test')
    print(f"Second call (cached): {result}")

    # Stats
    print(f"\nCache stats: {cache.get_stats()}")

    # Test decorator
    @cached(ttl=10, namespace='test')
    def fetch_data(item_id):
        print(f"  Fetching {item_id}...")
        return {'id': item_id, 'value': item_id * 10}

    print("\nDecorator test:")
    print(f"First call: {fetch_data(42)}")
    print(f"Second call (cached): {fetch_data(42)}")
    print(f"Different arg: {fetch_data(43)}")


if __name__ == "__main__":
    test_cache()
