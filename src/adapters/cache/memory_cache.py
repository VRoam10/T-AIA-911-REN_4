"""Thread-safe in-memory cache implementation.

This cache replaces the scattered global caches throughout the codebase:
- strategies.py:64 (_GEOCODE_CACHE)
- hf_ner/ner.py:16-18 (_PIPELINES, _STATION_MAP)
- asr.py:23 (MODEL_CACHE)

Key improvements:
- Thread-safe with RLock
- Optional TTL (time-to-live)
- Explicit invalidation
- Proper logging
- Statistics tracking
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

T = TypeVar("T")


@dataclass
class InMemoryCache(Generic[T]):
    """Thread-safe in-memory cache with optional TTL.

    This cache implements the CachePort protocol and can be injected
    into adapters that need caching functionality.

    Attributes:
        default_ttl_seconds: Default time-to-live for entries (None = no expiry)
        max_size: Maximum number of entries (None = unlimited)
        name: Cache name for logging

    Example:
        cache = InMemoryCache[str](name="geocode", default_ttl_seconds=3600)
        result = cache.get_or_compute("paris", lambda: geocode("paris"))
    """

    default_ttl_seconds: Optional[float] = None
    max_size: Optional[int] = None
    name: str = "cache"

    _store: Dict[str, Tuple[Any, float]] = field(default_factory=dict, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _logger: logging.Logger = field(init=False, repr=False)

    # Statistics
    _hits: int = field(default=0, repr=False)
    _misses: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(f"cache.{self.name}")

    def get(self, key: str) -> Optional[T]:
        """Get a value from the cache.

        Args:
            key: The cache key.

        Returns:
            The cached value, or None if not found or expired.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            value, expiry = entry
            if self.default_ttl_seconds is not None and time.time() > expiry:
                # Entry has expired
                del self._store[key]
                self._logger.debug("Cache entry expired", extra={"key": key})
                self._misses += 1
                return None

            self._hits += 1
            return value

    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Set a value in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Optional TTL override for this entry.
        """
        with self._lock:
            # Check max size and evict if needed (simple FIFO eviction)
            if self.max_size is not None and len(self._store) >= self.max_size:
                if key not in self._store:
                    # Evict oldest entry
                    oldest_key = next(iter(self._store))
                    del self._store[oldest_key]
                    self._logger.debug(
                        "Cache evicted entry",
                        extra={"key": oldest_key, "reason": "max_size"},
                    )

            effective_ttl = ttl if ttl is not None else self.default_ttl_seconds
            if effective_ttl is not None:
                expiry = time.time() + effective_ttl
            else:
                expiry = float("inf")

            self._store[key] = (value, expiry)
            self._logger.debug(
                "Cache entry set",
                extra={"key": key, "ttl": effective_ttl},
            )

    def get_or_compute(self, key: str, compute_fn: Callable[[], T]) -> T:
        """Get from cache or compute and cache the value.

        This is the primary method for the cache-aside pattern.

        Args:
            key: The cache key.
            compute_fn: Function to compute the value if not cached.

        Returns:
            The cached or computed value.
        """
        # Check cache first (with lock)
        value = self.get(key)
        if value is not None:
            self._logger.debug("Cache hit", extra={"key": key})
            return value

        # Compute value (outside lock to avoid blocking)
        self._logger.debug("Cache miss, computing", extra={"key": key})
        computed = compute_fn()

        # Store in cache
        self.set(key, computed)
        return computed

    def clear(self) -> int:
        """Clear all entries from the cache.

        Returns:
            Number of entries that were cleared.
        """
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._hits = 0
            self._misses = 0
            self._logger.info("Cache cleared", extra={"entries_cleared": count})
            return count

    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry.

        Args:
            key: The cache key to invalidate.

        Returns:
            True if the key existed and was removed.
        """
        with self._lock:
            if key in self._store:
                del self._store[key]
                self._logger.debug("Cache entry invalidated", extra={"key": key})
                return True
            return False

    def size(self) -> int:
        """Return the number of entries in the cache.

        Returns:
            Current number of cached entries.
        """
        with self._lock:
            return len(self._store)

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dictionary with hit/miss counts and size.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                "size": len(self._store),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_percent": round(hit_rate, 1),
            }

    def keys(self) -> list[str]:
        """Return all keys in the cache.

        Returns:
            List of cache keys.
        """
        with self._lock:
            return list(self._store.keys())
