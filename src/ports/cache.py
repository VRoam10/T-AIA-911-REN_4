"""Cache port - Injectable caching abstraction.

This protocol defines the contract for caching, replacing the scattered
global caches throughout the codebase with a unified, testable interface.

Replaces:
- strategies.py:64 (_GEOCODE_CACHE)
- hf_ner/ner.py:16-18 (_PIPELINES, _STATION_MAP)
- asr.py:23 (MODEL_CACHE)
"""

from __future__ import annotations

from typing import Callable, Generic, Optional, Protocol, TypeVar

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class CachePort(Protocol[T]):
    """Port for caching.

    Implementations:
    - adapters/cache/memory_cache.py (InMemoryCache) - Production
    - adapters/cache/null_cache.py (NullCache) - Testing

    The cache port allows dependency injection of caching behavior,
    making it easy to disable caching in tests or swap implementations.
    """

    def get(self, key: str) -> Optional[T]:
        """Get a value from the cache.

        Args:
            key: The cache key.

        Returns:
            The cached value, or None if not found or expired.
        """
        ...

    def set(self, key: str, value: T) -> None:
        """Set a value in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
        """
        ...

    def get_or_compute(self, key: str, compute_fn: Callable[[], T]) -> T:
        """Get from cache or compute and cache the value.

        This is the primary method for cache-aside pattern:
        1. Check if key exists in cache
        2. If yes, return cached value
        3. If no, call compute_fn, cache result, return result

        Args:
            key: The cache key.
            compute_fn: Function to compute the value if not cached.

        Returns:
            The cached or computed value.
        """
        ...

    def clear(self) -> int:
        """Clear all entries from the cache.

        Returns:
            Number of entries that were cleared.
        """
        ...

    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry.

        Args:
            key: The cache key to invalidate.

        Returns:
            True if the key existed and was removed, False otherwise.
        """
        ...

    def size(self) -> int:
        """Return the number of entries in the cache.

        Returns:
            Current number of cached entries.
        """
        ...
