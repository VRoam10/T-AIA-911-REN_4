"""Null cache implementation for testing.

This cache always misses, ensuring that tests don't accidentally
depend on cached state from previous tests. Use this cache in
test fixtures to ensure isolation.

Example:
    @pytest.fixture
    def test_container(null_cache):
        container = Container()
        container.register(CachePort, lambda: null_cache)
        return container
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class NullCache(Generic[T]):
    """No-op cache for testing - always misses.

    This cache implements the CachePort protocol but never actually
    caches anything. Every get() returns None, every get_or_compute()
    calls the compute function.

    This is useful for:
    - Testing without cache pollution between tests
    - Debugging to rule out caching issues
    - Development when you want fresh computations
    """

    name: str = "null"

    def get(self, key: str) -> Optional[T]:
        """Always returns None (cache miss).

        Args:
            key: The cache key (ignored).

        Returns:
            Always None.
        """
        return None

    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Does nothing.

        Args:
            key: The cache key (ignored).
            value: The value (ignored).
            ttl: Optional TTL (ignored).
        """
        pass

    def get_or_compute(self, key: str, compute_fn: Callable[[], T]) -> T:
        """Always calls compute_fn.

        Args:
            key: The cache key (ignored).
            compute_fn: Function to compute the value.

        Returns:
            The computed value (never cached).
        """
        return compute_fn()

    def clear(self) -> int:
        """Does nothing, returns 0.

        Returns:
            Always 0.
        """
        return 0

    def invalidate(self, key: str) -> bool:
        """Does nothing, returns False.

        Args:
            key: The cache key (ignored).

        Returns:
            Always False.
        """
        return False

    def size(self) -> int:
        """Always returns 0.

        Returns:
            Always 0.
        """
        return 0

    def stats(self) -> Dict[str, int]:
        """Return empty stats.

        Returns:
            Dictionary with all zeros.
        """
        return {
            "size": 0,
            "hits": 0,
            "misses": 0,
            "hit_rate_percent": 0,
        }

    def keys(self) -> list[str]:
        """Return empty list.

        Returns:
            Empty list.
        """
        return []
