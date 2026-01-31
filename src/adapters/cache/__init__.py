"""Cache adapters - Implementations of the CachePort.

Available implementations:
- InMemoryCache: Thread-safe in-memory cache with optional TTL
- NullCache: No-op cache for testing (always misses)
"""

from .memory_cache import InMemoryCache
from .null_cache import NullCache

__all__ = ["InMemoryCache", "NullCache"]
