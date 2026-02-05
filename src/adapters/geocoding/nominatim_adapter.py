"""Nominatim geocoder adapter.

This adapter wraps the Nominatim geocoding from strategies.py with:
- Proper caching via CachePort
- Configuration injection
- Better error handling with logging
- Rate limiting

Replaces: strategies.py:48-98 (geocoder initialization and _cached_geocode)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

from ...config import GeocodingConfig, get_config
from ...domain.errors import GeocodingError
from ...domain.models import City, GeoLocation
from ...ports.cache import CachePort
from ..cache.memory_cache import InMemoryCache


@dataclass
class NominatimGeocoderAdapter:
    """Nominatim geocoder adapter with caching and rate limiting.

    This adapter implements GeocoderPort using OpenStreetMap's Nominatim
    geocoding service.

    Attributes:
        config: Geocoding configuration
        cache: Cache for geocoding results
    """

    config: GeocodingConfig = field(default_factory=lambda: get_config().geocoding)
    cache: CachePort[Optional[City]] = field(
        default_factory=lambda: InMemoryCache(name="geocode")
    )

    _geolocator: Optional[Nominatim] = field(default=None, repr=False)
    _geocode_fn: Optional[Any] = field(default=None, repr=False)
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def _get_geocoder(self) -> Any:
        """Get or initialize the geocoder with rate limiting."""
        if self._geocode_fn is not None:
            return self._geocode_fn

        self._logger.debug(
            "Initializing Nominatim geocoder",
            extra={
                "user_agent": self.config.user_agent,
                "timeout": self.config.timeout_seconds,
            },
        )

        self._geolocator = Nominatim(
            user_agent=self.config.user_agent,
            timeout=self.config.timeout_seconds,
        )

        self._geocode_fn = RateLimiter(
            self._geolocator.geocode,
            min_delay_seconds=self.config.rate_limit_delay,
            max_retries=self.config.max_retries,
            error_wait_seconds=self.config.error_wait_seconds,
            swallow_exceptions=True,  # Prevents crashes, we handle None
        )

        return self._geocode_fn

    def geocode(self, query: str, language: str = "fr") -> Optional[City]:
        """Geocode a location query.

        Args:
            query: The location name to geocode.
            language: Language code for results.

        Returns:
            City with coordinates and metadata, or None if not found.
        """
        if not query or not query.strip():
            return None

        cache_key = f"{query.strip().lower()}:{language}"

        # Check cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            self._logger.debug("Geocode cache hit", extra={"query": query})
            return cached

        # Try to geocode
        try:
            geocode_fn = self._get_geocoder()
            location = geocode_fn(query, language=language, addressdetails=True)

            if location is None:
                self._logger.debug(
                    "Geocode returned no result",
                    extra={"query": query},
                )
                # Cache the miss to avoid repeated lookups
                self.cache.set(cache_key, None)
                return None

            # Extract city information from address
            address = location.raw.get("address", {})
            city_name = (
                address.get("city")
                or address.get("municipality")
                or address.get("town")
                or address.get("village")
            )

            if not city_name:
                self._logger.debug(
                    "Geocode result is not a city",
                    extra={"query": query, "address": address},
                )
                self.cache.set(cache_key, None)
                return None

            city = City(
                name=str(city_name),
                location=GeoLocation(
                    latitude=float(location.latitude),
                    longitude=float(location.longitude),
                ),
                country=address.get("country", ""),
                admin_area=address.get("state") or address.get("region"),
            )

            self._logger.debug(
                "Geocode success",
                extra={"query": query, "city": city.name},
            )

            self.cache.set(cache_key, city)
            return city

        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            self._logger.warning(
                "Geocode service error",
                extra={"query": query, "error": str(e)},
            )
            return None
        except Exception as e:
            self._logger.error(
                "Geocode unexpected error",
                extra={"query": query, "error": str(e)},
            )
            return None

    def reverse_geocode(self, location: GeoLocation) -> Optional[City]:
        """Reverse geocode coordinates to city information.

        Args:
            location: GPS coordinates to look up.

        Returns:
            City information for the coordinates, or None if not found.
        """
        if self._geolocator is None:
            self._get_geocoder()

        try:
            result = self._geolocator.reverse(  # type: ignore
                (location.latitude, location.longitude),
                language="fr",
                addressdetails=True,
            )

            if result is None:
                return None

            address = result.raw.get("address", {})
            city_name = (
                address.get("city")
                or address.get("municipality")
                or address.get("town")
                or address.get("village")
            )

            if not city_name:
                return None

            return City(
                name=str(city_name),
                location=location,
                country=address.get("country", ""),
                admin_area=address.get("state") or address.get("region"),
            )

        except Exception as e:
            self._logger.warning(
                "Reverse geocode failed",
                extra={
                    "lat": location.latitude,
                    "lon": location.longitude,
                    "error": str(e),
                },
            )
            return None
