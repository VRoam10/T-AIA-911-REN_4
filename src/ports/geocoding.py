"""Geocoding port - Abstraction for location validation and coordinates.

This protocol defines the contract for geocoding services, allowing
different implementations (Nominatim, Google Maps, etc.) to be used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from ..domain.models import City, GeoLocation


class GeocoderPort(Protocol):
    """Port for geocoding services.

    Replaces: strategies.py:82-98 (_cached_geocode)
    Implementation: adapters/geocoding/nominatim_adapter.py

    Geocoding converts location names to coordinates and validates
    that locations are actual cities/towns.
    """

    def geocode(self, query: str, language: str = "fr") -> Optional[City]:
        """Geocode a location query to coordinates and metadata.

        Args:
            query: The location name to geocode (e.g., "Paris", "Rennes").
            language: Language code for results (default: French).

        Returns:
            City with coordinates and metadata, or None if not found.
        """
        ...

    def reverse_geocode(self, location: GeoLocation) -> Optional[City]:
        """Reverse geocode coordinates to city information.

        Args:
            location: GPS coordinates to look up.

        Returns:
            City information for the coordinates, or None if not found.
        """
        ...
