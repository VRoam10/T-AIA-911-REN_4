"""Geocoding adapters - Implementations of GeocoderPort.

Available implementations:
- NominatimGeocoderAdapter: OpenStreetMap Nominatim geocoding
"""

from .nominatim_adapter import NominatimGeocoderAdapter

__all__ = ["NominatimGeocoderAdapter"]
