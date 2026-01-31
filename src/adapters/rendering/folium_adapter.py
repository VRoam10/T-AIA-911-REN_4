"""Folium map renderer adapter.

This adapter wraps the Folium map rendering from viz/map.py with:
- Domain model input (Station objects)
- Better error handling
- Configuration injection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ...domain.errors import RenderingError
from ...domain.models import Station


@dataclass
class FoliumMapRenderer:
    """Folium-based interactive map renderer.

    This adapter implements MapRendererPort using Folium for
    generating interactive HTML maps.
    """

    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def render(
        self,
        stations: Sequence[Station],
        output_path: Path,
    ) -> Path:
        """Render a route on a map and save to file.

        Args:
            stations: Sequence of stations forming the route.
            output_path: Where to save the rendered map.

        Returns:
            Path to the generated map file.

        Raises:
            RenderingError: If rendering fails.
        """
        if not stations:
            raise RenderingError(
                "Cannot render empty route",
                output_path=str(output_path),
                renderer_type="folium",
            )

        self._logger.info(
            "Rendering route map",
            extra={
                "stations": len(stations),
                "output_path": str(output_path),
            },
        )

        try:
            import folium

            # Calculate map center
            lats = [s.location.latitude for s in stations]
            lons = [s.location.longitude for s in stations]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)

            # Create map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

            # Add markers for each station
            for i, station in enumerate(stations):
                icon_color = "green" if i == 0 else "red" if i == len(stations) - 1 else "blue"
                folium.Marker(
                    location=[station.location.latitude, station.location.longitude],
                    popup=f"{station.name} ({station.city})",
                    tooltip=station.code,
                    icon=folium.Icon(color=icon_color),
                ).add_to(m)

            # Add route line
            if len(stations) >= 2:
                route_coords = [
                    [s.location.latitude, s.location.longitude] for s in stations
                ]
                folium.PolyLine(
                    route_coords,
                    weight=3,
                    color="blue",
                    opacity=0.8,
                ).add_to(m)

            # Save map
            output_path.parent.mkdir(parents=True, exist_ok=True)
            m.save(str(output_path))

            self._logger.info(
                "Map rendered successfully",
                extra={"output_path": str(output_path)},
            )

            return output_path

        except ImportError as e:
            raise RenderingError(
                "Folium not installed",
                output_path=str(output_path),
                renderer_type="folium",
                cause=e,
            )
        except Exception as e:
            self._logger.error(
                "Map rendering failed",
                extra={"error": str(e), "output_path": str(output_path)},
            )
            raise RenderingError(
                f"Map rendering failed: {e}",
                output_path=str(output_path),
                renderer_type="folium",
                cause=e,
            )
