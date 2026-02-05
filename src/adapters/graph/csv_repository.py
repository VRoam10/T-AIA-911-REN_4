"""CSV Graph Repository adapter.

This adapter wraps the existing graph loading logic and adds:
- Configuration injection (paths from config)
- Caching support
- Station metadata loading
- Better error handling
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ...config import GraphConfig, get_config
from ...domain.errors import GraphError, StationNotFoundError
from ...domain.models import GeoLocation, Station
from ...ports.graph import Graph


@dataclass
class CSVGraphRepository:
    """Graph repository that loads from CSV files.

    This adapter implements GraphRepositoryPort and wraps the logic
    from graph/load_graph.py with additional features.

    Attributes:
        config: Graph configuration (paths, file names)
    """

    config: GraphConfig = field(default_factory=lambda: get_config().graph)
    _logger: logging.Logger = field(init=False, repr=False)

    # Cached data
    _graph: Optional[Graph] = field(default=None, repr=False)
    _stations: Optional[Dict[str, Station]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def load(self) -> Graph:
        """Load the transportation graph from CSV files.

        Returns:
            The graph as a mapping of station codes to neighbors.

        Raises:
            GraphError: If the graph cannot be loaded.
        """
        if self._graph is not None:
            return self._graph

        self._logger.debug(
            "Loading graph",
            extra={
                "stations_path": str(self.config.stations_path),
                "edges_path": str(self.config.edges_path),
            },
        )

        try:
            graph = self._load_graph_from_csv()
            self._graph = graph
            self._logger.info(
                "Graph loaded",
                extra={"nodes": len(graph)},
            )
            return graph
        except (OSError, KeyError, ValueError) as e:
            raise GraphError(
                f"Failed to load graph: {e}",
                file_path=str(self.config.stations_path),
                cause=e,
            )

    def _load_graph_from_csv(self) -> Dict[str, List[Tuple[str, float]]]:
        """Internal method to load graph from CSV files."""
        graph: Dict[str, List[Tuple[str, float]]] = {}

        # Load stations (vertices)
        with self.config.stations_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                station_id = row.get("station_id", "").strip()
                if station_id:
                    graph[station_id] = []

        # Load edges
        with self.config.edges_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                from_id = row.get("from_station_id", "").strip()
                to_id = row.get("to_station_id", "").strip()
                distance_str = row.get("distance_km", "").strip()

                if not from_id or not to_id or not distance_str:
                    continue

                distance = float(distance_str)
                graph.setdefault(from_id, []).append((to_id, distance))

        return graph

    def get_station(self, code: str) -> Optional[Station]:
        """Get station details by code.

        Args:
            code: The station code to look up.

        Returns:
            Station with full details, or None if not found.
        """
        stations = self._load_stations()
        return stations.get(code)

    def get_station_or_raise(self, code: str) -> Station:
        """Get station details by code, raising if not found.

        Args:
            code: The station code to look up.

        Returns:
            Station with full details.

        Raises:
            StationNotFoundError: If the station is not found.
        """
        station = self.get_station(code)
        if station is None:
            raise StationNotFoundError(
                f"Station not found: {code}",
                station_code=code,
            )
        return station

    def list_stations(self) -> Sequence[Station]:
        """List all stations.

        Returns:
            Sequence of all stations with their details.
        """
        stations = self._load_stations()
        return list(stations.values())

    def _load_stations(self) -> Dict[str, Station]:
        """Load station metadata from CSV."""
        if self._stations is not None:
            return self._stations

        stations: Dict[str, Station] = {}

        try:
            with self.config.stations_path.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    code = row.get("station_id", "").strip()
                    name = row.get("station_name", "").strip()
                    city = row.get("city", "").strip()
                    lat_str = row.get("lat", "").strip()
                    lon_str = row.get("lon", "").strip()

                    if not code:
                        continue

                    # Handle missing coordinates gracefully
                    try:
                        lat = float(lat_str) if lat_str else 0.0
                        lon = float(lon_str) if lon_str else 0.0
                    except ValueError:
                        lat, lon = 0.0, 0.0

                    stations[code] = Station(
                        code=code,
                        name=name or code,
                        city=city or name or code,
                        location=GeoLocation(latitude=lat, longitude=lon),
                    )

            self._stations = stations
        except OSError as e:
            self._logger.warning(
                "Failed to load station metadata",
                extra={"error": str(e)},
            )
            self._stations = {}

        return self._stations

    def clear_cache(self) -> None:
        """Clear cached graph and station data."""
        self._graph = None
        self._stations = None
        self._logger.debug("Graph cache cleared")
