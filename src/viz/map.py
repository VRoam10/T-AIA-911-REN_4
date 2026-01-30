from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Union

import folium


@dataclass(frozen=True)
class StationPoint:
    station_id: str
    station_name: str
    city: str
    lat: float
    lon: float


def load_station_points(stations_csv: Union[str, Path]) -> Dict[str, StationPoint]:
    """Load station coordinates from a CSV.

    Expected columns: station_id, station_name, city, lat, lon
    """
    stations_path = Path(stations_csv)
    points: Dict[str, StationPoint] = {}

    with stations_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            station_id = (row.get("station_id") or "").strip()
            if not station_id:
                continue

            points[station_id] = StationPoint(
                station_id=station_id,
                station_name=(row.get("station_name") or "").strip(),
                city=(row.get("city") or "").strip(),
                lat=float(row.get("lat") or 0.0),
                lon=float(row.get("lon") or 0.0),
            )

    return points


def plot_path(
    path: Sequence[str],
    stations_csv: Union[str, Path],
    output_html: Union[str, Path],
) -> Path:
    """Render a station_id path into an interactive Folium map and save it.

    Args:
        path: Ordered list of station_id (Dijkstra output).
        stations_csv: CSV file containing station coordinates.
        output_html: Destination HTML filepath.
    """
    if not path:
        raise ValueError("path is empty")

    points = load_station_points(stations_csv)
    missing = [station_id for station_id in path if station_id not in points]
    if missing:
        raise KeyError(f"Missing station_id(s) in stations CSV: {missing!r}")

    coordinates = [
        (points[station_id].lat, points[station_id].lon) for station_id in path
    ]

    m = folium.Map(location=coordinates[0], zoom_start=6, control_scale=True)

    for idx, station_id in enumerate(path, start=1):
        p = points[station_id]
        label = p.station_name or p.city or station_id
        popup = f"{idx}. {label} ({station_id})"
        folium.Marker(location=(p.lat, p.lon), popup=popup).add_to(m)

    folium.PolyLine(locations=coordinates, color="blue", weight=5, opacity=0.8).add_to(m)

    if len(coordinates) >= 2:
        m.fit_bounds(coordinates)

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    return output_path
