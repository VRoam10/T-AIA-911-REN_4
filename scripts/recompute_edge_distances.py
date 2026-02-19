#!/usr/bin/env python3
"""Recompute graph edge weights from real road routing distances.

This script keeps the current graph topology from data/edges.csv and replaces
`distance_km` by routing distances computed from station coordinates in
data/stations.csv.

Provider:
- OSRM public API (`router.project-osrm.org`) by default.

Usage example:
    python scripts/recompute_edge_distances.py \
      --stations data/stations.csv \
      --edges data/edges.csv \
      --output data/edges.real.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


OSRM_BASE_URL = "https://router.project-osrm.org/route/v1/driving/"


@dataclass(frozen=True)
class StationCoords:
    lat: float
    lon: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute edge distance_km values using road routing between station "
            "coordinates."
        )
    )
    parser.add_argument(
        "--stations",
        type=Path,
        default=Path("data/stations.csv"),
        help="Path to stations.csv",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path("data/edges.csv"),
        help="Path to existing edges.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/edges.real.csv"),
        help="Path to output CSV with updated distance_km values",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(".cache/edge_distances_osrm.json"),
        help="Persistent request cache file (for resume/retry)",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.05,
        help="Delay between HTTP requests to avoid rate limiting",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=10.0,
        help="HTTP timeout per request",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per failed request",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N edges (0 = all). Useful for quick validation.",
    )
    parser.add_argument(
        "--overwrite-input",
        action="store_true",
        help="Overwrite --edges file in-place instead of writing to --output.",
    )
    parser.add_argument(
        "--keep-old-on-failure",
        action="store_true",
        help="If routing fails, keep original distance_km instead of skipping edge.",
    )
    return parser.parse_args()


def load_stations(path: Path) -> Dict[str, StationCoords]:
    stations: Dict[str, StationCoords] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            station_id = (row.get("station_id") or "").strip()
            lat_raw = (row.get("lat") or "").strip()
            lon_raw = (row.get("lon") or "").strip()
            if not station_id or not lat_raw or not lon_raw:
                continue
            stations[station_id] = StationCoords(lat=float(lat_raw), lon=float(lon_raw))
    return stations


def load_edges(path: Path) -> List[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_cache(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): float(v) for k, v in data.items()}


def save_cache(path: Path, cache: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def route_distance_osrm_km(
    a: StationCoords,
    b: StationCoords,
    timeout_seconds: float,
) -> float:
    # OSRM expects lon,lat coordinate order.
    coords = f"{a.lon},{a.lat};{b.lon},{b.lat}"
    query = urllib.parse.urlencode({"overview": "false"})
    url = f"{OSRM_BASE_URL}{coords}?{query}"

    request = urllib.request.Request(
        url=url,
        headers={"User-Agent": "travel-order-resolver/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))

    routes = payload.get("routes") or []
    if not routes:
        raise RuntimeError(f"No route found. Response code={payload.get('code')}")

    distance_m = float(routes[0]["distance"])
    return round(distance_m / 1000.0, 3)


def key_for_edge(from_id: str, to_id: str) -> str:
    return f"{from_id}->{to_id}"


def compute_distance_with_retry(
    from_id: str,
    to_id: str,
    stations: Dict[str, StationCoords],
    timeout_seconds: float,
    max_retries: int,
    sleep_seconds: float,
) -> float:
    if from_id not in stations:
        raise KeyError(f"Missing station coordinates for from_station_id={from_id}")
    if to_id not in stations:
        raise KeyError(f"Missing station coordinates for to_station_id={to_id}")

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return route_distance_osrm_km(
                stations[from_id],
                stations[to_id],
                timeout_seconds=timeout_seconds,
            )
        except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(sleep_seconds * attempt)
            else:
                break

    assert last_error is not None
    raise last_error


def write_edges(path: Path, rows: List[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    stations = load_stations(args.stations)
    edges = load_edges(args.edges)
    cache = load_cache(args.cache)

    total = len(edges)
    if args.limit and args.limit > 0:
        edges = edges[: args.limit]

    processed = 0
    updated = 0
    failures = 0

    out_rows: List[dict] = []

    for row in edges:
        from_id = (row.get("from_station_id") or "").strip()
        to_id = (row.get("to_station_id") or "").strip()
        old_distance = (row.get("distance_km") or "").strip()
        if not from_id or not to_id:
            failures += 1
            continue

        edge_key = key_for_edge(from_id, to_id)
        if edge_key in cache:
            new_distance = cache[edge_key]
        else:
            try:
                new_distance = compute_distance_with_retry(
                    from_id=from_id,
                    to_id=to_id,
                    stations=stations,
                    timeout_seconds=args.timeout_seconds,
                    max_retries=args.max_retries,
                    sleep_seconds=args.sleep_seconds,
                )
                cache[edge_key] = new_distance
                save_cache(args.cache, cache)
                time.sleep(args.sleep_seconds)
            except Exception as exc:  # pragma: no cover - integration behavior
                failures += 1
                if args.keep_old_on_failure and old_distance:
                    out = dict(row)
                    out["distance_km"] = old_distance
                    out_rows.append(out)
                print(f"[WARN] Failed {edge_key}: {exc}")
                processed += 1
                if processed % 100 == 0:
                    print(
                        f"[PROGRESS] {processed}/{len(edges)} processed "
                        f"(updated={updated}, failures={failures})"
                    )
                continue

        out = dict(row)
        out["distance_km"] = f"{new_distance:.3f}"
        out_rows.append(out)
        updated += 1
        processed += 1

        if processed % 100 == 0:
            print(
                f"[PROGRESS] {processed}/{len(edges)} processed "
                f"(updated={updated}, failures={failures})"
            )

    target = args.edges if args.overwrite_input else args.output
    write_edges(target, out_rows)

    print("")
    print("[SUMMARY]")
    print(f"- stations loaded: {len(stations)}")
    print(f"- edges total in input: {total}")
    print(f"- edges processed this run: {len(edges)}")
    print(f"- edges updated: {updated}")
    print(f"- failures: {failures}")
    print(f"- cache entries: {len(cache)}")
    print(f"- written file: {target}")

    return 0 if updated > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

