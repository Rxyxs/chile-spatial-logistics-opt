"""Real driving distances from OSRM's table service, to compare with the simulated matrix.

`network_distance.simulate_road_network_matrix` multiplies straight-line
distance by a random circuity factor (1.15-1.35 inside a comuna, 1.25-1.55
across). This module asks a real road network instead: the public OSRM demo
server (router.project-osrm.org, OpenStreetMap data, car profile). The demo
server caps a table request at about 100 x 100 locations, so the matrix is
built in blocks, one request per second, and cached to disk so a rerun does
not query it again.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np

OSRM_URL = "https://router.project-osrm.org/table/v1/driving/"
BLOCK = 90


def table_url(locations: list[tuple[float, float]], sources: list[int], destinations: list[int],
              base: str = OSRM_URL) -> str:
    """OSRM wants lon,lat pairs; sources/destinations index into the coordinate list."""
    idx = sorted(set(sources) | set(destinations))
    pos = {i: k for k, i in enumerate(idx)}
    coords = ";".join(f"{locations[i][1]:.6f},{locations[i][0]:.6f}" for i in idx)
    s = ";".join(str(pos[i]) for i in sources)
    d = ";".join(str(pos[i]) for i in destinations)
    return f"{base}{coords}?sources={s}&destinations={d}&annotations=distance"


def parse_distances_km(payload: dict) -> np.ndarray:
    if payload.get("code") != "Ok":
        raise RuntimeError(f"OSRM error: {payload.get('code')} {payload.get('message', '')}")
    return np.array(payload["distances"], dtype=float) / 1000.0


def road_distance_matrix(locations: list[tuple[float, float]], cache: Path | None = None,
                         block: int = BLOCK, pause_s: float = 1.0) -> np.ndarray:
    """Full n x n driving-distance matrix in km (asymmetric: one-way streets)."""
    if cache is not None and cache.exists():
        m = np.load(cache)
        if m.shape == (len(locations), len(locations)):
            return m
    n = len(locations)
    m = np.zeros((n, n))
    chunks = [list(range(i, min(i + block, n))) for i in range(0, n, block)]
    for src in chunks:
        for dst in chunks:
            url = table_url(locations, src, dst)
            with urllib.request.urlopen(url, timeout=60) as r:
                m[np.ix_(src, dst)] = parse_distances_km(json.load(r))
            time.sleep(pause_s)
    np.fill_diagonal(m, 0.0)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, m)
    return m


def circuity(road_km: np.ndarray, straight_km: np.ndarray, min_straight_km: float = 0.5) -> np.ndarray:
    """Road / straight-line distance for every ordered pair at least `min_straight_km` apart."""
    mask = straight_km >= min_straight_km
    return road_km[mask] / straight_km[mask]
