"""Matriz de distancias por red vial simulada ("OSRM"), sobre una base de
Haversine vectorizada.

Por qué simulada y no un servidor OSRM real: levantar OSRM de verdad requiere
un extracto `.osm.pbf` de la red vial de Santiago (cientos de MB) más un
proceso de pre-procesamiento (`osrm-extract` + `osrm-contract`) -- infraestructura
pesada e innecesaria para el propósito de este proyecto (demostrar que el
motor de ruteo puede consumir *cualquier* matriz de distancias, no solo
distancia en línea recta). En su lugar, se simula lo que una consulta real de
distancia por red vial devolvería: la distancia Haversine multiplicada por un
**factor de circuidad** realista (la relación distancia-real-en-calles /
distancia-en-línea-recta), documentado abajo con sus valores de referencia.

"Haversine acelerada": el cálculo original (`vrp_solver._build_distance_matrix_km`)
usa un doble loop en Python puro (O(n²) llamadas a `math.sin/cos/asin`). Acá se
vectoriza con NumPy (broadcasting sobre todos los pares a la vez) -- ver
`benchmark_haversine_speedup` para la medición real del speedup, no solo la
afirmación de que "debería ser más rápido".
"""

from __future__ import annotations

import math
import time

import numpy as np

EARTH_RADIUS_KM = 6371.0088

# Factor de circuidad (road distance / straight-line distance) por tipo de
# viaje. La literatura de transporte urbano reporta circuidad tipica entre
# 1.2 y 1.4 para trayectos dentro de una misma zona urbana (grilla de calles,
# rotondas, calles de sentido unico); se usa un rango algo mas amplio para
# trayectos que cruzan comunas (mas probable que dependan de una arteria
# principal en vez de una ruta directa).
INTRA_ZONE_CIRCUITY_RANGE = (1.15, 1.35)
INTER_ZONE_CIRCUITY_RANGE = (1.25, 1.55)
MIN_CIRCUITY = 1.0  # la distancia por calles nunca puede ser menor que la recta


def vectorized_haversine_matrix(locations: list[tuple[float, float]]) -> np.ndarray:
    """Matriz de distancias Haversine (km) entre todos los pares de
    `locations`, calculada con NumPy vectorizado en vez de un doble loop en
    Python puro."""
    coords = np.radians(np.asarray(locations, dtype=float))
    lat, lon = coords[:, 0], coords[:, 1]

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    a = np.clip(a, 0.0, 1.0)  # errores de punto flotante pueden dar a levemente > 1
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _loop_haversine_matrix(locations: list[tuple[float, float]]) -> list[list[float]]:
    """Implementación de referencia (doble loop puro en Python) -- se usa
    solo para el test de equivalencia numérica y el benchmark de velocidad,
    no en el pipeline principal."""

    def haversine_km(lat1, lon1, lat2, lon2):
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))

    n = len(locations)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = haversine_km(*locations[i], *locations[j])
    return matrix


def benchmark_haversine_speedup(locations: list[tuple[float, float]]) -> dict:
    """Mide el speedup real de la versión vectorizada vs. el doble loop, sobre
    la misma lista de ubicaciones -- no una afirmación, un número medido."""
    start = time.perf_counter()
    _loop_haversine_matrix(locations)
    loop_seconds = time.perf_counter() - start

    start = time.perf_counter()
    vectorized_haversine_matrix(locations)
    vectorized_seconds = time.perf_counter() - start

    return {
        "n_locations": len(locations),
        "loop_seconds": loop_seconds,
        "vectorized_seconds": vectorized_seconds,
        "speedup_x": loop_seconds / vectorized_seconds if vectorized_seconds > 0 else float("inf"),
    }


def simulate_road_network_matrix(
    locations: list[tuple[float, float]],
    zone_labels: list[str] | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Simula una matriz de distancias por red vial (km), aplicando un factor
    de circuidad realista sobre la distancia Haversine vectorizada.

    Si `zone_labels` se entrega (una etiqueta de comuna/zona por ubicación),
    los pares que cruzan de zona reciben un factor de circuidad mayor
    (`INTER_ZONE_CIRCUITY_RANGE`) que los pares dentro de la misma zona
    (`INTRA_ZONE_CIRCUITY_RANGE`) -- un viaje que cruza comunas depende más de
    una arteria principal que de calles locales directas. El factor se
    sortea por par (simétrico: mismo factor para i->j que para j->i, una
    simplificación explícita -- una red vial real puede ser levemente
    asimétrica por calles de sentido único) y nunca es menor a
    `MIN_CIRCUITY` (la distancia por calles nunca es más corta que la recta).
    """
    haversine = vectorized_haversine_matrix(locations)
    n = len(locations)
    rng = np.random.default_rng(seed)

    if zone_labels is not None:
        same_zone = np.array([[a == b for b in zone_labels] for a in zone_labels])
    else:
        same_zone = np.ones((n, n), dtype=bool)

    # Solo se necesita sortear el triangulo superior (excluyendo la diagonal)
    # y reflejarlo, para garantizar simetria exacta.
    intra_low, intra_high = INTRA_ZONE_CIRCUITY_RANGE
    inter_low, inter_high = INTER_ZONE_CIRCUITY_RANGE
    random_intra = rng.uniform(intra_low, intra_high, size=(n, n))
    random_inter = rng.uniform(inter_low, inter_high, size=(n, n))
    factor_upper = np.triu(np.where(same_zone, random_intra, random_inter), k=1)
    factor = factor_upper + factor_upper.T
    np.fill_diagonal(factor, 0.0)  # la diagonal de haversine ya es 0

    factor = np.maximum(factor, MIN_CIRCUITY)
    network_km = haversine * factor
    np.fill_diagonal(network_km, 0.0)
    return network_km
