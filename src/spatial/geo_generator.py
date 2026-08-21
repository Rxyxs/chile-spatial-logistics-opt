"""Generador de datos sinteticos de demanda para dark stores en el Gran Santiago.

Crea pedidos con coordenadas dentro de comunas seleccionadas de la Region
Metropolitana, los indexa en celdas H3 (resolucion 8) y agrega la demanda por
celda para identificar zonas de alta densidad -- candidatas a ubicacion de
Dark Stores.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import geopandas as gpd
import h3
import pandas as pd
from shapely.geometry import Point, Polygon

# Centro aproximado (lat, lon) y radio efectivo (km) de cada comuna.
COMUNAS: dict[str, dict[str, float]] = {
    "Las Condes": {"lat": -33.4089, "lon": -70.5693, "radius_km": 4.0},
    "Providencia": {"lat": -33.4314, "lon": -70.6093, "radius_km": 2.0},
    "Santiago Centro": {"lat": -33.4372, "lon": -70.6506, "radius_km": 2.5},
    "Maipu": {"lat": -33.5111, "lon": -70.7580, "radius_km": 4.5},
    "San Bernardo": {"lat": -33.5928, "lon": -70.6997, "radius_km": 3.5},
}

H3_RESOLUTION = 8

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def _random_point_in_radius(
    lat: float, lon: float, radius_km: float, rng: random.Random
) -> tuple[float, float]:
    """Punto aleatorio uniforme dentro de un circulo de `radius_km` en torno a (lat, lon)."""
    r = radius_km * math.sqrt(rng.random())
    theta = rng.uniform(0, 2 * math.pi)
    dlat = (r * math.cos(theta)) / 111.32
    dlon = (r * math.sin(theta)) / (111.32 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def generate_synthetic_demand(
    n_per_comuna: int = 40,
    seed: int = 42,
    min_demand: int = 1,
    max_demand: int = 8,
) -> gpd.GeoDataFrame:
    """Genera pedidos sinteticos distribuidos en las comunas de `COMUNAS`.

    Cada fila es un pedido individual con una coordenada dentro del radio
    urbano de su comuna y una demanda (unidades) aleatoria.
    """
    rng = random.Random(seed)
    records = []
    order_id = 0
    for comuna, params in COMUNAS.items():
        for _ in range(n_per_comuna):
            lat, lon = _random_point_in_radius(
                params["lat"], params["lon"], params["radius_km"], rng
            )
            records.append(
                {
                    "order_id": order_id,
                    "comuna": comuna,
                    "lat": lat,
                    "lon": lon,
                    "demand": rng.randint(min_demand, max_demand),
                }
            )
            order_id += 1

    df = pd.DataFrame.from_records(records)
    geometry = [Point(lon, lat) for lat, lon in zip(df["lat"], df["lon"])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def assign_h3_index(gdf: gpd.GeoDataFrame, resolution: int = H3_RESOLUTION) -> gpd.GeoDataFrame:
    """Anade la columna `h3_cell` indexando cada punto a su celda H3."""
    gdf = gdf.copy()
    gdf["h3_cell"] = [
        h3.latlng_to_cell(lat, lon, resolution) for lat, lon in zip(gdf["lat"], gdf["lon"])
    ]
    return gdf


def _h3_cell_to_polygon(cell: str) -> Polygon:
    boundary = h3.cell_to_boundary(cell)  # tupla de (lat, lon)
    return Polygon([(lon, lat) for lat, lon in boundary])


def aggregate_by_h3(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Agrega la demanda por celda H3: total de unidades, num. de pedidos y centroide."""
    agg = (
        gdf.groupby("h3_cell")
        .agg(
            total_demand=("demand", "sum"),
            order_count=("demand", "count"),
            comuna=("comuna", lambda s: s.mode().iat[0]),
        )
        .reset_index()
    )
    centroids = agg["h3_cell"].apply(h3.cell_to_latlng)
    agg["centroid_lat"] = centroids.apply(lambda c: c[0])
    agg["centroid_lon"] = centroids.apply(lambda c: c[1])
    agg["geometry"] = agg["h3_cell"].apply(_h3_cell_to_polygon)
    return gpd.GeoDataFrame(agg, geometry="geometry", crs="EPSG:4326")


def select_dark_store_candidates(h3_agg: gpd.GeoDataFrame, n_stores: int = 1) -> gpd.GeoDataFrame:
    """Selecciona las `n_stores` celdas H3 con mayor demanda como candidatas a Dark Store."""
    return h3_agg.sort_values("total_demand", ascending=False).head(n_stores).reset_index(drop=True)


def save_dataset(gdf: gpd.GeoDataFrame, h3_agg: gpd.GeoDataFrame) -> None:
    """Persiste el dataset de pedidos y la agregacion H3 en data/processed."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    gdf.drop(columns="geometry").to_csv(PROCESSED_DIR / "demand_points.csv", index=False)
    gdf.to_file(PROCESSED_DIR / "demand_points.geojson", driver="GeoJSON")
    h3_agg.to_file(PROCESSED_DIR / "h3_demand_agg.geojson", driver="GeoJSON")


def main() -> None:
    gdf = assign_h3_index(generate_synthetic_demand())
    h3_agg = aggregate_by_h3(gdf)
    candidates = select_dark_store_candidates(h3_agg, n_stores=3)

    save_dataset(gdf, h3_agg)

    print(f"Pedidos generados: {len(gdf)}")
    print(f"Celdas H3 (res {H3_RESOLUTION}) con demanda: {len(h3_agg)}")
    print("Top 3 celdas candidatas a Dark Store:")
    print(candidates[["h3_cell", "comuna", "total_demand", "order_count"]].to_string(index=False))
    print(f"\nArchivos guardados en: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
