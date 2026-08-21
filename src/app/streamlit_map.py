"""Dashboard interactivo (Streamlit + Folium) para visualizar la demanda por
celda H3 en Santiago y las rutas de reparto optimizadas desde el Dark Store
seleccionado.

Ejecutar desde la raiz del repositorio con:
    streamlit run src/app/streamlit_map.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Asegura que la raiz del repo este en sys.path sin importar como Streamlit
# invoque este script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import folium
import streamlit as st
from streamlit_folium import st_folium

from src.optimization.vrp_solver import build_instance_from_h3, solve_vrp
from src.spatial.geo_generator import (
    H3_RESOLUTION,
    aggregate_by_h3,
    assign_h3_index,
    generate_synthetic_demand,
    select_dark_store_candidates,
)

SANTIAGO_CENTER = (-33.45, -70.65)
VEHICLE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#9a6324",
]

st.set_page_config(page_title="Chile Spatial Logistics Optimizer", layout="wide")


@st.cache_data
def load_demand(n_per_comuna: int, seed: int):
    gdf = assign_h3_index(generate_synthetic_demand(n_per_comuna=n_per_comuna, seed=seed))
    h3_agg = aggregate_by_h3(gdf)
    return gdf, h3_agg


def demand_color(value: float, vmin: float, vmax: float) -> str:
    ratio = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)
    r = 255
    g = int(255 * (1 - ratio))
    b = 60
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_h3_layer(fmap: folium.Map, h3_agg) -> None:
    vmin, vmax = h3_agg["total_demand"].min(), h3_agg["total_demand"].max()
    for _, row in h3_agg.iterrows():
        boundary = [(lat, lon) for lon, lat in row["geometry"].exterior.coords]
        folium.Polygon(
            locations=boundary,
            color="#333333",
            weight=1,
            fill=True,
            fill_color=demand_color(row["total_demand"], vmin, vmax),
            fill_opacity=0.55,
            tooltip=(
                f"Celda H3: {row['h3_cell']}<br>Comuna: {row['comuna']}<br>"
                f"Demanda total: {row['total_demand']}<br>Pedidos: {row['order_count']}"
            ),
        ).add_to(fmap)


def draw_depot(fmap: folium.Map, depot: tuple[float, float]) -> None:
    folium.Marker(
        location=depot,
        popup="Dark Store",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(fmap)


def draw_routes(fmap: folium.Map, solution, locations: list[tuple[float, float]]) -> None:
    for route in solution.routes:
        color = VEHICLE_COLORS[route.vehicle_id % len(VEHICLE_COLORS)]
        points = [locations[i] for i in route.stops]
        folium.PolyLine(
            points,
            color=color,
            weight=4,
            opacity=0.85,
            tooltip=f"Vehiculo {route.vehicle_id} · {route.distance_km} km · carga {route.load}",
        ).add_to(fmap)
        for stop_idx in route.stops[1:-1]:
            folium.CircleMarker(
                location=locations[stop_idx],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=1.0,
            ).add_to(fmap)


def main() -> None:
    st.title("🚚 Chile Spatial Logistics Optimizer")
    st.caption(
        "Optimizacion de ubicacion de Dark Stores y rutas de ultima milla — "
        "Region Metropolitana de Santiago"
    )

    with st.sidebar:
        st.header("Parametros")
        n_per_comuna = st.slider("Pedidos por comuna", 10, 100, 40, step=5)
        seed = st.number_input("Semilla aleatoria", value=42, step=1)
        n_dark_stores = st.slider("Candidatos a Dark Store", 1, 5, 1)
        num_vehicles = st.slider("Numero de furgones", 1, 15, 8)
        vehicle_capacity = st.slider("Capacidad por furgon (unidades)", 20, 300, 150, step=10)
        run_optimization = st.button("🧭 Optimizar rutas", type="primary")

    gdf, h3_agg = load_demand(n_per_comuna, seed)

    st.subheader(f"Demanda agregada · {len(h3_agg)} celdas H3 (resolucion {H3_RESOLUTION})")

    col_map, col_metrics = st.columns([3, 1])

    fmap = folium.Map(location=SANTIAGO_CENTER, zoom_start=11, tiles="cartodbpositron")
    draw_h3_layer(fmap, h3_agg)

    candidates = select_dark_store_candidates(h3_agg, n_stores=n_dark_stores)
    depot_row = candidates.iloc[0]
    depot = (depot_row["centroid_lat"], depot_row["centroid_lon"])
    draw_depot(fmap, depot)

    solution = None
    if run_optimization:
        demand_cells = h3_agg[h3_agg["h3_cell"] != depot_row["h3_cell"]].reset_index(drop=True)
        instance = build_instance_from_h3(demand_cells, depot, num_vehicles, int(vehicle_capacity))
        with st.spinner("Resolviendo VRP con OR-Tools..."):
            solution = solve_vrp(instance)
        if solution is not None:
            draw_routes(fmap, solution, instance.locations)
        else:
            st.error("No se encontro una solucion factible con los parametros actuales.")

    with col_map:
        st_folium(fmap, height=600, returned_objects=[])

    with col_metrics:
        st.metric("Pedidos totales", len(gdf))
        st.metric("Demanda total (unidades)", int(h3_agg["total_demand"].sum()))
        st.metric("Celdas candidatas a Dark Store", n_dark_stores)
        if solution is not None:
            st.metric("Distancia total ruteo", f"{solution.total_distance_km} km")
            st.metric("Furgones utilizados", len(solution.routes))
            if solution.dropped:
                st.warning(
                    f"{len(solution.dropped)} celdas sin asignar "
                    "(revisar capacidad/ventanas de tiempo)."
                )

    st.subheader("Demanda total por comuna")
    st.bar_chart(gdf.groupby("comuna")["demand"].sum())


if __name__ == "__main__":
    main()
