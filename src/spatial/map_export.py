"""Exportadores de mapas HTML interactivos (Folium y Pydeck) para las rutas
óptimas de un `MultiDepotVRPSolution`, coloreadas por depósito.

Ambos exportadores son standalone (no dependen de Streamlit): generan un
archivo `.html` autocontenido que se puede abrir directamente en un navegador
o embeber en un notebook.
"""

from __future__ import annotations

from pathlib import Path

import folium
import pydeck as pdk

# Una familia de color por depósito (no por vehículo individual): el punto de
# esta paleta es visualizar el *balance de carga entre depósitos*, la
# pregunta que motiva el multi-depósito en primer lugar -- todas las rutas
# que salen del mismo depósito comparten color.
DEPOT_COLORS_HEX = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6"]
DEPOT_COLORS_RGB = [
    (230, 25, 75), (60, 180, 75), (67, 99, 216), (245, 130, 49), (145, 30, 180), (66, 212, 244), (240, 50, 230),
]


def depot_color_index(depot_index: int, depot_indices: list[int]) -> int:
    return depot_indices.index(depot_index) % len(DEPOT_COLORS_HEX)


def export_folium_map(
    h3_agg,
    depots: list[tuple[float, float]],
    depot_labels: list[str],
    solution,
    locations: list[tuple[float, float]],
    output_path: str | Path,
    map_center: tuple[float, float] | None = None,
) -> Path:
    """Renderiza la demanda por celda H3, un marcador por depósito, y cada
    ruta coloreada según su depósito de origen, y guarda el resultado como un
    archivo HTML standalone."""
    depot_indices = list(range(len(depots)))
    center = map_center or (
        sum(d[0] for d in depots) / len(depots),
        sum(d[1] for d in depots) / len(depots),
    )

    fmap = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")

    vmin, vmax = h3_agg["total_demand"].min(), h3_agg["total_demand"].max()
    for _, row in h3_agg.iterrows():
        boundary = [(lat, lon) for lon, lat in row["geometry"].exterior.coords]
        ratio = 0.0 if vmax == vmin else (row["total_demand"] - vmin) / (vmax - vmin)
        fill = f"#{255:02x}{int(255 * (1 - ratio)):02x}{60:02x}"
        folium.Polygon(
            locations=boundary, color="#333333", weight=1, fill=True,
            fill_color=fill, fill_opacity=0.5,
            tooltip=f"Comuna: {row.get('comuna', '?')}<br>Demanda: {row['total_demand']}<br>Pedidos: {row['order_count']}",
        ).add_to(fmap)

    for depot_idx, (depot, label) in enumerate(zip(depots, depot_labels)):
        color_hex = DEPOT_COLORS_HEX[depot_color_index(depot_idx, depot_indices)]
        folium.Marker(
            location=depot, popup=f"Dark Store: {label}",
            icon=folium.Icon(color="black", icon="home", prefix="fa"),
        ).add_to(fmap)
        folium.CircleMarker(
            location=depot, radius=10, color=color_hex, fill=True, fill_opacity=0.9,
            tooltip=f"Deposito {label}",
        ).add_to(fmap)

    for route in solution.routes:
        color_hex = DEPOT_COLORS_HEX[depot_color_index(route.depot_index, depot_indices)]
        points = [locations[i] for i in route.stops]
        folium.PolyLine(
            points, color=color_hex, weight=4, opacity=0.85,
            tooltip=f"Vehiculo {route.vehicle_id} (deposito {depot_labels[depot_indices.index(route.depot_index)]}) "
                    f"· {route.distance_km} km · carga {route.load}",
        ).add_to(fmap)
        for stop_idx in route.stops[1:-1]:
            folium.CircleMarker(
                location=locations[stop_idx], radius=4, color=color_hex, fill=True, fill_opacity=1.0,
            ).add_to(fmap)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_path))
    return output_path


def export_pydeck_map(
    depots: list[tuple[float, float]],
    depot_labels: list[str],
    solution,
    locations: list[tuple[float, float]],
    output_path: str | Path,
    map_center: tuple[float, float] | None = None,
) -> Path:
    """Visualización alternativa con Pydeck: un `ArcLayer` por tramo de ruta
    (codificado por color de depósito, con altura proporcional a la
    distancia del tramo) más un `ScatterplotLayer` para los depósitos --
    guarda un HTML standalone con `deck.to_html`."""
    depot_indices = list(range(len(depots)))
    center = map_center or (
        sum(d[0] for d in depots) / len(depots),
        sum(d[1] for d in depots) / len(depots),
    )

    arcs = []
    for route in solution.routes:
        color = DEPOT_COLORS_RGB[depot_color_index(route.depot_index, depot_indices)]
        for a, b in zip(route.stops[:-1], route.stops[1:]):
            lat_a, lon_a = locations[a]
            lat_b, lon_b = locations[b]
            arcs.append({
                "source": [lon_a, lat_a], "target": [lon_b, lat_b],
                "color": list(color), "vehicle_id": route.vehicle_id,
                "depot": depot_labels[depot_indices.index(route.depot_index)],
            })

    depot_points = [
        {"position": [lon, lat], "label": label, "color": DEPOT_COLORS_RGB[depot_color_index(i, depot_indices)]}
        for i, ((lat, lon), label) in enumerate(zip(depots, depot_labels))
    ]

    arc_layer = pdk.Layer(
        "ArcLayer", data=arcs, get_source_position="source", get_target_position="target",
        get_source_color="color", get_target_color="color", get_width=3, pickable=True,
    )
    depot_layer = pdk.Layer(
        "ScatterplotLayer", data=depot_points, get_position="position", get_fill_color="color",
        get_radius=180, pickable=True,
    )
    view_state = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=10.5, pitch=45)
    deck = pdk.Deck(
        layers=[arc_layer, depot_layer], initial_view_state=view_state,
        map_style="light",
        tooltip={"text": "Vehiculo {vehicle_id} · Deposito {depot}\nDeposito: {label}"},
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deck.to_html(str(output_path), open_browser=False, notebook_display=False)
    return output_path
