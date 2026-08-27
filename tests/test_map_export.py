import pandas as pd
import pytest

from src.optimization.multi_depot_vrp import build_multi_depot_instance, solve_multi_depot_vrp
from src.spatial.map_export import export_folium_map, export_pydeck_map


@pytest.fixture(scope="module")
def solved_case(tmp_path_factory):
    h3_agg = pd.DataFrame({
        "centroid_lat": [-33.401, -33.402, -33.551, -33.552],
        "centroid_lon": [-70.601, -70.602, -70.751, -70.752],
        "total_demand": [10, 12, 8, 9],
        "order_count": [3, 4, 2, 3],
        "comuna": ["A", "A", "B", "B"],
        "h3_cell": ["c1", "c2", "c3", "c4"],
        "geometry": [
            _square(-70.601, -33.401), _square(-70.602, -33.402),
            _square(-70.751, -33.551), _square(-70.752, -33.552),
        ],
    })
    depots = [(-33.40, -70.60), (-33.55, -70.75)]
    depot_labels = ["Deposito A", "Deposito B"]
    instance = build_multi_depot_instance(h3_agg, depots, vehicles_per_depot=[1, 1], vehicle_capacity=50)
    solution = solve_multi_depot_vrp(instance, time_limit_seconds=10)
    return h3_agg, depots, depot_labels, solution, instance.locations


def _square(lon, lat, half_side=0.001):
    from shapely.geometry import Polygon

    return Polygon([
        (lon - half_side, lat - half_side), (lon + half_side, lat - half_side),
        (lon + half_side, lat + half_side), (lon - half_side, lat + half_side),
    ])


def test_export_folium_map_creates_file_with_depot_labels(solved_case, tmp_path):
    h3_agg, depots, depot_labels, solution, locations = solved_case
    assert solution is not None

    out_path = tmp_path / "routes.html"
    result_path = export_folium_map(h3_agg, depots, depot_labels, solution, locations, out_path)

    assert result_path.exists()
    content = result_path.read_text(encoding="utf-8")
    assert "Deposito A" in content
    assert "Deposito B" in content
    assert len(content) > 1000  # un mapa real, no un archivo vacio


def test_export_pydeck_map_creates_file(solved_case, tmp_path):
    h3_agg, depots, depot_labels, solution, locations = solved_case
    assert solution is not None

    out_path = tmp_path / "routes_pydeck.html"
    result_path = export_pydeck_map(depots, depot_labels, solution, locations, out_path)

    assert result_path.exists()
    content = result_path.read_text(encoding="utf-8")
    assert len(content) > 1000
