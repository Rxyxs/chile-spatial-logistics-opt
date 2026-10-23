"""Real-data modules: population calibration, the OSRM client (no network) and the study helpers."""
import h3
import numpy as np
import pandas as pd
import pytest

from src.real_santiago import route_km, spatial_cv
from src.spatial.osrm import circuity, parse_distances_km, table_url
from src.spatial.real_population import calibrate_to_census, cells_frame, comuna_comparison, demand_from_population


@pytest.fixture
def cells():
    base = h3.latlng_to_cell(-33.45, -70.65, 8)
    ring = list(h3.grid_disk(base, 2))[:12]
    comuna = ["A"] * 6 + ["B"] * 6
    pop = [100, 200, 300, 400, 500, 600, 10, 20, 30, 40, 50, 60]
    return cells_frame(ring, pop, comuna)


def test_calibration_matches_census_and_keeps_the_shape(cells):
    out = calibrate_to_census(cells, {"A": 4200, "B": 420})
    assert out.groupby("comuna")["population"].sum().to_dict() == pytest.approx({"A": 4200, "B": 420})
    a = out[out.comuna == "A"]
    np.testing.assert_allclose(a["population"] / a["kontur_population"], 2.0)


def test_comuna_comparison_reports_the_gap(cells):
    t = comuna_comparison(cells, {"A": 1050, "B": 420}).set_index("comuna")
    assert t.loc["A", "kontur_vs_censo_pct"] == pytest.approx(100.0)
    assert t.loc["B", "kontur_vs_censo_pct"] == pytest.approx(-50.0)


def test_demand_sums_to_the_target_and_follows_population(cells):
    out = demand_from_population(calibrate_to_census(cells, {"A": 2100, "B": 210}), total_units=100, drop_zero=False)
    assert out["total_demand"].sum() == 100
    assert out.loc[out.kontur_population.idxmax(), "total_demand"] == out["total_demand"].max()


def test_table_url_uses_lon_lat_and_local_indices():
    locs = [(-33.4, -70.6), (-33.5, -70.7), (-33.6, -70.8)]
    url = table_url(locs, sources=[2], destinations=[0, 2])
    assert "-70.600000,-33.400000;-70.800000,-33.600000" in url
    assert "sources=1&destinations=0;1" in url


def test_parse_distances_and_errors():
    m = parse_distances_km({"code": "Ok", "distances": [[0, 1500], [1600, 0]]})
    np.testing.assert_allclose(m, [[0, 1.5], [1.6, 0]])
    with pytest.raises(RuntimeError):
        parse_distances_km({"code": "TooBig", "message": "too many"})


def test_circuity_skips_short_pairs():
    straight = np.array([[0, 0.2, 2.0], [0.2, 0, 1.0], [2.0, 1.0, 0]])
    road = straight * 1.4
    c = circuity(road, straight, min_straight_km=0.5)
    assert c.size == 4 and np.allclose(c, 1.4)


def test_route_km_measures_a_plan_on_another_matrix():
    class R:
        def __init__(self, stops):
            self.stops = stops

    class S:
        routes = [R([0, 1, 2, 0])]

    m = np.array([[0, 1, 5], [1, 0, 2], [5, 2, 0]], dtype=float)
    assert route_km(S(), m) == 8.0


def test_spatial_cv_is_harder_than_random_folds():
    """A smooth spatial field: neighbours predict each other, distant blocks do not."""
    rng = np.random.default_rng(0)
    base = h3.latlng_to_cell(-33.5, -70.7, 8)
    cells = list(h3.grid_disk(base, 14))
    ll = np.array([h3.cell_to_latlng(c) for c in cells])
    d2 = (ll[:, 0] + 33.5) ** 2 + (ll[:, 1] + 70.7) ** 2
    pop = 3000 * np.exp(-d2 / (2 * 0.02 ** 2)) + 500 + rng.normal(0, 50, len(cells))
    df = pd.DataFrame({"h3_cell": cells, "comuna": "A", "centroid_lat": ll[:, 0],
                       "centroid_lon": ll[:, 1], "total_demand": pop})
    cv = spatial_cv(df, parent_res=6)
    assert cv["random_folds"]["random_forest"] > 0.5
    assert cv["random_folds"]["random_forest"] > cv["spatial_blocks"]["random_forest"] + 0.2
