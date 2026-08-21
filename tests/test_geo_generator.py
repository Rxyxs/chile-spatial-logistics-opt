import h3
import pytest

from src.spatial.geo_generator import (
    COMUNAS,
    H3_RESOLUTION,
    aggregate_by_h3,
    assign_h3_index,
    generate_synthetic_demand,
    select_dark_store_candidates,
)

# Bounding box aproximado de la Region Metropolitana de Santiago.
RM_LAT_RANGE = (-33.80, -33.25)
RM_LON_RANGE = (-70.90, -70.40)


@pytest.fixture(scope="module")
def demand_gdf():
    return assign_h3_index(generate_synthetic_demand(n_per_comuna=15, seed=1))


def test_generated_points_within_santiago_bounds(demand_gdf):
    assert demand_gdf["lat"].between(*RM_LAT_RANGE).all()
    assert demand_gdf["lon"].between(*RM_LON_RANGE).all()


def test_expected_row_count(demand_gdf):
    assert len(demand_gdf) == 15 * len(COMUNAS)


def test_all_comunas_present(demand_gdf):
    assert set(demand_gdf["comuna"]) == set(COMUNAS)


def test_h3_cells_have_expected_resolution(demand_gdf):
    assert all(h3.get_resolution(cell) == H3_RESOLUTION for cell in demand_gdf["h3_cell"])


def test_h3_cell_matches_point(demand_gdf):
    sample = demand_gdf.iloc[0]
    expected = h3.latlng_to_cell(sample["lat"], sample["lon"], H3_RESOLUTION)
    assert sample["h3_cell"] == expected


def test_aggregate_preserves_total_demand(demand_gdf):
    h3_agg = aggregate_by_h3(demand_gdf)
    assert h3_agg["total_demand"].sum() == demand_gdf["demand"].sum()
    assert h3_agg["order_count"].sum() == len(demand_gdf)


def test_select_dark_store_candidates_orders_by_demand(demand_gdf):
    h3_agg = aggregate_by_h3(demand_gdf)
    candidates = select_dark_store_candidates(h3_agg, n_stores=3)
    assert len(candidates) == 3
    assert candidates["total_demand"].is_monotonic_decreasing
