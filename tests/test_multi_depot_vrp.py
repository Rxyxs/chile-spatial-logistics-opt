import numpy as np
import pandas as pd
import pytest

from src.optimization.multi_depot_vrp import (
    MultiDepotVRPInstance,
    build_multi_depot_instance,
    solve_multi_depot_vrp,
    summarize_by_depot,
)
from src.spatial.network_distance import vectorized_haversine_matrix


def _toy_locations():
    # 2 depositos (indices 0, 1) + 6 puntos de demanda, en dos grupos
    # geograficos claramente separados, uno cerca de cada deposito.
    depot_a = (-33.40, -70.60)
    depot_b = (-33.55, -70.75)
    near_a = [(-33.401, -70.601), (-33.402, -70.602), (-33.403, -70.603)]
    near_b = [(-33.551, -70.751), (-33.552, -70.752), (-33.553, -70.753)]
    return [depot_a, depot_b] + near_a + near_b


def _toy_instance(vehicles_per_depot=(2, 2), vehicle_capacity=50):
    locations = _toy_locations()
    demands = [0, 0] + [10, 10, 10, 10, 10, 10]
    distance_matrix_km = vectorized_haversine_matrix(locations)
    return MultiDepotVRPInstance(
        locations=locations, demands=demands, depot_indices=[0, 1],
        vehicles_per_depot=list(vehicles_per_depot), vehicle_capacity=vehicle_capacity,
        distance_matrix_km=distance_matrix_km,
    )


def test_instance_rejects_nonzero_depot_demand():
    locations = _toy_locations()
    demands = [5, 0] + [10] * 6  # deposito 0 con demanda != 0
    with pytest.raises(ValueError, match="deposito"):
        MultiDepotVRPInstance(
            locations=locations, demands=demands, depot_indices=[0, 1],
            vehicles_per_depot=[1, 1], vehicle_capacity=50,
            distance_matrix_km=vectorized_haversine_matrix(locations),
        )


def test_instance_rejects_mismatched_vehicles_per_depot_length():
    locations = _toy_locations()
    demands = [0, 0] + [10] * 6
    with pytest.raises(ValueError, match="vehicles_per_depot"):
        MultiDepotVRPInstance(
            locations=locations, demands=demands, depot_indices=[0, 1],
            vehicles_per_depot=[1, 1, 1], vehicle_capacity=50,  # 3 valores, 2 depositos
            distance_matrix_km=vectorized_haversine_matrix(locations),
        )


def test_instance_rejects_wrong_shaped_distance_matrix():
    locations = _toy_locations()
    demands = [0, 0] + [10] * 6
    with pytest.raises(ValueError, match="distance_matrix_km"):
        MultiDepotVRPInstance(
            locations=locations, demands=demands, depot_indices=[0, 1],
            vehicles_per_depot=[1, 1], vehicle_capacity=50,
            distance_matrix_km=np.zeros((3, 3)),
        )


def test_vehicle_starts_groups_by_depot_in_order():
    instance = _toy_instance(vehicles_per_depot=(2, 3))
    assert instance.vehicle_starts == [0, 0, 1, 1, 1]
    assert instance.num_vehicles == 5


def test_solve_finds_a_feasible_solution_with_no_dropped_stops():
    instance = _toy_instance()
    solution = solve_multi_depot_vrp(instance, time_limit_seconds=10)
    assert solution is not None
    assert solution.dropped == []


def test_every_route_starts_and_ends_at_its_own_depot():
    instance = _toy_instance()
    solution = solve_multi_depot_vrp(instance, time_limit_seconds=10)
    for route in solution.routes:
        assert route.stops[0] == route.depot_index
        assert route.stops[-1] == route.depot_index
        assert route.depot_index in instance.depot_indices


def test_nearby_demand_is_served_by_the_nearby_depot():
    """With demand points clustered tightly around each depot and far from
    the other, the solver should assign each cluster to its own nearby
    depot rather than crossing the city -- a sanity check that multi-depot
    assignment is actually working, not just defaulting to depot 0."""
    instance = _toy_instance()
    solution = solve_multi_depot_vrp(instance, time_limit_seconds=10)

    visited_by_depot = {0: set(), 1: set()}
    for route in solution.routes:
        for stop in route.stops[1:-1]:
            visited_by_depot[route.depot_index].add(stop)

    # puntos 2,3,4 estan cerca del deposito 0; puntos 5,6,7 cerca del deposito 1
    assert visited_by_depot[0] == {2, 3, 4}
    assert visited_by_depot[1] == {5, 6, 7}


def test_summarize_by_depot_totals_match_solution():
    instance = _toy_instance()
    solution = solve_multi_depot_vrp(instance, time_limit_seconds=10)
    summary = summarize_by_depot(solution, instance.depot_indices, depot_labels=["A", "B"])

    assert sum(row["total_load"] for row in summary) == sum(r.load for r in solution.routes)
    assert sum(row["n_vehicles_used"] for row in summary) == len(solution.routes)
    assert pytest.approx(sum(row["total_distance_km"] for row in summary), abs=1e-6) == pytest.approx(
        sum(r.distance_km for r in solution.routes), abs=1e-6
    )


def test_build_multi_depot_instance_from_h3_like_dataframe():
    h3_agg = pd.DataFrame({
        "centroid_lat": [-33.401, -33.551],
        "centroid_lon": [-70.601, -70.751],
        "total_demand": [15, 20],
        "comuna": ["A", "B"],
    })
    depots = [(-33.40, -70.60), (-33.55, -70.75)]
    instance = build_multi_depot_instance(h3_agg, depots, vehicles_per_depot=[1, 1], vehicle_capacity=50)

    assert instance.locations[:2] == depots
    assert instance.demands[:2] == [0, 0]
    assert instance.demands[2:] == [15, 20]
    assert instance.distance_matrix_km.shape == (4, 4)
