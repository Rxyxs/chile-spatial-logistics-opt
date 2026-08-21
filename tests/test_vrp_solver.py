import pytest

from src.optimization.vrp_solver import VRPInstance, haversine_km, solve_vrp


def test_haversine_known_distance():
    # Las Condes -> Santiago Centro, aprox 9-10 km en linea recta.
    dist = haversine_km(-33.4089, -70.5693, -33.4372, -70.6506)
    assert 8.0 < dist < 11.0


@pytest.fixture
def small_instance():
    depot = (-33.45, -70.65)
    stops = [
        (-33.44, -70.64),
        (-33.46, -70.66),
        (-33.43, -70.63),
        (-33.47, -70.67),
        (-33.45, -70.62),
    ]
    demands = [0, 10, 15, 5, 20, 10]
    return VRPInstance(
        locations=[depot, *stops],
        demands=demands,
        num_vehicles=2,
        vehicle_capacity=40,
    )


def test_solve_vrp_returns_feasible_solution(small_instance):
    solution = solve_vrp(small_instance, time_limit_seconds=10)
    assert solution is not None
    assert not solution.dropped


def test_solve_vrp_routes_start_and_end_at_depot(small_instance):
    solution = solve_vrp(small_instance, time_limit_seconds=10)
    for route in solution.routes:
        assert route.stops[0] == small_instance.depot
        assert route.stops[-1] == small_instance.depot


def test_solve_vrp_respects_vehicle_capacity(small_instance):
    solution = solve_vrp(small_instance, time_limit_seconds=10)
    for route in solution.routes:
        assert route.load <= small_instance.vehicle_capacity


def test_solve_vrp_visits_each_customer_once(small_instance):
    solution = solve_vrp(small_instance, time_limit_seconds=10)
    visited = [stop for route in solution.routes for stop in route.stops[1:-1]]
    assert sorted(visited) == [1, 2, 3, 4, 5]


def test_solve_vrp_uses_multiple_vehicles_when_needed():
    depot = (-33.45, -70.65)
    stops = [(-33.44, -70.64), (-33.46, -70.66), (-33.43, -70.63)]
    demands = [0, 30, 30, 30]
    instance = VRPInstance(
        locations=[depot, *stops], demands=demands, num_vehicles=3, vehicle_capacity=35
    )
    solution = solve_vrp(instance, time_limit_seconds=10)
    assert solution is not None
    assert len(solution.routes) >= 2
