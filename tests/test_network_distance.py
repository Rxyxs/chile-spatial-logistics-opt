import numpy as np
import pytest

from src.spatial.network_distance import (
    _loop_haversine_matrix,
    simulate_road_network_matrix,
    vectorized_haversine_matrix,
)

SANTIAGO_POINTS = [(-33.45, -70.65), (-33.41, -70.57), (-33.51, -70.75), (-33.59, -70.70)]


def test_vectorized_matches_loop_implementation():
    vec = vectorized_haversine_matrix(SANTIAGO_POINTS)
    loop = np.array(_loop_haversine_matrix(SANTIAGO_POINTS))
    assert np.allclose(vec, loop, atol=1e-9)


def test_vectorized_haversine_diagonal_is_zero():
    vec = vectorized_haversine_matrix(SANTIAGO_POINTS)
    assert np.allclose(np.diag(vec), 0.0)


def test_vectorized_haversine_is_symmetric():
    vec = vectorized_haversine_matrix(SANTIAGO_POINTS)
    assert np.allclose(vec, vec.T)


def test_network_matrix_never_shorter_than_straight_line():
    haversine = vectorized_haversine_matrix(SANTIAGO_POINTS)
    network = simulate_road_network_matrix(SANTIAGO_POINTS, seed=1)
    assert np.all(network >= haversine - 1e-9)


def test_network_matrix_is_symmetric():
    network = simulate_road_network_matrix(SANTIAGO_POINTS, seed=1)
    assert np.allclose(network, network.T)


def test_network_matrix_diagonal_is_zero():
    network = simulate_road_network_matrix(SANTIAGO_POINTS, seed=1)
    assert np.allclose(np.diag(network), 0.0)


def test_network_matrix_is_deterministic_given_seed():
    a = simulate_road_network_matrix(SANTIAGO_POINTS, seed=7)
    b = simulate_road_network_matrix(SANTIAGO_POINTS, seed=7)
    assert np.array_equal(a, b)


def test_inter_zone_trips_have_higher_average_circuity_than_intra_zone():
    """With zone labels provided, cross-zone pairs should on average get a
    larger circuity factor than same-zone pairs -- checked over many seeds
    to average out the per-pair randomness, not asserted on a single draw."""
    zones = ["A", "A", "B", "B"]
    haversine = vectorized_haversine_matrix(SANTIAGO_POINTS)

    intra_ratios, inter_ratios = [], []
    for seed in range(30):
        network = simulate_road_network_matrix(SANTIAGO_POINTS, zone_labels=zones, seed=seed)
        ratio = np.divide(network, haversine, out=np.ones_like(network), where=haversine > 0)
        for i in range(len(zones)):
            for j in range(i + 1, len(zones)):
                (intra_ratios if zones[i] == zones[j] else inter_ratios).append(ratio[i, j])

    assert np.mean(inter_ratios) > np.mean(intra_ratios)


def test_benchmark_reports_a_real_speedup():
    from src.spatial.network_distance import benchmark_haversine_speedup

    locations = [(-33.4 - i * 0.001, -70.6 - i * 0.001) for i in range(150)]
    result = benchmark_haversine_speedup(locations)
    assert result["speedup_x"] > 1.0
