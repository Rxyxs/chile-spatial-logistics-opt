"""Vehicle Routing Problem multi-depósito con ventanas de tiempo (Multi-Depot
VRPTW): extiende `vrp_solver.py` (un único depósito) a varios, usando el
soporte nativo de OR-Tools para múltiples inicios/fines de vehículo
(`starts`/`ends` en `RoutingIndexManager`).

Esto es optimización multi-depósito real, no una heurística de
pre-clustering: el solver decide *conjuntamente* qué vehículo -- de qué
depósito -- atiende cada parada, dentro de una sola resolución, no se asigna
cada celda de demanda a "su" depósito más cercano antes de rutear (lo cual
podría dejar un vehículo de un depósito sobrecargado mientras uno de otro
depósito, geográficamente más lejos pero con capacidad libre, queda ocioso).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.spatial.network_distance import simulate_road_network_matrix


@dataclass
class MultiDepotVRPInstance:
    """`locations` debe listar todos los depósitos primero (en el mismo orden
    que `depot_indices`), y luego los puntos de demanda -- igual que
    `vrp_solver.VRPInstance` pero generalizado a N depósitos en vez de 1."""

    locations: list[tuple[float, float]]
    demands: list[int]
    depot_indices: list[int]
    vehicles_per_depot: list[int]
    vehicle_capacity: int
    distance_matrix_km: np.ndarray
    time_windows: list[tuple[int, int]] | None = None
    service_time_min: int = 5
    vehicle_speed_kmh: float = 25.0

    def __post_init__(self) -> None:
        n = len(self.locations)
        if len(self.demands) != n:
            raise ValueError("demands debe tener la misma longitud que locations")
        if self.distance_matrix_km.shape != (n, n):
            raise ValueError(f"distance_matrix_km debe ser {n}x{n}, es {self.distance_matrix_km.shape}")
        if len(self.vehicles_per_depot) != len(self.depot_indices):
            raise ValueError("vehicles_per_depot debe tener un valor por cada depot_indices")
        if any(self.demands[d] != 0 for d in self.depot_indices):
            raise ValueError("la demanda de un deposito debe ser 0")
        if self.time_windows is None:
            self.time_windows = [(0, 24 * 60)] * n
        elif len(self.time_windows) != n:
            raise ValueError("time_windows debe tener la misma longitud que locations")

    @property
    def num_vehicles(self) -> int:
        return sum(self.vehicles_per_depot)

    @property
    def vehicle_starts(self) -> list[int]:
        """Índice del depósito (en `locations`) desde el que parte cada
        vehículo -- un vehículo por posición, agrupados por depósito."""
        starts: list[int] = []
        for depot_idx, n_vehicles in zip(self.depot_indices, self.vehicles_per_depot):
            starts.extend([depot_idx] * n_vehicles)
        return starts


@dataclass
class MultiDepotVehicleRoute:
    vehicle_id: int
    depot_index: int  # posición en `locations` del depósito de origen
    stops: list[int]
    distance_km: float
    load: int


@dataclass
class MultiDepotVRPSolution:
    routes: list[MultiDepotVehicleRoute]
    total_distance_km: float
    dropped: list[int] = field(default_factory=list)


def build_multi_depot_instance(
    h3_agg,
    depots: list[tuple[float, float]],
    vehicles_per_depot: list[int],
    vehicle_capacity: int,
    demand_column: str = "total_demand",
    lat_column: str = "centroid_lat",
    lon_column: str = "centroid_lon",
    zone_column: str | None = "comuna",
    open_min: int = 0,
    close_min: int = 480,
    seed: int = 42,
) -> MultiDepotVRPInstance:
    """Construye una `MultiDepotVRPInstance` a partir de la demanda agregada
    por H3 y una lista de depósitos (candidatos, típicamente de
    `geo_generator.select_dark_store_candidates`)."""
    depot_indices = list(range(len(depots)))
    demand_locations = list(zip(h3_agg[lat_column], h3_agg[lon_column]))
    locations = list(depots) + demand_locations

    demands = [0] * len(depots) + h3_agg[demand_column].astype(int).tolist()
    time_windows = [(open_min, close_min)] * len(locations)

    zone_labels = None
    if zone_column is not None and zone_column in h3_agg.columns:
        zone_labels = ["__depot__"] * len(depots) + h3_agg[zone_column].tolist()

    distance_matrix_km = simulate_road_network_matrix(locations, zone_labels=zone_labels, seed=seed)

    return MultiDepotVRPInstance(
        locations=locations,
        demands=demands,
        depot_indices=depot_indices,
        vehicles_per_depot=vehicles_per_depot,
        vehicle_capacity=vehicle_capacity,
        distance_matrix_km=distance_matrix_km,
        time_windows=time_windows,
    )


def solve_multi_depot_vrp(instance: MultiDepotVRPInstance, time_limit_seconds: int = 30) -> MultiDepotVRPSolution | None:
    n = len(instance.locations)
    starts = instance.vehicle_starts
    ends = list(starts)  # cada vehiculo regresa a su propio deposito de origen

    distance_m = np.round(instance.distance_matrix_km * 1000).astype(int)

    manager = pywrapcp.RoutingIndexManager(n, instance.num_vehicles, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_m[from_node, to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    routing.AddDimension(transit_callback_index, 0, 10_000_000, True, "Distance")
    routing.GetDimensionOrDie("Distance").SetGlobalSpanCostCoefficient(100)

    def demand_callback(from_index: int) -> int:
        return instance.demands[manager.IndexToNode(from_index)]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, [instance.vehicle_capacity] * instance.num_vehicles, True, "Capacity",
    )

    speed_km_per_min = instance.vehicle_speed_kmh / 60.0

    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel_min = instance.distance_matrix_km[from_node, to_node] / speed_km_per_min
        return round(travel_min) + instance.service_time_min

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(time_callback_index, 60, 24 * 60, False, "Time")
    time_dimension = routing.GetDimensionOrDie("Time")

    for location_idx, (start, end) in enumerate(instance.time_windows):
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(start, end)

    for vehicle_id in range(instance.num_vehicles):
        depot_node = starts[vehicle_id]
        depot_window = instance.time_windows[depot_node]
        time_dimension.CumulVar(routing.Start(vehicle_id)).SetRange(*depot_window)
        time_dimension.CumulVar(routing.End(vehicle_id)).SetRange(*depot_window)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(time_limit_seconds)

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        return None

    return _extract_multi_depot_solution(manager, routing, solution, instance)


def _extract_multi_depot_solution(manager, routing, solution, instance: MultiDepotVRPInstance) -> MultiDepotVRPSolution:
    routes: list[MultiDepotVehicleRoute] = []
    total_distance = 0.0
    depot_for_vehicle = instance.vehicle_starts

    for vehicle_id in range(instance.num_vehicles):
        index = routing.Start(vehicle_id)
        stops: list[int] = []
        route_distance = 0.0
        route_load = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            stops.append(node)
            route_load += instance.demands[node]
            index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(index)
            route_distance += instance.distance_matrix_km[node, next_node]
        stops.append(manager.IndexToNode(index))

        if len(stops) > 2:
            routes.append(
                MultiDepotVehicleRoute(
                    vehicle_id=vehicle_id,
                    depot_index=depot_for_vehicle[vehicle_id],
                    stops=stops,
                    distance_km=round(route_distance, 3),
                    load=route_load,
                )
            )
            total_distance += route_distance

    visited = {s for route in routes for s in route.stops if s not in instance.depot_indices}
    dropped = [i for i in range(len(instance.locations)) if i not in instance.depot_indices and i not in visited]

    return MultiDepotVRPSolution(routes=routes, total_distance_km=round(total_distance, 3), dropped=dropped)


def summarize_by_depot(
    solution: MultiDepotVRPSolution,
    depot_indices: list[int],
    depot_labels: list[str] | None = None,
) -> list[dict]:
    """Balance de carga por depósito: distancia total, carga total, número de
    vehículos usados y de paradas atendidas, uno por depósito."""
    labels = depot_labels or [f"Deposito {i}" for i in range(len(depot_indices))]
    rows = []
    for label, depot_idx in zip(labels, depot_indices):
        depot_routes = [r for r in solution.routes if r.depot_index == depot_idx]
        rows.append({
            "depot_index": depot_idx,
            "depot_label": label,
            "n_vehicles_used": len(depot_routes),
            "total_distance_km": round(sum(r.distance_km for r in depot_routes), 3),
            "total_load": sum(r.load for r in depot_routes),
            "n_stops": sum(len(r.stops) - 2 for r in depot_routes),
        })
    return rows
