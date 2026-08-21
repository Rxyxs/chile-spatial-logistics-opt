"""Solucionador de Vehicle Routing Problem (VRP) con capacidad y ventanas de
tiempo para la ultima milla desde un Dark Store hacia puntos de demanda
agregados en celdas H3.

Minimiza la distancia total recorrida (haversine, en km) y balancea la carga
de trabajo entre los vehiculos disponibles usando Google OR-Tools.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en linea recta (gran circulo) entre dos coordenadas, en km."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@dataclass
class VRPInstance:
    """Instancia de un problema VRP con capacidades y ventanas de tiempo.

    `locations[0]` (y `demands[0]`, que debe ser 0) representan siempre el
    deposito (Dark Store) desde el cual parten y regresan todos los vehiculos.
    """

    locations: list[tuple[float, float]]
    demands: list[int]
    num_vehicles: int
    vehicle_capacity: int
    time_windows: list[tuple[int, int]] | None = None  # minutos desde apertura
    service_time_min: int = 5
    vehicle_speed_kmh: float = 25.0
    depot: int = 0

    def __post_init__(self) -> None:
        n = len(self.locations)
        if len(self.demands) != n:
            raise ValueError("demands debe tener la misma longitud que locations")
        if self.time_windows is None:
            self.time_windows = [(0, 24 * 60)] * n
        elif len(self.time_windows) != n:
            raise ValueError("time_windows debe tener la misma longitud que locations")


@dataclass
class VehicleRoute:
    vehicle_id: int
    stops: list[int]  # indices de `locations`; incluye el deposito al inicio y al final
    distance_km: float
    load: int


@dataclass
class VRPSolution:
    routes: list[VehicleRoute]
    total_distance_km: float
    dropped: list[int] = field(default_factory=list)


def _build_distance_matrix_km(locations: list[tuple[float, float]]) -> list[list[float]]:
    n = len(locations)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = haversine_km(*locations[i], *locations[j])
    return matrix


def build_instance_from_h3(
    h3_agg,
    depot: tuple[float, float],
    num_vehicles: int,
    vehicle_capacity: int,
    demand_column: str = "total_demand",
    lat_column: str = "centroid_lat",
    lon_column: str = "centroid_lon",
    open_min: int = 0,
    close_min: int = 240,
) -> VRPInstance:
    """Construye una `VRPInstance` a partir de la demanda agregada por H3 y un deposito.

    `h3_agg` puede ser cualquier DataFrame/GeoDataFrame con las columnas de
    demanda y centroide (por ejemplo, el resultado de `aggregate_by_h3`).
    """
    locations = [depot] + list(zip(h3_agg[lat_column], h3_agg[lon_column]))
    demands = [0] + h3_agg[demand_column].astype(int).tolist()
    time_windows = [(open_min, close_min)] * len(locations)
    return VRPInstance(
        locations=locations,
        demands=demands,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        time_windows=time_windows,
    )


def solve_vrp(instance: VRPInstance, time_limit_seconds: int = 20) -> VRPSolution | None:
    """Resuelve el VRP minimizando distancia total y balanceando carga entre vehiculos."""
    n = len(instance.locations)
    distance_km = _build_distance_matrix_km(instance.locations)
    # OR-Tools trabaja mejor con enteros -> se usan metros para costos/dimensiones.
    distance_m = [[round(d * 1000) for d in row] for row in distance_km]

    manager = pywrapcp.RoutingIndexManager(n, instance.num_vehicles, instance.depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_m[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # --- Dimension de distancia: permite balancear el largo de ruta entre vehiculos ---
    routing.AddDimension(transit_callback_index, 0, 10_000_000, True, "Distance")
    distance_dimension = routing.GetDimensionOrDie("Distance")
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # --- Dimension de capacidad ---
    def demand_callback(from_index: int) -> int:
        return instance.demands[manager.IndexToNode(from_index)]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [instance.vehicle_capacity] * instance.num_vehicles,
        True,
        "Capacity",
    )

    # --- Dimension de tiempo (ventanas de entrega) ---
    speed_km_per_min = instance.vehicle_speed_kmh / 60.0

    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel_min = distance_km[from_node][to_node] / speed_km_per_min
        return round(travel_min) + instance.service_time_min

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(time_callback_index, 60, 24 * 60, False, "Time")
    time_dimension = routing.GetDimensionOrDie("Time")

    for location_idx, (start, end) in enumerate(instance.time_windows):
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(start, end)

    depot_window = instance.time_windows[instance.depot]
    for vehicle_id in range(instance.num_vehicles):
        time_dimension.CumulVar(routing.Start(vehicle_id)).SetRange(*depot_window)
        time_dimension.CumulVar(routing.End(vehicle_id)).SetRange(*depot_window)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(time_limit_seconds)

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        return None

    return _extract_solution(manager, routing, solution, instance, distance_km)


def _extract_solution(manager, routing, solution, instance: VRPInstance, distance_km) -> VRPSolution:
    routes: list[VehicleRoute] = []
    total_distance = 0.0

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
            route_distance += distance_km[node][next_node]
        stops.append(manager.IndexToNode(index))  # deposito final

        if len(stops) > 2:  # el vehiculo atendio al menos una parada real
            routes.append(
                VehicleRoute(
                    vehicle_id=vehicle_id,
                    stops=stops,
                    distance_km=round(route_distance, 3),
                    load=route_load,
                )
            )
            total_distance += route_distance

    visited = {s for route in routes for s in route.stops if s != instance.depot}
    dropped = [i for i in range(len(instance.locations)) if i != instance.depot and i not in visited]

    return VRPSolution(routes=routes, total_distance_km=round(total_distance, 3), dropped=dropped)


def main() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.spatial.geo_generator import aggregate_by_h3, assign_h3_index, generate_synthetic_demand, select_dark_store_candidates

    gdf = assign_h3_index(generate_synthetic_demand())
    h3_agg = aggregate_by_h3(gdf)

    depot_row = select_dark_store_candidates(h3_agg, n_stores=1).iloc[0]
    depot = (depot_row["centroid_lat"], depot_row["centroid_lon"])
    demand_cells = h3_agg[h3_agg["h3_cell"] != depot_row["h3_cell"]].reset_index(drop=True)

    instance = build_instance_from_h3(demand_cells, depot, num_vehicles=4, vehicle_capacity=120)
    solution = solve_vrp(instance)

    if solution is None:
        print("No se encontro una solucion factible.")
        return

    print(f"Deposito (Dark Store): {depot}")
    print(f"Distancia total: {solution.total_distance_km} km")
    for route in solution.routes:
        print(
            f"Vehiculo {route.vehicle_id}: {len(route.stops) - 2} paradas, "
            f"{route.distance_km} km, carga {route.load}"
        )
    if solution.dropped:
        print(f"Celdas sin asignar: {solution.dropped}")


if __name__ == "__main__":
    main()
