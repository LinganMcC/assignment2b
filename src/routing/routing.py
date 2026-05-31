from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
ROUTING_ROOT = SRC_ROOT / "routing"
PARTA_ROOT = ROUTING_ROOT / "partA"
ALGORITHMS_ROOT = PARTA_ROOT / "algorithms"

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PARTA_ROOT))
sys.path.insert(0, str(ALGORITHMS_ROOT))

from partA.graph import Graph
from partA.algorithms.Astar import AStar
from partA.algorithms.bfs import BFS
from partA.algorithms.dfs import DFS
from partA.algorithms.gbfs import GBFS
from partA.algorithms.cus1 import CUS1
from partA.algorithms.cus2 import CUS2

GRAPH_FILE = PROJECT_ROOT / "data" / "processed" / "boroondara_graph.json"

FLOW_CAPACITY = 400.0
SPEED_CAPACITY = 32.0
SPEED_LIMIT = 60.0   
SPEED_MIN = 1.0
INTERSECTION_DELAY_S = 30.0

_A = -FLOW_CAPACITY / (SPEED_CAPACITY ** 2)
_B = -2.0 * SPEED_CAPACITY * _A
_B2 = _B ** 2


@dataclass(order=True)
class Route:
    total_time_s: float
    path: list[int] = field(compare=False)
    edges_km: list[float] = field(compare=False)
    speeds_kmh: list[float] = field(compare=False)
    algorithm: str = field(default="astar", compare=False)
    nodes_created: int = field(default=0, compare=False)
    goal: Optional[int] = field(default=None, compare=False)

    @property
    def total_time_min(self) -> float:
        return self.total_time_s / 60.0

    @property
    def total_distance_km(self) -> float:
        return sum(self.edges_km)

    def __repr__(self) -> str:
        sites = " → ".join(str(s) for s in self.path)
        return (
            f"Route({self.total_time_min:.1f} min, "
            f"{self.total_distance_km:.2f} km, "
            f"{len(self.path) - 1} links | {sites})"
        )


def flow_to_speed(flow_veh_per_15min: float) -> float:
    q = max(0.0, min(float(flow_veh_per_15min), FLOW_CAPACITY))

    discriminant = _B2 + 4.0 * _A * q
    if discriminant < 0:
        return SPEED_CAPACITY

    sqrt_d = math.sqrt(discriminant)
    speed = (-_B - sqrt_d) / (2.0 * _A)
    return max(SPEED_MIN, min(speed, SPEED_LIMIT))


def edge_travel_time_s(distance_km: float, flow_veh_per_15min: int) -> float:
    speed_kmh = flow_to_speed(flow_veh_per_15min)
    return (distance_km / speed_kmh) * 3600.0 + INTERSECTION_DELAY_S


def get_predict_function():
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    if str(SRC_ROOT / "ml") not in sys.path:
        sys.path.insert(0, str(SRC_ROOT / "ml"))
    from ml.predict import predict
    return predict


def load_boroondara_json(path: Path = GRAPH_FILE) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_parta_graph(
    origin: int,
    dest: int,
    datetime_str: str,
    model: str,
    predict_fn: Optional[Callable] = None,
    json_path: Path = GRAPH_FILE,
) -> tuple[Graph, dict[tuple[int, int], float], dict[int, int]]:
    data = load_boroondara_json(json_path)

    graph = Graph()
    graph.origin = int(origin)
    graph.destinations = [int(dest)]

    for node_id, info in data["nodes"].items():
        sid = int(node_id)
        graph.nodes[sid] = (float(info["lon"]), float(info["lat"]))

    pred = predict_fn or get_predict_function()
    flow_cache: dict[int, int] = {}
    distance_lookup: dict[tuple[int, int], float] = {}

    def get_flow(site_id: int) -> int:
        if site_id not in flow_cache:
            try:
                flow_cache[site_id] = max(0, int(pred(site_id, datetime_str, model)))
            except Exception:
                flow_cache[site_id] = 0
        return flow_cache[site_id]

    for from_id, edge_list in data["edges"].items():
        u = int(from_id)
        graph.edges[u] = []
        for edge in edge_list:
            v = int(edge["to"])
            distance_km = float(edge["distance_km"])
            flow = get_flow(u)
            cost = edge_travel_time_s(distance_km, flow)
            graph.edges[u].append((v, cost))
            distance_lookup[(u, v)] = distance_km

    for node_id in graph.nodes:
        graph.edges.setdefault(node_id, [])

    return graph, distance_lookup, flow_cache


def run_algorithm(graph: Graph, algorithm: str):
    algorithms = {
        "astar": AStar,
        "bfs": BFS,
        "dfs": DFS,
        "gbfs": GBFS,
        "cus1": CUS1,
        "cus2": CUS2,
    }
    alg = algorithm.lower()
    if alg not in algorithms:
        raise ValueError(f"Invalid algorithm: {algorithm}")
    return algorithms[alg](graph).search()


def build_route(
    path: list[int],
    goal: Optional[int],
    nodes_created: int,
    algorithm: str,
    distance_lookup: dict[tuple[int, int], float],
    flow_cache: dict[int, int],
) -> Route:
    total_time_s = 0.0
    edges_km = []
    speeds_kmh = []

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        distance_km = distance_lookup.get((u, v), 0.0)
        flow_15min = flow_cache.get(u, 0)
        speed = flow_to_speed(flow_15min)
        time_s = edge_travel_time_s(distance_km, flow_15min)
        edges_km.append(distance_km)
        speeds_kmh.append(speed)
        total_time_s += time_s

    return Route(
        total_time_s=total_time_s,
        path=path,
        edges_km=edges_km,
        speeds_kmh=speeds_kmh,
        algorithm=algorithm,
        nodes_created=nodes_created,
        goal=goal,
    )


def find_routes(
    origin: int,
    dest: int,
    datetime_str: str,
    model: str = "gru",
    k: int = 1,
    algorithm: str = "astar",
    predict_fn: Optional[Callable] = None,
) -> list[Route]:
    graph, distance_lookup, flow_cache = build_parta_graph(
        origin=origin,
        dest=dest,
        datetime_str=datetime_str,
        model=model,
        predict_fn=predict_fn,
    )
    goal, nodes_created, path = run_algorithm(graph, algorithm)
    if not path:
        return []
    route = build_route(
        path=path,
        goal=goal,
        nodes_created=nodes_created,
        algorithm=algorithm,
        distance_lookup=distance_lookup,
        flow_cache=flow_cache,
    )
    return [route]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PartA search algorithms on routing graph")
    parser.add_argument("--origin", type=int, default=970)
    parser.add_argument("--dest", type=int, default=2200)
    parser.add_argument("--datetime", type=str, default="2006-10-15 08:30:00")
    parser.add_argument("--model", type=str, default="gru", choices=["lstm", "gru", "transformer"])
    parser.add_argument("--algorithm", type=str, default="astar",
                        choices=["astar", "bfs", "dfs", "gbfs", "cus1", "cus2"])
    args = parser.parse_args()

    routes = find_routes(
        origin=args.origin,
        dest=args.dest,
        datetime_str=args.datetime,
        model=args.model,
        algorithm=args.algorithm,
    )

    if not routes:
        print("No route found.")
    else:
        for i, route in enumerate(routes, 1):
            print(f"Route {i}: {route}")
            print(f"Algorithm: {route.algorithm.upper()}")
            print(f"Goal: {route.goal}")
            print(f"Nodes created: {route.nodes_created}")
            print(f"Path: {' -> '.join(str(s) for s in route.path)}")
            print(f"Total time: {route.total_time_min:.2f} min")
            print(f"Total distance: {route.total_distance_km:.2f} km")
            print()
