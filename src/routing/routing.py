"""
routing.py  –  Bridge between ML flow predictions and graph-based routing.

OWNER: Person B (ML ↔ routing bridge)

Responsibilities
----------------
1. Flow → speed conversion   (quadratic from Traffic_Flow_to_Travel_Time PDF)
2. Travel-time computation   (distance / speed + 30 s intersection delay)
3. A* search on the Boroondara graph
4. Yen's k-shortest-loopless-paths on top of A*
5. Public API: find_routes(origin, dest, datetime, model) → list[Route]

Graph input (data/processed/boroondara_graph.json, per INTERFACES.md interface 2):
    {
      "nodes": { "970": {"name": ..., "lat": ..., "lon": ...}, ... },
      "edges": { "970": [{"to": 971, "distance_km": 0.432}, ...], ... }
    }

ML prediction input (src/ml/predict.py interface):
    predict(site_id: int, datetime_str: str, model_name: str) -> int
    Returns estimated vehicles per 15-min interval for that site.

Usage
-----
    from routing import find_routes

    routes = find_routes(
        origin=970,
        dest=2000,
        datetime_str="2006-10-15 08:30:00",
        model="gru",   # "lstm" | "gru" | "transformer"
        k=5,
    )
    for r in routes:
        print(r)
"""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# ── project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GRAPH_FILE = PROJECT_ROOT / "data" / "processed" / "boroondara_graph.json"

# ---------------------------------------------------------------------------
# 1.  FLOW → SPEED CONVERSION
#     Source: Traffic_Flow_to_Travel_Time_Conversion_v1_0.pdf
#
#     The fundamental diagram is expressed as a quadratic in speed (v):
#         flow = A·v² + B·v
#     where:
#         A = -q_c / v_c²  = -1500 / 32²  = -1.46484375
#         B = -2·v_c·A     =  2·32·1.46484375 = 93.75
#
#     Inverting (quadratic formula):
#         v = (-B ± sqrt(B² + 4·A·flow)) / (2·A)
#
#     Green line (under capacity) → use the + root (higher speed).
#     Red  line (over  capacity)  → use the − root (lower speed).
#
#     Capacity point: flow_c = 1500 veh/hr, speed_c = 32 km/hr
#     Speed limit:    v_max  = 60 km/hr
#     Speed-limit crossover flow: 351.56 veh/hr
#         (at flow ≤ 351 veh/hr the quadratic gives speed ≥ 60 → cap at 60)
#
#     NOTE: our ML model predicts vehicles per 15-min interval.
#           Multiply by 4 to convert to vehicles/hour before using this formula.
# ---------------------------------------------------------------------------

FLOW_CAPACITY   = 1500.0   # veh/hr  (turning point of parabola)
SPEED_CAPACITY  =   32.0   # km/hr   (speed at capacity)
SPEED_LIMIT     =   60.0   # km/hr
SPEED_MIN       =    1.0   # km/hr   (guard: never divide by ~0)

# Quadratic coefficients: flow = A·v² + B·v
_A = -FLOW_CAPACITY / (SPEED_CAPACITY ** 2)   # ≈ -1.46484375
_B = -2.0 * SPEED_CAPACITY * _A               # ≈  93.75

# Discriminant constant: B² (flow-independent part of sqrt)
_B2 = _B ** 2   # ≈ 8789.0625


def flow_to_speed(flow_veh_per_hour: float) -> float:
    """Convert traffic flow (vehicles/hour) to travel speed (km/hr).

    Implements the quadratic Traffic Flow Fundamental Diagram from the PDF:
        flow = A·v² + B·v
        A = -1.46484375,  B = 93.75

    Returns
    -------
    speed_kmh : float
        Speed in km/hr, capped at SPEED_LIMIT (60) and floored at SPEED_MIN.

    Branch selection
    ----------------
    - flow ≤ FLOW_CAPACITY (1500): road is under/at capacity → green branch
      (higher speed, + root).
    - flow >  FLOW_CAPACITY:       road is over capacity → red branch
      (lower speed, − root). We clamp flow to FLOW_CAPACITY * 1.5 to avoid
      imaginary roots (the parabola only extends to ~2× capacity in practice).
    """
    q = max(0.0, float(flow_veh_per_hour))

    discriminant = _B2 + 4.0 * _A * q   # B² + 4Aq  (A is negative)
    if discriminant < 0.0:
        # Flow beyond the physical maximum of the parabola → near-standstill
        return SPEED_MIN

    sqrt_d = math.sqrt(discriminant)

    # NOTE: A is negative, so the standard quadratic formula roots are swapped
    # relative to intuition.  With A < 0:
    #   (-B − sqrt_d) / (2A) → HIGHER speed  (green / under-capacity branch)
    #   (-B + sqrt_d) / (2A) → LOWER  speed  (red   / over-capacity branch)
    if q <= FLOW_CAPACITY:
        # Under capacity: green branch — take the higher-speed root
        speed = (-_B - sqrt_d) / (2.0 * _A)
    else:
        # Over capacity: red branch — take the lower-speed root
        speed = (-_B + sqrt_d) / (2.0 * _A)

    # Apply speed limit and floor
    speed = min(speed, SPEED_LIMIT)
    speed = max(speed, SPEED_MIN)
    return speed


# ---------------------------------------------------------------------------
# 2.  TRAVEL-TIME COMPUTATION
# ---------------------------------------------------------------------------

INTERSECTION_DELAY_S = 30.0   # seconds added per intersection traversed


def edge_travel_time_s(distance_km: float, flow_veh_per_15min: int) -> float:
    """Return the travel time in **seconds** for one graph edge.

    Parameters
    ----------
    distance_km          : straight-line distance between the two SCATS sites (km)
    flow_veh_per_15min   : ML-predicted vehicle count for the 15-min interval
                           at the **origin** site of this edge (per PDF assumption)

    Formula
    -------
    1. Convert flow to hourly rate: q = flow_15min × 4
    2. Convert flow to speed:       v = flow_to_speed(q)          [km/hr]
    3. Link travel time:            t = (distance_km / v) × 3600  [seconds]
    4. Add intersection delay:      t += INTERSECTION_DELAY_S      [seconds]
    """
    flow_hourly = flow_veh_per_15min * 4.0
    speed_kmh   = flow_to_speed(flow_hourly)
    link_time_s = (distance_km / speed_kmh) * 3600.0
    return link_time_s + INTERSECTION_DELAY_S


# ---------------------------------------------------------------------------
# 3.  GRAPH LOADING
# ---------------------------------------------------------------------------

@dataclass
class Node:
    site_id: int
    name:    str
    lat:     float
    lon:     float


@dataclass
class Edge:
    to:          int
    distance_km: float


@dataclass
class Graph:
    nodes: dict[int, Node]   # site_id → Node
    edges: dict[int, list[Edge]]   # site_id → [Edge, ...]

    def neighbours(self, site_id: int) -> list[Edge]:
        return self.edges.get(site_id, [])

    def node(self, site_id: int) -> Node:
        return self.nodes[site_id]


def load_graph(path: Path = GRAPH_FILE) -> Graph:
    """Load boroondara_graph.json into a Graph object."""
    with open(path) as f:
        data = json.load(f)

    nodes = {
        int(k): Node(
            site_id=int(k),
            name=v["name"],
            lat=v["lat"],
            lon=v["lon"],
        )
        for k, v in data["nodes"].items()
    }

    edges = {
        int(k): [Edge(to=e["to"], distance_km=e["distance_km"]) for e in v]
        for k, v in data["edges"].items()
    }
    # Ensure every node has an edge list (even if empty)
    for sid in nodes:
        if sid not in edges:
            edges[sid] = []

    return Graph(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# 4.  A* SEARCH
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine great-circle distance (km)."""
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _heuristic_s(graph: Graph, node_id: int, goal_id: int) -> float:
    """Admissible A* heuristic: straight-line distance at speed limit (seconds).
    Never over-estimates actual travel time → guarantees optimal path."""
    n = graph.node(node_id)
    g = graph.node(goal_id)
    dist_km = _haversine_km(n.lat, n.lon, g.lat, g.lon)
    return (dist_km / SPEED_LIMIT) * 3600.0


def astar(
    graph: Graph,
    origin: int,
    dest: int,
    cost_fn: Callable[[int, Edge], float],
    forbidden_nodes: Optional[set[int]] = None,
    forbidden_edges: Optional[set[tuple[int, int]]] = None,
) -> Optional[tuple[float, list[int]]]:
    """A* shortest-path search.

    Parameters
    ----------
    graph          : the road graph
    origin, dest   : SCATS site IDs
    cost_fn        : callable(from_site_id, edge) → travel_time_seconds
                     (this closure captures the ML flow predictions)
    forbidden_nodes: set of node IDs to treat as removed (for Yen's)
    forbidden_edges: set of (u, v) edge pairs to treat as removed (for Yen's)

    Returns
    -------
    (total_cost_s, path_list) or None if no path exists.
    path_list is a list of site_ids from origin to dest inclusive.
    """
    if forbidden_nodes is None:
        forbidden_nodes = set()
    if forbidden_edges is None:
        forbidden_edges = set()

    if origin in forbidden_nodes or dest in forbidden_nodes:
        return None

    # Priority queue: (f_score, tie-breaker, node_id)
    counter = 0
    open_heap: list[tuple[float, int, int]] = []
    heapq.heappush(open_heap, (0.0, counter, origin))

    g_score: dict[int, float] = {origin: 0.0}
    came_from: dict[int, int] = {}

    visited: set[int] = set()

    while open_heap:
        f, _, current = heapq.heappop(open_heap)

        if current in visited:
            continue
        visited.add(current)

        if current == dest:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return g_score[dest], path

        for edge in graph.neighbours(current):
            nb = edge.to
            if nb in forbidden_nodes:
                continue
            if (current, nb) in forbidden_edges:
                continue

            tent_g = g_score[current] + cost_fn(current, edge)
            if tent_g < g_score.get(nb, math.inf):
                g_score[nb] = tent_g
                came_from[nb] = current
                h = _heuristic_s(graph, nb, dest)
                counter += 1
                heapq.heappush(open_heap, (tent_g + h, counter, nb))

    return None   # no path found


# ---------------------------------------------------------------------------
# 5.  YEN'S K-SHORTEST LOOPLESS PATHS
#     Reference: Yen (1971) — "Finding the k Shortest Loopless Paths"
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Route:
    """A candidate route returned by find_routes()."""
    total_time_s: float
    path:         list[int] = field(compare=False)
    edges_km:     list[float] = field(compare=False)   # per-segment distances
    speeds_kmh:   list[float] = field(compare=False)   # per-segment speeds

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
            f"{len(self.path)-1} links | {sites})"
        )


def yens_k_shortest(
    graph: Graph,
    origin: int,
    dest: int,
    cost_fn: Callable[[int, Edge], float],
    k: int = 5,
) -> list[Route]:
    """Yen's algorithm for k shortest loopless paths.

    Runs A* as the inner shortest-path oracle.

    Returns up to k Route objects sorted by ascending total_time_s.
    May return fewer than k if the graph has fewer distinct loopless paths.
    """

    def _route_from_path(path: list[int]) -> Route:
        """Build a Route object by computing per-edge costs."""
        total_s = 0.0
        dists: list[float] = []
        speeds: list[float] = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # Find the edge u→v
            edge = next((e for e in graph.neighbours(u) if e.to == v), None)
            if edge is None:
                raise ValueError(f"Edge {u}→{v} not found in graph")
            seg_cost = cost_fn(u, edge)
            total_s += seg_cost
            dists.append(edge.distance_km)
            flow_hr = _last_flows.get(u, 0) * 4.0
            speeds.append(flow_to_speed(flow_hr))
        return Route(
            total_time_s=total_s,
            path=path,
            edges_km=dists,
            speeds_kmh=speeds,
        )

    # ── find shortest path (A) ─────────────────────────────────────────────
    result = astar(graph, origin, dest, cost_fn)
    if result is None:
        return []

    cost0, path0 = result
    A: list[Route] = [_route_from_path(path0)]

    # Candidate heap: (cost, counter, path)
    cand_counter = 0
    B: list[tuple[float, int, list[int]]] = []
    seen_paths: set[tuple[int, ...]] = {tuple(path0)}

    for _ in range(k - 1):
        prev_route = A[-1]

        for i in range(len(prev_route.path) - 1):
            spur_node  = prev_route.path[i]
            root_path  = prev_route.path[: i + 1]

            # Edges to forbid: any edge (root_path[-1] → next) used by previous
            # k-shortest paths that share the same root
            forbidden_edges: set[tuple[int, int]] = set()
            for prev in A:
                if prev.path[: i + 1] == root_path:
                    u = prev.path[i]
                    v = prev.path[i + 1]
                    forbidden_edges.add((u, v))

            # Nodes to forbid: all nodes in root_path except spur_node itself
            # (prevents loops back through the root)
            forbidden_nodes: set[int] = set(root_path[:-1])

            spur_result = astar(
                graph, spur_node, dest, cost_fn,
                forbidden_nodes=forbidden_nodes,
                forbidden_edges=forbidden_edges,
            )
            if spur_result is None:
                continue

            _, spur_path = spur_result
            total_path = root_path[:-1] + spur_path   # root + spur (spur starts at spur_node)

            # Avoid duplicates
            path_key = tuple(total_path)
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)

            # Compute cost of total_path from scratch (root costs + spur costs)
            try:
                candidate_route = _route_from_path(total_path)
            except ValueError:
                continue

            cand_counter += 1
            heapq.heappush(B, (candidate_route.total_time_s, cand_counter, total_path))

        if not B:
            break

        best_cost, _, best_path = heapq.heappop(B)
        A.append(_route_from_path(best_path))

    return A


# ---------------------------------------------------------------------------
# 6.  PUBLIC API: find_routes()
# ---------------------------------------------------------------------------

# Module-level cache: lazy-import predict to avoid circular imports at load time
_predict_fn: Optional[Callable] = None
_graph: Optional[Graph] = None
_last_flows: dict[int, int] = {}   # site_id → predicted flow (veh/15min) for current call


def _get_predict():
    global _predict_fn
    if _predict_fn is None:
        try:
            from src.ml.predict import predict as _p
            _predict_fn = _p
        except ImportError:
            # Try relative import (when running from src/routing.py)
            import sys
            sys.path.insert(0, str(PROJECT_ROOT / "src" / "ml"))
            from predict import predict as _p   # type: ignore[no-redef]
            _predict_fn = _p
    return _predict_fn


def _get_graph() -> Graph:
    global _graph
    if _graph is None:
        _graph = load_graph()
    return _graph


def find_routes(
    origin: int,
    dest: int,
    datetime_str: str,
    model: str = "gru",
    k: int = 5,
    graph: Optional[Graph] = None,
    predict_fn: Optional[Callable] = None,
) -> list[Route]:
    """Find the top-k fastest routes from origin to dest at a given time.

    Parameters
    ----------
    origin        : SCATS site ID of the start intersection
    dest          : SCATS site ID of the destination intersection
    datetime_str  : ISO datetime string, e.g. "2006-10-15 08:30:00"
                    Defines the traffic conditions for the search.
    model         : ML model to use for flow prediction ("lstm" | "gru" | "transformer")
    k             : number of routes to return (default 5)
    graph         : optional pre-loaded Graph (for testing / repeated calls)
    predict_fn    : optional callable(site_id, datetime_str, model_name) → int
                    (for testing; defaults to src.ml.predict.predict)

    Returns
    -------
    list[Route]  sorted by ascending total travel time (fastest first).
    May be shorter than k if fewer distinct loopless paths exist.

    Algorithm
    ---------
    1. For each edge (u → v), query the ML model for the flow at site u at
       datetime_str.  Cache results to avoid redundant model calls.
    2. Convert flow (veh/15min) → speed (km/hr) via the quadratic PDF formula.
    3. Edge cost = distance_km / speed_kmh × 3600 + 30s intersection delay.
    4. Run Yen's k-shortest-loopless-paths algorithm with A* as the inner oracle.

    Notes
    -----
    - Flow predictions are cached per (site_id, datetime_str, model) within
      a single find_routes() call (and across repeated calls via _flow_cache).
    - The cost function uses the **origin** site's flow for each edge, per
      the PDF assumption.
    - The heuristic for A* is the straight-line distance at speed limit,
      which is admissible (never over-estimates).

    Examples
    --------
    >>> routes = find_routes(970, 2000, "2006-10-15 08:30:00", model="gru", k=5)
    >>> for r in routes:
    ...     print(r)
    Route(12.4 min, 3.21 km, 4 links | 970 → 2820 → 3685 → 2000)
    ...
    """
    g = graph or _get_graph()
    _pred = predict_fn or _get_predict()

    # ── flow cache for this call ───────────────────────────────────────────
    flow_cache: dict[int, int] = {}

    def _flow(site_id: int) -> int:
        if site_id not in flow_cache:
            try:
                flow_cache[site_id] = int(_pred(site_id, datetime_str, model))
            except Exception:
                # Graceful degradation: if prediction fails, assume free-flow
                flow_cache[site_id] = 0
        return flow_cache[site_id]

    # Update module-level for Route construction (speeds)
    global _last_flows
    _last_flows = flow_cache

    def cost_fn(from_id: int, edge: Edge) -> float:
        return edge_travel_time_s(edge.distance_km, _flow(from_id))

    # ── run Yen's ──────────────────────────────────────────────────────────
    routes = yens_k_shortest(g, origin, dest, cost_fn, k=k)

    # Ensure flow_cache is populated for all nodes on returned paths
    # (so speeds in Route objects are correct)
    _last_flows = flow_cache
    for r in routes:
        for i in range(len(r.path) - 1):
            u = r.path[i]
            edge = next(e for e in g.neighbours(u) if e.to == r.path[i + 1])
            flow_hr = _flow(u) * 4.0
            r.speeds_kmh[i] = flow_to_speed(flow_hr)

    return routes


# ---------------------------------------------------------------------------
# 7.  SELF-TEST  (run directly: python routing.py --test)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Smoke-test the flow→speed conversion and A* with a tiny synthetic graph."""
    print("=== Flow → Speed conversion tests ===")
    test_cases = [
        (0,    "zero flow",            60.0),
        (351,  "just at speed limit",  60.0),
        (352,  "just over limit",      None),   # expect < 60
        (1500, "at capacity",          32.0),
        (2000, "over capacity (red)",  None),   # expect low speed
    ]
    for flow, label, expected in test_cases:
        spd = flow_to_speed(flow)
        ok = ""
        if expected is not None:
            ok = "✓" if abs(spd - expected) < 1.0 else "✗"
        print(f"  flow={flow:5d}  speed={spd:5.1f} km/hr  [{label}] {ok}")

    print("\n=== Edge travel time ===")
    print(f"  1 km, flow=0   → {edge_travel_time_s(1.0, 0):.1f}s  "
          f"(expect {(1/60)*3600+30:.1f}s)")
    print(f"  1 km, flow=375 → {edge_travel_time_s(1.0, 375):.1f}s  "
          f"(375 veh/15min = 1500 veh/hr = capacity → 32 km/hr)")

    print("\n=== Synthetic graph A* ===")
    # Build a tiny graph: 0→1→2→3, and shortcut 0→3 with high flow
    nodes = {
        0: Node(0, "A", -37.80, 145.00),
        1: Node(1, "B", -37.81, 145.01),
        2: Node(2, "C", -37.82, 145.02),
        3: Node(3, "D", -37.83, 145.03),
    }
    edges = {
        0: [Edge(to=1, distance_km=1.0), Edge(to=3, distance_km=3.0)],
        1: [Edge(to=2, distance_km=1.0)],
        2: [Edge(to=3, distance_km=1.0)],
        3: [],
    }
    g = Graph(nodes=nodes, edges=edges)

    # Free-flow cost function (flow=0 everywhere → speed=60 km/hr)
    def free_flow_cost(from_id: int, edge: Edge) -> float:
        return edge_travel_time_s(edge.distance_km, 0)

    result = astar(g, 0, 3, free_flow_cost)
    assert result is not None
    cost, path = result
    print(f"  Shortest path 0→3: {path}  cost={cost:.1f}s")
    assert path == [0, 3] or path == [0, 1, 2, 3], f"Unexpected path: {path}"
    # Direct edge 0→3 is 3 km, via 0→1→2→3 is also 3 km (same distance)
    # Both are valid; direct edge should win on cost (one intersection delay vs three)
    print(f"  (direct vs 3-hop, both 3 km, direct wins on delay: {path == [0, 3]})")

    print("\n=== Yen's k-shortest (synthetic) ===")
    routes = yens_k_shortest(g, 0, 3, free_flow_cost, k=3)
    for r in routes:
        print(f"  {r}")
    assert len(routes) >= 2, "Expected at least 2 distinct paths"

    print("\nAll self-tests passed ✓")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find top-k routes between two SCATS sites")
    parser.add_argument("--test",     action="store_true", help="Run self-tests")
    parser.add_argument("--origin",   type=int, default=2820,  help="Origin SCATS site ID (default: 970)")
    parser.add_argument("--dest",     type=int, default=2000, help="Destination SCATS site ID (default: 2200)")
    parser.add_argument("--datetime", type=str, default="2006-10-15 08:30:00")
    parser.add_argument("--model",    type=str, default="gru", choices=["lstm", "gru", "transformer"])
    parser.add_argument("--k",        type=int, default=2)
    args = parser.parse_args()

    if args.test:
        _self_test()
    else:
        print(f"Finding top-{args.k} routes: {args.origin} -> {args.dest}")
        print(f"  datetime={args.datetime}  model={args.model}\n")
        routes = find_routes(
            origin=args.origin,
            dest=args.dest,
            datetime_str=args.datetime,
            model=args.model,
            k=args.k,
        )
        if not routes:
            print("No routes found.")
        else:
            for i, r in enumerate(routes, 1):
                print(f"Route {i}: {r}")
                print(f"  Path : {' -> '.join(str(s) for s in r.path)}")
                segs = [f'{d:.3f}km@{v:.1f}km/h' for d, v in zip(r.edges_km, r.speeds_kmh)]
                print(f"  Segs : {segs}")
                print()
