"""
Build the Boroondara road graph from SCATS site locations + intersection names.

OWNER: Person A

Input:  data/raw/Scats_Data_October_2006.xls   (for the Location strings)
        data/processed/site_locations.csv      (site centroids from parse_scats.py)
Output: data/processed/boroondara_graph.json   (per INTERFACES.md interface 2)

Adjacency rule (FIRST PASS — needs validation against the Boroondara map PDF):
    Two SCATS sites A and B are connected by a directed edge in BOTH directions if:
      (1) A and B share at least one road name, AND
      (2) No other SCATS site C lies strictly between A and B along that road
          (where "between" means C also sits on that road, and C's projection onto
          the A-B great-circle line falls strictly between A and B).

This produces a reasonable first-pass graph but WILL contain errors. Person A
should overlay the resulting edges on the Boroondara map PDF and manually
remove/add edges as needed.

Usage:
    python -m src.data.build_graph
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "Scats_Data_October_2006.xls"
LOCATIONS_FILE = PROJECT_ROOT / "data" / "processed" / "site_locations.csv"
OUT_GRAPH = PROJECT_ROOT / "data" / "processed" / "boroondara_graph.json"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

EARTH_R_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) pairs in degrees."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Road-name extraction
# ---------------------------------------------------------------------------

# Location strings look like:
#   "WARRIGAL_RD N of TOORAK_RD"
#   "BURWOOD_HWY E of WARRIGAL_RD"
#   "HIGH_ST NE OF HARP_ST"
#   "WARRIGAL_RD N of HIGH STREET_RD"   <- some names contain internal spaces
# Pattern: anchor on " <DIR> [oO][fF] " separator (DIR is N|S|E|W and combos).
LOCATION_RE = re.compile(
    r"^\s*(?P<road>.+?)\s+(?P<dir>[NSEW]{1,2})\s+[oO][fF]\s+(?P<cross>.+?)\s*$"
)


def normalise_road_name(name: str) -> str:
    """Collapse minor naming variations so the same road matches across sites.

    Examples:
        WARRIGAL_RD       -> WARRIGAL
        WARRIGAL_HWY      -> WARRIGAL  (treat HWY/RD/ST suffixes as same road)
        HIGH STREET_RD    -> HIGH_STREET
        HIGH_STREET_RD    -> HIGH_STREET
    """
    n = name.upper().strip().replace(" ", "_")
    # collapse double underscores
    while "__" in n:
        n = n.replace("__", "_")
    # strip a single trailing road-type suffix
    for suffix in ("_RD", "_ST", "_STREET", "_HWY", "_HIGHWAY", "_AV", "_AVE",
                   "_DR", "_DRIVE", "_PDE", "_PARADE", "_PL"):
        if n.endswith(suffix):
            return n[: -len(suffix)]
    return n


def extract_roads_at_site(locations: list[str]) -> set[str]:
    """Given the list of detector location strings at a site, return the set
    of (normalised) road names that pass through that intersection."""
    roads: set[str] = set()
    for loc in locations:
        m = LOCATION_RE.match(loc)
        if not m:
            continue
        roads.add(normalise_road_name(m.group("road")))
        roads.add(normalise_road_name(m.group("cross")))
    return roads


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------

def is_between(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float],
               tol_km: float = 0.15) -> bool:
    """True if c's projection onto the segment a-b falls strictly between a and b
    AND c lies within `tol_km` of the line a-b. Used to detect a SCATS site that
    sits on the road between two others."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    # vector from a to b in degrees (good enough for short distances)
    abx, aby = bx - ax, by - ay
    acx, acy = cx - ax, cy - ay
    ab_sq = abx * abx + aby * aby
    if ab_sq == 0:
        return False
    t = (acx * abx + acy * aby) / ab_sq
    if not (0.0 < t < 1.0):
        return False
    # perpendicular distance from c to segment (in km, rough)
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    perp_km = haversine_km(cx, cy, proj_x, proj_y)
    return perp_km <= tol_km


def build_adjacency(
    sites: pd.DataFrame,
    roads_per_site: dict[int, set[str]],
) -> dict[int, list[tuple[int, float]]]:
    """For each site, find its nearest neighbour along each shared road, with no
    other site in between. Returns adjacency as {site_id: [(neighbour, dist_km)]}."""
    site_ids = sites["site_id"].tolist()
    coords = {int(r.site_id): (float(r.lat), float(r.lon)) for r in sites.itertuples()}

    adj: dict[int, set[tuple[int, float]]] = {sid: set() for sid in site_ids}

    # For every pair (A, B) sharing a road: candidate edge.
    # Reject the edge if any other site C also on that road lies between A and B.
    for i, a in enumerate(site_ids):
        a_roads = roads_per_site.get(a, set())
        if not a_roads:
            continue
        for b in site_ids[i + 1:]:
            shared = a_roads & roads_per_site.get(b, set())
            if not shared:
                continue
            # Check at least one road has no intermediate site
            has_clear_road = False
            for road in shared:
                blocked = False
                for c in site_ids:
                    if c == a or c == b:
                        continue
                    if road not in roads_per_site.get(c, set()):
                        continue
                    if is_between(coords[a], coords[b], coords[c]):
                        blocked = True
                        break
                if not blocked:
                    has_clear_road = True
                    break
            if has_clear_road:
                d = haversine_km(*coords[a], *coords[b])
                adj[a].add((b, round(d, 3)))
                adj[b].add((a, round(d, 3)))

    return {sid: sorted(list(neighbours)) for sid, neighbours in adj.items()}


def main() -> None:
    print("Loading site locations ...")
    sites = pd.read_csv(LOCATIONS_FILE)
    print(f"  {len(sites)} sites")

    print("Extracting roads at each site from raw data ...")
    raw = pd.read_excel(RAW_FILE, sheet_name="Data", header=1, usecols=["SCATS Number", "Location"])
    raw = raw.drop_duplicates()
    roads_per_site: dict[int, set[str]] = {}
    for sid, group in raw.groupby("SCATS Number"):
        roads_per_site[int(sid)] = extract_roads_at_site(list(group["Location"]))
    sample_sid = next(iter(roads_per_site))
    print(f"  Example: site {sample_sid} -> roads {sorted(roads_per_site[sample_sid])}")

    print("Building adjacency ...")
    adj = build_adjacency(sites, roads_per_site)
    edge_count = sum(len(v) for v in adj.values())
    print(f"  Total directed edges: {edge_count}")
    avg_degree = edge_count / len(sites) if sites.size else 0
    print(f"  Average out-degree: {avg_degree:.2f}")

    # Build JSON
    nodes_json = {}
    for r in sites.itertuples():
        nodes_json[str(int(r.site_id))] = {
            "name": r.name,
            "lat": round(float(r.lat), 6),
            "lon": round(float(r.lon), 6),
        }

    edges_json = {}
    for sid, neighbours in adj.items():
        edges_json[str(sid)] = [
            {"to": int(n), "distance_km": float(d)} for n, d in neighbours
        ]

    OUT_GRAPH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_GRAPH, "w") as f:
        json.dump({"nodes": nodes_json, "edges": edges_json}, f, indent=2)

    print(f"\nWrote {OUT_GRAPH}")
    print("\nNEXT STEP for Person A:")
    print("  Open the Boroondara map PDF and visually verify each edge.")
    print("  Likely manual fixes: missing edges across freeways, spurious edges")
    print("  along roads that don't actually connect (e.g. one-way streets).")


if __name__ == "__main__":
    main()
