from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import xlrd
import xlrd.book


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LISTING_FILE = PROJECT_ROOT / "data" / "raw" / "scats_site_listing.xls"
LOCATIONS_FILE = PROJECT_ROOT / "data" / "processed" / "site_locations.csv"
OUT_GRAPH = PROJECT_ROOT / "data" / "processed" / "boroondara_graph.json"


# ---------------------------------------------------------------------------
# Road-name aliases — extend when the map shows a misalignment.
# Key = token as it appears in the listing's "Location Description" (between
# the slashes); value = canonical road key used for adjacency matching.
# ---------------------------------------------------------------------------

ROAD_ALIASES: dict[str, str] = {
    # Same road, different name segments
    "EASTERN FWY": "EASTERN",
    "S.E.ARTERIAL": "MONASH",
    "CHANDLER HWY": "CHANDLER",
    "PRINCESS ST": "PRINCESS",       # 2820 vs 3662 listing inconsistency
    "MAROONDAH": "WHITEHORSE",       # Maroondah Hwy is the eastward continuation
                                     # of Whitehorse Rd at Mont Albert Rd. Links
                                     # isolated 2200 (Maroondah/Union) into the
                                     # Whitehorse Rd chain.
    # NOTE: BURWOOD HWY is NOT aliased to BURWOOD. Per IBM Boroondara map
    # verification, the two roads do not form a continuous corridor between
    # the SCATS sites in this dataset (intermediates 4049/4050 etc. exist on
    # the map but are absent from our 40-site subset). Site 2000 only links
    # to other Warrigal Rd sites.
}

# Tokens that appear in descriptions but aren't real roads — filtered out.
NON_ROAD_TOKENS: set[str] = {
    "ON RAMP",
    "OFF RAMP",
}

# Per-site road-set overrides. Use when the master listing's road name is
# ambiguous (same name, geographically different road) — e.g. site 2846's
# "HIGH" is High St Malvern/Caulfield (on the Monash Fwy), NOT the same as
# High St Kew that 3001/3662/4321/4335/4030 sit on. Without this override
# the script generates spurious 6-7 km edges between them.
SITE_ROAD_OVERRIDES: dict[int, set[str]] = {
    # 2846 (MONASH/HIGH/WILLS): the "HIGH" here is actually High Street Road
    # in Glen Iris, NOT High St Kew. Without this override 2846 spuriously links
    # to every Kew "HIGH" site (3001/3662/4321/4335/4030) at 6+km. With the
    # explicit "HIGH ST RD" we get a single 3km link to 970 (Warrigal/High St Rd)
    # which is geographically correct.
    2846: {"MONASH", "WILLS", "HIGH ST RD"},
}


# ---------------------------------------------------------------------------
# Manually-added edges — for sites that share a real road but the master
# listing's road labels don't reveal it (typically short connector streets).
# Distances are computed from lat/lon at build time. Each entry adds a
# bidirectional edge.
# ---------------------------------------------------------------------------

MANUAL_EDGES: list[tuple[int, int]] = [
    # 4812 (Swan/Madden, inner Burnley): Swan St connects to Bridge Rd via
    # short cross-streets; 4262 is the nearest in-dataset peer at 0.82km.
    (4812, 4262),
    # 4821 (Burnley/Victoria/Walmer, inner Richmond): Walmer/Burnley feed
    # into Bridge Rd; 4262 again the nearest in-dataset peer at 1.12km.
    (4821, 4262),
]


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


def is_between(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float],
               tol_km: float | None = None) -> bool:
    """True if c's projection onto segment a-b falls strictly between a and b
    AND c lies within `tol_km` of the line.

    Tolerance scales with segment length to handle curved roads: a 7km edge
    along Burwood Rd kinks south enough that intermediates sit ~700m off the
    great-circle line. Default = max(0.2, 0.1 * ab_km).
    """
    ax, ay = a
    bx, by = b
    cx, cy = c
    abx, aby = bx - ax, by - ay
    acx, acy = cx - ax, cy - ay
    ab_sq = abx * abx + aby * aby
    if ab_sq == 0:
        return False
    t = (acx * abx + acy * aby) / ab_sq
    if not (0.0 < t < 1.0):
        return False
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    perp_km = haversine_km(cx, cy, proj_x, proj_y)
    if tol_km is None:
        ab_km = haversine_km(ax, ay, bx, by)
        tol_km = max(0.2, 0.1 * ab_km)
    return perp_km <= tol_km


# ---------------------------------------------------------------------------
# Master listing parser
# ---------------------------------------------------------------------------

def _read_listing_rows() -> list[tuple[int, str]]:
    """Yield (site_id, description) tuples from the master listing.

    The .xls is a legacy file with a malformed Defined Names section that
    crashes xlrd's epilogue step. We monkey-patch the offending method to
    skip it; the spreadsheet data itself is fine to read.
    """
    if not LISTING_FILE.exists():
        sys.exit(
            f"ERROR: master site listing not found at {LISTING_FILE}\n"
            f"  Save 'SCATSSiteListingSpreadsheet_VicRoads.xls' there and re-run.\n"
            f"  PowerShell: Copy-Item 'E:\\downloads\\SCATSSiteListingSpreadsheet_VicRoads(1).xls' "
            f"'{LISTING_FILE}'"
        )

    xlrd.book.Book.names_epilogue = lambda self: None
    bk = xlrd.open_workbook(str(LISTING_FILE), ignore_workbook_corruption=True)
    sheet = bk.sheet_by_name("SCATS Site Numbers")

    rows: list[tuple[int, str]] = []
    # Header is at row 9; data begins at row 10
    for r in range(10, sheet.nrows):
        raw_id = sheet.cell_value(r, 0)
        desc = sheet.cell_value(r, 1)
        if not raw_id or not desc:
            continue
        try:
            sid = int(raw_id)
        except (TypeError, ValueError):
            continue
        rows.append((sid, str(desc).strip()))
    return rows


def _parse_road_set(description: str) -> set[str]:
    """'BURWOOD HWY/WARRIGAL' -> {'BURWOOD', 'WARRIGAL'} after alias mapping."""
    raw_tokens = [t.strip().upper() for t in description.split("/")]
    roads: set[str] = set()
    for tok in raw_tokens:
        if not tok or tok in NON_ROAD_TOKENS:
            continue
        roads.add(ROAD_ALIASES.get(tok, tok))
    return roads


def load_roads_per_site(needed_sites: set[int]) -> dict[int, set[str]]:
    """Return {site_id -> normalised road set} for every site in needed_sites."""
    out: dict[int, set[str]] = {}
    for sid, desc in _read_listing_rows():
        if sid in needed_sites:
            out[sid] = _parse_road_set(desc)
    # Apply per-site overrides last so they always win over the parsed set.
    for sid, override in SITE_ROAD_OVERRIDES.items():
        if sid in out:
            out[sid] = set(override)
    return out


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------

def build_adjacency(
    sites: pd.DataFrame,
    roads_per_site: dict[int, set[str]],
) -> dict[int, list[tuple[int, float]]]:
    """For each site, find nearest neighbour along each shared road, with no
    other site in between."""
    site_ids = sites["site_id"].tolist()
    coords = {int(r.site_id): (float(r.lat), float(r.lon)) for r in sites.itertuples()}

    adj: dict[int, set[tuple[int, float]]] = {sid: set() for sid in site_ids}

    for i, a in enumerate(site_ids):
        a_roads = roads_per_site.get(a, set())
        if not a_roads:
            continue
        for b in site_ids[i + 1:]:
            shared = a_roads & roads_per_site.get(b, set())
            if not shared:
                continue
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
    print(f"Loading site locations from {LOCATIONS_FILE} ...")
    sites = pd.read_csv(LOCATIONS_FILE)
    print(f"  {len(sites)} sites")

    needed = {int(s) for s in sites["site_id"]}
    print(f"Reading road descriptions from {LISTING_FILE} ...")
    roads_per_site = load_roads_per_site(needed)
    missing = needed - set(roads_per_site)
    if missing:
        print(f"  WARNING: {len(missing)} sites not found in master listing: {sorted(missing)}")
    print(f"  Got road sets for {len(roads_per_site)}/{len(needed)} sites")
    sample_sid = next(iter(roads_per_site))
    print(f"  Example: site {sample_sid} -> roads {sorted(roads_per_site[sample_sid])}")

    print("Building adjacency ...")
    adj = build_adjacency(sites, roads_per_site)

    # Apply manually-added edges (for connections that don't appear via shared
    # road names — short connector streets etc).
    coords = {int(r.site_id): (float(r.lat), float(r.lon)) for r in sites.itertuples()}
    if MANUAL_EDGES:
        print(f"Adding {len(MANUAL_EDGES)} manual edge(s):")
    for a, b in MANUAL_EDGES:
        if a not in coords or b not in coords:
            print(f"  SKIP ({a}, {b}): one or both sites not in dataset")
            continue
        d = round(haversine_km(*coords[a], *coords[b]), 3)
        # Insert in sorted position; skip if the edge already exists
        if not any(n == b for n, _ in adj[a]):
            adj[a].append((b, d))
            adj[a].sort()
        if not any(n == a for n, _ in adj[b]):
            adj[b].append((a, d))
            adj[b].sort()
        print(f"  + {a} <-> {b} ({d:.2f}km)")
    edge_count = sum(len(v) for v in adj.values())
    isolated = [sid for sid, n in adj.items() if not n]
    longest = max(
        ((sid, n, d) for sid, edges in adj.items() for n, d in edges),
        key=lambda t: t[2],
        default=(None, None, 0),
    )
    print(f"  Total directed edges: {edge_count}")
    print(f"  Average out-degree: {edge_count / len(sites):.2f}")
    print(f"  Isolated nodes ({len(isolated)}): {isolated}")
    print(f"  Longest edge: {longest[0]} -> {longest[1]} = {longest[2]:.2f} km")

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

if __name__ == "__main__":
    main()
