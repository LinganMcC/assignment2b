# Boroondara graph — manual fixes and design decisions

This file documents every non-automatic intervention applied during graph
construction (`src/data/build_graph.py`). It exists so the report can cite
specific decisions and so future contributors understand why the heuristic
needed help in particular places.

**Final graph:** 40 nodes, 51 undirected edges, 0 isolated nodes, longest
edge 3.06 km (3682 ↔ 3804 along Riversdale Rd).

---

## 1. Switch from per-detector strings to master listing

**What changed.** The original `build_graph.py` parsed the per-detector
`Location` strings in `Scats_Data_October_2006.xls` (e.g.
`"WARRIGAL_RD N of TOORAK_RD"`), ran a regex over them, and stripped
suffixes (`_RD`, `_HWY`, ...) to recover canonical road names.

**Why it was a problem.** Suffix stripping was inconsistent
(`HIGH STREET_RD` collapsed to `HIGH_STREET` while `HIGH_ST` became
`HIGH` — same road, different keys). Some sites had only one detector,
yielding very thin road sets. Names like `S.E.ARTERIAL` had no suffix to
strip and stayed unique. The result was many phantom 6–7 km edges
(initially 9 edges > 6 km).

**Fix.** Switched to the VicRoads SCATS Site Listing
(`data/raw/scats_site_listing.xls`), which gives one canonical
slash-separated string per intersection (e.g. `BURWOOD HWY/WARRIGAL`).
Tokens are split on `/` and used directly.

---

## 2. ROAD_ALIASES (collisions in the master listing)

The master listing isn't perfectly consistent — the same road can appear
under different names at different sites. These aliases unify them:

| Listing token | Canonical key | Reason |
|---------------|---------------|--------|
| `EASTERN FWY` | `EASTERN` | Sites 2825 and 2827 both on Eastern Fwy |
| `S.E.ARTERIAL` | `MONASH` | Same road designation |
| `CHANDLER HWY` | `CHANDLER` | Suffix variant |
| `PRINCESS ST` | `PRINCESS` | Site 2820 vs 3662 listing inconsistency |
| `MAROONDAH` | `WHITEHORSE` | Maroondah Hwy is the eastward continuation of Whitehorse Rd at Mont Albert Rd; links isolated 2200 into the Whitehorse chain |

**Deliberately NOT aliased:** `BURWOOD HWY → BURWOOD`. IBM map verification
showed the two roads do not form a continuous corridor between the SCATS
sites in this dataset; intermediate sites (4049, 4050, etc.) exist on the
map but are absent from our 40-site subset. Adding the alias produced a
spurious 5.36 km edge from 2000 (Warrigal/Burwood Hwy) to 4266
(Burwood/Auburn), which the map showed was not a direct connection.

---

## 3. SITE_ROAD_OVERRIDES (ambiguous names)

Per-site overrides apply after the listing-derived road sets are built.
Used when the same road name in the master listing refers to
geographically different roads.

| Site | Listing description | Override road set | Reason |
|------|---------------------|-------------------|--------|
| 2846 | `MONASH/HIGH/WILLS` | `{MONASH, WILLS, HIGH ST RD}` | The "HIGH" at 2846 is High Street Road (Glen Iris), NOT the same as High St (Kew) at sites 3001/3662/4321/4335/4030. Without this override, 2846 would link to every Kew "HIGH" site at 6+ km. With `HIGH ST RD` explicitly listed, 2846 correctly links to site 970 (Warrigal/High St Rd) at 3.03 km. |

---

## 4. Length-scaling `is_between()` tolerance

**Problem.** The original tolerance for "is site C between A and B?" was
fixed at 150 m perpendicular distance from the great-circle line. This
worked for straight roads but failed on curved ones — Burwood Rd kinks
south as it goes east, so intermediate sites 4263, 4264, and 4266 sat
233 m, 477 m, and 723 m off the straight line from 2000 to 4262 and
weren't blocking the spurious 7.74 km edge.

**Fix.** Tolerance now scales with segment length:
`tol_km = max(0.2, 0.1 * ab_km)`. For a 7 km edge this gives 700 m of
slack — enough to handle realistic road curvature without producing
false positives. Floor of 200 m ensures short edges aren't over-tight.

---

## 5. MANUAL_EDGES (connections invisible to the heuristic)

Two edges are added by hand after `build_adjacency()` because the
sites involved share a real road link in Melbourne, but the master
listing's road labels don't name the connector.

| Edge | Distance | Reason |
|------|----------|--------|
| 4812 ↔ 4262 | 0.82 km | 4812 (Swan/Madden, Burnley) connects to 4262 (Bridge/Burwood, Hawthorn) via Hawthorn Bridge. Bridge Rd is named because it crosses the Yarra here. |
| 4821 ↔ 4262 | 1.11 km | 4821 (Walmer/Victoria/Burnley, Richmond) reaches 4262 via the same Bridge Rd corridor through Burnley. |

Without these manual edges, 4812 and 4821 would be unreachable in the
graph because their road names (Swan St, Madden Gv, Walmer St, etc.)
don't appear at any other in-dataset site.

---

## 6. Resolution summary for the four originally-isolated sites

After all of the above, every site has at least one neighbour.

| Site | Originally isolated | Now linked via |
|------|---------------------|----------------|
| 2200 (Maroondah/Union) | yes | `MAROONDAH → WHITEHORSE` alias → 4063 (1.60 km) |
| 2846 (Monash/Wills) | yes | `HIGH ST RD` added to override → 970 (3.03 km) |
| 4812 (Swan/Madden) | yes | manual edge → 4262 (0.82 km) |
| 4821 (Walmer/Victoria) | yes | manual edge → 4262 (1.11 km) |

---

## 7. Known limitations (kept as-is, documented for the report)

| Issue | Why kept |
|-------|----------|
| Sites 3804 and 4035 absent from the IBM Boroondara map PDF (July 2009) | Both have 2976 flow rows in the October 2006 .xls — they are real 2006-era sites, likely decommissioned by 2009. Centroid lat/lon used for graph construction. |
| 18 of 40 sites not visible on IBM map | Same 3-year gap between data (2006) and map (2009). Master listing + lat/lon used as authoritative source. |
| Edges drawn as straight lines in any visualisation | Great-circle from centroid to centroid; real roads curve. Does not affect graph correctness. |
| Detector-centroid lat/lons offset 50–200 m from the actual intersection | Each detector sits on an approach lane. Centroid is the mean. Does not affect adjacency. |
| Edges are unconditionally bidirectional | One-way streets are not modelled. Acceptable for this scale; mentioned as future work. |
| Distances are Haversine (great-circle), not road distances | Slightly underestimates real travel distance. Acceptable approximation per the assignment. |

---

## Where each fix lives in the code

| Item | File / lines |
|------|--------------|
| Master listing reader | `src/data/build_graph.py` — `_read_listing_rows()`, `_parse_road_set()`, `load_roads_per_site()` |
| `ROAD_ALIASES` dict | `src/data/build_graph.py:52-67` |
| `SITE_ROAD_OVERRIDES` dict | `src/data/build_graph.py:80-87` |
| `MANUAL_EDGES` list | `src/data/build_graph.py:97-104` |
| Length-scaling `is_between()` | `src/data/build_graph.py:123-149` |
| Manual edge application | `src/data/build_graph.py` — inside `main()`, after `build_adjacency()` |
