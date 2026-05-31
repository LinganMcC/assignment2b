from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMESERIES = PROJECT_ROOT / "data" / "processed" / "flow_timeseries.csv"
LOCATIONS = PROJECT_ROOT / "data" / "processed" / "site_locations.csv"
GRAPH = PROJECT_ROOT / "data" / "processed" / "boroondara_graph.json"

EXPECTED_SITES = 40                  # 40 SCATS sites in Boroondara dataset
EXPECTED_ROWS_PER_SITE = 31 * 96     # 31 days x 96 fifteen-min intervals
EXPECTED_TIMESERIES_ROWS = EXPECTED_SITES * EXPECTED_ROWS_PER_SITE


class TestDataParsing(unittest.TestCase):
    """Tests on parse_scats.py outputs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ts = pd.read_csv(TIMESERIES)
        cls.locs = pd.read_csv(LOCATIONS)

    # ---------- TC-A1 -------------------------------------------------------
    def test_a1_site_count_matches_expected(self) -> None:
        """TC-A1: parse_scats produces 40 sites from the raw .xls."""
        # Count unique site IDs in the timeseries and confirm the locations
        # file lists exactly the same number — catches dropped/duplicated sites.
        self.assertEqual(self.ts["site_id"].nunique(), EXPECTED_SITES)
        self.assertEqual(len(self.locs), EXPECTED_SITES)

    # ---------- TC-A2 -------------------------------------------------------
    def test_a2_flow_values_are_valid(self) -> None:
        """TC-A2: All flow values are non-negative integers, no NaN."""
        # Three independent sanity checks on the flow column: no missing
        # values, no negatives (vehicles/hour can't be < 0), and the dtype
        # is an integer (rules out accidentally-stringified or float values).
        self.assertEqual(self.ts["flow"].isna().sum(), 0, "no NaN flow values")
        self.assertGreaterEqual(self.ts["flow"].min(), 0, "no negative flows")
        self.assertTrue(
            pd.api.types.is_integer_dtype(self.ts["flow"]),
            "flow column is integer dtype",
        )

    # ---------- TC-A3 -------------------------------------------------------
    def test_a3_timeseries_row_count(self) -> None:
        """TC-A3: Timeseries has 40 sites x 31 days x 96 intervals = 119,040 rows."""
        # Allow +/- 5% slack because a few sites have missing detector readings
        # on some days. Outside that range = something is genuinely wrong with
        # the parser (e.g. missing days or duplicated rows).
        self.assertGreaterEqual(len(self.ts), EXPECTED_TIMESERIES_ROWS * 0.95)
        self.assertLessEqual(len(self.ts), EXPECTED_TIMESERIES_ROWS * 1.05)

    # ---------- TC-A4 -------------------------------------------------------
    def test_a4_site_4266_has_valid_centroid(self) -> None:
        """TC-A4: Site 4266 (which has a (0,0) sentinel detector in the raw
        .xls) still has a valid lat/lon centroid after filtering."""
        # Site 4266 specifically because its raw data contains a detector row
        # with NB_LATITUDE=0, NB_LONGITUDE=0 (VicRoads' missing-data marker).
        # The parser must drop those rows before averaging — otherwise the
        # centroid gets pulled toward (0,0). This test confirms the filter.
        row = self.locs[self.locs["site_id"] == 4266]
        self.assertEqual(len(row), 1, "site 4266 present in locations")
        lat = float(row["lat"].iloc[0])
        lon = float(row["lon"].iloc[0])
        self.assertNotEqual(lat, 0.0, "lat is not the (0,0) sentinel")
        self.assertNotEqual(lon, 0.0, "lon is not the (0,0) sentinel")
        # Boroondara is in inner-east Melbourne; tight bounding box catches
        # any centroid that's been dragged off by stray (0,0) detectors.
        self.assertTrue(-38.0 < lat < -37.5, "lat within Melbourne range")
        self.assertTrue(144.9 < lon < 145.3, "lon within Melbourne range")


class TestGraphIntegrity(unittest.TestCase):
    """Tests on build_graph.py output."""

    @classmethod
    def setUpClass(cls) -> None:
        with open(GRAPH) as f:
            cls.g = json.load(f)

    # ---------- TC-A5 -------------------------------------------------------
    def test_a5_graph_has_zero_isolated_nodes(self) -> None:
        """TC-A5: After all manual fixes, no site is isolated."""
        # Collect any sites whose adjacency list is empty. The four originally-
        # isolated sites (2200, 2846, 4812, 4821) should be linked now via
        # ROAD_ALIASES / SITE_ROAD_OVERRIDES / MANUAL_EDGES in build_graph.py.
        isolated = [sid for sid, edges in self.g["edges"].items() if not edges]
        self.assertEqual(len(isolated), 0,
                         f"isolated sites should be empty, got {isolated}")

    # ---------- TC-A6 -------------------------------------------------------
    def test_a6_all_edges_are_symmetric(self) -> None:
        """TC-A6: For every edge A->B in the graph, B->A also exists with the
        same distance."""
        # Walk every edge and look for the reverse in the other site's list.
        # Asymmetry would silently break routing (e.g. A* might find a path
        # one direction but not the other). 1e-6 tolerance allows for float
        # rounding when the same haversine distance is stored on both sides.
        edges = self.g["edges"]
        asymmetric = []
        for src, elist in edges.items():
            for e in elist:
                back = [b for b in edges[str(e["to"])] if b["to"] == int(src)]
                if not back:
                    asymmetric.append(f"{src}->{e['to']} has no return")
                elif abs(back[0]["distance_km"] - e["distance_km"]) > 1e-6:
                    asymmetric.append(f"{src}<->{e['to']} distance mismatch")
        self.assertEqual(len(asymmetric), 0,
                         f"asymmetric edges: {asymmetric[:5]}")

    # ---------- TC-A7 -------------------------------------------------------
    def test_a7_graph_json_schema(self) -> None:
        """TC-A7: Graph JSON has the schema routing.py expects (nodes with
        name/lat/lon; edges with to/distance_km)."""
        # Top-level structure check first — routing.py does
        # `data["nodes"]` and `data["edges"]`, so both keys must exist.
        self.assertIn("nodes", self.g)
        self.assertIn("edges", self.g)
        # Then spot-check a node has the three fields routing.py reads
        # (name for display, lat/lon to build the Graph object).
        sample_node = next(iter(self.g["nodes"].values()))
        self.assertIn("name", sample_node)
        self.assertIn("lat", sample_node)
        self.assertIn("lon", sample_node)
        # And spot-check an edge has its two fields (`to` for the neighbour
        # site id, `distance_km` for the travel-time calculation).
        sample_edge = next(iter(e for elist in self.g["edges"].values()
                                for e in elist))
        self.assertIn("to", sample_edge)
        self.assertIn("distance_km", sample_edge)


if __name__ == "__main__":
    unittest.main(verbosity=2)
