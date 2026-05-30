"""
route_finder.py  ─  Standalone Route Finder GUI (Task 4 only)

Yen's k-shortest paths with A* on the Boroondara SCATS network,
flow-speed travel-time estimation via ML prediction.

Run from the assignment2b-main directory (or pass --project-root):
    python route_finder.py
    python route_finder.py --project-root /path/to/assignment2b-main
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# Palette
# ═══════════════════════════════════════════════════════════════════════════
C = {
    "bg":     "#1e1e2e",
    "surf":   "#313244",
    "surf2":  "#45475a",
    "border": "#585b70",
    "text":   "#cdd6f4",
    "sub":    "#a6adc8",
    "accent": "#89b4fa",
    "green":  "#a6e3a1",
    "yellow": "#f9e2af",
    "red":    "#f38ba8",
    "mauve":  "#cba6f7",
    "lstm":   "#89b4fa",
    "gru":    "#a6e3a1",
    "transformer": "#f9e2af",
}

MODEL_CLR = {"lstm": C["lstm"], "gru": C["gru"], "transformer": C["transformer"]}
ROUTE_CLR = ["#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8", "#cba6f7"]

# ═══════════════════════════════════════════════════════════════════════════
# Traffic physics constants
# ═══════════════════════════════════════════════════════════════════════════
FLOW_CAP   = 1500.0
SPD_CAP    = 32.0
SPD_LIM    = 60.0
SPD_MIN    = 5.0
INTER_DELAY = 30.0
_A  = -FLOW_CAP / SPD_CAP**2
_B  = -2 * SPD_CAP * _A
_B2 = _B**2


def flow_to_speed(q_hr: float) -> float:
    q    = min(max(0.0, q_hr), FLOW_CAP)  # cap at capacity
    disc = _B2 + 4 * _A * q
    if disc < 0:
        return SPD_MIN
    sq  = math.sqrt(disc)
    spd = (-_B - sq) / (2 * _A)
    return min(max(spd, SPD_MIN), SPD_LIM)


def edge_tt(dist_km: float, flow15: int) -> float:
    return (dist_km / flow_to_speed(flow15 * 4)) * 3600 + INTER_DELAY


# ═══════════════════════════════════════════════════════════════════════════
# Project bootstrap
# ═══════════════════════════════════════════════════════════════════════════

def setup_project(root: Path):
    for sub in ("ml", "routing"):
        p = str(root / "src" / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


def find_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [
        here,
        here / "assignment2b-main",
        here.parent / "assignment2b-main",
        Path.cwd(),
        Path.cwd() / "assignment2b-main",
    ]:
        if (candidate / "data" / "processed" / "ml_artefacts" / "gru_best.pt").exists():
            return candidate
    return here


# ═══════════════════════════════════════════════════════════════════════════
# Widget helpers
# ═══════════════════════════════════════════════════════════════════════════

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(C["surf"])
    ax.tick_params(colors=C["sub"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(C["border"]); sp.set_linewidth(0.5)
    ax.grid(True, color=C["border"], linewidth=0.4, alpha=0.5)
    if title:  ax.set_title(title,  color=C["text"], fontsize=10)
    if xlabel: ax.set_xlabel(xlabel, color=C["sub"],  fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=C["sub"],  fontsize=8)


def make_canvas(parent, fig):
    cv = FigureCanvasTkAgg(fig, master=parent)
    cv.get_tk_widget().pack(fill="both", expand=True)
    NavigationToolbar2Tk(cv, parent).pack(fill="x")
    return cv


def section_label(parent, text):
    tk.Label(parent, text=text.upper(), font=("Segoe UI", 7, "bold"),
             bg=C["surf"], fg=C["sub"]).pack(anchor="w", padx=12, pady=(10, 1))


def entry_field(parent, var, width=22):
    return tk.Entry(parent, textvariable=var, width=width,
                    bg=C["surf2"], fg=C["text"], insertbackground=C["text"],
                    relief="flat", font=("Segoe UI", 9))


def run_button(parent, text, cmd):
    tk.Button(parent, text=text, command=cmd,
              bg=C["accent"], fg=C["bg"],
              font=("Segoe UI", 10, "bold"),
              relief="flat", pady=6, cursor="hand2",
              activebackground=C["text"], activeforeground=C["bg"],
              ).pack(fill="x", padx=12, pady=8)


def scrolled_text(parent, height=8):
    frame = tk.Frame(parent, bg=C["surf2"])
    frame.pack(fill="both", expand=True, padx=8, pady=4)
    sb = tk.Scrollbar(frame)
    sb.pack(side="right", fill="y")
    t = tk.Text(frame, yscrollcommand=sb.set, height=height,
                bg=C["surf2"], fg=C["text"], font=("Consolas", 9),
                relief="flat", wrap="word")
    t.pack(fill="both", expand=True)
    sb.config(command=t.yview)
    return t


def combo(parent, var, values, width=28):
    style = ttk.Style()
    style.configure("Dark.TCombobox",
                    fieldbackground=C["surf2"], background=C["surf2"],
                    foreground=C["text"], selectbackground=C["accent"],
                    selectforeground=C["bg"])
    w = ttk.Combobox(parent, textvariable=var, values=values,
                     width=width, state="readonly", style="Dark.TCombobox")
    w.pack(padx=12, pady=3, fill="x")
    return w


def radio_group(parent, options, var, color_map=None):
    for opt in options:
        clr = (color_map or {}).get(opt, C["text"])
        tk.Radiobutton(parent, text=opt.upper(), variable=var, value=opt,
                       bg=C["surf"], fg=clr, selectcolor=C["surf2"],
                       activebackground=C["surf"],
                       font=("Segoe UI", 10, "bold"),
                       ).pack(anchor="w", padx=16, pady=1)


# ═══════════════════════════════════════════════════════════════════════════
# Application
# ═══════════════════════════════════════════════════════════════════════════

class RouteFinder(tk.Tk):
    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        setup_project(root)

        from predict import FlowPredictor
        self.FlowPredictor = FlowPredictor

        # ── data ──────────────────────────────────────────────────────────
        self.sites = pd.read_csv(root / "data/processed/site_locations.csv")
        with open(root / "data/processed/boroondara_graph.json") as f:
            self.graph_data = json.load(f)

        self._site_opts = [
            f"{r.site_id}  –  {r['name'][:32]}"
            for _, r in self.sites.iterrows()
        ]
        self._site_ids = [int(r.site_id) for _, r in self.sites.iterrows()]

        # ── window ────────────────────────────────────────────────────────
        self.title("TBRGS — Route Finder")
        self.configure(bg=C["bg"])
        self.geometry("1240x820")
        self.minsize(900, 620)

        self._build_ui()

    # ── Layout ────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Banner
        ban = tk.Frame(self, bg=C["bg"])
        ban.pack(fill="x", padx=16, pady=(8, 4))
        tk.Label(ban, text="🚦  Route Finder — Boroondara Traffic Network",
                 font=("Segoe UI", 15, "bold"),
                 bg=C["bg"], fg=C["text"]).pack(side="left")
        tk.Label(ban, text="Yen's k-shortest  •  A*  •  SCATS Oct 2006",
                 font=("Segoe UI", 9), bg=C["bg"], fg=C["sub"]).pack(
                 side="left", padx=14)

        # Main body
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        lp = tk.Frame(body, bg=C["surf"], width=285)
        lp.pack(side="left", fill="y", padx=(0, 4))
        lp.pack_propagate(False)

        rp = tk.Frame(body, bg=C["bg"])
        rp.pack(side="left", fill="both", expand=True)

        self._build_controls(lp)
        self._build_map(rp)

    def _build_controls(self, lp):
        section_label(lp, "Origin")
        self._rt_origin = tk.StringVar(value=self._site_opts[3])
        combo(lp, self._rt_origin, self._site_opts)

        section_label(lp, "Destination")
        self._rt_dest = tk.StringVar(value=self._site_opts[1])
        combo(lp, self._rt_dest, self._site_opts)

        section_label(lp, "Date & Time")
        self._rt_date = tk.StringVar(value="2006-10-28")
        self._rt_time = tk.StringVar(value="08:30")
        for var, hint in [(self._rt_date, "YYYY-MM-DD"), (self._rt_time, "HH:MM")]:
            tk.Label(lp, text=hint, bg=C["surf"], fg=C["sub"],
                     font=("Segoe UI", 8)).pack(anchor="w", padx=14)
            entry_field(lp, var).pack(fill="x", padx=12, pady=1)

        section_label(lp, "ML Model")
        self._rt_model = tk.StringVar(value="gru")
        radio_group(lp, ["lstm", "gru", "transformer"], self._rt_model, MODEL_CLR)

        section_label(lp, "number of routes")
        self._rt_k = tk.IntVar(value=3)
        tk.Scale(lp, from_=1, to=5, orient="horizontal",
                 variable=self._rt_k,
                 bg=C["surf"], fg=C["text"], troughcolor=C["surf2"],
                 highlightthickness=0, bd=0, font=("Segoe UI", 9),
                 ).pack(fill="x", padx=12, pady=2)

        run_button(lp, "▶  Find Routes", self._run_routing)

        section_label(lp, "Results")
        self._rt_txt = scrolled_text(lp, height=18)

    def _build_map(self, rp):
        self._rt_fig = Figure(facecolor=C["bg"])
        self._rt_cv  = make_canvas(rp, self._rt_fig)
        self._draw_placeholder()

    # ── Placeholder ───────────────────────────────────────────────────────

    def _draw_placeholder(self):
        self._rt_fig.clf()
        ax = self._rt_fig.add_subplot(111)
        style_ax(ax,
                 "Boroondara Network — select origin / destination and click Find Routes",
                 "Longitude", "Latitude")
        nodes = self.graph_data["nodes"]
        edges = self.graph_data["edges"]

        for src_str, edge_list in edges.items():
            n1 = nodes[src_str]
            for e in edge_list:
                dst_str = str(e["to"])
                if dst_str not in nodes:
                    continue
                n2 = nodes[dst_str]
                ax.plot([n1["lon"], n2["lon"]], [n1["lat"], n2["lat"]],
                        color=C["surf2"], linewidth=0.8, alpha=0.6, zorder=1)

        ax.scatter([nodes[k]["lon"] for k in nodes],
                   [nodes[k]["lat"] for k in nodes],
                   c=C["sub"], s=30, zorder=3,
                   edgecolors=C["bg"], linewidths=0.4)
        for sid_str, nd in nodes.items():
            ax.annotate(sid_str, (nd["lon"], nd["lat"]),
                        textcoords="offset points", xytext=(3, 2),
                        color=C["sub"], fontsize=6)

        self._rt_fig.tight_layout(pad=1.5)
        self._rt_cv.draw()

    # ── Route search ──────────────────────────────────────────────────────

    def _run_routing(self):
        origin  = int(self._rt_origin.get().split()[0])
        dest    = int(self._rt_dest.get().split()[0])
        dt_str  = f"{self._rt_date.get()} {self._rt_time.get()}:00"
        model   = self._rt_model.get()
        k       = self._rt_k.get()

        if origin == dest:
            messagebox.showwarning("Same node",
                                   "Origin and destination must be different.")
            return

        self._rt_txt.config(state="normal")
        self._rt_txt.delete("1.0", "end")
        self._rt_txt.insert("end",
            f"Searching  {origin} → {dest}  [{model.upper()}]  k={k}...\n")
        self._rt_txt.config(state="disabled")
        self.update_idletasks()

        try:
            from routing import find_routes as _find_one, build_parta_graph, run_algorithm, build_route
            import heapq as _hq
            import tempfile as _tmp, json as _json, os as _os
            from pathlib import Path as _Path

            def _spur_route(spur, dest, removed_nodes, removed_edges, dt_str, model):
                """Find shortest path from spur to dest using routing.py, with
                certain nodes/edges removed. Writes a pruned graph to a temp
                file so build_parta_graph reads the right topology."""
                pruned = {
                    "nodes": {k: v for k, v in self.graph_data["nodes"].items()
                              if int(k) not in removed_nodes},
                    "edges": {}
                }
                for u_str, edge_list in self.graph_data["edges"].items():
                    uid = int(u_str)
                    if uid in removed_nodes:
                        continue
                    pruned["edges"][u_str] = [
                        e for e in edge_list
                        if int(e["to"]) not in removed_nodes
                        and (uid, int(e["to"])) not in removed_edges
                    ]
                tf = _tmp.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
                try:
                    _json.dump(pruned, tf); tf.close()
                    graph, dist_lk, flow_cache = build_parta_graph(
                        origin=spur, dest=dest,
                        datetime_str=dt_str, model=model,
                        json_path=_Path(tf.name))
                    goal, nodes_created, path = run_algorithm(graph, "astar")
                    if not path:
                        return None
                    return build_route(path, goal, nodes_created, "astar",
                                       dist_lk, flow_cache)
                finally:
                    _os.unlink(tf.name)

            first = _find_one(origin=origin, dest=dest,
                              datetime_str=dt_str, model=model, k=1)
            if not first or k == 1:
                routes = first
            else:
                best = list(first)
                seen = {tuple(first[0].path)}
                cands = []
                for _ in range(k - 1):
                    prev_path = best[-1].path
                    for i in range(len(prev_path) - 1):
                        spur_node = prev_path[i]
                        root = prev_path[:i+1]
                        rem_edges = set()
                        rem_nodes = set()
                        for r in best:
                            rp = r.path
                            if len(rp) > i and rp[:i+1] == root:
                                rem_edges.add((rp[i], rp[i+1]))
                        for node in root[:-1]:
                            rem_nodes.add(node)
                        spur_rt = _spur_route(
                            spur_node, dest, rem_nodes, rem_edges, dt_str, model)
                        if spur_rt:
                            full_path = root[:-1] + spur_rt.path
                            key = tuple(full_path)
                            if key not in seen:
                                seen.add(key)
                                # Cost the root segment using the first route's data,
                                # then add the spur route's cost for a total
                                root_time = sum(
                                    best[0].edges_km[j] / best[0].speeds_kmh[j] * 3600
                                    + INTER_DELAY
                                    for j in range(i)
                                ) if i > 0 else 0.0
                                total_time = root_time + spur_rt.total_time_s
                                _hq.heappush(cands, (total_time, full_path, spur_rt, i))
                    if not cands:
                        break
                    _, next_path, spur_rt, split_i = _hq.heappop(cands)
                    seen.add(tuple(next_path))
                    # Build the full route by combining root + spur via routing.py
                    # Write the full path as a forced route using a temp graph that
                    # only contains the edges on next_path
                    forced_edges = set(zip(next_path, next_path[1:]))
                    forced = {
                        "nodes": {str(n): self.graph_data["nodes"][str(n)]
                                  for n in next_path if str(n) in self.graph_data["nodes"]},
                        "edges": {}
                    }
                    for u_str, edge_list in self.graph_data["edges"].items():
                        uid = int(u_str)
                        if uid not in next_path:
                            continue
                        forced["edges"][u_str] = [
                            e for e in edge_list
                            if (uid, int(e["to"])) in forced_edges
                        ]
                    tf = _tmp.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
                    try:
                        _json.dump(forced, tf); tf.close()
                        graph2, dist_lk2, flow_cache2 = build_parta_graph(
                            origin=next_path[0], dest=next_path[-1],
                            datetime_str=dt_str, model=model,
                            json_path=_Path(tf.name))
                        goal2, nc2, path2 = run_algorithm(graph2, "astar")
                        if path2:
                            full_rt = build_route(path2, goal2, nc2, "astar",
                                                  dist_lk2, flow_cache2)
                            best.append(full_rt)
                    finally:
                        _os.unlink(tf.name)
                routes = best
        except Exception as e:
            messagebox.showerror("Routing error", str(e))
            return

        # ── Result text ───────────────────────────────────────────────────
        self._rt_txt.config(state="normal")
        self._rt_txt.delete("1.0", "end")

        if not routes:
            self._rt_txt.insert("end", "No routes found.\n")
        else:
            self._rt_txt.insert("end",
                f"{'─'*40}\n"
                f"  {origin} → {dest}   [{model.upper()}]   {dt_str}\n"
                f"{'─'*40}\n\n")
            for i, r in enumerate(routes, 1):
                self._rt_txt.insert("end",
                    f"Route {i}  ●  {r.total_time_min:.1f} min"
                    f"  |  {r.total_distance_km:.2f} km"
                    f"  |  {len(r.path)-1} link(s)\n",
                    f"r{i}")
                path_str = " → ".join(str(s) for s in r.path)
                self._rt_txt.insert("end", f"  Path : {path_str}\n")
                for j, (d, v) in enumerate(zip(r.edges_km, r.speeds_kmh)):
                    u, nxt = r.path[j], r.path[j + 1]
                    tt = (d / v) * 3600 + INTER_DELAY
                    self._rt_txt.insert("end",
                        f"    {u:>4} → {nxt:<4}  {d:.3f} km"
                        f"  @{v:.1f} km/h  ({tt:.0f} s)\n")
                self._rt_txt.insert("end", "\n")
                self._rt_txt.tag_config(
                    f"r{i}",
                    foreground=ROUTE_CLR[i - 1],
                    font=("Consolas", 9, "bold"))

        self._rt_txt.config(state="disabled")

        # ── Map ───────────────────────────────────────────────────────────
        self._rt_fig.clf()
        ax = self._rt_fig.add_subplot(111)
        style_ax(ax,
                 f"Top-{len(routes)} routes: {origin} → {dest}  [{model.upper()}]",
                 "Longitude", "Latitude")

        nodes      = self.graph_data["nodes"]
        edges_data = self.graph_data["edges"]

        # background network
        for src_str, edge_list in edges_data.items():
            n1 = nodes[src_str]
            for e in edge_list:
                dst_str = str(e["to"])
                if dst_str not in nodes:
                    continue
                n2 = nodes[dst_str]
                ax.plot([n1["lon"], n2["lon"]], [n1["lat"], n2["lat"]],
                        color=C["surf2"], linewidth=0.7, alpha=0.5, zorder=1)

        # all nodes (muted)
        ax.scatter([nodes[k]["lon"] for k in nodes],
                   [nodes[k]["lat"] for k in nodes],
                   c=C["surf2"], s=20, zorder=2,
                   edgecolors=C["bg"], linewidths=0.3)
        for sid_str, nd in nodes.items():
            ax.annotate(sid_str, (nd["lon"], nd["lat"]),
                        textcoords="offset points", xytext=(3, 2),
                        color=C["sub"], fontsize=6)

        # coloured routes
        legend_patches = []
        for i, r in enumerate(routes):
            clr = ROUTE_CLR[i]
            for j in range(len(r.path) - 1):
                u_str = str(r.path[j])
                v_str = str(r.path[j + 1])
                if u_str not in nodes or v_str not in nodes:
                    continue
                n1, n2 = nodes[u_str], nodes[v_str]
                offset = (i - len(routes) / 2) * 0.0003
                ax.annotate("",
                    xy=(n2["lon"] + offset, n2["lat"] + offset),
                    xytext=(n1["lon"] + offset, n1["lat"] + offset),
                    arrowprops=dict(
                        arrowstyle="-|>", color=clr,
                        lw=2.0 - i * 0.2, mutation_scale=12),
                    zorder=5 - i)
            legend_patches.append(mpatches.Patch(
                color=clr,
                label=f"Route {i+1}: {r.total_time_min:.1f} min  "
                      f"{r.total_distance_km:.2f} km"))

        # origin / destination highlights
        for sid, label, clr in [(origin, "O", C["green"]),
                                 (dest,   "D", C["red"])]:
            sid_str = str(sid)
            if sid_str in nodes:
                nd = nodes[sid_str]
                ax.scatter(nd["lon"], nd["lat"], c=clr, s=120,
                           zorder=8, edgecolors=C["bg"], linewidths=0.8)
                ax.annotate(f"{label}:{sid}", (nd["lon"], nd["lat"]),
                            xytext=(5, 5), textcoords="offset points",
                            color=clr, fontsize=8, fontweight="bold")

        ax.legend(handles=legend_patches,
                  facecolor=C["surf"], edgecolor="none",
                  labelcolor=C["text"], fontsize=8, loc="best")
        self._rt_fig.tight_layout(pad=1.5)
        self._rt_cv.draw()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone Route Finder GUI for Boroondara SCATS network")
    parser.add_argument("--project-root", type=Path, default=None,
                        help="Path to assignment2b-main directory")
    args = parser.parse_args()

    rp = args.project_root or find_root()
    if not (rp / "data/processed/ml_artefacts/gru_best.pt").exists():
        print(f"ERROR: model artefacts not found under {rp}")
        print("Pass --project-root /path/to/assignment2b-main")
        sys.exit(1)

    app = RouteFinder(root=rp)
    app.mainloop()