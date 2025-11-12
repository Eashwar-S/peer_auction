#!/usr/bin/env python3
"""
Refactored Multi-Trip Router (Parameter-Free), testable + SOLID-friendly.

Preserves logic from the latest version:
  - Cycle-proof single-trip growth (frontier-progress + no-backtrack)
  - Coverage-aware finisher (relocation-with-coverage; orienteering-style on meta-graph)
  - Start-depot retargeting INSIDE the same trip (no standalone relocations)
  - Correct per-trip cost accounting
  - No new parameters

Design for testability:
  - Small abstractions (Protocols) + dependency injection
  - Pure data objects (Trip, PlanSummary)
  - No hardcoded scenario path (provide path to `load_scenario`)
  - Orchestrator returns a result; CLI wrapper prints it
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Set, Tuple

import networkx as nx
from math import exp

# ========= Data Models =========

Node = int
Edge = Tuple[int, int]  # always sorted

@dataclass(frozen=True)
class Trip:
    start: Node
    end: Node
    route: List[Node]
    cost: float
    covered_edges: Set[Edge]

@dataclass(frozen=True)
class PlanSummary:
    trips: List[Trip]
    total_cost: float
    required_initial: int
    required_remaining: int


# ========= Parsing =========

def load_scenario(path: str) -> Tuple[nx.Graph, List[Edge], List[Node], float]:
    """
    Minimal GDB-like parser. Expected keys:
      NUMBER OF VERTICES
      VEHICLE CAPACITY
      DEPOT: 1,2,3...
      LIST_REQUIRED_EDGES / LIST_NON_REQUIRED_EDGES: lines "(u,v) edge weight w"
    Returns: (Graph, required_edges(sorted), depots, capacity)
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    def extract_int(key: str) -> Optional[int]:
        for ln in lines:
            if ln.upper().startswith(key.upper()):
                m = re.findall(r"(-?\d+)", ln)
                if m:
                    return int(m[0])
        return None

    def extract_float(key: str) -> Optional[float]:
        for ln in lines:
            if ln.upper().startswith(key.upper()):
                m = re.findall(r"(-?\d+(?:\.\d+)?)", ln)
                if m:
                    return float(m[0])
        return None

    def extract_depots() -> List[Node]:
        for ln in lines:
            if ln.upper().startswith("DEPOT"):
                nums = re.findall(r"(\d+)", ln)
                return [int(x) for x in nums]
        return []

    nV = extract_int("NUMBER OF VERTICES")
    capacity = float(extract_float("VEHICLE CAPACITY") or 0.0)
    depots = extract_depots()

    required_edges: List[Edge] = []
    weights: Dict[Edge, float] = {}

    req_section = False
    nonreq_section = False

    for ln in lines:
        if ln.upper().startswith("LIST_REQUIRED_EDGES"):
            req_section, nonreq_section = True, False
            continue
        if ln.upper().startswith("LIST_NON_REQUIRED_EDGES"):
            req_section, nonreq_section = False, True
            continue
        if ln.upper().startswith("FAILURE_SCENARIO"):
            req_section = nonreq_section = False
            continue

        m = re.match(r"\((\d+),\s*(\d+)\)\s*edge\s*weight\s*([0-9.]+)", ln, re.IGNORECASE)
        if m:
            u, v, w = int(m.group(1)), int(m.group(2)), float(m.group(3))
            e = tuple(sorted((u, v)))
            weights[e] = w
            if req_section:
                required_edges.append(e)

    G = nx.Graph()
    if nV:
        G.add_nodes_from(range(1, nV + 1))
    for (a, b), w in weights.items():
        G.add_edge(a, b, weight=w)

    return G, required_edges, depots, capacity


# ========= Abstractions (Protocols) =========

class IGraphMetrics(Protocol):
    def dist(self, a: Node, b: Node) -> float: ...
    def shortest_path(self, a: Node, b: Node) -> List[Node]: ...
    def min_dist_to_any(self, a: Node, targets: Set[Node]) -> float: ...


class ITripPlanner(Protocol):
    def plan_single_trip(
        self, required_edges: Set[Edge], *, verbose: bool = False
    ) -> Tuple[List[Node], float, Set[Edge]]:
        """
        Returns (route nodes, trip cost, covered required edges).
        Route starts at the planner's configured start depot and must end at a depot.
        """


class IFinisher(Protocol):
    def finish(
        self,
        start_node: Node,
        base_cost: float,
        required_remaining: Set[Edge],
    ) -> Tuple[List[Node], float, Set[Edge]]:
        """
        Coverage-aware finisher: within remaining capacity, return a segment from start_node
        that ends at ANY depot, possibly covering multiple required edges; returns
        (tail nodes, new_total_cost, newly_covered_edges). When concatenated to the
        current route, the whole trip must end at a depot.
        """


class IStartRetargetor(Protocol):
    def retarget_and_plan(
        self,
        current_start: Node,
        required_remaining: Set[Edge],
    ) -> Optional[Trip]:
        """
        If planning from current_start would cover 0 edges, try all depots s*:
        pay deadhead(current_start->s*) as a prefix INSIDE the same trip, reduce capacity,
        plan covering leg from s*. Return one trip (prefix+cover) with coverage>0, or None.
        """


# ========= Concrete: Graph Metrics =========

class APSPMetrics(IGraphMetrics):
    """All-pairs shortest path metrics, immutable and shareable."""
    def __init__(self, G: nx.Graph):
        self.G = G
        self._dist: Dict[Node, Dict[Node, float]] = dict(
            nx.all_pairs_dijkstra_path_length(G, weight="weight")
        )

    def dist(self, a: Node, b: Node) -> float:
        return self._dist.get(a, {}).get(b, float("inf"))

    def shortest_path(self, a: Node, b: Node) -> List[Node]:
        if a == b:
            return [a]
        return nx.shortest_path(self.G, a, b, weight="weight")

    def min_dist_to_any(self, a: Node, targets: Set[Node]) -> float:
        row = self._dist.get(a, {})
        best = float("inf")
        for t in targets:
            d = row.get(t, float("inf"))
            if d < best:
                best = d
        return best


# ========= Concrete: Single-Trip Planner (Magnetic Field Growth) =========

class MagneticFieldTripPlanner(ITripPlanner):
    """
    Single-trip router preserving the original rules:
      - Frontier progress unless covering a new required edge
      - No immediate backtrack unless covering
      - Free-end depot
      - Uses a coverage-aware finisher object to close the trip
    """
    def __init__(
        self,
        G: nx.Graph,
        metrics: IGraphMetrics,
        depots: Sequence[Node],
        capacity: float,
        start_depot: Node,
        finisher: IFinisher,
    ):
        self.G = G
        self.M = metrics
        self.depots: Set[Node] = set(depots)
        self.capacity = float(capacity)
        self.start_depot = start_depot
        self.finisher = finisher

    def _dist_to_frontier(self, node: Node, req: Set[Edge]) -> float:
        best = float("inf")
        for a, b in req:
            da = self.M.dist(node, a)
            db = self.M.dist(node, b)
            if da < best: best = da
            if db < best: best = db
            if best == 0: break
        return best

    def _edge_score(
        self, u: Node, v: Node, req_set_remaining: Set[Edge], current_length: float, is_new_required: bool
    ) -> Tuple[float, float]:
        w_uv = self.G[u][v]["weight"]
        w_cap = min(max(current_length / max(self.capacity, 1e-9), 0.0), 1.0)

        if is_new_required:
            P = 1.0
        else:
            Fv = self._dist_to_frontier(v, req_set_remaining)
            P = 0.0 if math.isinf(Fv) else exp(-1.0 * Fv)

        d_v = self.M.min_dist_to_any(v, self.depots)
        D = 0.0 if math.isinf(d_v) else exp(-1.0 * (d_v / max(self.capacity, 1e-9)))

        final = (1.0 - w_cap) * P + w_cap * D
        return w_uv, final

    def plan_single_trip(
        self, required_edges: Set[Edge], *, verbose: bool = False
    ) -> Tuple[List[Node], float, Set[Edge]]:
        current_route: List[Node] = [self.start_depot]
        current_length: float = 0.0
        covered: Set[Edge] = set()
        req_all = set(required_edges)
        depot_depot_covered_this_trip: Set[Edge] = set()
        max_iterations = len(self.G.edges()) * 10
        it = 0

        while len(covered) < len(req_all) and it < max_iterations:
            it += 1
            u = current_route[-1]
            remaining = req_all - covered
            if not remaining:
                break
            Fu = self._dist_to_frontier(u, remaining)

            candidates = []
            for v in self.G.neighbors(u):
                e_sorted = tuple(sorted((u, v)))
                w_uv = self.G[u][v]["weight"]
                depot_after = self.M.min_dist_to_any(v, self.depots)
                if current_length + w_uv + depot_after > self.capacity:
                    continue

                is_new_required = (e_sorted in remaining)

                # No immediate backtrack unless covering
                if len(current_route) >= 2 and v == current_route[-2] and not is_new_required:
                    continue

                # Avoid reusing depot–depot required edge in same trip unless covering
                if (u in self.depots and v in self.depots and
                        e_sorted in depot_depot_covered_this_trip and not is_new_required):
                    continue

                # Frontier progress unless covering
                Fv = self._dist_to_frontier(v, remaining)
                if not is_new_required and not (Fv < Fu):
                    continue

                w_uv2, score = self._edge_score(u, v, remaining, current_length, is_new_required)
                candidates.append((not is_new_required, Fv, -score, v, e_sorted, w_uv2, is_new_required))

            if not candidates:
                # Force move toward nearest frontier endpoint if feasible
                best_path, best_len = None, float("inf")
                for (a, b) in remaining:
                    for t in (a, b):
                        try:
                            path = self.M.shortest_path(u, t)
                            plen = sum(self.G[path[i]][path[i+1]]["weight"] for i in range(len(path)-1))
                            depot_after = self.M.min_dist_to_any(t, self.depots)
                            if current_length + plen + depot_after <= self.capacity and plen < best_len:
                                best_path, best_len = path[1:], plen
                        except Exception:
                            continue
                if best_path:
                    for nxt in best_path:
                        prev = current_route[-1]
                        e_sorted2 = tuple(sorted((prev, nxt)))
                        if prev in self.depots and nxt in self.depots and e_sorted2 in remaining:
                            depot_depot_covered_this_trip.add(e_sorted2)
                        current_route.append(nxt)
                        current_length += self.G[prev][nxt]["weight"]
                    continue
                break  # growth done; delegate to finisher

            candidates.sort()
            _, _, _, v, e_sorted, w_uv, is_new_required = candidates[0]
            current_route.append(v)
            current_length += w_uv

            if is_new_required:
                covered.add(e_sorted)
                if current_route[-2] in self.depots and v in self.depots:
                    depot_depot_covered_this_trip.add(e_sorted)
                if verbose:
                    print(f"✓ Covered required: {(current_route[-2], v)} ({len(covered)}/{len(req_all)})")

        # Coverage-aware finisher (same logic; just injected)
        tail_nodes, new_total, newly_covered = self.finisher.finish(
            start_node=current_route[-1],
            base_cost=current_length,
            required_remaining=req_all - covered,
        )
        if tail_nodes and len(tail_nodes) > 1:
            current_route.extend(tail_nodes[1:])
            current_length = new_total
            covered |= newly_covered

        return current_route, current_length, covered


# ========= Concrete: Coverage-Aware Finisher =========

class CoverageAwareFinisher(IFinisher):
    """
    Orienteering-style meta-graph DP over required-edge endpoints + depots + start.
    Maximizes prize = sum(weight of covered required edges) within remaining budget,
    must end at ANY depot. Materializes meta path via APSP into real nodes.
    """
    def __init__(self, G: nx.Graph, metrics: IGraphMetrics, depots: Sequence[Node], capacity: float):
        self.G = G
        self.M = metrics
        self.depots: Set[Node] = set(depots)
        self.capacity = float(capacity)

    def finish(
        self,
        start_node: Node,
        base_cost: float,
        required_remaining: Set[Edge],
    ) -> Tuple[List[Node], float, Set[Edge]]:
        budget = self.capacity - base_cost
        if budget <= 1e-12:
            return [start_node], base_cost, set()

        req_list = sorted(list(required_remaining))
        if not req_list:
            # No remaining required: go to nearest depot if possible
            d, dlen = self._nearest_depot_within_budget(start_node, budget)
            if d is None:
                return [start_node], base_cost, set()
            path = self.M.shortest_path(start_node, d)
            return path, base_cost + dlen, set()

        idx_to_edge: Dict[int, Tuple[Node, Node, float]] = {}
        endpoint_to_edges: Dict[Node, List[int]] = {}
        for i, (a, b) in enumerate(req_list):
            w = self.G[a][b]["weight"]
            idx_to_edge[i] = (a, b, w)
            endpoint_to_edges.setdefault(a, []).append(i)
            endpoint_to_edges.setdefault(b, []).append(i)

        meta_nodes: Set[Node] = set()
        for (a, b) in req_list:
            meta_nodes.add(a)
            meta_nodes.add(b)
        meta_nodes |= self.depots
        meta_nodes.add(start_node)

        from collections import deque
        Label = Tuple[Node, frozenset]
        best: Dict[Label, Tuple[float, float]] = {}
        parent: Dict[Label, Tuple[Optional[Label], Tuple[str, int]]] = {}

        def dominates(a_cost: float, a_prize: float, b_cost: float, b_prize: float) -> bool:
            return (a_cost <= b_cost and a_prize >= b_prize) and (a_cost < b_cost or a_prize > b_prize)

        start_label: Label = (start_node, frozenset())
        best[start_label] = (0.0, 0.0)  # (cost_used, prize)
        parent[start_label] = (None, ("start", -1))
        q = deque([start_label])

        while q:
            node, covered = q.popleft()
            cost_used, prize = best[(node, covered)]
            if budget - cost_used < -1e-12:
                continue

            # Cover a required edge if at its endpoint
            for e_idx in endpoint_to_edges.get(node, []):
                if e_idx in covered:
                    continue
                a, b, w_e = idx_to_edge[e_idx]
                nxt = b if node == a else a if node == b else None
                if nxt is None:
                    continue
                new_cost = cost_used + w_e
                if new_cost <= budget + 1e-12:
                    new_cov = frozenset(set(covered) | {e_idx})
                    new_prize = prize + w_e
                    lbl = (nxt, new_cov)
                    old = best.get(lbl)
                    if old is None or dominates(new_cost, new_prize, old[0], old[1]):
                        best[lbl] = (new_cost, new_prize)
                        parent[lbl] = ((node, covered), ("cover", e_idx))
                        q.append(lbl)

            # Move to another meta node
            for t in meta_nodes:
                if t == node:
                    continue
                d_nt = self.M.dist(node, t)
                if not math.isfinite(d_nt):
                    continue
                new_cost = cost_used + d_nt
                if new_cost <= budget + 1e-12:
                    lbl = (t, covered)
                    old = best.get(lbl)
                    if old is None or dominates(new_cost, prize, old[0], old[1]):
                        best[lbl] = (new_cost, prize)
                        parent[lbl] = ((node, covered), ("move", t))
                        q.append(lbl)

        # Choose label that can reach ANY depot; score by max prize, then shorter tail, then lower total cost
        best_end: Optional[Label] = None
        best_end_depot: Optional[Node] = None
        best_score: Optional[Tuple[float, float, float]] = None  # (-prize, tail_len, total_cost)

        for (node, covered), (cost_used, prize) in best.items():
            if node in self.depots:
                tail = 0.0
                total_cost = base_cost + cost_used
                score = (-prize, tail, total_cost)
                if best_score is None or score < best_score:
                    best_score, best_end, best_end_depot = score, (node, covered), node
            else:
                for d in self.depots:
                    dlen = self.M.dist(node, d)
                    if not math.isfinite(dlen):
                        continue
                    if cost_used + dlen <= budget + 1e-12:
                        total_cost = base_cost + cost_used + dlen
                        score = (-prize, dlen, total_cost)
                        if best_score is None or score < best_score:
                            best_score, best_end, best_end_depot = score, (node, covered), d

        if best_end is None:
            return [start_node], base_cost, set()

        # Reconstruct meta path and materialize into real nodes
        route_nodes = [best_end[0]]
        cur = best_end
        while parent[cur][0] is not None:
            prev, _ = parent[cur]
            route_nodes.append(prev[0])
            cur = prev
        route_nodes.reverse()

        real_path = [start_node]
        for i in range(1, len(route_nodes)):
            a, b = route_nodes[i - 1], route_nodes[i]
            seg = self.M.shortest_path(a, b)
            real_path.extend(seg[1:])
        if route_nodes[-1] != best_end_depot:
            seg = self.M.shortest_path(route_nodes[-1], best_end_depot)
            real_path.extend(seg[1:])

        # Compute final cost and newly covered
        new_cost = base_cost
        newly_covered: Set[Edge] = set()
        req_set_remaining = set(required_remaining)
        for i in range(1, len(real_path)):
            a, b = real_path[i - 1], real_path[i]
            new_cost += self.G[a][b]["weight"]
            e_sorted = tuple(sorted((a, b)))
            if e_sorted in req_set_remaining:
                newly_covered.add(e_sorted)

        return real_path, new_cost, newly_covered

    def _nearest_depot_within_budget(self, u: Node, budget: float) -> Tuple[Optional[Node], float]:
        best_d, best_len = None, float("inf")
        for d in self.depots:
            L = self.M.dist(u, d)
            if math.isfinite(L) and L <= budget and L < best_len:
                best_d, best_len = d, L
        return best_d, best_len


# ========= Concrete: Start-Depot Retargeting =========

class StartDepotRetargetor(IStartRetargetor):
    """
    If planning from current_start would cover 0 edges, try all depots s*:
    deadhead(current_start->s*) as prefix INSIDE the same trip, then plan from s*
    with reduced capacity. Choose (deadhead + cover) minimal with positive coverage.
    """
    def __init__(
        self,
        G: nx.Graph,
        metrics: IGraphMetrics,
        depots: Sequence[Node],
        capacity: float,
    ):
        self.G = G
        self.M = metrics
        self.depots = list(depots)
        self.capacity = float(capacity)

    def retarget_and_plan(
        self, current_start: Node, required_remaining: Set[Edge]
    ) -> Optional[Trip]:
        # Precompute distances from current_start
        best: Optional[Tuple[float, int, List[Node], Set[Edge]]] = None

        for s_star in self.depots:
            D = self.M.dist(current_start, s_star)
            if not math.isfinite(D) or D >= self.capacity - 1e-12:
                continue

            # plan from s* with reduced capacity
            metrics_reduced = _ReducedCapacityMetrics(self.G, self.M, cap_shift=D)
            finisher = CoverageAwareFinisher(self.G, metrics_reduced, self.depots, self.capacity - D)
            inner = MagneticFieldTripPlanner(
                G=self.G,
                metrics=metrics_reduced,
                depots=self.depots,
                capacity=self.capacity - D,
                start_depot=s_star,
                finisher=finisher,
            )
            route2, cost2, covered2 = inner.plan_single_trip(required_remaining, verbose=False)
            if not covered2:
                continue

            # stitch full route (deadhead + covering)
            if s_star == current_start:
                full_route = route2
                full_cost = cost2  # D == 0
            else:
                deadhead_path = self.M.shortest_path(current_start, s_star)
                full_route = deadhead_path + route2[1:]
                full_cost = D + cost2

            cand = (full_cost, -len(covered2), s_star, full_route, covered2)
            if best is None or cand < best:
                best = cand

        if best is None:
            return None

        full_cost, _, _, full_route, covered2 = best
        return Trip(
            start=full_route[0],
            end=full_route[-1],
            route=full_route,
            cost=full_cost,
            covered_edges=covered2,
        )


class _ReducedCapacityMetrics(IGraphMetrics):
    """
    Decorator over metrics that does NOT change distances/paths;
    it exists to represent the reduced capacity context (shifted by deadhead D).
    Trip planning logic reads capacity from the planner, not from metrics; this
    class is only here to keep signatures uniform and SRP intact.
    """
    def __init__(self, G: nx.Graph, base: IGraphMetrics, cap_shift: float):
        self.G = G
        self.base = base
        self.cap_shift = cap_shift

    def dist(self, a: Node, b: Node) -> float:
        return self.base.dist(a, b)

    def shortest_path(self, a: Node, b: Node) -> List[Node]:
        return self.base.shortest_path(a, b)

    def min_dist_to_any(self, a: Node, targets: Set[Node]) -> float:
        return self.base.min_dist_to_any(a, targets)


# ========= Orchestrator =========

class MultiTripPlanner:
    """
    High-level loop:
      - Plan a trip from current_start;
      - If coverage==0, use retargetor to build one trip (deadhead+cover) and commit it;
      - Else commit the normal trip.
      - Repeat until all required edges are covered or guard trips exhausted.
    """
    def __init__(
        self,
        G: nx.Graph,
        depots: Sequence[Node],
        capacity: float,
        metrics: Optional[IGraphMetrics] = None,
    ):
        self.G = G
        self.depots = list(depots)
        self.capacity = float(capacity)
        self.M = metrics or APSPMetrics(G)

    def plan(
        self,
        start_depot: Node,
        required_edges: Sequence[Edge],
        *,
        verbose: bool = False,
        max_trips_guard: Optional[int] = None,
    ) -> PlanSummary:
        req_remaining: Set[Edge] = set(tuple(sorted(e)) for e in required_edges)
        trips: List[Trip] = []
        total_cost = 0.0
        current_start = start_depot
        guard = max_trips_guard or max(2 * len(req_remaining), 50)

        while req_remaining and len(trips) < guard:
            finisher = CoverageAwareFinisher(self.G, self.M, self.depots, self.capacity)
            single = MagneticFieldTripPlanner(
                G=self.G,
                metrics=self.M,
                depots=self.depots,
                capacity=self.capacity,
                start_depot=current_start,
                finisher=finisher,
            )
            route, trip_cost, covered_now = single.plan_single_trip(req_remaining, verbose=verbose)

            # compute coverage set from route to be consistent
            if not covered_now:
                # Retarget inside the SAME trip; no standalone relocation
                retarget = StartDepotRetargetor(self.G, self.M, self.depots, self.capacity)
                trip = retarget.retarget_and_plan(current_start, req_remaining)
                if trip is None:
                    # No feasible coverage from any depot in one cycle; stop
                    break
                trips.append(trip)
                total_cost += trip.cost
                req_remaining -= trip.covered_edges
                current_start = trip.end
                continue

            # Normal covering trip
            trip = Trip(
                start=current_start,
                end=route[-1],
                route=route,
                cost=trip_cost,
                covered_edges=set(covered_now),
            )
            trips.append(trip)
            total_cost += trip_cost
            req_remaining -= covered_now
            current_start = trip.end

        return PlanSummary(
            trips=trips,
            total_cost=total_cost,
            required_initial=len(required_edges),
            required_remaining=len(req_remaining),
        )


# ========= CLI helper (optional) =========

def print_summary(summary: PlanSummary) -> None:
    print("\n=== Multi-Trip Summary ===")
    print(f"Trips committed: {len(summary.trips)}")
    print(f"Total cost (no standalone relocations): {summary.total_cost:.2f}")
    print(f"Required edges initially: {summary.required_initial}")
    print(f"Remaining uncovered edges: {summary.required_remaining}")
    for i, t in enumerate(summary.trips, 1):
        print(f"Trip {i}: start={t.start} end={t.end} "
              f"cost={t.cost:.2f} steps={len(t.route)-1} covered={len(t.covered_edges)} "
              f"route={t.route}")


# ========= Example main (kept minimal for tests) =========

def main(path: str) -> None:
    G, req_edges, depots, cap = load_scenario(path)
    planner = MultiTripPlanner(G, depots, cap)
    summary = planner.plan(start_depot=depots[0], required_edges=req_edges, verbose=False)
    print_summary(summary)


if __name__ == "__main__":
    for sce in range(111, 112):
        print(f'\nRunning scenario - {sce}')
        # scenario_txt_path = f"gdb_failure_scenarios/gdb.{sce}.txt"   # change to your file name if needed
        # scenario_txt_path = f"bccm_failure_scenarios/bccm.{sce}.txt"   # change to your file name if needed
        scenario_txt_path = f"eglese_failure_scenarios/eglese.{sce}.txt"   # change to your file name if needed
        main(scenario_txt_path)
    # pass
