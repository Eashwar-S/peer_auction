#!/usr/bin/env python3
"""
Multi-Trip Router with Coverage-Aware Finisher (Parameter-Free)

- Parses a GDB-style instance (e.g., gdb.1.txt).
- Runs a cycle-proof single-trip growth (frontier-progress, no backtrack).
- At the end of each trip, uses a general "relocation-with-coverage" finisher:
  _finish_with_coverage_orienteering(...) which finds the best path to ANY depot
  under remaining budget that may cover multiple required edges on the way.
- Multi-trip loop until all required edges are covered.

No new tunables. Uses APSP, shortest paths, and required edge weights only.
"""

import re
import math
import networkx as nx
from math import exp
from typing import List, Tuple, Set, Dict, Optional

# ---------- Parsing ----------

def parse_txt_file(path: str):
    """
    Parse a GDB-like instance text.
    Minimal fields used:
      NUMBER OF VERTICES
      VEHICLE CAPACITY
      NUMBER OF VEHICLES (optional)
      RECHARGE TIME (optional)
      DEPOT: comma-separated ints
      LIST_REQUIRED_EDGES:
        (u,v) edge weight w
      LIST_NON_REQUIRED_EDGES:
        (u,v) edge weight w
    Returns:
      G, required_edges, depots, capacity, recharge, nveh, fail_vs, fail_ts
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    def extract_int(key):
        for ln in lines:
            if ln.upper().startswith(key.upper()):
                m = re.findall(r"(-?\d+)", ln)
                if m:
                    return int(m[0])
        return None

    def extract_float(key):
        for ln in lines:
            if ln.upper().startswith(key.upper()):
                m = re.findall(r"(-?\d+(?:\.\d+)?)", ln)
                if m:
                    return float(m[0])
        return None

    def extract_list_depots():
        for ln in lines:
            if ln.upper().startswith("DEPOT"):
                nums = re.findall(r"(\d+)", ln)
                return [int(x) for x in nums]
        return []

    nV = extract_int("NUMBER OF VERTICES")
    capacity = extract_float("VEHICLE CAPACITY") or 0.0
    nveh = extract_int("NUMBER OF VEHICLES") or 1
    recharge = extract_float("RECHARGE TIME") or 0.0
    depots = extract_list_depots()

    req_section = False
    nonreq_section = False
    required_edges = []
    weights = {}

    for ln in lines:
        if ln.upper().startswith("LIST_REQUIRED_EDGES"):
            req_section = True; nonreq_section = False; continue
        if ln.upper().startswith("LIST_NON_REQUIRED_EDGES"):
            nonreq_section = True; req_section = False; continue
        if ln.upper().startswith("FAILURE_SCENARIO"):
            req_section = False; nonreq_section = False; continue

        m = re.match(r"\((\d+),\s*(\d+)\)\s*edge\s*weight\s*([0-9.]+)", ln, re.IGNORECASE)
        if m:
            u, v, w = int(m.group(1)), int(m.group(2)), float(m.group(3))
            weights[tuple(sorted((u, v)))] = w
            if req_section:
                required_edges.append((u, v))

    # Build graph
    G = nx.Graph()
    if nV:
        G.add_nodes_from(range(1, nV + 1))
    for e, w in weights.items():
        a, b = e
        G.add_edge(a, b, weight=w)

    # Optional failures (not used in routing)
    fail_vs, fail_ts = [], []
    for ln in lines:
        m = re.match(r"Vehicle\s+(\d+)\s+will\s+fail\s+in\s+(\d+)", ln, re.IGNORECASE)
        if m:
            fail_vs.append(int(m.group(1)))
            fail_ts.append(float(m.group(2)))

    return G, required_edges, depots, capacity, recharge, nveh, fail_vs, fail_ts

# ---------- Router ----------

class MagneticFieldRouter:
    """
    Single-trip router:
      - Frontier progress rule (must move closer to uncovered edges unless covering).
      - No immediate backtrack (unless covering).
      - Free-end depot.
      - NEW: coverage-aware finisher that relocates to any depot and may cover multiple
             required edges on the way (orienteering-style on a reduced meta-graph).
    """

    def __init__(self, graph: nx.Graph, start_depot: int, depots: List[int], capacity: float,
                 alpha: float = 1.0, gamma: float = 1.0):
        self.graph = graph
        self.start_depot = start_depot
        self.depots = set(depots)
        self.capacity = float(capacity)
        self.alpha = alpha
        self.gamma = gamma

        # APSP distances for metric lookups
        self._dist: Dict[int, Dict[int, float]] = dict(
            nx.all_pairs_dijkstra_path_length(self.graph, weight='weight')
        )

    # ---- Dist helpers ----

    def _min_dist_to_any_depot(self, node: int) -> float:
        INF = float('inf')
        best = INF
        row = self._dist.get(node, {})
        for d in self.depots:
            v = row.get(d, INF)
            if v < best:
                best = v
        return best

    def _dist_to_frontier(self, node: int, req_set_remaining: Set[Tuple[int, int]]) -> float:
        """Distance from node to nearest endpoint of any remaining required edge."""
        INF = float('inf')
        best = INF
        row = self._dist.get(node, {})
        for (a, b) in req_set_remaining:
            da = row.get(a, INF); db = row.get(b, INF)
            if da < best: best = da
            if db < best: best = db
            if best == 0: break
        return best

    # ---- Scoring (parameter-free) ----

    def calculate_edge_score(self, edge: Tuple[int, int],
                             req_remaining: List[Tuple[int, int]],
                             current_length: float,
                             is_new_required: bool):
        """
        Simple score:
          final = (1-w)*P + w*D, w = current_length / capacity
          P: reward for “new required” (1.0) else exp(-frontier_distance)
          D: exp(-return-to-depot distance / capacity)
        """
        u, v = edge
        w_uv = self.graph[u][v]['weight']
        w_cap = min(max(current_length / max(self.capacity, 1e-9), 0.0), 1.0)

        if is_new_required:
            P = 1.0
        else:
            req_set = {tuple(sorted(e)) for e in req_remaining}
            Fv = self._dist_to_frontier(v, req_set)
            P = 0.0 if math.isinf(Fv) else exp(-1.0 * Fv)

        d_v = self._min_dist_to_any_depot(v)
        D = 0.0 if math.isinf(d_v) else exp(-1.0 * (d_v / max(self.capacity, 1e-9)))

        final = (1.0 - w_cap) * P + w_cap * D
        return {"edge_weight": w_uv, "final_score": final}

    # ---- Helpers for MST lower bound (not strictly needed in the finisher, but kept) ----

    def _frontier_nodes(self, req_set: Set[Tuple[int, int]]) -> Set[int]:
        nodes = set()
        for a, b in req_set:
            nodes.add(a); nodes.add(b)
        return nodes

    def _mst_metric_lb(self, frontier_nodes: Set[int], depot: int) -> float:
        pts = list(frontier_nodes | {depot})
        if len(pts) <= 1:
            return 0.0
        H = nx.Graph()
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                a, b = pts[i], pts[j]
                d = self._dist.get(a, {}).get(b, float('inf'))
                if d < float('inf'):
                    H.add_edge(a, b, weight=d)
        if H.number_of_edges() == 0:
            return 0.0
        T = nx.minimum_spanning_tree(H, weight='weight')
        return sum(d['weight'] for _, _, d in T.edges(data=True))

    # ---- Main single-trip planner body (cycle-proof growth) ----

    def find_route_with_magnetic_scoring(self, required_edges: List[Tuple[int, int]], verbose: bool = False):
        current_route: List[int] = [self.start_depot]
        current_length: float = 0.0
        required_covered: Set[Tuple[int, int]] = set()
        max_iterations = len(self.graph.edges()) * 10
        iteration_count = 0

        req_set_all = {tuple(sorted(e)) for e in required_edges}
        depot_depot_covered_this_trip = set()

        while len(required_covered) < len(req_set_all) and iteration_count < max_iterations:
            iteration_count += 1
            u = current_route[-1]
            req_set_remaining = req_set_all - required_covered
            if not req_set_remaining:
                break
            Fu = self._dist_to_frontier(u, req_set_remaining)

            candidates = []
            for v in self.graph.neighbors(u):
                edge = (u, v)
                edge_sorted = tuple(sorted(edge))
                w_uv = self.graph[u][v]['weight']

                depot_after = self._min_dist_to_any_depot(v)
                if current_length + w_uv + depot_after > self.capacity:
                    continue

                is_new_required = (edge_sorted in req_set_remaining)

                # No immediate backtrack unless covering
                if len(current_route) >= 2 and v == current_route[-2] and not is_new_required:
                    continue

                # Avoid reusing depot–depot required edge in same trip unless covering
                if (u in self.depots and v in self.depots and
                    edge_sorted in depot_depot_covered_this_trip and not is_new_required):
                    continue

                # Frontier progress unless covering
                Fv = self._dist_to_frontier(v, req_set_remaining)
                if not is_new_required and not (Fv < Fu):
                    continue

                sd = self.calculate_edge_score(edge, list(req_set_remaining), current_length, is_new_required)
                candidates.append({'edge': edge, 'neighbor': v, 'is_new_required': is_new_required,
                                   'score_data': sd, 'Fv': Fv})

            if not candidates:
                # Force-move toward nearest frontier endpoint while preserving depot return feasibility
                best_path, best_len = None, float('inf')
                for (a, b) in req_set_remaining:
                    for t in (a, b):
                        try:
                            path = nx.shortest_path(self.graph, u, t, weight='weight')
                            plen = nx.shortest_path_length(self.graph, u, t, weight='weight')
                            depot_after = self._min_dist_to_any_depot(t)
                            if current_length + plen + depot_after <= self.capacity and plen < best_len:
                                best_path, best_len = path[1:], plen
                        except nx.NetworkXNoPath:
                            continue
                if best_path:
                    for nxt in best_path:
                        prev = current_route[-1]
                        e_sorted = tuple(sorted((prev, nxt)))
                        if prev in self.depots and nxt in self.depots and e_sorted in req_set_remaining:
                            depot_depot_covered_this_trip.add(e_sorted)
                        current_route.append(nxt)
                        current_length += self.graph[prev][nxt]['weight']
                    continue

                # Fallback: finish at nearest depot NOW (we will still run the coverage-aware finisher next)
                break

            # Prefer covering, then lower frontier distance, then higher score
            candidates.sort(key=lambda c: (not c['is_new_required'], c['Fv'], -c['score_data']['final_score']))
            best = candidates[0]
            v = best['neighbor']
            current_route.append(v)
            current_length += best['score_data']['edge_weight']

            if best['is_new_required']:
                e_sorted = tuple(sorted(best['edge']))
                required_covered.add(e_sorted)
                if best['edge'][0] in self.depots and best['edge'][1] in self.depots:
                    depot_depot_covered_this_trip.add(e_sorted)
                if verbose:
                    print(f"✓ Covered required edge: {best['edge']} "
                          f"({len(required_covered)}/{len(req_set_all)})")

            if verbose:
                left = len(req_set_all - required_covered)
                print(f"Step: {best['edge']} -> {v}, len={current_length:.2f}, left={left}")

        # === Coverage-aware finisher ===
        u = current_route[-1]
        req_set_remaining = req_set_all - required_covered

        finish_route, finish_cost, newly_covered = self._finish_with_coverage_orienteering(
            start_node=u,
            base_cost=current_length,
            req_set_remaining=set(req_set_remaining)
        )
        # Stitch the finisher path (if any)
        if finish_route and len(finish_route) > 1:
            # Append the new nodes, compute exact appended cost
            for i in range(1, len(finish_route)):
                a, b = finish_route[i-1], finish_route[i]
                current_route.append(b)
            # Update length and coverage from finisher
            current_length = finish_cost
            required_covered |= newly_covered

        return current_route, current_length, len(required_covered)

    # ---------- Coverage-aware finisher ----------
    #
    # Core idea: within remaining budget (capacity - base_cost), find a path that
    # ends at ANY depot while maximizing the total weight of remaining required
    # edges covered. We model this on a reduced meta-graph with nodes = endpoints
    # of remaining required edges ∪ {all depots} ∪ {start}, and use a label-setting
    # DP with dominance pruning. Then materialize back via APSP paths.

    def _finish_with_coverage_orienteering(self,
                                           start_node: int,
                                           base_cost: float,
                                           req_set_remaining: Set[Tuple[int, int]]):
        """
        Returns:
          (route_segment_nodes, new_total_cost, covered_edges_set)
        where route_segment_nodes starts at start_node and ends at some depot.

        If no feasible movement exists, returns ([start_node], base_cost, set()).
        """
        budget = self.capacity - base_cost
        if budget <= 1e-12:
            # Already at/near capacity; ensure end at nearest depot if not already depot
            if start_node in self.depots:
                return [start_node], base_cost, set()
            # go to nearest depot if possible within zero budget (likely not), else do nothing
            return [start_node], base_cost, set()

        # Normalize remaining required as indexed list for bitset-like tracking
        req_list = sorted(list({tuple(sorted(e)) for e in req_set_remaining}))
        if not req_list:
            # No remaining required edges: just go to nearest depot
            end_d, dlen = self._nearest_depot_within_budget(start_node, budget)
            if end_d is None:
                return [start_node], base_cost, set()
            path = self._shortest_path_nodes(start_node, end_d)
            return path, base_cost + dlen, set()

        # Precompute helpers
        # edge index -> (a,b,weight)
        idx_to_edge = {}
        endpoint_to_edges: Dict[int, List[int]] = {}
        for i, (a, b) in enumerate(req_list):
            w = self.graph[a][b]['weight']
            idx_to_edge[i] = (a, b, w)
            endpoint_to_edges.setdefault(a, []).append(i)
            endpoint_to_edges.setdefault(b, []).append(i)

        # Meta-nodes we allow to stand on: all endpoints + all depots + start
        meta_nodes = set()
        for (a, b) in req_list:
            meta_nodes.add(a); meta_nodes.add(b)
        meta_nodes |= set(self.depots)
        meta_nodes.add(start_node)

        # Quick distance accessor
        def dist(x, y):
            return self._dist.get(x, {}).get(y, float('inf'))

        # ----- Label-setting DP -----
        # State: (node, covered_bitmask) -> best_cost, best_prize
        # But we need to keep multiple Pareto-optimal labels (cost, prize) per (node, bitmask)
        # We'll implement dominance pruning per node, but track different bitmasks

        from collections import defaultdict, deque

        # Represent covered set as frozenset indices (keeps it simple & clear)
        # Each label:
        #   node: current meta node
        #   covered: frozenset of edge indices covered so far
        #   cost_used: float
        #   prize: float (sum of weights of covered edges)
        Label = Tuple[int, frozenset]

        best: Dict[Label, Tuple[float, float]] = {}  # label -> (cost_used, prize)
        parent: Dict[Label, Tuple[Optional[Label], Tuple[str, int]]] = {}
        # action encoded as ("move", next_node) or ("cover", edge_idx)

        start_label: Label = (start_node, frozenset())
        best[start_label] = (0.0, 0.0)
        parent[start_label] = (None, ("start", -1))

        q = deque([start_label])

        def dominates(a_cost, a_prize, b_cost, b_prize):
            # a dominates b if a is no worse in cost and no worse in prize and strictly better in one
            return (a_cost <= b_cost and a_prize >= b_prize) and (a_cost < b_cost or a_prize > b_prize)

        # For pruning: at each (node, covered) we only keep the best (lowest cost, highest prize)
        # Also, limit exploration to meta_nodes only (we stitch real nodes at the end).
        while q:
            node, covered = q.popleft()
            cost_used, prize = best[(node, covered)]
            remaining = budget - cost_used
            if remaining < -1e-12:
                continue

            # 1) Try to "cover" a required edge if we're at one of its endpoints
            if node in endpoint_to_edges:
                for e_idx in endpoint_to_edges[node]:
                    if e_idx in covered:
                        continue
                    a, b, w_e = idx_to_edge[e_idx]
                    # Can traverse the required edge only if we're at a or b
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

            # 2) Try to "move" to any other meta node (endpoints or depots) (no prize)
            for t in meta_nodes:
                if t == node:
                    continue
                d_nt = dist(node, t)
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

        # Among all labels, pick one that can reach ANY depot within leftover budget
        # Score by: (max prize), tie-break by minimal added tail-to-depot, then minimal total cost
        best_end = None
        best_score = None  # (-prize, tail_len, total_cost)  lexicographically smaller is better
        best_end_depot = None

        for (node, covered), (cost_used, prize) in best.items():
            # If already at a depot, tail = 0, else need to reach a depot
            if node in self.depots:
                tail = 0.0
                total_cost = base_cost + cost_used
                score = (-prize, tail, total_cost)
                if best_score is None or score < best_score:
                    best_score = score
                    best_end = (node, covered)
                    best_end_depot = node
            else:
                # Try each depot
                for d in self.depots:
                    dlen = dist(node, d)
                    if not math.isfinite(dlen):
                        continue
                    if cost_used + dlen <= budget + 1e-12:
                        total_cost = base_cost + cost_used + dlen
                        score = (-prize, dlen, total_cost)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_end = (node, covered)
                            best_end_depot = d

        if best_end is None:
            # Could not move anywhere; if already depot, done; else go to nearest depot if possible
            if start_node in self.depots:
                return [start_node], base_cost, set()
            d, dlen = self._nearest_depot_within_budget(start_node, budget)
            if d is None:
                return [start_node], base_cost, set()
            path = self._shortest_path_nodes(start_node, d)
            return path, base_cost + dlen, set()

        # Reconstruct meta path from parent pointers up to best_end; then add final jump to depot (if any)
        route_nodes = [best_end[0]]
        covered_set = set(best_end[1])
        cur = best_end
        while parent[cur][0] is not None:
            prev, act = parent[cur]
            route_nodes.append(prev[0])
            cur = prev
        route_nodes = list(reversed(route_nodes))

        # Materialize into original graph nodes by stitching APSP paths and required edges
        real_path = [start_node]
        for i in range(1, len(route_nodes)):
            a, b = route_nodes[i - 1], route_nodes[i]
            # If this step corresponds to a "move", stitch shortest path a->b
            # If "cover", it is exactly the required edge (a,b), which should exist directly.
            # We infer the action by checking if (sorted(a,b)) is a required edge in idx_to_edge.
            if (a in self._dist) and (b in self._dist[a]):
                seg = self._shortest_path_nodes(a, b)
                real_path.extend(seg[1:])
            else:
                # Should not happen if APSP exists; as a fallback, add direct b
                real_path.append(b)

        # If not already at chosen depot, add final depot hop
        end_node = route_nodes[-1]
        if end_node != best_end_depot:
            seg = self._shortest_path_nodes(end_node, best_end_depot)
            real_path.extend(seg[1:])

        # Compute final cost and the set of newly covered required edges
        new_cost = base_cost
        newly_covered = set()
        for i in range(1, len(real_path)):
            a, b = real_path[i - 1], real_path[i]
            w = self.graph[a][b]['weight']
            new_cost += w
            e_sorted = tuple(sorted((a, b)))
            if e_sorted in req_set_remaining:
                newly_covered.add(e_sorted)

        return real_path, new_cost, newly_covered

    # ---- Small utilities ----

    def _shortest_path_nodes(self, a: int, b: int) -> List[int]:
        if a == b:
            return [a]
        return nx.shortest_path(self.graph, a, b, weight='weight')

    def _nearest_depot_within_budget(self, u: int, budget: float):
        best_d, best_len = None, float('inf')
        row = self._dist.get(u, {})
        for d in self.depots:
            L = row.get(d, float('inf'))
            if math.isfinite(L) and L <= budget and L < best_len:
                best_d, best_len = d, L
        return best_d, best_len

# ---------- Multi-trip loop with relocation coalescing ----------

def run_intelligent_tuning_demo():
    scenario_txt_path = "gdb_failure_scenarios/gdb.1.txt"  # change to your file name if needed
    G, req_edges, depots, cap, recharge, nveh, fail_vs, fail_ts = parse_txt_file(scenario_txt_path)

    req_edges_remaining = [tuple(sorted(e)) for e in req_edges]
    req_set_remaining = set(req_edges_remaining)

    START_DEPOT = depots[0]
    MAX_VEHICLE_CAPACITY = cap

    trips = []
    total_cost = 0.0
    trip_idx = 0
    MAX_TRIPS_GUARD = max(2 * len(req_edges_remaining), 50)

    current_start = START_DEPOT

    while req_set_remaining and trip_idx < MAX_TRIPS_GUARD:
        router = MagneticFieldRouter(
            graph=G,
            start_depot=current_start,
            depots=depots,
            capacity=MAX_VEHICLE_CAPACITY,
            alpha=1.0, gamma=1.0
        )

        route, cost, _ncovered_reported = router.find_route_with_magnetic_scoring(list(req_set_remaining), verbose=False)

        if not route or len(route) < 2:
            print(f"[Plan] No feasible route found from {current_start}. Stopping.")
            break

        route_edge_seq = [tuple(sorted((route[i], route[i+1]))) for i in range(len(route)-1)]
        covered_now = set(e for e in route_edge_seq if e in req_set_remaining)

        # If this leg covered nothing: treat as relocation (not a trip); advance start and continue
        if not covered_now:
            print(f"[Relocate] Zero-coverage move {current_start}->{route[-1]} "
                  f"(cost {cost - total_cost:.2f}); not counted.")
            total_cost += (cost - total_cost) if trips else cost  # ensure monotonic
            current_start = route[-1]
            continue

        # Commit trip
        trip_idx += 1
        req_set_remaining -= covered_now
        leg_cost = cost - total_cost if trips else cost
        total_cost = cost if not trips else total_cost + leg_cost  # keep sum consistent

        print(f"[Trip {trip_idx}] start={current_start} -> end={route[-1]} | "
              f"cost={leg_cost:.2f} | covered={len(covered_now)} | remaining={len(req_set_remaining)}")

        trips.append({
            "start": current_start,
            "end": route[-1],
            "route": route,
            "cost": leg_cost,
            "covered_edges": covered_now,
        })

        current_start = route[-1]

    # Summary
    print("\n=== Multi-Trip Summary ===")
    print(f"Trips committed: {len(trips)}")
    print(f"Total cost (including relocations): {total_cost:.2f}")
    print(f"Required edges initially: {len(req_edges)}")
    print(f"Remaining uncovered edges: {len(req_set_remaining)}")

    for i, t in enumerate(trips, 1):
        print(f"Trip {i}: start={t['start']} end={t['end']} "
              f"cost={t['cost']:.2f} steps={len(t['route'])-1} covered={len(t['covered_edges'])} "
              f"route={t['route']}")

if __name__ == "__main__":
    run_intelligent_tuning_demo()
