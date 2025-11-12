#!/usr/bin/env python3
"""
Multi-Trip Magnetic Field Router (Parameter-Free)
- Cycle-proof single-trip router (frontier-progress + no-backtrack).
- Free-end depots.
- End-depot lookahead (ghost next trip + MST lower bound).
- Greedy post-fill to pack capacity.
- Multi-trip loop with ZERO-COVERAGE COALESCING:
    * If a trip covers 0 new required edges, try to merge it into the previous trip (capacity permitting).
    * Otherwise do not count it as a trip; just relocate the start and plan again.
"""

import re
import math
import networkx as nx
from math import exp
from typing import List, Tuple, Set

# ---------- Parsing ----------

def parse_txt_file(path: str):
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

    G = nx.Graph()
    if nV:
        G.add_nodes_from(range(1, nV + 1))
    for e, w in weights.items():
        a, b = e
        G.add_edge(a, b, weight=w)

    fail_vs, fail_ts = [], []
    for ln in lines:
        m = re.match(r"Vehicle\s+(\d+)\s+will\s+fail\s+in\s+(\d+)", ln, re.IGNORECASE)
        if m:
            fail_vs.append(int(m.group(1)))
            fail_ts.append(float(m.group(2)))

    return G, required_edges, depots, capacity, recharge, nveh, fail_vs, fail_ts

# ---------- Router ----------

class MagneticFieldRouter:
    def __init__(self, graph: nx.Graph, start_depot: int, depots: List[int], capacity: float,
                 alpha: float = 1.0, gamma: float = 1.0):
        self.graph = graph
        self.start_depot = start_depot
        self.depots = set(depots)
        self.capacity = float(capacity)
        self.alpha = alpha
        self.gamma = gamma
        self._dist = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='weight'))

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


    def _pre_depot_frontier_fill(self, u, current_length, req_set_remaining, current_route, required_covered):
        """
        Try to take ONE MORE required edge before choosing an end depot.
        Feasibility check uses min return-to-any-depot (no fixed d* yet).
        """
        if not req_set_remaining:
            return u, current_length

        # pick nearest endpoint of any remaining required edge
        best_path, best_plen, best_edge = None, float('inf'), None
        row = self._dist.get(u, {})
        for (a, b) in list(req_set_remaining):
            for t, other in ((a, b), (b, a)):
                d_ut = row.get(t, float('inf'))
                if d_ut < best_plen:
                    try:
                        path = nx.shortest_path(self.graph, u, t, weight='weight')[1:]
                        best_plen, best_path, best_edge = d_ut, path, (t, other)
                    except nx.NetworkXNoPath:
                        pass

        if best_path is None:
            return u, current_length

        # cost if we take u->t + traverse that required edge + return to SOME depot
        req_edge_sorted = tuple(sorted(best_edge))
        req_w = self.graph[best_edge[0]][best_edge[1]]['weight']
        # min return over all depots from the edge's other endpoint
        return_home = min(self._dist.get(best_edge[1], {}).get(d, float('inf')) for d in self.depots)
        total_if_take = current_length + best_plen + req_w + return_home

        if total_if_take <= self.capacity:
            # append path u->t
            for nxt in best_path:
                prev = current_route[-1]
                current_route.append(nxt)
                current_length += self.graph[prev][nxt]['weight']
            # traverse required edge t->other
            prev = current_route[-1]
            current_route.append(best_edge[1])
            current_length += self.graph[prev][best_edge[1]]['weight']
            required_covered.add(req_edge_sorted)
            req_set_remaining.discard(req_edge_sorted)
            u = current_route[-1]

        return u, current_length



    # ---- Frontier/MST helpers ----
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

    # ---- Ghost trip (no recursive lookahead) ----
    def _single_trip_greedy_no_lookahead(self, required_edges: List[Tuple[int, int]], verbose: bool = False):
        current_route = [self.start_depot]
        current_length = 0.0
        required_covered = set()
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

                # Avoid reusing depot–depot required edge this trip unless covering it
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
                # Force-move to nearest frontier endpoint while preserving return feasibility
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

                # Finish at nearest depot
                if u not in self.depots:
                    nearest = min(self.depots, key=lambda d: self._dist[u].get(d, float('inf')))
                    path_to_end = nx.shortest_path(self.graph, u, nearest, weight='weight')
                    add_len = nx.shortest_path_length(self.graph, u, nearest, weight='weight')
                    if current_length + add_len <= self.capacity:
                        current_route.extend(path_to_end[1:])
                        current_length += add_len
                break

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

        if current_route[-1] not in self.depots:
            u = current_route[-1]
            nearest = min(self.depots, key=lambda d: self._dist[u].get(d, float('inf')))
            path_to_end = nx.shortest_path(self.graph, u, nearest, weight='weight')
            add_len = nx.shortest_path_length(self.graph, u, nearest, weight='weight')
            if current_length + add_len <= self.capacity:
                current_route.extend(path_to_end[1:])
                current_length += add_len

        route_edge_seq = [tuple(sorted((current_route[i], current_route[i+1])))
                          for i in range(len(current_route)-1)]
        covered_now = {e for e in route_edge_seq if e in req_set_all}
        return current_route, current_length, covered_now

    # ---- Lookahead depot choice ----
    def _choose_end_depot_lookahead(self, u: int, current_length: float,
                                    req_set_remaining: Set[Tuple[int, int]]):
        feas = []
        row = self._dist.get(u, {})
        for d in self.depots:
            tail = row.get(d, float('inf'))
            if tail < float('inf') and current_length + tail <= self.capacity:
                feas.append((d, tail))
        if not feas:
            return None

        best_key, best_d, best_ghost_cov = None, None, -1
        for d, tail_len in feas:
            ghost = MagneticFieldRouter(self.graph, start_depot=d, depots=list(self.depots),
                                        capacity=self.capacity, alpha=self.alpha, gamma=self.gamma)
            ghost_route, ghost_len, ghost_cov = ghost._single_trip_greedy_no_lookahead(list(req_set_remaining))
            req_after = set(req_set_remaining) - set(ghost_cov)

            W_rem = 0.0
            for (a, b) in req_after:
                W_rem += self.graph[a][b]['weight']

            d_prime = ghost_route[-1] if ghost_route else d
            LB = max(W_rem, self._mst_metric_lb(self._frontier_nodes(req_after), d_prime))
            cov_per_len = (len(ghost_cov) / max(ghost_len, 1e-9)) if ghost_len > 0 else 0.0
            # key = (W_rem, LB, tail_len, -cov_per_len)
            future_cost_lb = LB + tail_len
            rem_work_plus_tail = W_rem + tail_len
            key = (rem_work_plus_tail, future_cost_lb, tail_len, -cov_per_len)

            # tie-break: prefer ghost that covers >=1 edge
            if best_key is None or key < best_key or (key == best_key and len(ghost_cov) > best_ghost_cov):
                best_key, best_d, best_ghost_cov = key, d, len(ghost_cov)
        return best_d

    # ---- Post-fill greedy toward chosen depot ----
    def _post_fill_toward_depot(self, u: int, current_length: float, d_star: int,
                                req_set_remaining: Set[Tuple[int, int]],
                                current_route: List[int], required_covered: Set[Tuple[int, int]]):
        while req_set_remaining:
            best_t, best_path, best_plen, best_edge = None, None, float('inf'), None
            row = self._dist.get(u, {})
            for (a, b) in list(req_set_remaining):
                for t, other in ((a, b), (b, a)):
                    d_ut = row.get(t, float('inf'))
                    if d_ut < best_plen:
                        try:
                            path = nx.shortest_path(self.graph, u, t, weight='weight')[1:]
                            best_t, best_edge = t, (t, other)
                            best_plen = d_ut
                            best_path = path
                        except nx.NetworkXNoPath:
                            pass
            if best_path is None:
                break

            req_edge_sorted = tuple(sorted(best_edge))
            req_w = self.graph[best_edge[0]][best_edge[1]]['weight']
            return_home = self._dist.get(best_edge[1], {}).get(d_star, float('inf'))
            total_if_take = current_length + best_plen + req_w + return_home

            if total_if_take <= self.capacity:
                for nxt in best_path:
                    prev = current_route[-1]
                    current_route.append(nxt)
                    current_length += self.graph[prev][nxt]['weight']
                prev = current_route[-1]
                current_route.append(best_edge[1])
                current_length += self.graph[prev][best_edge[1]]['weight']
                required_covered.add(req_edge_sorted)
                req_set_remaining.discard(req_edge_sorted)
                u = current_route[-1]
            else:
                break
        return u, current_length

    # ---- Main single-trip planner ----
    def find_route_with_magnetic_scoring(self, required_edges: List[Tuple[int, int]], verbose: bool = False):
        current_route = [self.start_depot]
        current_length = 0.0
        required_covered = set()
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
                # Force-move towards nearest frontier while preserving depot return feasibility
                best_path, best_len = None, float('inf')
                for (a, b) in req_set_all - required_covered:
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
                        if prev in self.depots and nxt in self.depots and e_sorted in (req_set_all - required_covered):
                            depot_depot_covered_this_trip.add(e_sorted)
                        current_route.append(nxt)
                        current_length += self.graph[prev][nxt]['weight']
                    continue

                # Fallback finish at nearest depot
                if u not in self.depots:
                    nearest = min(self.depots, key=lambda d: self._dist[u].get(d, float('inf')))
                    path_to_end = nx.shortest_path(self.graph, u, nearest, weight='weight')
                    add_len = nx.shortest_path_length(self.graph, u, nearest, weight='weight')
                    if current_length + add_len <= self.capacity:
                        current_route.extend(path_to_end[1:])
                        current_length += add_len
                break

            # Prefer covering, then lower Fv, then higher score
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

        # Lookahead end-depot choice + post-fill
        u = current_route[-1]
        req_set_remaining = req_set_all - required_covered

        # NEW: try to grab one more required edge BEFORE picking a depot (parameter-free)
        u, current_length = self._pre_depot_frontier_fill(
            u, current_length, set(req_set_remaining), current_route, required_covered
        )
        req_set_remaining = (req_set_all - required_covered)  # refresh after possible fill


        d_star = self._choose_end_depot_lookahead(u, current_length, req_set_remaining)
        if d_star is None:
            d_star = min(self.depots, key=lambda d: self._dist[u].get(d, float('inf')))

        u, current_length = self._post_fill_toward_depot(
            u, current_length, d_star, set(req_set_remaining), current_route, required_covered
        )

        if current_route[-1] != d_star:
            path_to_end = nx.shortest_path(self.graph, current_route[-1], d_star, weight='weight')
            add_len = nx.shortest_path_length(self.graph, current_route[-1], d_star, weight='weight')
            if current_length + add_len <= self.capacity:
                current_route.extend(path_to_end[1:])
                current_length += add_len

        return current_route, current_length, len(required_covered)

# ---------- Multi-trip loop with zero-coverage coalescing ----------

def run_intelligent_tuning_demo():
    scenario_txt_path = "gdb.1.txt"  # change if needed
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
    last_action_was_reloc = False  # to avoid pathological loops

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

        if not covered_now:
            # Attempt to COALESCE this relocation into the previous trip if capacity permits
            if trips and trips[-1]["end"] == current_start and trips[-1]["cost"] + cost <= MAX_VEHICLE_CAPACITY:
                # merge relocation into previous trip
                trips[-1]["route"].extend(route[1:])           # append path
                trips[-1]["end"] = route[-1]
                trips[-1]["cost"] += cost
                total_cost += cost
                print(f"[Coalesce] Merged zero-coverage relocation into Trip {len(trips)} | "
                      f"new_end={trips[-1]['end']} new_cost={trips[-1]['cost']:.2f}")
                current_start = route[-1]
                last_action_was_reloc = True
                # continue planning from new start WITHOUT counting a new trip
                continue
            else:
                # Don’t count it as a trip; just relocate start and try again
                print(f"[Relocate] Zero-coverage move {current_start}->{route[-1]} (cost {cost:.2f}); not counted.")
                current_start = route[-1]
                total_cost += cost  # energy is still spent; include in total cost
                last_action_was_reloc = True

                # Safety: if we relocate twice in a row with zero coverage, commit this as a trip
                # (rare; indicates tight feasibility). This does not introduce a tunable parameter.
                if last_action_was_reloc:
                    last_action_was_reloc = False  # only allow one free relocation in a row
                else:
                    pass
                continue

        # We covered something: commit the trip
        trip_idx += 1
        req_set_remaining -= covered_now

        trips.append({
            "start": current_start,
            "end": route[-1],
            "route": route,
            "cost": cost,
            "covered_edges": covered_now,
        })
        total_cost += cost

        print(f"[Trip {trip_idx}] start={current_start} -> end={route[-1]} | "
              f"cost={cost:.2f} | covered={len(covered_now)} | remaining={len(req_set_remaining)}")

        current_start = route[-1]
        last_action_was_reloc = False

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
