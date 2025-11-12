"""
Intelligent Capacity Tuning for Magnetic Field Vehicle Routing Algorithm
========================================================================
This script varies vehicle capacity to find optimal capacity values that ensure
all required edges are traversed while minimizing trip cost.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from math import exp, sqrt
import seaborn as sns
from itertools import permutations
import random
from collections import defaultdict
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Create a simpler test instance for clearer demonstration
def create_simple_graph():
    """Create a simple graph for magnetic field demonstration"""
    G = nx.Graph()
    G.add_weighted_edges_from([
        (0, 1, 3), (0, 2, 4), (0, 3, 5),
        (1, 2, 2), (1, 4, 4), 
        (2, 3, 3), (2, 4, 2),
        (3, 4, 3), (3, 5, 4),
        (4, 5, 2), (4, 6, 3),
    ])
    return G

# Simple test instance
SIMPLE_GRAPH = create_simple_graph()
START_DEPOT = 0
END_DEPOT = 5
MAX_VEHICLE_CAPACITY = 32
# REQUIRED_EDGES = [(1, 2), (3, 4)]  # Only 3 required edges for clarity
# FAILED_EDGES = [(2, 3), (4, 6), (3,5)]  # Only 2 failed edges to handle
REQUIRED_EDGES = [(1, 2), (3, 5), (4,6)]  # Only 3 required edges for clarity
FAILED_EDGES = [(2, 4), (1, 4)]  # Only 2 failed edges to handle


class MagneticFieldRouter:
    def __init__(self, graph, start_depot, depots, capacity, alpha=1.0, gamma=1.0):
        self.graph = graph
        self.start_depot = start_depot
        self.depots = set(depots)          # <— list/set of allowed end depots
        self.capacity = capacity
        self.alpha = alpha
        self.gamma = gamma
        # self.pos = self._create_layout()
        self.max_edge_weight = max(d['weight'] for u, v, d in graph.edges(data=True))
        # precompute all-pairs shortest-path lengths once
        self._dist = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='weight'))

    def _min_dist_to_any_depot(self, node):
        vals = [self._dist[node].get(d, float('inf')) for d in self.depots]
        return min(vals) if vals else float('inf')

    def calculate_distances(self):
        # keep method for backward compatibility, but use precomputed
        return self._dist

    def calculate_required_edge_influence(self, required_edges_to_cover):
        distances = self._dist
        influences = {}
        for edge in self.graph.edges():
            influences[edge] = {}
            influences[edge[::-1]] = {}
            for i, req_edge in enumerate(required_edges_to_cover):
                u_req, v_req = req_edge
                u_edge, v_edge = edge
                d1 = min(distances[u_edge].get(u_req, float('inf')),
                         distances[u_edge].get(v_req, float('inf')))
                d2 = min(distances[v_edge].get(u_req, float('inf')),
                         distances[v_edge].get(v_req, float('inf')))
                influence = 0.5 * (exp(-self.alpha * d1) + exp(-self.alpha * d2)) \
                            if d1 != float('inf') and d2 != float('inf') else 0.0
                influences[edge][f'req_{i}'] = influence
                influences[edge[::-1]][f'req_{i}'] = influence
        return influences

    def calculate_depot_influence(self):
        # “finish pull” grows as you near capacity, using nearest depot
        influences = {}
        for edge in self.graph.edges():
            u, v = edge
            d_u = self._min_dist_to_any_depot(u)
            d_v = self._min_dist_to_any_depot(v)
            if d_u != float('inf') and d_v != float('inf'):
                # scale by capacity similarly to your original idea
                influence = 0.5 * (exp(-self.gamma * d_u / max(self.capacity, 1e-9)) +
                                   exp(-self.gamma * d_v / max(self.capacity, 1e-9)))
            else:
                influence = 0.1
            influences[edge] = influence
            influences[edge[::-1]] = influence
        return influences

    def calculate_edge_score(self, edge, required_to_cover, current_length, is_new_required=False):
        req_influences = self.calculate_required_edge_influence(required_to_cover)
        depot_influences = self.calculate_depot_influence()
        
        P = max(req_influences[edge].values()) if req_influences[edge] else 0.0
        D = depot_influences[edge]
        w = current_length / self.capacity if self.capacity > 0 else 0
        S = (1 - w) * P + w * D
        
        if is_new_required:
            final_score = 1000 + S  # Large bonus for uncovered required edges
        else:
            final_score = S
        
        print(f'P - {P}, D - {D}, w - {w}, S - {S}, edge - {edge}, final score - {final_score}')
        return {
            'P': P,
            'D': D,
            'w': w,
            'S': S,
            'final_score': final_score,
            'edge_weight': self.graph[edge[0]][edge[1]]['weight'],
            'normalized_weight': self.graph[edge[0]][edge[1]]['weight'] / self.max_edge_weight
        }

    def find_route_with_magnetic_scoring(self, required_edges, verbose=False):
        current_route = [self.start_depot]
        current_length = 0.0
        required_covered = set()
        max_iterations = len(self.graph.edges()) * 10
        iteration_count = 0

        while len(required_covered) < len(required_edges) and iteration_count < max_iterations:
            iteration_count += 1
            current_node = current_route[-1]
            candidates = []

            required_to_cover = [req for req in required_edges
                                 if tuple(sorted(req)) not in required_covered]

            # try all neighbors from current_node
            for neighbor in self.graph.neighbors(current_node):
                edge = (current_node, neighbor)
                edge_sorted = tuple(sorted(edge))
                edge_weight = self.graph[current_node][neighbor]['weight']

                # Feasibility: must still be able to reach SOME depot after taking this edge
                try:
                    path_to_nearest_depot_after = self._min_dist_to_any_depot(neighbor)
                    if current_length + edge_weight + path_to_nearest_depot_after > self.capacity:
                        continue
                except nx.NetworkXNoPath:
                    continue

                is_new_required = (edge_sorted in [tuple(sorted(req)) for req in required_edges]
                                   and edge_sorted not in required_covered)

                score_data = self.calculate_edge_score(
                    edge, required_to_cover, current_length, is_new_required
                )

                candidates.append({
                    'edge': edge,
                    'neighbor': neighbor,
                    'is_new_required': is_new_required,
                    'score_data': score_data
                })

            if not candidates:
                if verbose:
                    print(f"No direct candidates at node {current_node} - seeking uncovered required edge...")

                # Try to “force” a path to the nearest *reachable* required edge while preserving ability to end at a depot
                best_path = None
                best_length = float('inf')
                target_edge = None
                for req_edge in required_to_cover:
                    u, v = req_edge
                    for target in (u, v):
                        try:
                            path = nx.shortest_path(self.graph, current_node, target, weight='weight')
                            path_len = nx.shortest_path_length(self.graph, current_node, target, weight='weight')
                            # must still allow a depot after reaching target
                            depot_after = self._min_dist_to_any_depot(target)
                            total_len = current_length + path_len + depot_after
                            if total_len <= self.capacity and path_len < best_length:
                                best_path = path[1:]  # exclude current_node
                                best_length = path_len
                                target_edge = req_edge
                        except nx.NetworkXNoPath:
                            continue

                if best_path:
                    if verbose:
                        print(f"Forcing path {best_path} toward required edge {target_edge}")
                    for nxt in best_path:
                        current_route.append(nxt)
                        current_length += self.graph[current_route[-2]][nxt]['weight']
                    continue

                # Otherwise, just finish at the nearest depot (guaranteed feasible by earlier checks)
                if current_node not in self.depots:
                    # append path to nearest depot (feasible by construction)
                    nearest = min(self.depots, key=lambda d: self._dist[current_node].get(d, float('inf')))
                    path_to_end = nx.shortest_path(self.graph, current_node, nearest, weight='weight')
                    add_len = nx.shortest_path_length(self.graph, current_node, nearest, weight='weight')
                    if current_length + add_len <= self.capacity:
                        current_route.extend(path_to_end[1:])
                        current_length += add_len
                break

            # Prefer new required edges, then higher final_score
            candidates.sort(key=lambda x: (not x['is_new_required'], -x['score_data']['final_score']))
            best = candidates[0]
            current_route.append(best['neighbor'])
            current_length += best['score_data']['edge_weight']

            if best['is_new_required']:
                required_covered.add(tuple(sorted(best['edge'])))
                if verbose:
                    print(f"✓ Covered required edge: {best['edge']} "
                          f"({len(required_covered)}/{len(required_edges)})")

            if verbose:
                print(f"Step: {best['edge']} -> Node {best['neighbor']}, "
                      f"Length: {current_length:.2f}, Required: {len(required_covered)}/{len(required_edges)}")

        # Ensure we finish at a depot (pick nearest feasible)
        if current_route[-1] not in self.depots:
            current_node = current_route[-1]
            nearest = min(self.depots, key=lambda d: self._dist[current_node].get(d, float('inf')))
            path_to_end = nx.shortest_path(self.graph, current_node, nearest, weight='weight')
            add_len = nx.shortest_path_length(self.graph, current_node, nearest, weight='weight')
            if current_length + add_len <= self.capacity:
                current_route.extend(path_to_end[1:])
                current_length += add_len
            else:
                # If this ever triggers, feasibility filters need tightening above.
                return None, float('inf'), len(required_covered)

        if verbose:
            print(f"Final route: {current_route}, Length: {current_length:.2f}")

        return current_route, current_length, len(required_covered)


def parse_txt_file(file_path):
    """Parse the .txt failure scenario file to extract graph and parameters"""
    import re  # local import so the signature matches your snippet

    G = nx.Graph()
    required_edges = []
    depot_nodes = []
    vehicle_capacity = None
    recharge_time = None
    num_vehicles = None
    failure_vehicles = []
    vehicle_failure_times = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    current_section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("NUMBER OF VERTICES"):
            num_vertices = int(line.split(":")[1].strip())
            G.add_nodes_from(range(1, num_vertices + 1))
        elif line.startswith("VEHICLE CAPACITY"):
            vehicle_capacity = float(line.split(":")[1].strip())
        elif line.startswith("RECHARGE TIME"):
            recharge_time = float(line.split(":")[1].strip())
        elif line.startswith("NUMBER OF VEHICLES"):
            num_vehicles = int(line.split(":")[1].strip())
        elif line.startswith("DEPOT:"):
            depot_line = line.split(":", 1)[1].strip()
            depot_nodes = [int(x.strip()) for x in depot_line.split(",")]
        elif line.startswith("LIST_REQUIRED_EDGES:"):
            current_section = "required_edges"
        elif line.startswith("LIST_NON_REQUIRED_EDGES:"):
            current_section = "non_required_edges"
        elif line.startswith("FAILURE_SCENARIO:"):
            current_section = "failure_scenario"
        elif line.startswith("(") and current_section in ["required_edges", "non_required_edges"]:
            parts = line.split(") ")
            if len(parts) >= 2:
                edge_part = parts[0].strip("(")
                u, v = map(int, edge_part.split(","))
                weight_part = parts[1].strip()
                weight = float(weight_part.split()[-1])
                G.add_edge(u, v, weight=weight)
                if current_section == "required_edges":
                    required_edges.append([u, v])
        elif current_section == "failure_scenario" and line.startswith("Vehicle"):
            match = re.search(r"Vehicle (\d+) will fail in (\d+) time units", line)
            if match:
                vehicle_id = int(match.group(1)) - 1  # 0-index
                failure_time = float(match.group(2))
                failure_vehicles.append(vehicle_id)
                vehicle_failure_times.append(failure_time)

    return (
        G,
        required_edges,
        depot_nodes,
        vehicle_capacity,
        recharge_time,
        num_vehicles,
        failure_vehicles,
        vehicle_failure_times,
    )


# def run_intelligent_tuning_demo():
#     scenario_txt_path = "gdb.1.txt"
#     G, req_edges, depots, cap, recharge, nveh, fail_vs, fail_ts = parse_txt_file(scenario_txt_path)

#     START_DEPOT = depots[0]   # keep your chosen start
#     MAX_VEHICLE_CAPACITY = cap
    
#     best_router = MagneticFieldRouter(
#         graph=G,
#         start_depot=START_DEPOT,
#         depots=depots,              # <— pass ALL depots here
#         capacity=MAX_VEHICLE_CAPACITY,
#         alpha=1.0, gamma=1.0
#     )

#     route, cost, required_covered = best_router.find_route_with_magnetic_scoring(req_edges, verbose=True)
    
#     if route:
#         print(f"Best route: {route}")
#         print(f"Cost: {cost:.2f}")
#         print(f"Required edges covered: {required_covered}/{len(REQUIRED_EDGES + FAILED_EDGES)}")
#         # print(f"Capacity utilization: {cost:.2f}/{tuner.best_capacity:.2f} ({100*cost/tuner.best_capacity:.1f}%)")
#     else:
#         print("No feasible route found with best capacity")
    
def run_intelligent_tuning_demo():
    # scenario_txt_path = "gdb.1.txt"
    scenario_txt_path = "bccm.1.txt"
    G, req_edges, depots, cap, recharge, nveh, fail_vs, fail_ts = parse_txt_file(scenario_txt_path)

    print(f'Number of req edges - {len(req_edges)}')
    # Normalize required edges to sorted tuples once
    req_edges_remaining = [tuple(sorted(e)) for e in req_edges]
    req_set_remaining = set(req_edges_remaining)

    START_DEPOT = depots[0]       # initial start
    MAX_VEHICLE_CAPACITY = cap

    trips = []
    total_cost = 0.0
    trip_idx = 0
    MAX_TRIPS_GUARD = max(2*len(req_edges_remaining), 50)  # generous guard

    current_start = START_DEPOT
    while req_set_remaining and trip_idx < MAX_TRIPS_GUARD:
        trip_idx += 1

        router = MagneticFieldRouter(
            graph=G,
            start_depot=current_start,
            depots=depots,             # allow free end at any depot
            capacity=MAX_VEHICLE_CAPACITY,
            alpha=1.0, gamma=1.0
        )

        # Plan one trip against the *current* remaining required edges
        route, cost, _ncovered_reported = router.find_route_with_magnetic_scoring(
            list(req_set_remaining), verbose=False
        )

        if not route or len(route) < 2:
            print(f"[Trip {trip_idx}] No feasible route found from {current_start}. Stopping.")
            break

        # Compute which remaining required edges were actually covered by this route
        route_edge_seq = [tuple(sorted((route[i], route[i+1]))) for i in range(len(route)-1)]
        covered_now = set(e for e in route_edge_seq if e in req_set_remaining)

        # Progress guard: if we didn't cover anything, stop to avoid infinite loop
        if not covered_now:
            # If we ended at a different depot, still accept the relocation as a zero-coverage hop;
            # but if this repeats, we'll stop next time.
            
            print(f"[Trip {trip_idx}] Planned a feasible trip but covered 0 new required edges. Stopping.")
            print(f'trip - {route}')
            break

        # Remove covered edges from remaining set
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

        # Next trip starts where this one ended (which should be a depot)
        current_start = route[-1]

    # Summary
    print("\n=== Multi-Trip Summary ===")
    print(f"Trips planned: {len(trips)}")
    print(f"Total cost: {total_cost:.2f}")
    print(f"Required edges initially: {len(req_edges)}")
    print(f"Remaining uncovered edges: {len(req_set_remaining)}")

    for i, t in enumerate(trips, 1):
        print(f"Trip {i}: start={t['start']} end={t['end']} "
              f"cost={t['cost']:.2f} trip={t['route']} steps={len(t['route'])-1} covered={len(t['covered_edges'])}")


if __name__ == "__main__":
    run_intelligent_tuning_demo()