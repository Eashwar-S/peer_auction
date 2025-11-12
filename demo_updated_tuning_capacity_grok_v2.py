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
    def __init__(self, graph, start_depot, end_depot, capacity, alpha=1.0, gamma=1.0):
        self.graph = graph
        self.start_depot = start_depot
        self.end_depot = end_depot
        self.capacity = capacity
        self.alpha = alpha
        self.gamma = gamma
        self.pos = self._create_layout()
        self.max_edge_weight = max(d['weight'] for u, v, d in graph.edges(data=True))
        
    def _create_layout(self):
        return nx.spring_layout(self.graph, seed=42, k=2, iterations=50)
    
    def calculate_distances(self):
        return dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='weight'))
    
    def calculate_required_edge_influence(self, required_edges_to_cover):
        distances = self.calculate_distances()
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
                
                if d1 != float('inf') and d2 != float('inf'):
                    influence = 0.5 * (exp(-self.alpha * d1) + exp(-self.alpha * d2))
                else:
                    influence = 0.0
                
                influences[edge][f'req_{i}'] = influence
                influences[edge[::-1]][f'req_{i}'] = influence
        
        return influences
    
    def calculate_depot_influence(self):
        distances = self.calculate_distances()
        influences = {}
        
        for edge in self.graph.edges():
            u, v = edge
            d_u = min(distances[u].get(self.start_depot, float('inf')),
                     distances[u].get(self.end_depot, float('inf')))
            d_v = min(distances[v].get(self.start_depot, float('inf')),
                     distances[v].get(self.end_depot, float('inf')))
            
            if d_u != float('inf') and d_v != float('inf'):
                influence = 0.5 * (exp(-self.gamma * d_u / self.capacity) + 
                                 exp(-self.gamma * d_v / self.capacity))
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
        current_length = 0
        required_covered = set()
        max_iterations = len(self.graph.edges()) * 10
        iteration_count = 0
        
        while len(required_covered) < len(required_edges) and iteration_count < max_iterations:
            current_node = current_route[-1]
            candidates = []
            iteration_count += 1
            
            # Update required edges to cover
            required_to_cover = [req for req in required_edges if tuple(sorted(req)) not in required_covered]
            
            # Get all possible next edges
            for neighbor in self.graph.neighbors(current_node):
                edge = (current_node, neighbor)
                edge_sorted = tuple(sorted(edge))
                edge_weight = self.graph[current_node][neighbor]['weight']
                
                # Capacity check
                try:
                    if neighbor == self.end_depot:
                        path_to_end_length = 0
                    else:
                        path_to_end_length = nx.shortest_path_length(
                            self.graph, neighbor, self.end_depot, weight='weight'
                        )
                    
                    if current_length + edge_weight + path_to_end_length > self.capacity:
                        continue
                except nx.NetworkXNoPath:
                    continue
                
                is_new_required = (edge_sorted in [tuple(sorted(req)) for req in required_edges] 
                                and edge_sorted not in required_covered)
                
                score_data = self.calculate_edge_score(edge, required_to_cover, current_length, is_new_required)
                
                candidates.append({
                    'edge': edge,
                    'neighbor': neighbor,
                    'is_new_required': is_new_required,
                    'score_data': score_data
                })
            
            # Handle no candidates
            if not candidates:
                if verbose:
                    print(f"No direct candidates at node {current_node} - seeking uncovered required edges...")
                
                uncovered_required = required_to_cover
                
                if uncovered_required and current_node != self.end_depot:
                    best_path = None
                    best_length = float('inf')
                    target_edge = None
                    
                    for req_edge in uncovered_required:
                        u, v = req_edge
                        for target_node in [u, v]:
                            try:
                                path = nx.shortest_path(self.graph, current_node, target_node, weight='weight')
                                path_length = nx.shortest_path_length(self.graph, current_node, target_node, weight='weight')
                                
                                try:
                                    final_path_length = nx.shortest_path_length(self.graph, target_node, self.end_depot, weight='weight')
                                    total_length = current_length + path_length + final_path_length
                                    
                                    if total_length <= self.capacity and path_length < best_length:
                                        best_path = path[1:]
                                        best_length = path_length
                                        target_edge = req_edge
                                except nx.NetworkXNoPath:
                                    continue
                            except nx.NetworkXNoPath:
                                continue
                    
                    if best_path:
                        if verbose:
                            print(f"Forcing path {best_path} to reach required edge {target_edge}")
                        for next_node in best_path:
                            edge_weight = self.graph[current_route[-1]][next_node]['weight']
                            current_route.append(next_node)
                            current_length += edge_weight
                            if verbose:
                                print(f"Forced step: ({current_route[-2]}, {next_node}) -> Node {next_node}, Length: {current_length:.2f}")
                        continue
                
                # Go to end depot if no further progress possible
                if current_node != self.end_depot:
                    try:
                        path_to_end = nx.shortest_path(self.graph, current_node, self.end_depot, weight='weight')
                        additional_length = nx.shortest_path_length(self.graph, current_node, self.end_depot, weight='weight')
                        if current_length + additional_length <= self.capacity:
                            current_route.extend(path_to_end[1:])
                            current_length += additional_length
                            if verbose:
                                print(f"Final path to end depot: {path_to_end[1:]}")
                    except nx.NetworkXNoPath:
                        pass
                break
            
            # Sort candidates: prioritize new required edges
            candidates.sort(key=lambda x: (
                not x['is_new_required'],
                -x['score_data']['final_score']
            ))
            
            best = candidates[0]
            current_route.append(best['neighbor'])
            current_length += best['score_data']['edge_weight']
            
            if best['is_new_required']:
                required_covered.add(tuple(sorted(best['edge'])))
                if verbose:
                    print(f"✓ Covered required edge: {best['edge']} ({len(required_covered)}/{len(required_edges)})")
            
            if verbose:
                print(f"Step: {best['edge']} -> Node {best['neighbor']}, "
                    f"Length: {current_length:.2f}, Required: {len(required_covered)}/{len(required_edges)}")
        
        # Ensure end at depot
        if current_route[-1] != self.end_depot:
            try:
                path_to_end = nx.shortest_path(self.graph, current_route[-1], self.end_depot, weight='weight')
                additional_length = nx.shortest_path_length(self.graph, current_route[-1], self.end_depot, weight='weight')
                current_route.extend(path_to_end[1:])
                current_length += additional_length
            except nx.NetworkXNoPath:
                return None, float('inf'), len(required_covered)
        
        if verbose:
            print(f"Final route: {current_route}, Length: {current_length:.2f}")
            print(f"Required covered: {len(required_covered)}/{len(required_edges)}")
        
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


def run_intelligent_tuning_demo():
    
    scenario_txt_path = "gdb.1.txt"
    G, req_edges, depots, cap, recharge, nveh, fail_vs, fail_ts = parse_txt_file(scenario_txt_path)
    
    SIMPLE_GRAPH = G
    REQUIRED_EDGES = req_edges
    START_DEPOT = depots[0]
    END_DEPOT = depots[3]
    MAX_VEHICLE_CAPACITY = cap
    FAILED_EDGES = []
    best_router = MagneticFieldRouter(SIMPLE_GRAPH, START_DEPOT, END_DEPOT, 
                                    MAX_VEHICLE_CAPACITY, alpha=1.0, gamma=1.0)
    route, cost, required_covered = best_router.find_route_with_magnetic_scoring(REQUIRED_EDGES + FAILED_EDGES, verbose=True)
    
    if route:
        print(f"Best route: {route}")
        print(f"Cost: {cost:.2f}")
        print(f"Required edges covered: {required_covered}/{len(REQUIRED_EDGES + FAILED_EDGES)}")
        # print(f"Capacity utilization: {cost:.2f}/{tuner.best_capacity:.2f} ({100*cost/tuner.best_capacity:.1f}%)")
    else:
        print("No feasible route found with best capacity")
    
    # tuner.visualize_results()
    # return tuner

if __name__ == "__main__":
    run_intelligent_tuning_demo()