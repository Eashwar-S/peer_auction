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
    # Simple 6-node graph
    G.add_weighted_edges_from([
        (0, 1, 3), (0, 2, 4), (0, 3, 5),
        (1, 2, 2), (1, 4, 4), 
        (2, 3, 3), (2, 4, 2),
        (3, 4, 3), (3, 5, 4),
        (4, 5, 2)
    ])
    return G

# Simple test instance
SIMPLE_GRAPH = create_simple_graph()
START_DEPOT = 0
END_DEPOT = 5
MAX_VEHICLE_CAPACITY = 305
REQUIRED_EDGES = [(1, 2), (3, 5)]  # Only 3 required edges for clarity
FAILED_EDGES = [(2, 4), (1, 4)]  # Only 2 failed edges to handle
class MagneticFieldRouter:
    """
    Magnetic Field Vehicle Routing Algorithm with fixed alpha and gamma
    """
    
    def __init__(self, graph, start_depot, end_depot, capacity, alpha=1.0, gamma=1.0):
        self.graph = graph
        self.start_depot = start_depot
        self.end_depot = end_depot
        self.capacity = capacity
        self.alpha = alpha  # Fixed at 1.0
        self.gamma = gamma  # Fixed at 1.0
        self.pos = self._create_layout()
        self.max_edge_weight = max(d['weight'] for u, v, d in graph.edges(data=True))
        
    def _create_layout(self):
        """Create consistent layout for visualizations"""
        return nx.spring_layout(self.graph, seed=42, k=2, iterations=50)
    
    def calculate_distances(self):
        """Calculate all shortest path distances"""
        return dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='weight'))
    
    def calculate_required_edge_influence(self, required_edges):
        """Calculate influence of required edges on all other edges"""
        distances = self.calculate_distances()
        influences = {}
        
        for edge in self.graph.edges():
            influences[edge] = {}
            influences[edge[::-1]] = {}  # Both directions
            
            for i, req_edge in enumerate(required_edges):
                u_req, v_req = req_edge
                u_edge, v_edge = edge
                
                # Distance from edge endpoints to required edge endpoints
                d1 = min(distances[u_edge].get(u_req, float('inf')), 
                        distances[u_edge].get(v_req, float('inf')))
                d2 = min(distances[v_edge].get(u_req, float('inf')), 
                        distances[v_edge].get(v_req, float('inf')))
                
                # Calculate influence using exponential decay
                if d1 != float('inf') and d2 != float('inf'):
                    influence = 0.5 * (exp(-self.alpha * d1) + exp(-self.alpha * d2))
                else:
                    influence = 0.0
                
                influences[edge][f'req_{i}'] = influence
                influences[edge[::-1]][f'req_{i}'] = influence
        
        return influences
    
    def calculate_depot_influence(self):
        """Calculate influence of depots on all edges"""
        distances = self.calculate_distances()
        influences = {}
        
        for edge in self.graph.edges():
            u, v = edge
            
            # Distance to nearest depot
            d_u = min(distances[u].get(self.start_depot, float('inf')),
                     distances[u].get(self.end_depot, float('inf')))
            d_v = min(distances[v].get(self.start_depot, float('inf')),
                     distances[v].get(self.end_depot, float('inf')))
            
            if d_u != float('inf') and d_v != float('inf'):
                # Using gamma parameter as in the original implementation
                influence = 0.5 * (exp(-self.gamma * d_u / self.capacity) + 
                                 exp(-self.gamma * d_v / self.capacity))
            else:
                influence = 0.1
            
            influences[edge] = influence
            influences[edge[::-1]] = influence
        
        return influences
    
    def calculate_edge_score(self, edge, required_edges, current_length, is_new_required=False):
        """Calculate the magnetic field score for an edge"""
        req_influences = self.calculate_required_edge_influence(required_edges)
        depot_influences = self.calculate_depot_influence()
        
        # Get maximum required edge influence for this edge (P in the formula)
        P = max(req_influences[edge].values()) if req_influences[edge] else 0.0
        
        # Get depot influence (D in the formula)
        D = depot_influences[edge]
        
        # Calculate w - normalized current trip length
        w = current_length / self.capacity if self.capacity > 0 else 0
        
        # Calculate base score S = (1-w)*P + w*D
        S = (1 - w) * P + w * D
        score = S
            
        return {
            'P': P,
            'D': D,
            'w': w,
            'S': S,
            'final_score': score,
            'edge_weight': self.graph[edge[0]][edge[1]]['weight'],
            'normalized_weight': self.graph[edge[0]][edge[1]]['weight'] / self.max_edge_weight
        }
    
    # def find_route_with_magnetic_scoring(self, required_edges, verbose=False):
    #     """Find route using magnetic field scoring, ensuring we can always reach end depot"""
    #     # Start building route from start depot
    #     current_route = [self.start_depot]
    #     current_length = 0
    #     visited_edges = set()
    #     required_covered = set()
        
    #     while True:
    #         current_node = current_route[-1]
    #         candidates = []
            
    #         # If we're at the end depot and have covered some required edges, we can stop
    #         # if current_node == self.end_depot and len(required_covered) > 0:
    #         #     break
                
    #         # If we're at the end depot but haven't covered any required edges yet, continue exploring
    #         # if current_node == self.end_depot and len(required_covered) == 0:
    #         #     # Only continue if we have capacity and there are unvisited edges
    #         #     pass
            
    #         # Get all possible next edges
    #         for neighbor in self.graph.neighbors(current_node):
    #             edge = (current_node, neighbor)
    #             edge_sorted = tuple(sorted(edge))
                
    #             # Skip if already visited this edge
    #             if edge_sorted in visited_edges:
    #                 continue
                
    #             edge_weight = self.graph[current_node][neighbor]['weight']
                
    #             # Critical check: Can we reach end depot after taking this edge?
    #             try:
    #                 # Calculate shortest path from neighbor to end depot
    #                 if neighbor == self.end_depot:
    #                     path_to_end_length = 0
    #                 else:
    #                     path_to_end_length = nx.shortest_path_length(
    #                         self.graph, neighbor, self.end_depot, weight='weight'
    #                     )
                    
    #                 # Check if we can take this edge AND still reach end depot within capacity
    #                 total_required_capacity = current_length + edge_weight + path_to_end_length
                    
    #                 if total_required_capacity > self.capacity:
    #                     continue  # Skip this edge as it would prevent reaching end depot
                        
    #             except nx.NetworkXNoPath:
    #                 # If no path exists from neighbor to end depot, skip this edge
    #                 continue
                
    #             # Check if this is a new required edge
    #             is_new_required = (edge_sorted in [tuple(sorted(req)) for req in required_edges] 
    #                             and edge_sorted not in required_covered)
                
    #             # Calculate magnetic field score
    #             score_data = self.calculate_edge_score(edge, required_edges, current_length, is_new_required)
                
    #             candidates.append({
    #                 'edge': edge,
    #                 'neighbor': neighbor,
    #                 'is_new_required': is_new_required,
    #                 'score_data': score_data,
    #                 'path_to_end_length': path_to_end_length
    #             })
            
    #         # If no valid candidates, we must go directly to end depot
    #         if not candidates:
    #             if current_node != self.end_depot:
    #                 try:
    #                     path_to_end = nx.shortest_path(self.graph, current_node, self.end_depot, weight='weight')
    #                     additional_length = nx.shortest_path_length(self.graph, current_node, self.end_depot, weight='weight')
                        
    #                     if current_length + additional_length <= self.capacity:
    #                         current_route.extend(path_to_end[1:])
    #                         current_length += additional_length
    #                     else:
    #                         # This shouldn't happen with our improved logic, but safety check
    #                         if verbose:
    #                             print("ERROR: Cannot reach end depot within capacity!")
    #                         return None, float('inf'), len(required_covered)
    #                 except nx.NetworkXNoPath:
    #                     if verbose:
    #                         print("ERROR: No path to end depot!")
    #                     return None, float('inf'), len(required_covered)
    #             break
            
    #         # Sort candidates by score (higher is better), then by normalized edge weight (lower is better)
    #         candidates.sort(key=lambda x: (x['score_data']['final_score'], 
    #                                     -x['score_data']['normalized_weight']), reverse=True)
            
    #         # Select best candidate
    #         best = candidates[0]
    #         selected_edge = best['edge']
    #         selected_neighbor = best['neighbor']
            
    #         # Update route
    #         current_route.append(selected_neighbor)
    #         current_length += best['score_data']['edge_weight']
    #         visited_edges.add(tuple(sorted(selected_edge)))
            
    #         if best['is_new_required']:
    #             required_covered.add(tuple(sorted(selected_edge)))
    #             if verbose:
    #                 print(f"Covered required edge: {selected_edge}")
            
    #         if verbose:
    #             print(f"Step: {selected_edge} -> Node {selected_neighbor}, "
    #                 f"Length: {current_length:.2f}, Required: {len(required_covered)}/{len(required_edges)}")
        
    #     # Ensure we end at the end depot (should already be there with improved logic)
    #     if current_route[-1] != self.end_depot:
    #         try:
    #             path_to_end = nx.shortest_path(self.graph, current_route[-1], self.end_depot, weight='weight')
    #             additional_length = nx.shortest_path_length(self.graph, current_route[-1], self.end_depot, weight='weight')
    #             current_route.extend(path_to_end[1:])
    #             current_length += additional_length
    #         except nx.NetworkXNoPath:
    #             if verbose:
    #                 print("Final ERROR: No path to end depot!")
    #             return None, float('inf'), len(required_covered)
        
    #     if verbose:
    #         print(f"Final route: {current_route}, Length: {current_length:.2f}, "
    #             f"Required covered: {len(required_covered)}/{len(required_edges)}")
        
    #     return current_route, current_length, len(required_covered)

    # v2
    # def find_route_with_magnetic_scoring(self, required_edges, verbose=False):
    #     """Find route using magnetic field scoring, allowing multiple passes through end depot"""
    #     # Start building route from start depot
    #     current_route = [self.start_depot]
    #     current_length = 0
    #     visited_edges = set()
    #     required_covered = set()
        
    #     # Track if we've made progress in recent iterations to avoid infinite loops
    #     iterations_without_progress = 0
    #     max_iterations_without_progress = len(self.graph.edges()) * 2  # Safety limit
        
    #     while True:
    #         current_node = current_route[-1]
    #         candidates = []
    #         made_progress_this_iteration = False
            
    #         # Check if we should terminate:
    #         # 1. All required edges covered AND at end depot
    #         # 2. No progress for too long AND at end depot (safety)
    #         if current_node == self.end_depot:
    #             if len(required_covered) == len(required_edges):
    #                 if verbose:
    #                     print("SUCCESS: All required edges covered and at end depot!")
    #                 break
    #             elif iterations_without_progress >= max_iterations_without_progress:
    #                 if verbose:
    #                     print(f"TERMINATION: No progress for {max_iterations_without_progress} iterations")
    #                 break
            
    #         # Get all possible next edges
    #         for neighbor in self.graph.neighbors(current_node):
    #             edge = (current_node, neighbor)
    #             edge_sorted = tuple(sorted(edge))
                
    #             # Skip if already visited this edge
    #             if edge_sorted in visited_edges:
    #                 continue
                
    #             edge_weight = self.graph[current_node][neighbor]['weight']
                
    #             # Critical check: Can we reach end depot after taking this edge?
    #             try:
    #                 # Calculate shortest path from neighbor to end depot
    #                 if neighbor == self.end_depot:
    #                     path_to_end_length = 0
    #                 else:
    #                     path_to_end_length = nx.shortest_path_length(
    #                         self.graph, neighbor, self.end_depot, weight='weight'
    #                     )
                    
    #                 # Check if we can take this edge AND still reach end depot within capacity
    #                 total_required_capacity = current_length + edge_weight + path_to_end_length
                    
    #                 if total_required_capacity > self.capacity:
    #                     if verbose:
    #                         print(f"Skipping edge {edge} - exceeds capacity to end depot")
    #                     continue  # Skip this edge as it would prevent reaching end depot
                        
    #             except nx.NetworkXNoPath:
    #                 # If no path exists from neighbor to end depot, skip this edge
    #                 if verbose:
    #                     print(f"Skipping edge {edge} - no path to end depot from {neighbor}")

    #                 continue
                
    #             # Check if this is a new required edge
    #             is_new_required = (edge_sorted in [tuple(sorted(req)) for req in required_edges] 
    #                             and edge_sorted not in required_covered)
                
    #             # Calculate magnetic field score
    #             score_data = self.calculate_edge_score(edge, required_edges, current_length, is_new_required)
                
    #             candidates.append({
    #                 'edge': edge,
    #                 'neighbor': neighbor,
    #                 'is_new_required': is_new_required,
    #                 'score_data': score_data,
    #                 'path_to_end_length': path_to_end_length
    #             })
    #             print(f"Candidate: {edge} -> Neighbor {neighbor}, Score: {score_data['final_score']}")
    #         # If no valid candidates, we must go directly to end depot
    #         if not candidates:
    #             if current_node != self.end_depot:
    #                 try:
    #                     path_to_end = nx.shortest_path(self.graph, current_node, self.end_depot, weight='weight')
    #                     additional_length = nx.shortest_path_length(self.graph, current_node, self.end_depot, weight='weight')
                        
    #                     if current_length + additional_length <= self.capacity:
    #                         current_route.extend(path_to_end[1:])
    #                         current_length += additional_length
    #                         if verbose:
    #                             print(f"No candidates - going directly to end depot via {path_to_end[1:]}")
    #                     else:
    #                         # This shouldn't happen with our improved logic, but safety check
    #                         if verbose:
    #                             print("ERROR: Cannot reach end depot within capacity!")
    #                         return None, float('inf'), len(required_covered)
    #                 except nx.NetworkXNoPath:
    #                     if verbose:
    #                         print("ERROR: No path to end depot!")
    #                     return None, float('inf'), len(required_covered)
                
    #             # We're at end depot with no candidates - check if we should continue or stop
    #             if len(required_covered) == len(required_edges):
    #                 if verbose:
    #                     print("All required edges covered - terminating at end depot")
    #                 break
    #             else:
    #                 if verbose:
    #                     print(f"At end depot but only {len(required_covered)}/{len(required_edges)} required edges covered")
    #                 break
            
    #         # Prioritize required edges that haven't been covered yet
    #         # Sort candidates: 1) New required edges first, 2) Higher magnetic score, 3) Lower edge weight
    #         candidates.sort(key=lambda x: (
    #             not x['is_new_required'],  # False (new required) comes before True (not new required)
    #             -x['score_data']['final_score'], 
    #             x['score_data']['normalized_weight']
    #         ))
            
    #         # Select best candidate
    #         best = candidates[0]
    #         selected_edge = best['edge']
    #         selected_neighbor = best['neighbor']
            
    #         # Update route
    #         current_route.append(selected_neighbor)
    #         current_length += best['score_data']['edge_weight']
    #         visited_edges.add(tuple(sorted(selected_edge)))
            
    #         if best['is_new_required']:
    #             required_covered.add(tuple(sorted(selected_edge)))
    #             made_progress_this_iteration = True
    #             iterations_without_progress = 0  # Reset counter
    #             if verbose:
    #                 print(f"✓ Covered required edge: {selected_edge} ({len(required_covered)}/{len(required_edges)})")
    #         else:
    #             iterations_without_progress += 1
            
    #         if verbose:
    #             depot_indicator = " [END DEPOT]" if selected_neighbor == self.end_depot else ""
    #             print(f"Step: {selected_edge} -> Node {selected_neighbor}{depot_indicator}, "
    #                 f"Length: {current_length:.2f}, Required: {len(required_covered)}/{len(required_edges)}")
        
    #     # Ensure we end at the end depot (should already be there with improved logic)
    #     if current_route[-1] != self.end_depot:
    #         try:
    #             path_to_end = nx.shortest_path(self.graph, current_route[-1], self.end_depot, weight='weight')
    #             additional_length = nx.shortest_path_length(self.graph, current_route[-1], self.end_depot, weight='weight')
                
    #             if current_length + additional_length <= self.capacity:
    #                 current_route.extend(path_to_end[1:])
    #                 current_length += additional_length
    #                 if verbose:
    #                     print(f"Final path to end depot: {path_to_end[1:]}")
    #             else:
    #                 if verbose:
    #                     print("ERROR: Cannot reach end depot in final step!")
    #                 return None, float('inf'), len(required_covered)
    #         except nx.NetworkXNoPath:
    #             if verbose:
    #                 print("Final ERROR: No path to end depot!")
    #             return None, float('inf'), len(required_covered)
        
    #     if verbose:
    #         print(f"Final route: {current_route}, Length: {current_length:.2f}, "
    #             f"Required covered: {len(required_covered)}/{len(required_edges)}")
    #         if len(required_covered) == len(required_edges):
    #             print("🎉 SUCCESS: All required edges covered!")
    #         else:
    #             print(f"⚠️  PARTIAL: Only {len(required_covered)}/{len(required_edges)} required edges covered")
        
    #     return current_route, current_length, len(required_covered)

    # v3
    # def find_route_with_magnetic_scoring(self, required_edges, verbose=False):
    #     """Find route using magnetic field scoring, allowing edge revisits when necessary"""
    #     # Start building route from start depot
    #     current_route = [self.start_depot]
    #     current_length = 0
    #     required_covered = set()
        
    #     # Track iterations to prevent infinite loops
    #     max_iterations = len(self.graph.edges()) * 3  # Allow multiple passes
    #     iteration_count = 0
        
    #     while len(required_covered) < len(required_edges) and iteration_count < max_iterations:
    #         current_node = current_route[-1]
    #         candidates = []
    #         iteration_count += 1
            
    #         # Get all possible next edges (no visited_edges restriction)
    #         for neighbor in self.graph.neighbors(current_node):
    #             edge = (current_node, neighbor)
    #             edge_sorted = tuple(sorted(edge))
                
    #             edge_weight = self.graph[current_node][neighbor]['weight']
                
    #             # Critical check: Can we reach end depot after taking this edge?
    #             try:
    #                 # Calculate shortest path from neighbor to end depot
    #                 if neighbor == self.end_depot:
    #                     path_to_end_length = 0
    #                 else:
    #                     path_to_end_length = nx.shortest_path_length(
    #                         self.graph, neighbor, self.end_depot, weight='weight'
    #                     )
                    
    #                 # Check if we can take this edge AND still reach end depot within capacity
    #                 total_required_capacity = current_length + edge_weight + path_to_end_length
                    
    #                 if total_required_capacity > self.capacity:
    #                     continue  # Skip this edge as it would prevent reaching end depot
                        
    #             except nx.NetworkXNoPath:
    #                 # If no path exists from neighbor to end depot, skip this edge
    #                 continue
                
    #             # Check if this is a new required edge
    #             is_new_required = (edge_sorted in [tuple(sorted(req)) for req in required_edges] 
    #                             and edge_sorted not in required_covered)
                
    #             # Calculate magnetic field score
    #             score_data = self.calculate_edge_score(edge, required_edges, current_length, is_new_required)

    #             if is_new_required:
    #                 final_score = 1000 + score_data['final_score']  # Huge bonus
    #             else:
    #                 final_score = score_data['final_score']
                
    #             candidates.append({
    #                 'edge': edge,
    #                 'neighbor': neighbor,
    #                 'is_new_required': is_new_required,
    #                 'score_data': score_data,
    #                 'path_to_end_length': path_to_end_length
    #             })
            
    #         # If no valid candidates, we must go directly to end depot
    #         if not candidates:
    #             if current_node != self.end_depot:
    #                 try:
    #                     path_to_end = nx.shortest_path(self.graph, current_node, self.end_depot, weight='weight')
    #                     additional_length = nx.shortest_path_length(self.graph, current_node, self.end_depot, weight='weight')
                        
    #                     if current_length + additional_length <= self.capacity:
    #                         current_route.extend(path_to_end[1:])
    #                         current_length += additional_length
    #                     else:
    #                         if verbose:
    #                             print("ERROR: Cannot reach end depot within capacity!")
    #                         return None, float('inf'), len(required_covered)
    #                 except nx.NetworkXNoPath:
    #                     if verbose:
    #                         print("ERROR: No path to end depot!")
    #                     return None, float('inf'), len(required_covered)
    #             break
            
    #         # Sort candidates: prioritize new required edges, then by magnetic score
    #         candidates.sort(key=lambda x: (
    #             not x['is_new_required'],  # New required edges first
    #             -x['score_data']['final_score'], 
    #             x['score_data']['normalized_weight']
    #         ))
            
    #         # Select best candidate
    #         best = candidates[0]
    #         selected_edge = best['edge']
    #         selected_neighbor = best['neighbor']
            
    #         # Update route
    #         current_route.append(selected_neighbor)
    #         current_length += best['score_data']['edge_weight']
            
    #         if best['is_new_required']:
    #             required_covered.add(tuple(sorted(selected_edge)))
    #             if verbose:
    #                 print(f"Covered required edge: {selected_edge}")
            
    #         if verbose:
    #             print(f"Step: {selected_edge} -> Node {selected_neighbor}, "
    #                 f"Length: {current_length:.2f}, Required: {len(required_covered)}/{len(required_edges)}")
        
    #     # Ensure we end at the end depot
    #     if current_route[-1] != self.end_depot:
    #         try:
    #             path_to_end = nx.shortest_path(self.graph, current_route[-1], self.end_depot, weight='weight')
    #             additional_length = nx.shortest_path_length(self.graph, current_route[-1], self.end_depot, weight='weight')
    #             current_route.extend(path_to_end[1:])
    #             current_length += additional_length
    #         except nx.NetworkXNoPath:
    #             if verbose:
    #                 print("Final ERROR: No path to end depot!")
    #             return None, float('inf'), len(required_covered)
        
    #     if verbose:
    #         print(f"Final route: {current_route}, Length: {current_length:.2f}, "
    #             f"Required covered: {len(required_covered)}/{len(required_edges)}")
        
    #     return current_route, current_length, len(required_covered)

    def find_route_with_magnetic_scoring(self, required_edges, verbose=False):
        """Find route with aggressive required edge seeking"""
        current_route = [self.start_depot]
        current_length = 0
        required_covered = set()
        
        max_iterations = len(self.graph.edges()) * 10  # Much more exploration
        iteration_count = 0
        
        while len(required_covered) < len(required_edges) and iteration_count < max_iterations:
            current_node = current_route[-1]
            candidates = []
            iteration_count += 1
            
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
                
                score_data = self.calculate_edge_score(edge, required_edges, current_length, is_new_required)
                
                candidates.append({
                    'edge': edge,
                    'neighbor': neighbor,
                    'is_new_required': is_new_required,
                    'score_data': score_data
                })
            
            # If no candidates found, FORCE movement toward uncovered required edges
            if not candidates:
                if verbose:
                    print(f"No direct candidates - seeking uncovered required edges...")
                
                # Find uncovered required edges
                uncovered_required = []
                for req_edge in required_edges:
                    if tuple(sorted(req_edge)) not in required_covered:
                        uncovered_required.append(req_edge)
                
                if uncovered_required and current_node != self.end_depot:
                    # Find shortest path to any uncovered required edge
                    best_path = None
                    best_length = float('inf')
                    target_edge = None
                    
                    for req_edge in uncovered_required:
                        u, v = req_edge
                        # Try to reach either endpoint of the required edge
                        for target_node in [u, v]:
                            try:
                                path = nx.shortest_path(self.graph, current_node, target_node, weight='weight')
                                path_length = nx.shortest_path_length(self.graph, current_node, target_node, weight='weight')
                                
                                # Check if we can reach this target and still get to end depot
                                try:
                                    final_path_length = nx.shortest_path_length(self.graph, target_node, self.end_depot, weight='weight')
                                    total_length = current_length + path_length + final_path_length
                                    
                                    if total_length <= self.capacity and path_length < best_length:
                                        best_path = path[1:]  # Exclude current node
                                        best_length = path_length
                                        target_edge = req_edge
                                except nx.NetworkXNoPath:
                                    continue
                            except nx.NetworkXNoPath:
                                continue
                    
                    if best_path:
                        if verbose:
                            print(f"Forcing path {best_path} to reach required edge {target_edge}")
                        # Add the forced path
                        for next_node in best_path:
                            edge_weight = self.graph[current_route[-1]][next_node]['weight']
                            current_route.append(next_node)
                            current_length += edge_weight
                            if verbose:
                                print(f"Forced step: ({current_route[-2]}, {next_node}) -> Node {next_node}, Length: {current_length:.2f}")
                        continue
                
                # If we still can't find a path, go to end depot
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
            
            # Sort candidates: required edges get absolute priority
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

class IntelligentCapacityTuner:
    """
    Intelligent capacity tuner that explores vehicle capacity values
    to find minimum capacity that covers all required edges
    """
    
    def __init__(self, graph, start_depot, end_depot, max_capacity, required_edges):
        self.graph = graph
        self.start_depot = start_depot
        self.end_depot = end_depot
        self.max_capacity = max_capacity
        self.required_edges = required_edges
        self.results = []
        self.best_capacity = None
        self.best_score = float('inf')
        
    def evaluate_capacity(self, capacity):
        """Evaluate a specific capacity value"""
        router = MagneticFieldRouter(self.graph, self.start_depot, self.end_depot, 
                                   capacity, alpha=40.0, gamma=1.0)
        
        route, cost, required_covered = router.find_route_with_magnetic_scoring(self.required_edges)
        
        # Calculate fitness score
        if route is None:
            fitness = float('inf')
        else:
            if required_covered == len(self.required_edges):
                fitness = cost  # Minimize cost for complete coverage
            else:
                missing_edges = len(self.required_edges) - required_covered
                fitness = cost + 1000 * (missing_edges ** 2) + 500
        
        return {
            'capacity': capacity,
            'route': route,
            'cost': cost,
            'required_covered': required_covered,
            'fitness': fitness,
            'feasible': route is not None and required_covered == len(self.required_edges)
        }
    
    def random_search(self, n_iterations=500):
        """Random capacity search"""
        print(f"Starting random search with {n_iterations} iterations...")
        
        for i in range(n_iterations):
            # Generate random capacity between 1 and max_capacity
            capacity = random.uniform(1, self.max_capacity)
            
            result = self.evaluate_capacity(capacity)
            self.results.append(result)
            
            # Update best if this is better
            if result['fitness'] < self.best_score:
                self.best_score = result['fitness']
                self.best_capacity = capacity
            
            if (i + 1) % 50 == 0:
                print(f"Iteration {i+1}: Best fitness = {self.best_score:.2f}")
        
        return self.results
    
    def adaptive_search(self, n_iterations=500):
        """Adaptive capacity search that learns from good solutions"""
        print(f"Starting adaptive search with {n_iterations} iterations...")
        
        # Phase 1: Random exploration (first 25% of iterations)
        exploration_phase = n_iterations // 4
        for i in range(exploration_phase):
            capacity = random.uniform(1, self.max_capacity)
            
            result = self.evaluate_capacity(capacity)
            self.results.append(result)
            
            if result['fitness'] < self.best_score:
                self.best_score = result['fitness']
                self.best_capacity = capacity
        
        # Phase 2: Guided search based on good solutions
        # Find top 20% of feasible solutions
        feasible_results = [r for r in self.results if r['feasible']]
        if feasible_results:
            feasible_results.sort(key=lambda x: x['fitness'])
            top_results = feasible_results[:max(1, len(feasible_results) // 5)]
            
            # Calculate mean and std of good capacities
            capacity_values = [r['capacity'] for r in top_results]
            
            capacity_mean, capacity_std = np.mean(capacity_values), np.std(capacity_values)
            
            print(f"Good capacity region: {capacity_mean:.3f}±{capacity_std:.3f}")
            
            # Phase 2: Sample around good regions
            for i in range(exploration_phase, n_iterations):
                # Sample from normal distribution around good capacities
                capacity = np.clip(np.random.normal(capacity_mean, max(0.5, capacity_std)), 1, self.max_capacity)
                
                result = self.evaluate_capacity(capacity)
                self.results.append(result)
                
                if result['fitness'] < self.best_score:
                    self.best_score = result['fitness']
                    self.best_capacity = capacity
                
                if (i + 1) % 50 == 0:
                    print(f"Iteration {i+1}: Best fitness = {self.best_score:.2f}")
        
        return self.results
    
    def grid_search(self, n_points=50):
        """Grid search over capacity space"""
        print(f"Starting grid search with {n_points} capacity points...")
        
        capacity_values = np.linspace(1, self.max_capacity, n_points)
        
        for i, capacity in enumerate(capacity_values):
            result = self.evaluate_capacity(capacity)
            self.results.append(result)
            
            if result['fitness'] < self.best_score:
                self.best_score = result['fitness']
                self.best_capacity = capacity
            
            if (i + 1) % 10 == 0:
                print(f"Evaluation {i+1}/{n_points}: Best fitness = {self.best_score:.2f}")
        
        return self.results
    
    def binary_search(self):
        """Binary search to find minimum capacity for feasible solution"""
        print(f"Starting binary search for minimum feasible capacity...")
        
        low, high = 1, self.max_capacity
        best_feasible_capacity = None
        
        while high - low > 0.1:  # Precision threshold
            mid = (low + high) / 2
            
            result = self.evaluate_capacity(mid)
            self.results.append(result)
            
            if result['feasible']:
                best_feasible_capacity = mid
                high = mid  # Try smaller capacity
                print(f"Feasible at capacity {mid:.2f}, trying smaller...")
            else:
                low = mid + 0.1  # Try larger capacity
                print(f"Infeasible at capacity {mid:.2f}, trying larger...")
            
            if result['fitness'] < self.best_score:
                self.best_score = result['fitness']
                self.best_capacity = mid
        
        if best_feasible_capacity:
            print(f"Minimum feasible capacity found: {best_feasible_capacity:.2f}")
        else:
            print("No feasible capacity found in the given range")
        
        return self.results
    
    def bayesian_optimization(self, n_iterations=100):
        """Simple Bayesian optimization for capacity using scipy"""
        print(f"Starting Bayesian optimization with {n_iterations} iterations...")
        
        def objective(capacity_array):
            capacity = np.clip(capacity_array[0], 1, self.max_capacity)
            
            result = self.evaluate_capacity(capacity)
            self.results.append(result)
            
            if result['fitness'] < self.best_score:
                self.best_score = result['fitness']
                self.best_capacity = capacity
            
            return result['fitness']
        
        # Multiple random starts
        best_result = None
        for start in range(5):
            initial_guess = [random.uniform(1, self.max_capacity)]
            
            try:
                result = minimize(objective, initial_guess, 
                                bounds=[(1, self.max_capacity)],
                                method='L-BFGS-B',
                                options={'maxiter': n_iterations // 5})
                
                if best_result is None or result.fun < best_result.fun:
                    best_result = result
            except:
                continue
        
        return self.results
    
    def analyze_results(self):
        """Analyze the tuning results"""
        if not self.results:
            print("No results to analyze")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*60)
        print("CAPACITY TUNING ANALYSIS")
        print("="*60)
        
        # Overall statistics
        feasible_count = sum(1 for r in self.results if r['feasible'])
        print(f"Total evaluations: {len(self.results)}")
        print(f"Feasible solutions: {feasible_count} ({100*feasible_count/len(self.results):.1f}%)")
        
        if feasible_count > 0:
            feasible_df = df[df['feasible']]
            print(f"Best fitness: {self.best_score:.2f}")
            print(f"Best capacity: {self.best_capacity:.3f}")
            print(f"Average feasible cost: {feasible_df['cost'].mean():.2f}")
            print(f"Cost std: {feasible_df['cost'].std():.2f}")
            print(f"Minimum feasible capacity: {feasible_df['capacity'].min():.2f}")
            print(f"Maximum feasible capacity: {feasible_df['capacity'].max():.2f}")
        
        # Capacity analysis
        if feasible_count > 5:
            feasible_df = df[df['feasible']]
            top_10_percent = feasible_df.nsmallest(max(1, len(feasible_df) // 10), 'fitness')
            
            print(f"\nTop 10% solutions capacity range:")
            print(f"Capacity: {top_10_percent['capacity'].min():.3f} - {top_10_percent['capacity'].max():.3f}")
        
        return df
    
    def visualize_results(self):
        """Create visualizations of the tuning results"""
        if not self.results:
            print("No results to visualize")
            return
        
        df = pd.DataFrame(self.results)
        
        # Create a comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Capacity vs Fitness scatter plot
        if len(df) > 10:
            feasible_df = df[df['feasible']]
            infeasible_df = df[~df['feasible']]
            
            if len(infeasible_df) > 0:
                axes[0, 0].scatter(infeasible_df['capacity'], infeasible_df['fitness'], 
                                  c='red', alpha=0.5, s=20, label='Infeasible')
            if len(feasible_df) > 0:
                axes[0, 0].scatter(feasible_df['capacity'], feasible_df['fitness'], 
                                  c='green', alpha=0.7, s=20, label='Feasible')
            
            axes[0, 0].set_xlabel('Capacity')
            axes[0, 0].set_ylabel('Fitness')
            axes[0, 0].set_title('Capacity vs Fitness')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Mark best point
            if self.best_capacity:
                axes[0, 0].scatter(self.best_capacity, self.best_score, 
                                  color='blue', s=100, marker='*', label='Best', zorder=5)
                axes[0, 0].legend()
        
        # 2. Feasible vs infeasible capacity distribution
        feasible_df = df[df['feasible']]
        infeasible_df = df[~df['feasible']]
        
        if len(feasible_df) > 0 and len(infeasible_df) > 0:
            axes[0, 1].hist(infeasible_df['capacity'], bins=20, alpha=0.5, color='red', label='Infeasible')
            axes[0, 1].hist(feasible_df['capacity'], bins=20, alpha=0.7, color='green', label='Feasible')
            axes[0, 1].set_xlabel('Capacity')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Capacity Distribution')
            axes[0, 1].legend()
        
        # 3. Cost distribution for feasible solutions
        if len(feasible_df) > 0:
            axes[0, 2].hist(feasible_df['cost'], bins=20, alpha=0.7, color='blue')
            axes[0, 2].axvline(feasible_df['cost'].mean(), color='red', linestyle='--', 
                              label=f'Mean: {feasible_df["cost"].mean():.2f}')
            axes[0, 2].set_xlabel('Route Cost')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Cost Distribution (Feasible Solutions)')
            axes[0, 2].legend()
        
        # 4. Capacity vs Cost for feasible solutions
        if len(feasible_df) > 0:
            axes[1, 0].scatter(feasible_df['capacity'], feasible_df['cost'], alpha=0.6, color='blue')
            axes[1, 0].set_xlabel('Capacity')
            axes[1, 0].set_ylabel('Cost')
            axes[1, 0].set_title('Capacity vs Cost (Feasible Solutions)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add trend line if enough points
            if len(feasible_df) > 5:
                z = np.polyfit(feasible_df['capacity'], feasible_df['cost'], 1)
                p = np.poly1d(z)
                axes[1, 0].plot(feasible_df['capacity'], p(feasible_df['capacity']), "r--", alpha=0.8)
        
        # 5. Required edges coverage vs capacity
        if len(df) > 0:
            axes[1, 1].scatter(df['capacity'], df['required_covered'], alpha=0.6)
            axes[1, 1].set_xlabel('Capacity')
            axes[1, 1].set_ylabel('Required Edges Covered')
            axes[1, 1].set_title('Capacity vs Required Edge Coverage')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Convergence plot
        if len(df) > 10:
            best_so_far = []
            current_best = float('inf')
            for _, row in df.iterrows():
                if row['fitness'] < current_best:
                    current_best = row['fitness']
                best_so_far.append(current_best)
            
            axes[1, 2].plot(best_so_far, linewidth=2)
            axes[1, 2].set_xlabel('Iteration')
            axes[1, 2].set_ylabel('Best Fitness')
            axes[1, 2].set_title('Convergence Plot')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Capacity Tuning Analysis', fontsize=16, y=1.02)
        plt.show()

def run_intelligent_tuning_demo():
    """Run the complete intelligent capacity tuning demonstration"""
    print("Intelligent Capacity Tuning for Magnetic Field Routing")
    print("=" * 65)
    
    # Create tuner
    tuner = IntelligentCapacityTuner(SIMPLE_GRAPH, START_DEPOT, END_DEPOT, 
                                   MAX_VEHICLE_CAPACITY, REQUIRED_EDGES + FAILED_EDGES)
    
    print(f"Graph: {len(SIMPLE_GRAPH.nodes())} nodes, {len(SIMPLE_GRAPH.edges())} edges")
    print(f"Required edges: {REQUIRED_EDGES}")
    print(f"Failed edges: {FAILED_EDGES}")
    print(f"Maximum vehicle capacity: {MAX_VEHICLE_CAPACITY}")
    print(f"Alpha fixed at: 1.0")
    print(f"Gamma fixed at: 1.0")
    
    # Test different search strategies
    strategies = [
        ("Random Search", lambda: tuner.random_search(500)),
        ("Adaptive Search", lambda: tuner.adaptive_search(500)),
        ("Grid Search", lambda: tuner.grid_search(500)),
        ("Binary Search", lambda: tuner.binary_search()),
        ("Bayesian Optimization", lambda: tuner.bayesian_optimization(100))
    ]
    
    # Run one strategy (you can modify this to run multiple)
    strategy_name, strategy_func = strategies[1]  # Adaptive search

    # strategy_name, strategy_func = strategies[2]  # Grid search
    # strategy_name, strategy_func = strategies[3]  # Binary search
    # strategy_name, strategy_func = strategies[4]  # Bayesian optimization
    strategy_name, strategy_func = strategies[0]  # Random search

    print(f"\nRunning {strategy_name}...")
    results = strategy_func()
    
    # Analyze results
    df = tuner.analyze_results()
    
    # Test the best capacity
    if tuner.best_capacity:
        print(f"\nTesting best capacity: {tuner.best_capacity:.3f}")
        
        best_router = MagneticFieldRouter(SIMPLE_GRAPH, START_DEPOT, END_DEPOT, 
                                        tuner.best_capacity, alpha=40.0, gamma=1.0)
        
        route, cost, required_covered = best_router.find_route_with_magnetic_scoring(REQUIRED_EDGES + FAILED_EDGES, verbose=True)
        
        if route:
            print(f"Best route: {route}")
            print(f"Cost: {cost:.2f}")
            print(f"Required edges covered: {required_covered}/{len(REQUIRED_EDGES + FAILED_EDGES)}")
            print(f"Capacity utilization: {cost:.2f}/{tuner.best_capacity:.2f} ({100*cost/tuner.best_capacity:.1f}%)")
        else:
            print("No feasible route found with best capacity")
    
    # Visualize results
    tuner.visualize_results()
    
    return tuner, df

if __name__ == "__main__":
    tuner, results_df = run_intelligent_tuning_demo()