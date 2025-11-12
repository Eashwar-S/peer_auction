"""
Simple Multi-Vehicle Failure Recovery using Magnetic Field Approach
=================================================================
This demonstrates a focused 4-vehicle failure recovery system with detailed visualizations
showing initial routes, failure points, search radius based on battery capacity, and 
step-by-step recovery process.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from math import exp, sqrt, hypot
import seaborn as sns
from itertools import permutations
import random
import copy
from collections import defaultdict

# Create a simple network for clear demonstration
def create_simple_network():
    """Create a simple network with clear structure"""
    G = nx.Graph()
    
    # Add nodes with positions for clear visualization
    node_positions = {
        0: (0, 4),   # Depot A
        1: (2, 4),
        2: (4, 4),
        3: (6, 4),
        4: (8, 4),   # Depot B
        5: (8, 2),
        6: (6, 2),
        7: (4, 2),   # Depot C (center)
        8: (2, 2),
        9: (0, 2),
        10: (1, 3),
        11: (3, 3),
        12: (5, 3),
        13: (7, 3),
        14: (4, 1),
        15: (4, 5)
    }
    
    # Add nodes
    for node, pos in node_positions.items():
        G.add_node(node, pos=pos)
    
    # Add edges with weights
    edges = [
        (0, 1, 3), (1, 2, 3), (2, 3, 3), (3, 4, 3),  # Top horizontal
        (4, 5, 2), (5, 6, 2), (6, 7, 2), (7, 8, 2), (8, 9, 2),  # Bottom path
        (0, 9, 3), (9, 8, 2), (8, 7, 2), (7, 6, 2), (6, 5, 2), (5, 4, 2),  # Bottom horizontal
        (0, 10, 2), (1, 10, 2), (1, 11, 2), (2, 11, 2), (2, 12, 2), (3, 12, 2), (3, 13, 2), (4, 13, 2),  # Middle connections
        (10, 11, 2), (11, 12, 2), (12, 13, 2),  # Middle horizontal
        (7, 14, 2), (7, 15, 2), (2, 15, 2),  # Vertical connections
        (9, 10, 2), (10, 8, 3), (11, 7, 2), (12, 6, 3), (13, 5, 2),  # Cross connections
    ]
    
    G.add_weighted_edges_from(edges)
    return G, node_positions

# Create simple test instance
NETWORK, NODE_POSITIONS = create_simple_network()
DEPOTS = [0, 4, 7]  # Three depots: A, B, C
VEHICLE_CAPACITY = 30

# 4-vehicle scenario with detailed trips
VEHICLE_TRIPS = {
    0: {
        'route': [0, 1, 2, 15, 7, 14, 7],  # Depot A -> Depot C
        'current_position': 2,  # Vehicle is at node 2
        'required_edges': [(1, 2), (2, 15), (15, 7)],
        'depot': 0,
        'status': 'active'
    },
    1: {
        'route': [4, 13, 12, 11, 10, 0],  # Depot B -> Depot A  
        'current_position': 12,  # Vehicle is at node 12
        'required_edges': [(13, 12), (12, 11), (10, 0)],
        'depot': 4,
        'status': 'active'
    },
    2: {
        'route': [7, 8, 9, 0, 1, 11, 7],  # Depot C -> Depot C (round trip)
        'current_position': 9,  # Vehicle FAILED at node 9
        'required_edges': [(8, 9), (9, 0), (1, 11)],  # These need to be recovered
        'depot': 7,
        'status': 'failed'
    },
    3: {
        'route': [4, 5, 6, 7, 8, 9, 0],  # Depot B -> Depot A
        'current_position': 6,  # Vehicle is at node 6
        'required_edges': [(5, 6), (6, 7), (8, 9)],
        'depot': 4,
        'status': 'active'
    }
}

FAILED_VEHICLE = 2  # Vehicle 2 has failed

class SimpleFailureRecovery:
    """
    Simple Multi-Vehicle Failure Recovery System with Detailed Visualizations
    """
    
    def __init__(self, network, node_positions, vehicle_trips, failed_vehicle, capacity):
        self.network = network
        self.pos = node_positions
        self.vehicle_trips = copy.deepcopy(vehicle_trips)
        self.failed_vehicle = failed_vehicle
        self.capacity = capacity
        self.max_edge_weight = max(d['weight'] for u, v, d in network.edges(data=True))
        
        # Recovery data
        self.failed_edges = []
        self.search_zones = {}
        self.candidate_vehicles = []
        self.insertion_attempts = {}
        self.successful_insertions = []
        self.step_history = []
        
    def calculate_route_cost(self, route):
        """Calculate total cost of a route"""
        if len(route) < 2:
            return 0
        cost = 0
        for i in range(len(route) - 1):
            if self.network.has_edge(route[i], route[i+1]):
                cost += self.network[route[i]][route[i+1]]['weight']
            else:
                try:
                    cost += nx.shortest_path_length(self.network, route[i], route[i+1], weight='weight')
                except nx.NetworkXNoPath:
                    return float('inf')
        return cost
    
    def extract_failed_edges(self):
        """Extract required edges from failed vehicle"""
        failed_trip = self.vehicle_trips[self.failed_vehicle]
        self.failed_edges = failed_trip['required_edges'].copy()
        
        print(f"FAILURE ANALYSIS")
        print(f"================")
        print(f"Failed Vehicle: {self.failed_vehicle}")
        print(f"Failed at position: Node {failed_trip['current_position']}")
        print(f"Original route: {failed_trip['route']}")
        print(f"Required edges to recover: {self.failed_edges}")
        print(f"Total edges to recover: {len(self.failed_edges)}")
        
        return self.failed_edges
    
    def calculate_search_radius_from_edge(self, edge):
        """Calculate search radius from an edge based on vehicle capacity"""
        # Start from edge endpoints and find all reachable nodes within capacity
        reachable_nodes = set()
        reachable_edges = set()
        
        for start_node in edge:
            if start_node not in self.network.nodes():
                continue
                
            # Use Dijkstra to find all nodes reachable within capacity
            try:
                distances = nx.single_source_dijkstra_path_length(
                    self.network, start_node, cutoff=self.capacity, weight='weight'
                )
                reachable_nodes.update(distances.keys())
                
                # Add edges within the reachable area
                for node in distances.keys():
                    for neighbor in self.network.neighbors(node):
                        if neighbor in distances:
                            reachable_edges.add(tuple(sorted([node, neighbor])))
                            
            except Exception as e:
                print(f"Error calculating reachability from {start_node}: {e}")
                continue
        
        return reachable_nodes, reachable_edges
    
    def create_search_zones(self):
        """Create search zones for each failed edge based on battery capacity"""
        print(f"\nSEARCH ZONE ANALYSIS")
        print(f"===================")
        
        for i, edge in enumerate(self.failed_edges):
            reachable_nodes, reachable_edges = self.calculate_search_radius_from_edge(edge)
            
            self.search_zones[i] = {
                'edge': edge,
                'reachable_nodes': reachable_nodes,
                'reachable_edges': reachable_edges,
                'zone_size': len(reachable_nodes)
            }
            
            print(f"Edge {edge}:")
            print(f"  Reachable nodes: {len(reachable_nodes)} (within capacity {self.capacity})")
            print(f"  Reachable edges: {len(reachable_edges)}")
            print(f"  Zone nodes: {sorted(reachable_nodes)}")
    
    def find_candidate_vehicles(self):
        """Find vehicles whose routes intersect with search zones"""
        print(f"\nCANDIDATE VEHICLE ANALYSIS")
        print(f"==========================")
        
        candidates = []
        
        for vehicle_id, trip in self.vehicle_trips.items():
            if vehicle_id == self.failed_vehicle or trip['status'] == 'failed':
                continue
                
            route = trip['route']
            route_nodes = set(route)
            current_cost = self.calculate_route_cost(route)
            spare_capacity = self.capacity - current_cost
            
            # Check intersection with search zones
            intersecting_zones = []
            for zone_id, zone in self.search_zones.items():
                intersection = route_nodes.intersection(zone['reachable_nodes'])
                if intersection:
                    intersecting_zones.append({
                        'zone_id': zone_id,
                        'failed_edge': zone['edge'],
                        'intersection_nodes': intersection,
                        'intersection_size': len(intersection)
                    })
            
            if intersecting_zones:
                candidate = {
                    'vehicle_id': vehicle_id,
                    'route': route,
                    'current_cost': current_cost,
                    'spare_capacity': spare_capacity,
                    'intersecting_zones': intersecting_zones,
                    'feasible': spare_capacity > 5  # Minimum spare capacity needed
                }
                candidates.append(candidate)
                
                print(f"Vehicle {vehicle_id}:")
                print(f"  Route: {route}")
                print(f"  Current cost: {current_cost:.1f}/{self.capacity}")
                print(f"  Spare capacity: {spare_capacity:.1f}")
                print(f"  Intersects with {len(intersecting_zones)} search zones")
                for zone_info in intersecting_zones:
                    print(f"    Zone {zone_info['zone_id']} (edge {zone_info['failed_edge']}): {zone_info['intersection_nodes']}")
                print(f"  Feasible: {'Yes' if candidate['feasible'] else 'No'}")
        
        self.candidate_vehicles = candidates
        return candidates
    
    def calculate_magnetic_field_score(self, edge, vehicle_route, insertion_position):
        """Simplified magnetic field score calculation"""
        # Basic implementation - can be enhanced with full magnetic field logic
        
        # Required edge influence (P)
        P = 1.0 if edge in self.failed_edges else 0.0
        
        # Position in route influence
        route_progress = insertion_position / len(vehicle_route) if len(vehicle_route) > 0 else 0
        
        # Depot influence (D) - simplified
        D = 0.3  # Base depot influence
        
        # Calculate score: early in route favors required edges, later favors depot
        w = route_progress
        score = (1 - w) * P + w * D
        
        return {
            'P': P,
            'D': D,
            'w': w,
            'score': score
        }
    
    def attempt_insertions(self):
        """Attempt to insert failed edges into candidate vehicles"""
        print(f"\nINSERTION ATTEMPTS")
        print(f"==================")
        
        for candidate in self.candidate_vehicles:
            if not candidate['feasible']:
                continue
                
            vehicle_id = candidate['vehicle_id']
            vehicle_route = candidate['route']
            
            print(f"\nVehicle {vehicle_id} insertion attempts:")
            
            insertion_results = []
            
            # Try to insert each failed edge that intersects with this vehicle's zones
            for zone_info in candidate['intersecting_zones']:
                failed_edge = zone_info['failed_edge']
                zone_id = zone_info['zone_id']
                
                # Find best insertion position using magnetic field scoring
                best_position = None
                best_score = -1
                
                for pos in range(len(vehicle_route)):
                    score_data = self.calculate_magnetic_field_score(failed_edge, vehicle_route, pos)
                    
                    if score_data['score'] > best_score:
                        best_score = score_data['score']
                        best_position = pos
                
                if best_position is not None and best_score > 0.3:  # Minimum threshold
                    insertion_results.append({
                        'edge': failed_edge,
                        'position': best_position,
                        'score': best_score,
                        'zone_id': zone_id,
                        'estimated_cost': 4  # Simplified cost estimate
                    })
                    
                    print(f"  ✓ Edge {failed_edge}: Position {best_position}, Score {best_score:.3f}")
                else:
                    print(f"  ✗ Edge {failed_edge}: No suitable insertion (best score: {best_score:.3f})")
            
            self.insertion_attempts[vehicle_id] = insertion_results
    
    def resolve_conflicts_and_execute(self):
        """Resolve conflicts and execute successful insertions"""
        print(f"\nCONFLICT RESOLUTION")
        print(f"===================")
        
        # Group insertion attempts by edge
        edge_attempts = defaultdict(list)
        for vehicle_id, attempts in self.insertion_attempts.items():
            for attempt in attempts:
                edge = tuple(sorted(attempt['edge']))
                edge_attempts[edge].append({
                    'vehicle_id': vehicle_id,
                    'attempt': attempt
                })
        
        # Resolve conflicts
        for edge, attempts in edge_attempts.items():
            if len(attempts) > 1:
                print(f"Conflict for edge {edge}:")
                print(f"  Competing vehicles: {[a['vehicle_id'] for a in attempts]}")
                
                # Choose vehicle with highest score
                best_attempt = max(attempts, key=lambda x: x['attempt']['score'])
                winner = best_attempt['vehicle_id']
                
                print(f"  Winner: Vehicle {winner} (score: {best_attempt['attempt']['score']:.3f})")
                
                # Record successful insertion
                self.successful_insertions.append({
                    'vehicle_id': winner,
                    'edge': list(edge),
                    'insertion_data': best_attempt['attempt']
                })
            else:
                # No conflict
                attempt = attempts[0]
                self.successful_insertions.append({
                    'vehicle_id': attempt['vehicle_id'],
                    'edge': list(edge),
                    'insertion_data': attempt['attempt']
                })
                print(f"No conflict for edge {edge} -> Vehicle {attempt['vehicle_id']}")
        
        # Apply insertions to vehicle routes
        print(f"\nAPPLYING INSERTIONS")
        print(f"===================")
        
        for insertion in self.successful_insertions:
            vehicle_id = insertion['vehicle_id']
            edge = insertion['edge']
            
            # Add edge to vehicle's required edges
            self.vehicle_trips[vehicle_id]['required_edges'].append(edge)
            
            print(f"✓ Vehicle {vehicle_id} now handles edge {edge}")
    
    def run_recovery_process(self):
        """Run the complete recovery process with step tracking"""
        print(f"SIMPLE FAILURE RECOVERY DEMONSTRATION")
        print(f"=====================================")
        
        # Step 1: Extract failed edges
        self.step_history.append("Extracting failed edges")
        self.extract_failed_edges()
        
        # Step 2: Create search zones
        self.step_history.append("Creating search zones")
        self.create_search_zones()
        
        # Step 3: Find candidate vehicles
        self.step_history.append("Finding candidate vehicles")
        self.find_candidate_vehicles()
        
        # Step 4: Attempt insertions
        self.step_history.append("Attempting insertions")
        self.attempt_insertions()
        
        # Step 5: Resolve conflicts and execute
        self.step_history.append("Resolving conflicts")
        self.resolve_conflicts_and_execute()
        
        return self.successful_insertions

# def visualize_step_1_initial_state(recovery_system):
#     """Visualize initial state with all vehicle routes"""
#     fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
#     # Draw network
#     nx.draw_networkx_edges(recovery_system.network, recovery_system.pos, ax=ax, 
#                           edge_color='lightgray', width=1, alpha=0.5)
    
#     # Draw all nodes
#     nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos, ax=ax,
#                           node_color='lightblue', node_size=300, alpha=0.7)
    
#     # Highlight depots
#     nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos, 
#                           nodelist=DEPOTS, ax=ax, node_color='orange', 
#                           node_size=500, alpha=0.9)
    
#     # Draw vehicle routes
#     colors = ['blue', 'green', 'red', 'purple']
#     for vehicle_id, trip in recovery_system.vehicle_trips.items():
#         route = trip['route']
#         color = colors[vehicle_id % len(colors)]
        
#         # Draw route edges
#         for i in range(len(route) - 1):
#             u, v = route[i], route[i+1]
#             x1, y1 = recovery_system.pos[u]
#             x2, y2 = recovery_system.pos[v]
            
#             if trip['status'] == 'failed':
#                 ax.plot([x1, x2], [y1, y2], color='red', linewidth=3, alpha=0.8, linestyle='--')
#             else:
#                 ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)
        
#         # Mark current position
#         current_pos = trip['current_position']
#         x, y = recovery_system.pos[current_pos]
        
#         if trip['status'] == 'failed':
#             ax.scatter(x, y, c='red', s=400, marker='X', 
#                       label=f'Vehicle {vehicle_id} (FAILED)' if vehicle_id == recovery_system.failed_vehicle else "")
#         else:
#             ax.scatter(x, y, c=color, s=300, marker='o', 
#                       label=f'Vehicle {vehicle_id}' if vehicle_id != recovery_system.failed_vehicle else "")
    
#     # Add node labels
#     nx.draw_networkx_labels(recovery_system.network, recovery_system.pos, ax=ax, font_size=8)
    
#     # Add edge weights
#     edge_labels = nx.get_edge_attributes(recovery_system.network, 'weight')
#     nx.draw_networkx_edge_labels(recovery_system.network, recovery_system.pos, 
#                                 edge_labels, ax=ax, font_size=6)
    
#     ax.set_title('Initial State: Vehicle Routes and Positions', fontsize=14, fontweight='bold')
#     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     ax.set_aspect('equal')
#     plt.tight_layout()

# def visualize_step_2_failure_analysis(recovery_system):
#     """Visualize failure analysis with required edges highlighted"""
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
#     # Left: Failed vehicle details
#     ax1.set_title('Failed Vehicle Analysis', fontsize=14, fontweight='bold')
    
#     # Draw network
#     nx.draw_networkx_edges(recovery_system.network, recovery_system.pos, ax=ax1,
#                           edge_color='lightgray', width=1, alpha=0.3)
#     nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos, ax=ax1,
#                           node_color='lightgray', node_size=200, alpha=0.5)
    
#     # Highlight failed vehicle route
#     failed_trip = recovery_system.vehicle_trips[recovery_system.failed_vehicle]
#     route = failed_trip['route']
    
#     for i in range(len(route) - 1):
#         u, v = route[i], route[i+1]
#         x1, y1 = recovery_system.pos[u]
#         x2, y2 = recovery_system.pos[v]
#         ax1.plot([x1, x2], [y1, y2], 'red', linewidth=3, alpha=0.7, linestyle='--')
    
#     # Highlight required edges that need recovery
#     for edge in recovery_system.failed_edges:
#         u, v = edge
#         x1, y1 = recovery_system.pos[u]
#         x2, y2 = recovery_system.pos[v]
#         ax1.plot([x1, x2], [y1, y2], 'darkred', linewidth=5, alpha=0.9)
    
#     # Mark failure point
#     failure_node = failed_trip['current_position']
#     x, y = recovery_system.pos[failure_node]
#     ax1.scatter(x, y, c='red', s=500, marker='X', label='Failure Point')
    
#     # Mark depot
#     depot = failed_trip['depot']
#     x, y = recovery_system.pos[depot]
#     ax1.scatter(x, y, c='orange', s=400, marker='s', label='Depot')
    
#     nx.draw_networkx_labels(recovery_system.network, recovery_system.pos, ax=ax1, font_size=8)
#     ax1.legend()
#     ax1.set_aspect('equal')
    
#     # Right: Summary table
#     ax2.axis('off')
#     ax2.set_title('Failure Details', fontsize=14, fontweight='bold')
    
#     # Create summary text
#     summary_text = f"""
#     FAILURE SUMMARY
#     ===============
    
#     Failed Vehicle: {recovery_system.failed_vehicle}
#     Failure Location: Node {failed_trip['current_position']}
#     Original Route: {failed_trip['route']}
#     Route Length: {len(failed_trip['route'])} nodes
#     Original Depot: {failed_trip['depot']}
    
#     REQUIRED EDGES TO RECOVER:
#     """
    
#     for i, edge in enumerate(recovery_system.failed_edges):
#         u, v = edge
#         if recovery_system.network.has_edge(u, v):
#             weight = recovery_system.network[u][v]['weight']
#             summary_text += f"\n    {i+1}. Edge {edge} (weight: {weight})"
#         else:
#             summary_text += f"\n    {i+1}. Edge {edge} (indirect)"
    
#     summary_text += f"\n\n    Total edges to recover: {len(recovery_system.failed_edges)}"
#     summary_text += f"\n    Vehicle capacity: {recovery_system.capacity}"
    
#     ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=11,
#             verticalalignment='top', fontfamily='monospace',
#             bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))
    
#     plt.tight_layout()

# def visualize_step_3_search_zones(recovery_system):
#     """Visualize search zones for each failed edge"""
#     failed_edges = recovery_system.failed_edges
#     n_edges = len(failed_edges)
    
#     fig, axes = plt.subplots(1, n_edges, figsize=(6*n_edges, 8))
#     if n_edges == 1:
#         axes = [axes]
    
#     colors = ['red', 'green', 'blue', 'orange', 'purple']
    
#     for i, (edge_idx, zone) in enumerate(recovery_system.search_zones.items()):
#         ax = axes[i]
#         edge = zone['edge']
#         reachable_nodes = zone['reachable_nodes']
        
#         # Draw network
#         nx.draw_networkx_edges(recovery_system.network, recovery_system.pos, ax=ax,
#                               edge_color='lightgray', width=1, alpha=0.3)
        
#         # Draw all nodes in light gray
#         nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos, ax=ax,
#                               node_color='lightgray', node_size=150, alpha=0.5)
        
#         # Highlight reachable nodes in search zone
#         nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos,
#                               nodelist=list(reachable_nodes), ax=ax,
#                               node_color=colors[i % len(colors)], node_size=250, alpha=0.7)
        
#         # Highlight the failed edge
#         u, v = edge
#         x1, y1 = recovery_system.pos[u]
#         x2, y2 = recovery_system.pos[v]
#         ax.plot([x1, x2], [y1, y2], colors[i % len(colors)], linewidth=5, alpha=0.9)
        
#         # Mark edge endpoints
#         nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos,
#                               nodelist=[u, v], ax=ax,
#                               node_color='darkred', node_size=300, alpha=1.0)
        
#         # Add labels
#         nx.draw_networkx_labels(recovery_system.network, recovery_system.pos, ax=ax, font_size=8)
        
#         ax.set_title(f'Search Zone for Edge {edge}\n{len(reachable_nodes)} reachable nodes', 
#                     fontsize=12, fontweight='bold')
#         ax.set_aspect('equal')
    
#     plt.suptitle(f'Search Zones (Capacity-Based Reachability)', fontsize=14, fontweight='bold')
#     plt.tight_layout()

# def visualize_step_4_candidate_analysis(recovery_system):
#     """Visualize candidate vehicles and their intersections with search zones"""
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
#     # Left: All vehicles with search zone overlays
#     ax1.set_title('Candidate Vehicle Analysis', fontsize=14, fontweight='bold')
    
#     # Draw network
#     nx.draw_networkx_edges(recovery_system.network, recovery_system.pos, ax=ax1,
#                           edge_color='lightgray', width=1, alpha=0.3)
#     nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos, ax=ax1,
#                           node_color='lightgray', node_size=150, alpha=0.5)
    
#     # Draw search zones with transparency
#     zone_colors = ['red', 'green', 'blue', 'orange']
#     for zone_idx, zone in recovery_system.search_zones.items():
#         reachable_nodes = zone['reachable_nodes']
#         nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos,
#                               nodelist=list(reachable_nodes), ax=ax1,
#                               node_color=zone_colors[zone_idx % len(zone_colors)], 
#                               node_size=300, alpha=0.3)
    
#     # Draw vehicle routes
#     vehicle_colors = ['blue', 'green', 'purple', 'brown']
#     for candidate in recovery_system.candidate_vehicles:
#         vehicle_id = candidate['vehicle_id']
#         route = candidate['route']
#         color = vehicle_colors[vehicle_id % len(vehicle_colors)]
        
#         # Draw route
#         for i in range(len(route) - 1):
#             u, v = route[i], route[i+1]
#             x1, y1 = recovery_system.pos[u]
#             x2, y2 = recovery_system.pos[v]
#             ax1.plot([x1, x2], [y1, y2], color=color, linewidth=3, alpha=0.8)
        
#         # Mark vehicle position
#         current_pos = recovery_system.vehicle_trips[vehicle_id]['current_position']
#         x, y = recovery_system.pos[current_pos]
#         ax1.scatter(x, y, c=color, s=400, marker='o', 
#                    label=f'Vehicle {vehicle_id}' if candidate['feasible'] else f'Vehicle {vehicle_id} (No Capacity)')
    
#     nx.draw_networkx_labels(recovery_system.network, recovery_system.pos, ax=ax1, font_size=8)
#     ax1.legend()
#     ax1.set_aspect('equal')
    
#     # Right: Candidate summary
#     ax2.axis('off')
#     ax2.set_title('Candidate Analysis Results', fontsize=14, fontweight='bold')
    
#     summary_text = "CANDIDATE VEHICLE ANALYSIS\n" + "="*30 + "\n\n"
    
#     for candidate in recovery_system.candidate_vehicles:
#         vehicle_id = candidate['vehicle_id']
#         summary_text += f"Vehicle {vehicle_id}:\n"
#         summary_text += f"  Route: {candidate['route']}\n"
#         summary_text += f"  Current Cost: {candidate['current_cost']:.1f}/{recovery_system.capacity}\n"
#         summary_text += f"  Spare Capacity: {candidate['spare_capacity']:.1f}\n"
#         summary_text += f"  Feasible: {'✓' if candidate['feasible'] else '✗'}\n"
#         summary_text += f"  Intersecting Zones: {len(candidate['intersecting_zones'])}\n"
        
#         for zone_info in candidate['intersecting_zones']:
#             summary_text += f"    - Zone {zone_info['zone_id']} (edge {zone_info['failed_edge']})\n"
#             summary_text += f"      Intersection: {zone_info['intersection_nodes']}\n"
#         summary_text += "\n"
    
#     ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=10,
#             verticalalignment='top', fontfamily='monospace',
#             bbox=dict(boxstyle='round,pad=1', facecolor='lightcyan', alpha=0.8))
    
#     plt.tight_layout()

# def visualize_step_5_insertion_attempts(recovery_system):
#     """Visualize insertion attempts with magnetic field scores"""
#     n_candidates = len([c for c in recovery_system.candidate_vehicles if c['feasible']])
    
#     if n_candidates == 0:
#         print("No feasible candidates for insertion visualization")
#         return
    
#     fig, axes = plt.subplots(1, n_candidates, figsize=(8*n_candidates, 8))
#     if n_candidates == 1:
#         axes = [axes]
    
#     candidate_idx = 0
#     for candidate in recovery_system.candidate_vehicles:
#         if not candidate['feasible']:
#             continue
            
#         vehicle_id = candidate['vehicle_id']
#         ax = axes[candidate_idx]
        
#         # Draw network
#         nx.draw_networkx_edges(recovery_system.network, recovery_system.pos, ax=ax,
#                               edge_color='lightgray', width=1, alpha=0.3)
#         nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos, ax=ax,
#                               node_color='lightgray', node_size=150, alpha=0.5)
        
#         # Draw vehicle route
#         route = candidate['route']
#         for i in range(len(route) - 1):
#             u, v = route[i], route[i+1]
#             x1, y1 = recovery_system.pos[u]
#             x2, y2 = recovery_system.pos[v]
#             ax.plot([x1, x2], [y1, y2], 'blue', linewidth=3, alpha=0.7)
        
#         # Highlight insertion attempts
#         if vehicle_id in recovery_system.insertion_attempts:
#             attempts = recovery_system.insertion_attempts[vehicle_id]
            
#             for attempt in attempts:
#                 edge = attempt['edge']
#                 position = attempt['position']
#                 score = attempt['score']
                
#                 # Highlight the edge being inserted
#                 u, v = edge
#                 x1, y1 = recovery_system.pos[u]
#                 x2, y2 = recovery_system.pos[v]
#                 ax.plot([x1, x2], [y1, y2], 'red', linewidth=5, alpha=0.8)
                
#                 # Mark insertion position in route
#                 if position < len(route):
#                     insert_node = route[position]
#                     x, y = recovery_system.pos[insert_node]
#                     ax.scatter(x, y, c='green', s=300, marker='*', alpha=0.9)
                    
#                     # Add score annotation
#                     ax.annotate(f'Score: {score:.2f}', 
#                                xy=(x, y), xytext=(x+0.5, y+0.5),
#                                fontsize=10, fontweight='bold',
#                                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
#         # Mark vehicle position
#         current_pos = recovery_system.vehicle_trips[vehicle_id]['current_position']
#         x, y = recovery_system.pos[current_pos]
#         ax.scatter(x, y, c='blue', s=400, marker='o', alpha=0.9)
        
#         nx.draw_networkx_labels(recovery_system.network, recovery_system.pos, ax=ax, font_size=8)
#         ax.set_title(f'Vehicle {vehicle_id} Insertion Attempts', fontsize=12, fontweight='bold')
#         ax.set_aspect('equal')
        
#         candidate_idx += 1
    
#     plt.suptitle('Magnetic Field Insertion Attempts', fontsize=14, fontweight='bold')
#     plt.tight_layout()

# def visualize_step_6_final_results(recovery_system):
#     """Visualize final recovery results"""
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
#     # 1. Final state with successful insertions
#     ax1.set_title('Final Recovery State', fontsize=14, fontweight='bold')
    
#     # Draw network
#     nx.draw_networkx_edges(recovery_system.network, recovery_system.pos, ax=ax1,
#                           edge_color='lightgray', width=1, alpha=0.5)
#     nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos, ax=ax1,
#                           node_color='lightblue', node_size=200, alpha=0.7)
    
#     # Draw vehicle routes with updates
#     vehicle_colors = ['blue', 'green', 'purple', 'brown']
#     for vehicle_id, trip in recovery_system.vehicle_trips.items():
#         if trip['status'] == 'failed':
#             continue
            
#         route = trip['route']
#         color = vehicle_colors[vehicle_id % len(vehicle_colors)]
        
#         # Draw route
#         for i in range(len(route) - 1):
#             u, v = route[i], route[i+1]
#             x1, y1 = recovery_system.pos[u]
#             x2, y2 = recovery_system.pos[v]
#             ax1.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)
    
#     # Highlight successfully recovered edges
#     for insertion in recovery_system.successful_insertions:
#         edge = insertion['edge']
#         vehicle_id = insertion['vehicle_id']
#         color = vehicle_colors[vehicle_id % len(vehicle_colors)]
        
#         u, v = edge
#         x1, y1 = recovery_system.pos[u]
#         x2, y2 = recovery_system.pos[v]
#         ax1.plot([x1, x2], [y1, y2], color, linewidth=5, alpha=0.9)
        
#         # Add annotation
#         mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
#         ax1.annotate(f'V{vehicle_id}', xy=(mid_x, mid_y), 
#                     fontsize=8, fontweight='bold', ha='center',
#                     bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
    
#     nx.draw_networkx_labels(recovery_system.network, recovery_system.pos, ax=ax1, font_size=8)
#     ax1.set_aspect('equal')
    
#     # 2. Recovery success metrics
#     ax2.set_title('Recovery Success Analysis', fontsize=14, fontweight='bold')
    
#     total_failed = len(recovery_system.failed_edges)
#     recovered = len(recovery_system.successful_insertions)
#     remaining = total_failed - recovered
    
#     labels = ['Recovered', 'Remaining']
#     sizes = [recovered, remaining]
#     colors = ['green', 'red']
#     explode = (0.1, 0)  # Explode recovered slice
    
#     wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, explode=explode,
#                                        autopct='%1.1f%%', startangle=90)
    
#     # Add numbers to the chart
#     ax2.text(0, -1.3, f'Total Failed Edges: {total_failed}\nRecovered: {recovered}\nRemaining: {remaining}',
#             ha='center', va='center', fontsize=12,
#             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
#     # 3. Vehicle capacity utilization
#     ax3.set_title('Vehicle Capacity Utilization', fontsize=14, fontweight='bold')
    
#     vehicle_ids = []
#     original_costs = []
#     final_costs = []
    
#     for vehicle_id, trip in recovery_system.vehicle_trips.items():
#         if trip['status'] == 'failed':
#             continue
            
#         original_cost = recovery_system.calculate_route_cost(trip['route'])
        
#         # Calculate additional cost from insertions
#         additional_cost = 0
#         for insertion in recovery_system.successful_insertions:
#             if insertion['vehicle_id'] == vehicle_id:
#                 additional_cost += insertion['insertion_data']['estimated_cost']
        
#         final_cost = original_cost + additional_cost
        
#         vehicle_ids.append(f'V{vehicle_id}')
#         original_costs.append(original_cost)
#         final_costs.append(final_cost)
    
#     x = np.arange(len(vehicle_ids))
#     width = 0.35
    
#     bars1 = ax3.bar(x - width/2, original_costs, width, label='Original', color='lightblue', alpha=0.7)
#     bars2 = ax3.bar(x + width/2, final_costs, width, label='After Recovery', color='darkblue', alpha=0.7)
    
#     # Add capacity line
#     ax3.axhline(y=recovery_system.capacity, color='red', linestyle='--', 
#                label='Capacity Limit', alpha=0.7)
    
#     ax3.set_ylabel('Route Cost')
#     ax3.set_xticks(x)
#     ax3.set_xticklabels(vehicle_ids)
#     ax3.legend()
#     ax3.grid(True, alpha=0.3, axis='y')
    
#     # Add value labels
#     for bars in [bars1, bars2]:
#         for bar in bars:
#             height = bar.get_height()
#             ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
#                     f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
#     # 4. Insertion details table
#     ax4.axis('off')
#     ax4.set_title('Insertion Details', fontsize=14, fontweight='bold')
    
#     if recovery_system.successful_insertions:
#         # Create table data
#         table_data = [['Edge', 'Assigned Vehicle', 'Score', 'Position']]
        
#         for insertion in recovery_system.successful_insertions:
#             edge = insertion['edge']
#             vehicle_id = insertion['vehicle_id']
#             score = insertion['insertion_data']['score']
#             position = insertion['insertion_data']['position']
            
#             table_data.append([f'{edge}', f'Vehicle {vehicle_id}', f'{score:.3f}', f'{position}'])
        
#         # Create table
#         table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
#                          cellLoc='center', loc='center',
#                          bbox=[0.1, 0.3, 0.8, 0.6])
#         table.auto_set_font_size(False)
#         table.set_fontsize(10)
#         table.scale(1, 2)
        
#         # Style header
#         for i in range(len(table_data[0])):
#             table[(0, i)].set_facecolor('#4472C4')
#             table[(0, i)].set_text_props(weight='bold', color='white')
        
#         # Color rows alternately
#         for i in range(1, len(table_data)):
#             for j in range(len(table_data[0])):
#                 if i % 2 == 0:
#                     table[(i, j)].set_facecolor('#F2F2F2')
#     else:
#         ax4.text(0.5, 0.5, 'No successful insertions', ha='center', va='center',
#                 transform=ax4.transAxes, fontsize=16, style='italic')
    
#     plt.suptitle('Recovery Results Summary', fontsize=16, fontweight='bold')
#     plt.tight_layout()

# def create_step_by_step_summary(recovery_system):
#     """Create a summary visualization of all steps"""
#     fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
#     ax.set_title('Complete Recovery Process Summary', fontsize=16, fontweight='bold')
    
#     # Draw the final network state
#     nx.draw_networkx_edges(recovery_system.network, recovery_system.pos, ax=ax,
#                           edge_color='lightgray', width=1, alpha=0.5)
#     nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos, ax=ax,
#                           node_color='lightblue', node_size=300, alpha=0.7)
    
#     # Highlight depots
#     nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos,
#                           nodelist=DEPOTS, ax=ax, node_color='orange',
#                           node_size=500, alpha=0.9)
    
#     # Draw all active vehicle routes
#     vehicle_colors = ['blue', 'green', 'purple', 'brown']
#     for vehicle_id, trip in recovery_system.vehicle_trips.items():
#         if trip['status'] == 'failed':
#             continue
            
#         route = trip['route']
#         color = vehicle_colors[vehicle_id % len(vehicle_colors)]
        
#         for i in range(len(route) - 1):
#             u, v = route[i], route[i+1]
#             x1, y1 = recovery_system.pos[u]
#             x2, y2 = recovery_system.pos[v]
#             ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.6,
#                    label=f'Vehicle {vehicle_id}' if i == 0 else "")
    
#     # Highlight failed edges (original)
#     for edge in recovery_system.failed_edges:
#         u, v = edge
#         x1, y1 = recovery_system.pos[u]
#         x2, y2 = recovery_system.pos[v]
#         ax.plot([x1, x2], [y1, y2], 'red', linewidth=4, alpha=0.5, linestyle='--')
    
#     # Highlight successfully recovered edges
#     for insertion in recovery_system.successful_insertions:
#         edge = insertion['edge']
#         vehicle_id = insertion['vehicle_id']
#         color = vehicle_colors[vehicle_id % len(vehicle_colors)]
        
#         u, v = edge
#         x1, y1 = recovery_system.pos[u]
#         x2, y2 = recovery_system.pos[v]
#         ax.plot([x1, x2], [y1, y2], color, linewidth=5, alpha=0.9)
    
#     # Mark failure point
#     failed_trip = recovery_system.vehicle_trips[recovery_system.failed_vehicle]
#     failure_node = failed_trip['current_position']
#     x, y = recovery_system.pos[failure_node]
#     ax.scatter(x, y, c='red', s=600, marker='X', label='Failure Point')
    
#     nx.draw_networkx_labels(recovery_system.network, recovery_system.pos, ax=ax, font_size=10)
    
#     # Create legend
#     legend_elements = [
#         plt.Line2D([0], [0], color='red', linewidth=4, linestyle='--', alpha=0.5, label='Failed Edges'),
#         plt.Line2D([0], [0], color='black', linewidth=5, alpha=0.9, label='Recovered Edges'),
#         plt.scatter([], [], c='red', s=600, marker='X', label='Failure Point'),
#         plt.scatter([], [], c='orange', s=500, marker='o', label='Depots')
#     ]
    
#     # Add vehicle routes to legend
#     for vehicle_id, trip in recovery_system.vehicle_trips.items():
#         if trip['status'] != 'failed':
#             color = vehicle_colors[vehicle_id % len(vehicle_colors)]
#             legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=2, 
#                                             label=f'Vehicle {vehicle_id}'))
    
#     ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
#     ax.set_aspect('equal')
    
#     # Add summary statistics box
#     total_failed = len(recovery_system.failed_edges)
#     recovered = len(recovery_system.successful_insertions)
#     recovery_rate = (recovered / total_failed * 100) if total_failed > 0 else 0
    
#     summary_text = f"""
#     RECOVERY SUMMARY
#     ================
    
#     Failed Vehicle: {recovery_system.failed_vehicle}
#     Total Failed Edges: {total_failed}
#     Successfully Recovered: {recovered}
#     Recovery Rate: {recovery_rate:.1f}%
    
#     Search Zones Created: {len(recovery_system.search_zones)}
#     Candidate Vehicles: {len(recovery_system.candidate_vehicles)}
#     Successful Insertions: {len(recovery_system.successful_insertions)}
#     """
    
#     ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=11,
#            verticalalignment='top', fontfamily='monospace',
#            bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.9))
    
#     plt.tight_layout()

# def run_simple_failure_recovery_demo():
#     """Run the complete simple failure recovery demonstration with step-by-step visualizations"""
#     print("SIMPLE MULTI-VEHICLE FAILURE RECOVERY DEMONSTRATION")
#     print("="*60)
    
#     # Create recovery system
#     recovery_system = SimpleFailureRecovery(
#         network=NETWORK,
#         node_positions=NODE_POSITIONS,
#         vehicle_trips=VEHICLE_TRIPS,
#         failed_vehicle=FAILED_VEHICLE,
#         capacity=VEHICLE_CAPACITY
#     )
    
#     print(f"Network: {len(NETWORK.nodes())} nodes, {len(NETWORK.edges())} edges")
#     print(f"Vehicles: 4 total, 1 failed")
#     print(f"Depots: {DEPOTS}")
#     print(f"Vehicle capacity: {VEHICLE_CAPACITY}")
    
#     # Step 1: Show initial state
#     print(f"\nStep 1: Visualizing Initial State")
#     visualize_step_1_initial_state(recovery_system)
#     plt.show()
    
#     # Step 2: Analyze failure
#     print(f"\nStep 2: Analyzing Failure")
#     recovery_system.extract_failed_edges()
#     visualize_step_2_failure_analysis(recovery_system)
#     plt.show()
    
#     # Step 3: Create search zones
#     print(f"\nStep 3: Creating Search Zones")
#     recovery_system.create_search_zones()
#     visualize_step_3_search_zones(recovery_system)
#     plt.show()
    
#     # Step 4: Find candidates
#     print(f"\nStep 4: Finding Candidate Vehicles")
#     recovery_system.find_candidate_vehicles()
#     visualize_step_4_candidate_analysis(recovery_system)
#     plt.show()
    
#     # Step 5: Attempt insertions
#     print(f"\nStep 5: Attempting Insertions")
#     recovery_system.attempt_insertions()
#     visualize_step_5_insertion_attempts(recovery_system)
#     plt.show()
    
#     # Step 6: Execute recovery
#     print(f"\nStep 6: Executing Recovery")
#     recovery_system.resolve_conflicts_and_execute()
#     visualize_step_6_final_results(recovery_system)
#     plt.show()
    
#     # Step 7: Complete summary
#     print(f"\nStep 7: Complete Process Summary")
#     create_step_by_step_summary(recovery_system)
#     plt.show()
    
#     # Generate final report
#     print(f"\n{'='*60}")
#     print("FINAL RECOVERY REPORT")
#     print(f"{'='*60}")
    
#     total_failed = len(recovery_system.failed_edges)
#     recovered = len(recovery_system.successful_insertions)
#     recovery_rate = (recovered / total_failed * 100) if total_failed > 0 else 0
    
#     print(f"Recovery Performance:")
#     print(f"  Total failed edges: {total_failed}")
#     print(f"  Successfully recovered: {recovered}")
#     print(f"  Recovery rate: {recovery_rate:.1f}%")
#     print(f"  Search zones created: {len(recovery_system.search_zones)}")
#     print(f"  Candidate vehicles found: {len(recovery_system.candidate_vehicles)}")
    
#     print(f"\nRecovery Details:")
#     for insertion in recovery_system.successful_insertions:
#         edge = insertion['edge']
#         vehicle = insertion['vehicle_id']
#         score = insertion['insertion_data']['score']
#         print(f"  Edge {edge} -> Vehicle {vehicle} (score: {score:.3f})")
    
#     print(f"\nComputational Efficiency:")
#     total_vehicles = len(VEHICLE_TRIPS) - 1  # Exclude failed vehicle
#     candidates_found = len(recovery_system.candidate_vehicles)
#     efficiency = (candidates_found / total_vehicles * 100) if total_vehicles > 0 else 0
#     print(f"  Search radius eliminated {total_vehicles - candidates_found} vehicles")
#     print(f"  Computational focus: {efficiency:.1f}% of vehicles considered")
    
#     return recovery_system

# if __name__ == "__main__":
#     recovery_system = run_simple_failure_recovery_demo()
    
#     print(f"\n{'='*60}")
#     print("DEMONSTRATION COMPLETE")
#     print(f"{'='*60}")
#     print("\nKey Features Demonstrated:")
#     print("✓ Simple 4-vehicle scenario with clear visualization")
#     print("✓ Capacity-based search radius for each failed edge")
#     print("✓ Step-by-step process with detailed plots")
#     print("✓ Magnetic field scoring for insertion decisions")
#     print("✓ Conflict resolution between competing vehicles")
#     print("✓ Comprehensive recovery analysis and reporting")

"""
Simple Multi-Vehicle Failure Recovery using Magnetic Field Approach
=================================================================
This demonstrates a focused 4-vehicle failure recovery system with detailed visualizations
showing initial routes, failure points, search radius based on battery capacity, and 
step-by-step recovery process.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from math import exp, sqrt, hypot
import seaborn as sns
from itertools import permutations
import random
import copy
from collections import defaultdict

# Create a simple network for clear demonstration
def create_simple_network():
    """Create a simple network with clear structure"""
    G = nx.Graph()
    
    # Add nodes with positions for clear visualization
    node_positions = {
        0: (0, 4),   # Depot A
        1: (2, 4),
        2: (4, 4),
        3: (6, 4),
        4: (8, 4),   # Depot B
        5: (8, 2),
        6: (6, 2),
        7: (4, 2),   # Depot C (center)
        8: (2, 2),
        9: (0, 2),
        10: (1, 3),
        11: (3, 3),
        12: (5, 3),
        13: (7, 3),
        14: (4, 1),
        15: (4, 5)
    }
    
    # Add nodes
    for node, pos in node_positions.items():
        G.add_node(node, pos=pos)
    
    # Add edges with weights
    edges = [
        (0, 1, 3), (1, 2, 3), (2, 3, 3), (3, 4, 3),  # Top horizontal
        (4, 5, 2), (5, 6, 2), (6, 7, 2), (7, 8, 2), (8, 9, 2),  # Bottom path
        (0, 9, 3), (9, 8, 2), (8, 7, 2), (7, 6, 2), (6, 5, 2), (5, 4, 2),  # Bottom horizontal
        (0, 10, 2), (1, 10, 2), (1, 11, 2), (2, 11, 2), (2, 12, 2), (3, 12, 2), (3, 13, 2), (4, 13, 2),  # Middle connections
        (10, 11, 2), (11, 12, 2), (12, 13, 2),  # Middle horizontal
        (7, 14, 2), (7, 15, 2), (2, 15, 2),  # Vertical connections
        (9, 10, 2), (10, 8, 3), (11, 7, 2), (12, 6, 3), (13, 5, 2),  # Cross connections
    ]
    
    G.add_weighted_edges_from(edges)
    return G, node_positions

# Create simple test instance
NETWORK, NODE_POSITIONS = create_simple_network()
DEPOTS = [0, 4, 7]  # Three depots: A, B, C
VEHICLE_CAPACITY = 20

# 4-vehicle scenario with detailed trips
VEHICLE_TRIPS = {
    0: {
        'route': [0, 1, 2, 15, 7, 14, 7],  # Depot A -> Depot C
        'current_position': 2,  # Vehicle is at node 2
        'required_edges': [(1, 2), (2, 15), (15, 7)],
        'depot': 0,
        'status': 'active'
    },
    1: {
        'route': [4, 13, 12, 11, 10, 0],  # Depot B -> Depot A  
        'current_position': 12,  # Vehicle is at node 12
        'required_edges': [(13, 12), (12, 11), (10, 0)],
        'depot': 4,
        'status': 'active'
    },
    2: {
        'route': [7, 8, 9, 0, 1, 11, 7],  # Depot C -> Depot C (round trip)
        'current_position': 9,  # Vehicle FAILED at node 9
        'required_edges': [(8, 9), (9, 0), (1, 11)],  # These need to be recovered
        'depot': 7,
        'status': 'failed'
    },
    3: {
        'route': [4, 5, 6, 7, 8, 9, 0],  # Depot B -> Depot A
        'current_position': 6,  # Vehicle is at node 6
        'required_edges': [(5, 6), (6, 7), (8, 9)],
        'depot': 4,
        'status': 'active'
    }
}

FAILED_VEHICLE = 2  # Vehicle 2 has failed

class SimpleFailureRecovery:
    """
    Simple Multi-Vehicle Failure Recovery System with Detailed Visualizations
    """
    
    def __init__(self, network, node_positions, vehicle_trips, failed_vehicle, capacity):
        self.network = network
        self.pos = node_positions
        self.vehicle_trips = copy.deepcopy(vehicle_trips)
        self.failed_vehicle = failed_vehicle
        self.capacity = capacity
        self.max_edge_weight = max(d['weight'] for u, v, d in network.edges(data=True))
        
        # Recovery data
        self.failed_edges = []
        self.search_zones = {}
        self.candidate_vehicles = []
        self.insertion_attempts = {}
        self.successful_insertions = []
        self.step_history = []
        
    def calculate_route_cost(self, route):
        """Calculate total cost of a route"""
        if len(route) < 2:
            return 0
        cost = 0
        for i in range(len(route) - 1):
            if self.network.has_edge(route[i], route[i+1]):
                cost += self.network[route[i]][route[i+1]]['weight']
            else:
                try:
                    cost += nx.shortest_path_length(self.network, route[i], route[i+1], weight='weight')
                except nx.NetworkXNoPath:
                    return float('inf')
        return cost
    
    def extract_failed_edges(self):
        """Extract required edges from failed vehicle"""
        failed_trip = self.vehicle_trips[self.failed_vehicle]
        self.failed_edges = failed_trip['required_edges'].copy()
        
        print(f"FAILURE ANALYSIS")
        print(f"================")
        print(f"Failed Vehicle: {self.failed_vehicle}")
        print(f"Failed at position: Node {failed_trip['current_position']}")
        print(f"Original route: {failed_trip['route']}")
        print(f"Required edges to recover: {self.failed_edges}")
        print(f"Total edges to recover: {len(self.failed_edges)}")
        
        return self.failed_edges
    
    def calculate_search_radius_from_edge(self, edge):
        """Calculate search radius from an edge based on vehicle capacity"""
        # Start from edge endpoints and find all reachable nodes within capacity
        reachable_nodes = set()
        reachable_edges = set()
        
        for start_node in edge:
            if start_node not in self.network.nodes():
                continue
                
            # Use Dijkstra to find all nodes reachable within capacity
            try:
                distances = nx.single_source_dijkstra_path_length(
                    self.network, start_node, cutoff=self.capacity, weight='weight'
                )
                reachable_nodes.update(distances.keys())
                
                # Add edges within the reachable area
                for node in distances.keys():
                    for neighbor in self.network.neighbors(node):
                        if neighbor in distances:
                            reachable_edges.add(tuple(sorted([node, neighbor])))
                            
            except Exception as e:
                print(f"Error calculating reachability from {start_node}: {e}")
                continue
        
        return reachable_nodes, reachable_edges
    
    def create_search_zones(self):
        """Create search zones for each failed edge based on battery capacity"""
        print(f"\nSEARCH ZONE ANALYSIS")
        print(f"===================")
        
        for i, edge in enumerate(self.failed_edges):
            reachable_nodes, reachable_edges = self.calculate_search_radius_from_edge(edge)
            
            self.search_zones[i] = {
                'edge': edge,
                'reachable_nodes': reachable_nodes,
                'reachable_edges': reachable_edges,
                'zone_size': len(reachable_nodes)
            }
            
            print(f"Edge {edge}:")
            print(f"  Reachable nodes: {len(reachable_nodes)} (within capacity {self.capacity})")
            print(f"  Reachable edges: {len(reachable_edges)}")
            print(f"  Zone nodes: {sorted(reachable_nodes)}")
    
    def find_candidate_vehicles(self):
        """Find vehicles whose routes intersect with search zones"""
        print(f"\nCANDIDATE VEHICLE ANALYSIS")
        print(f"==========================")
        
        candidates = []
        
        for vehicle_id, trip in self.vehicle_trips.items():
            if vehicle_id == self.failed_vehicle or trip['status'] == 'failed':
                continue
                
            route = trip['route']
            route_nodes = set(route)
            current_cost = self.calculate_route_cost(route)
            spare_capacity = self.capacity - current_cost
            
            # Check intersection with search zones
            intersecting_zones = []
            for zone_id, zone in self.search_zones.items():
                intersection = route_nodes.intersection(zone['reachable_nodes'])
                if intersection:
                    intersecting_zones.append({
                        'zone_id': zone_id,
                        'failed_edge': zone['edge'],
                        'intersection_nodes': intersection,
                        'intersection_size': len(intersection)
                    })
            
            if intersecting_zones:
                candidate = {
                    'vehicle_id': vehicle_id,
                    'route': route,
                    'current_cost': current_cost,
                    'spare_capacity': spare_capacity,
                    'intersecting_zones': intersecting_zones,
                    'feasible': spare_capacity > 5  # Minimum spare capacity needed
                }
                candidates.append(candidate)
                
                print(f"Vehicle {vehicle_id}:")
                print(f"  Route: {route}")
                print(f"  Current cost: {current_cost:.1f}/{self.capacity}")
                print(f"  Spare capacity: {spare_capacity:.1f}")
                print(f"  Intersects with {len(intersecting_zones)} search zones")
                for zone_info in intersecting_zones:
                    print(f"    Zone {zone_info['zone_id']} (edge {zone_info['failed_edge']}): {zone_info['intersection_nodes']}")
                print(f"  Feasible: {'Yes' if candidate['feasible'] else 'No'}")
        
        self.candidate_vehicles = candidates
        return candidates
    
    def calculate_magnetic_field_score(self, edge, vehicle_route, insertion_position):
        """Simplified magnetic field score calculation"""
        # Basic implementation - can be enhanced with full magnetic field logic
        
        # Required edge influence (P)
        P = 1.0 if edge in self.failed_edges else 0.0
        
        # Position in route influence
        route_progress = insertion_position / len(vehicle_route) if len(vehicle_route) > 0 else 0
        
        # Depot influence (D) - simplified
        D = 0.3  # Base depot influence
        
        # Calculate score: early in route favors required edges, later favors depot
        w = route_progress
        score = (1 - w) * P + w * D
        
        return {
            'P': P,
            'D': D,
            'w': w,
            'score': score
        }
    
    def attempt_insertions(self):
        """Attempt to insert failed edges into candidate vehicles"""
        print(f"\nINSERTION ATTEMPTS")
        print(f"==================")
        
        for candidate in self.candidate_vehicles:
            if not candidate['feasible']:
                continue
                
            vehicle_id = candidate['vehicle_id']
            vehicle_route = candidate['route']
            
            print(f"\nVehicle {vehicle_id} insertion attempts:")
            
            insertion_results = []
            
            # Try to insert each failed edge that intersects with this vehicle's zones
            for zone_info in candidate['intersecting_zones']:
                failed_edge = zone_info['failed_edge']
                zone_id = zone_info['zone_id']
                
                # Find best insertion position using magnetic field scoring
                best_position = None
                best_score = -1
                
                for pos in range(len(vehicle_route)):
                    score_data = self.calculate_magnetic_field_score(failed_edge, vehicle_route, pos)
                    
                    if score_data['score'] > best_score:
                        best_score = score_data['score']
                        best_position = pos
                
                if best_position is not None and best_score > 0.3:  # Minimum threshold
                    insertion_results.append({
                        'edge': failed_edge,
                        'position': best_position,
                        'score': best_score,
                        'zone_id': zone_id,
                        'estimated_cost': 4  # Simplified cost estimate
                    })
                    
                    print(f"  ✓ Edge {failed_edge}: Position {best_position}, Score {best_score:.3f}")
                else:
                    print(f"  ✗ Edge {failed_edge}: No suitable insertion (best score: {best_score:.3f})")
            
            self.insertion_attempts[vehicle_id] = insertion_results
    
    def resolve_conflicts_and_execute(self):
        """Resolve conflicts and execute successful insertions"""
        print(f"\nCONFLICT RESOLUTION")
        print(f"===================")
        
        # Group insertion attempts by edge
        edge_attempts = defaultdict(list)
        for vehicle_id, attempts in self.insertion_attempts.items():
            for attempt in attempts:
                edge = tuple(sorted(attempt['edge']))
                edge_attempts[edge].append({
                    'vehicle_id': vehicle_id,
                    'attempt': attempt
                })
        
        # Resolve conflicts
        for edge, attempts in edge_attempts.items():
            if len(attempts) > 1:
                print(f"Conflict for edge {edge}:")
                print(f"  Competing vehicles: {[a['vehicle_id'] for a in attempts]}")
                
                # Choose vehicle with highest score
                best_attempt = max(attempts, key=lambda x: x['attempt']['score'])
                winner = best_attempt['vehicle_id']
                
                print(f"  Winner: Vehicle {winner} (score: {best_attempt['attempt']['score']:.3f})")
                
                # Record successful insertion
                self.successful_insertions.append({
                    'vehicle_id': winner,
                    'edge': list(edge),
                    'insertion_data': best_attempt['attempt']
                })
            else:
                # No conflict
                attempt = attempts[0]
                self.successful_insertions.append({
                    'vehicle_id': attempt['vehicle_id'],
                    'edge': list(edge),
                    'insertion_data': attempt['attempt']
                })
                print(f"No conflict for edge {edge} -> Vehicle {attempt['vehicle_id']}")
        
        # Apply insertions to vehicle routes
        print(f"\nAPPLYING INSERTIONS")
        print(f"===================")
        
        for insertion in self.successful_insertions:
            vehicle_id = insertion['vehicle_id']
            edge = insertion['edge']
            
            # Add edge to vehicle's required edges
            self.vehicle_trips[vehicle_id]['required_edges'].append(edge)
            
            print(f"✓ Vehicle {vehicle_id} now handles edge {edge}")
    
    def run_recovery_process(self):
        """Run the complete recovery process with step tracking"""
        print(f"SIMPLE FAILURE RECOVERY DEMONSTRATION")
        print(f"=====================================")
        
        # Step 1: Extract failed edges
        self.step_history.append("Extracting failed edges")
        self.extract_failed_edges()
        
        # Step 2: Create search zones
        self.step_history.append("Creating search zones")
        self.create_search_zones()
        
        # Step 3: Find candidate vehicles
        self.step_history.append("Finding candidate vehicles")
        self.find_candidate_vehicles()
        
        # Step 4: Attempt insertions
        self.step_history.append("Attempting insertions")
        self.attempt_insertions()
        
        # Step 5: Resolve conflicts and execute
        self.step_history.append("Resolving conflicts")
        self.resolve_conflicts_and_execute()
        
        return self.successful_insertions

def visualize_step_1_initial_state(recovery_system):
    """Visualize initial state with all vehicle routes"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Draw network
    nx.draw_networkx_edges(recovery_system.network, recovery_system.pos, ax=ax, 
                          edge_color='lightgray', width=1, alpha=0.5)
    
    # Draw all nodes
    nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos, ax=ax,
                          node_color='lightblue', node_size=300, alpha=0.7)
    
    # Highlight depots
    nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos, 
                          nodelist=DEPOTS, ax=ax, node_color='orange', 
                          node_size=500, alpha=0.9)
    
    # Draw vehicle routes
    colors = ['blue', 'green', 'red', 'purple']
    for vehicle_id, trip in recovery_system.vehicle_trips.items():
        route = trip['route']
        color = colors[vehicle_id % len(colors)]
        
        # Draw route edges
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            x1, y1 = recovery_system.pos[u]
            x2, y2 = recovery_system.pos[v]
            
            if trip['status'] == 'failed':
                ax.plot([x1, x2], [y1, y2], color='red', linewidth=3, alpha=0.8, linestyle='--')
            else:
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)
        
        # Mark current position
        current_pos = trip['current_position']
        x, y = recovery_system.pos[current_pos]
        
        if trip['status'] == 'failed':
            ax.scatter(x, y, c='red', s=400, marker='X', 
                      label=f'Vehicle {vehicle_id} (FAILED)' if vehicle_id == recovery_system.failed_vehicle else "")
        else:
            ax.scatter(x, y, c=color, s=300, marker='o', 
                      label=f'Vehicle {vehicle_id}' if vehicle_id != recovery_system.failed_vehicle else "")
    
    # Add node labels
    nx.draw_networkx_labels(recovery_system.network, recovery_system.pos, ax=ax, font_size=8)
    
    # Add edge weights
    edge_labels = nx.get_edge_attributes(recovery_system.network, 'weight')
    nx.draw_networkx_edge_labels(recovery_system.network, recovery_system.pos, 
                                edge_labels, ax=ax, font_size=6)
    
    ax.set_title('Initial State: Vehicle Routes and Positions', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_aspect('equal')
    plt.tight_layout()

def visualize_step_2_failure_analysis(recovery_system):
    """Visualize failure analysis with required edges highlighted"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left: Failed vehicle details
    ax1.set_title('Failed Vehicle Analysis', fontsize=14, fontweight='bold')
    
    # Draw network
    nx.draw_networkx_edges(recovery_system.network, recovery_system.pos, ax=ax1,
                          edge_color='lightgray', width=1, alpha=0.3)
    nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos, ax=ax1,
                          node_color='lightgray', node_size=200, alpha=0.5)
    
    # Highlight failed vehicle route
    failed_trip = recovery_system.vehicle_trips[recovery_system.failed_vehicle]
    route = failed_trip['route']
    
    for i in range(len(route) - 1):
        u, v = route[i], route[i+1]
        x1, y1 = recovery_system.pos[u]
        x2, y2 = recovery_system.pos[v]
        ax1.plot([x1, x2], [y1, y2], 'red', linewidth=3, alpha=0.7, linestyle='--')
    
    # Highlight required edges that need recovery
    for edge in recovery_system.failed_edges:
        u, v = edge
        x1, y1 = recovery_system.pos[u]
        x2, y2 = recovery_system.pos[v]
        ax1.plot([x1, x2], [y1, y2], 'darkred', linewidth=5, alpha=0.9)
    
    # Mark failure point
    failure_node = failed_trip['current_position']
    x, y = recovery_system.pos[failure_node]
    ax1.scatter(x, y, c='red', s=500, marker='X', label='Failure Point')
    
    # Mark depot
    depot = failed_trip['depot']
    x, y = recovery_system.pos[depot]
    ax1.scatter(x, y, c='orange', s=400, marker='s', label='Depot')
    
    nx.draw_networkx_labels(recovery_system.network, recovery_system.pos, ax=ax1, font_size=8)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Right: Summary table
    ax2.axis('off')
    ax2.set_title('Failure Details', fontsize=14, fontweight='bold')
    
    # Create summary text
    summary_text = f"""
    FAILURE SUMMARY
    ===============
    
    Failed Vehicle: {recovery_system.failed_vehicle}
    Failure Location: Node {failed_trip['current_position']}
    Original Route: {failed_trip['route']}
    Route Length: {len(failed_trip['route'])} nodes
    Original Depot: {failed_trip['depot']}
    
    REQUIRED EDGES TO RECOVER:
    """
    
    for i, edge in enumerate(recovery_system.failed_edges):
        u, v = edge
        if recovery_system.network.has_edge(u, v):
            weight = recovery_system.network[u][v]['weight']
            summary_text += f"\n    {i+1}. Edge {edge} (weight: {weight})"
        else:
            summary_text += f"\n    {i+1}. Edge {edge} (indirect)"
    
    summary_text += f"\n\n    Total edges to recover: {len(recovery_system.failed_edges)}"
    summary_text += f"\n    Vehicle capacity: {recovery_system.capacity}"
    
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()

def visualize_step_3_search_zones(recovery_system):
    """Visualize search zones for each failed edge"""
    failed_edges = recovery_system.failed_edges
    n_edges = len(failed_edges)
    
    fig, axes = plt.subplots(1, n_edges, figsize=(6*n_edges, 8))
    if n_edges == 1:
        axes = [axes]
    
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    for i, (edge_idx, zone) in enumerate(recovery_system.search_zones.items()):
        ax = axes[i]
        edge = zone['edge']
        reachable_nodes = zone['reachable_nodes']
        
        # Draw network
        nx.draw_networkx_edges(recovery_system.network, recovery_system.pos, ax=ax,
                              edge_color='lightgray', width=1, alpha=0.3)
        
        # Draw all nodes in light gray
        nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos, ax=ax,
                              node_color='lightgray', node_size=150, alpha=0.5)
        
        # Highlight reachable nodes in search zone
        nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos,
                              nodelist=list(reachable_nodes), ax=ax,
                              node_color=colors[i % len(colors)], node_size=250, alpha=0.7)
        
        # Highlight the failed edge
        u, v = edge
        x1, y1 = recovery_system.pos[u]
        x2, y2 = recovery_system.pos[v]
        ax.plot([x1, x2], [y1, y2], colors[i % len(colors)], linewidth=5, alpha=0.9)
        
        # Mark edge endpoints
        nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos,
                              nodelist=[u, v], ax=ax,
                              node_color='darkred', node_size=300, alpha=1.0)
        
        # Add labels
        nx.draw_networkx_labels(recovery_system.network, recovery_system.pos, ax=ax, font_size=8)
        
        ax.set_title(f'Search Zone for Edge {edge}\n{len(reachable_nodes)} reachable nodes', 
                    fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
    
    plt.suptitle(f'Search Zones (Capacity-Based Reachability)', fontsize=14, fontweight='bold')
    plt.tight_layout()

def visualize_step_4_candidate_analysis(recovery_system):
    """Visualize candidate vehicles and their intersections with search zones"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left: All vehicles with search zone overlays
    ax1.set_title('Candidate Vehicle Analysis', fontsize=14, fontweight='bold')
    
    # Draw network
    nx.draw_networkx_edges(recovery_system.network, recovery_system.pos, ax=ax1,
                          edge_color='lightgray', width=1, alpha=0.3)
    nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos, ax=ax1,
                          node_color='lightgray', node_size=150, alpha=0.5)
    
    # Draw search zones with transparency
    zone_colors = ['red', 'green', 'blue', 'orange']
    for zone_idx, zone in recovery_system.search_zones.items():
        reachable_nodes = zone['reachable_nodes']
        nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos,
                              nodelist=list(reachable_nodes), ax=ax1,
                              node_color=zone_colors[zone_idx % len(zone_colors)], 
                              node_size=300, alpha=0.3)
    
    # Draw vehicle routes
    vehicle_colors = ['blue', 'green', 'purple', 'brown']
    for candidate in recovery_system.candidate_vehicles:
        vehicle_id = candidate['vehicle_id']
        route = candidate['route']
        color = vehicle_colors[vehicle_id % len(vehicle_colors)]
        
        # Draw route
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            x1, y1 = recovery_system.pos[u]
            x2, y2 = recovery_system.pos[v]
            ax1.plot([x1, x2], [y1, y2], color=color, linewidth=3, alpha=0.8)
        
        # Mark vehicle position
        current_pos = recovery_system.vehicle_trips[vehicle_id]['current_position']
        x, y = recovery_system.pos[current_pos]
        ax1.scatter(x, y, c=color, s=400, marker='o', 
                   label=f'Vehicle {vehicle_id}' if candidate['feasible'] else f'Vehicle {vehicle_id} (No Capacity)')
    
    nx.draw_networkx_labels(recovery_system.network, recovery_system.pos, ax=ax1, font_size=8)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Right: Candidate summary
    ax2.axis('off')
    ax2.set_title('Candidate Analysis Results', fontsize=14, fontweight='bold')
    
    summary_text = "CANDIDATE VEHICLE ANALYSIS\n" + "="*30 + "\n\n"
    
    for candidate in recovery_system.candidate_vehicles:
        vehicle_id = candidate['vehicle_id']
        summary_text += f"Vehicle {vehicle_id}:\n"
        summary_text += f"  Route: {candidate['route']}\n"
        summary_text += f"  Current Cost: {candidate['current_cost']:.1f}/{recovery_system.capacity}\n"
        summary_text += f"  Spare Capacity: {candidate['spare_capacity']:.1f}\n"
        summary_text += f"  Feasible: {'✓' if candidate['feasible'] else '✗'}\n"
        summary_text += f"  Intersecting Zones: {len(candidate['intersecting_zones'])}\n"
        
        for zone_info in candidate['intersecting_zones']:
            summary_text += f"    - Zone {zone_info['zone_id']} (edge {zone_info['failed_edge']})\n"
            summary_text += f"      Intersection: {zone_info['intersection_nodes']}\n"
        summary_text += "\n"
    
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()

def visualize_step_5_insertion_attempts(recovery_system):
    """Visualize insertion attempts with magnetic field scores"""
    n_candidates = len([c for c in recovery_system.candidate_vehicles if c['feasible']])
    
    if n_candidates == 0:
        print("No feasible candidates for insertion visualization")
        return
    
    fig, axes = plt.subplots(1, n_candidates, figsize=(8*n_candidates, 8))
    if n_candidates == 1:
        axes = [axes]
    
    candidate_idx = 0
    for candidate in recovery_system.candidate_vehicles:
        if not candidate['feasible']:
            continue
            
        vehicle_id = candidate['vehicle_id']
        ax = axes[candidate_idx]
        
        # Draw network
        nx.draw_networkx_edges(recovery_system.network, recovery_system.pos, ax=ax,
                              edge_color='lightgray', width=1, alpha=0.3)
        nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos, ax=ax,
                              node_color='lightgray', node_size=150, alpha=0.5)
        
        # Draw vehicle route
        route = candidate['route']
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            x1, y1 = recovery_system.pos[u]
            x2, y2 = recovery_system.pos[v]
            ax.plot([x1, x2], [y1, y2], 'blue', linewidth=3, alpha=0.7)
        
        # Highlight insertion attempts
        if vehicle_id in recovery_system.insertion_attempts:
            attempts = recovery_system.insertion_attempts[vehicle_id]
            
            for attempt in attempts:
                edge = attempt['edge']
                position = attempt['position']
                score = attempt['score']
                
                # Highlight the edge being inserted
                u, v = edge
                x1, y1 = recovery_system.pos[u]
                x2, y2 = recovery_system.pos[v]
                ax.plot([x1, x2], [y1, y2], 'red', linewidth=5, alpha=0.8)
                
                # Mark insertion position in route
                if position < len(route):
                    insert_node = route[position]
                    x, y = recovery_system.pos[insert_node]
                    ax.scatter(x, y, c='green', s=300, marker='*', alpha=0.9)
                    
                    # Add score annotation
                    ax.annotate(f'Score: {score:.2f}', 
                               xy=(x, y), xytext=(x+0.5, y+0.5),
                               fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        # Mark vehicle position
        current_pos = recovery_system.vehicle_trips[vehicle_id]['current_position']
        x, y = recovery_system.pos[current_pos]
        ax.scatter(x, y, c='blue', s=400, marker='o', alpha=0.9)
        
        nx.draw_networkx_labels(recovery_system.network, recovery_system.pos, ax=ax, font_size=8)
        ax.set_title(f'Vehicle {vehicle_id} Insertion Attempts', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        
        candidate_idx += 1
    
    plt.suptitle('Magnetic Field Insertion Attempts', fontsize=14, fontweight='bold')
    plt.tight_layout()

def visualize_step_6_final_results(recovery_system):
    """Visualize final recovery results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Final state with successful insertions
    ax1.set_title('Final Recovery State', fontsize=14, fontweight='bold')
    
    # Draw network
    nx.draw_networkx_edges(recovery_system.network, recovery_system.pos, ax=ax1,
                          edge_color='lightgray', width=1, alpha=0.5)
    nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos, ax=ax1,
                          node_color='lightblue', node_size=200, alpha=0.7)
    
    # Draw vehicle routes with updates
    vehicle_colors = ['blue', 'green', 'purple', 'brown']
    for vehicle_id, trip in recovery_system.vehicle_trips.items():
        if trip['status'] == 'failed':
            continue
            
        route = trip['route']
        color = vehicle_colors[vehicle_id % len(vehicle_colors)]
        
        # Draw route
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            x1, y1 = recovery_system.pos[u]
            x2, y2 = recovery_system.pos[v]
            ax1.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)
    
    # Highlight successfully recovered edges
    for insertion in recovery_system.successful_insertions:
        edge = insertion['edge']
        vehicle_id = insertion['vehicle_id']
        color = vehicle_colors[vehicle_id % len(vehicle_colors)]
        
        u, v = edge
        x1, y1 = recovery_system.pos[u]
        x2, y2 = recovery_system.pos[v]
        ax1.plot([x1, x2], [y1, y2], color, linewidth=5, alpha=0.9)
        
        # Add annotation
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax1.annotate(f'V{vehicle_id}', xy=(mid_x, mid_y), 
                    fontsize=8, fontweight='bold', ha='center',
                    bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
    
    nx.draw_networkx_labels(recovery_system.network, recovery_system.pos, ax=ax1, font_size=8)
    ax1.set_aspect('equal')
    
    # 2. Recovery success metrics
    ax2.set_title('Recovery Success Analysis', fontsize=14, fontweight='bold')
    
    total_failed = len(recovery_system.failed_edges)
    recovered = len(recovery_system.successful_insertions)
    remaining = total_failed - recovered
    
    labels = ['Recovered', 'Remaining']
    sizes = [recovered, remaining]
    colors = ['green', 'red']
    explode = (0.1, 0)  # Explode recovered slice
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, explode=explode,
                                       autopct='%1.1f%%', startangle=90)
    
    # Add numbers to the chart
    ax2.text(0, -1.3, f'Total Failed Edges: {total_failed}\nRecovered: {recovered}\nRemaining: {remaining}',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    # 3. Vehicle capacity utilization
    ax3.set_title('Vehicle Capacity Utilization', fontsize=14, fontweight='bold')
    
    vehicle_ids = []
    original_costs = []
    final_costs = []
    
    for vehicle_id, trip in recovery_system.vehicle_trips.items():
        if trip['status'] == 'failed':
            continue
            
        original_cost = recovery_system.calculate_route_cost(trip['route'])
        
        # Calculate additional cost from insertions
        additional_cost = 0
        for insertion in recovery_system.successful_insertions:
            if insertion['vehicle_id'] == vehicle_id:
                additional_cost += insertion['insertion_data']['estimated_cost']
        
        final_cost = original_cost + additional_cost
        
        vehicle_ids.append(f'V{vehicle_id}')
        original_costs.append(original_cost)
        final_costs.append(final_cost)
    
    x = np.arange(len(vehicle_ids))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, original_costs, width, label='Original', color='lightblue', alpha=0.7)
    bars2 = ax3.bar(x + width/2, final_costs, width, label='After Recovery', color='darkblue', alpha=0.7)
    
    # Add capacity line
    ax3.axhline(y=recovery_system.capacity, color='red', linestyle='--', 
               label='Capacity Limit', alpha=0.7)
    
    ax3.set_ylabel('Route Cost')
    ax3.set_xticks(x)
    ax3.set_xticklabels(vehicle_ids)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Insertion details table
    ax4.axis('off')
    ax4.set_title('Insertion Details', fontsize=14, fontweight='bold')
    
    if recovery_system.successful_insertions:
        # Create table data
        table_data = [['Edge', 'Assigned Vehicle', 'Score', 'Position']]
        
        for insertion in recovery_system.successful_insertions:
            edge = insertion['edge']
            vehicle_id = insertion['vehicle_id']
            score = insertion['insertion_data']['score']
            position = insertion['insertion_data']['position']
            
            table_data.append([f'{edge}', f'Vehicle {vehicle_id}', f'{score:.3f}', f'{position}'])
        
        # Create table
        table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                         cellLoc='center', loc='center',
                         bbox=[0.1, 0.3, 0.8, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows alternately
        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F2F2F2')
    else:
        ax4.text(0.5, 0.5, 'No successful insertions', ha='center', va='center',
                transform=ax4.transAxes, fontsize=16, style='italic')
    
    plt.suptitle('Recovery Results Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()

def create_step_by_step_summary(recovery_system):
    """Create a summary visualization of all steps"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    ax.set_title('Complete Recovery Process Summary', fontsize=16, fontweight='bold')
    
    # Draw the final network state
    nx.draw_networkx_edges(recovery_system.network, recovery_system.pos, ax=ax,
                          edge_color='lightgray', width=1, alpha=0.5)
    nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos, ax=ax,
                          node_color='lightblue', node_size=300, alpha=0.7)
    
    # Highlight depots
    nx.draw_networkx_nodes(recovery_system.network, recovery_system.pos,
                          nodelist=DEPOTS, ax=ax, node_color='orange',
                          node_size=500, alpha=0.9)
    
    # Draw all active vehicle routes
    vehicle_colors = ['blue', 'green', 'purple', 'brown']
    for vehicle_id, trip in recovery_system.vehicle_trips.items():
        if trip['status'] == 'failed':
            continue
            
        route = trip['route']
        color = vehicle_colors[vehicle_id % len(vehicle_colors)]
        
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            x1, y1 = recovery_system.pos[u]
            x2, y2 = recovery_system.pos[v]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.6,
                   label=f'Vehicle {vehicle_id}' if i == 0 else "")
    
    # Highlight failed edges (original)
    for edge in recovery_system.failed_edges:
        u, v = edge
        x1, y1 = recovery_system.pos[u]
        x2, y2 = recovery_system.pos[v]
        ax.plot([x1, x2], [y1, y2], 'red', linewidth=4, alpha=0.5, linestyle='--')
    
    # Highlight successfully recovered edges
    for insertion in recovery_system.successful_insertions:
        edge = insertion['edge']
        vehicle_id = insertion['vehicle_id']
        color = vehicle_colors[vehicle_id % len(vehicle_colors)]
        
        u, v = edge
        x1, y1 = recovery_system.pos[u]
        x2, y2 = recovery_system.pos[v]
        ax.plot([x1, x2], [y1, y2], color, linewidth=5, alpha=0.9)
    
    # Mark failure point
    failed_trip = recovery_system.vehicle_trips[recovery_system.failed_vehicle]
    failure_node = failed_trip['current_position']
    x, y = recovery_system.pos[failure_node]
    ax.scatter(x, y, c='red', s=600, marker='X', label='Failure Point')
    
    nx.draw_networkx_labels(recovery_system.network, recovery_system.pos, ax=ax, font_size=10)
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', linewidth=4, linestyle='--', alpha=0.5, label='Failed Edges'),
        plt.Line2D([0], [0], color='black', linewidth=5, alpha=0.9, label='Recovered Edges'),
        plt.scatter([], [], c='red', s=600, marker='X', label='Failure Point'),
        plt.scatter([], [], c='orange', s=500, marker='o', label='Depots')
    ]
    
    # Add vehicle routes to legend
    for vehicle_id, trip in recovery_system.vehicle_trips.items():
        if trip['status'] != 'failed':
            color = vehicle_colors[vehicle_id % len(vehicle_colors)]
            legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=2, 
                                            label=f'Vehicle {vehicle_id}'))
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_aspect('equal')
    
    # Add summary statistics box
    total_failed = len(recovery_system.failed_edges)
    recovered = len(recovery_system.successful_insertions)
    recovery_rate = (recovered / total_failed * 100) if total_failed > 0 else 0
    
    summary_text = f"""
    RECOVERY SUMMARY
    ================
    
    Failed Vehicle: {recovery_system.failed_vehicle}
    Total Failed Edges: {total_failed}
    Successfully Recovered: {recovered}
    Recovery Rate: {recovery_rate:.1f}%
    
    Search Zones Created: {len(recovery_system.search_zones)}
    Candidate Vehicles: {len(recovery_system.candidate_vehicles)}
    Successful Insertions: {len(recovery_system.successful_insertions)}
    """
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()

def run_simple_failure_recovery_demo():
    """Run the complete simple failure recovery demonstration with step-by-step visualizations"""
    print("SIMPLE MULTI-VEHICLE FAILURE RECOVERY DEMONSTRATION")
    print("="*60)
    
    # Create recovery system
    recovery_system = SimpleFailureRecovery(
        network=NETWORK,
        node_positions=NODE_POSITIONS,
        vehicle_trips=VEHICLE_TRIPS,
        failed_vehicle=FAILED_VEHICLE,
        capacity=VEHICLE_CAPACITY
    )
    
    print(f"Network: {len(NETWORK.nodes())} nodes, {len(NETWORK.edges())} edges")
    print(f"Vehicles: 4 total, 1 failed")
    print(f"Depots: {DEPOTS}")
    print(f"Vehicle capacity: {VEHICLE_CAPACITY}")
    
    # Step 1: Show initial state
    print(f"\nStep 1: Visualizing Initial State")
    visualize_step_1_initial_state(recovery_system)
    plt.show()
    
    # Step 2: Analyze failure
    print(f"\nStep 2: Analyzing Failure")
    recovery_system.extract_failed_edges()
    visualize_step_2_failure_analysis(recovery_system)
    plt.show()
    
    # Step 3: Create search zones
    print(f"\nStep 3: Creating Search Zones")
    recovery_system.create_search_zones()
    visualize_step_3_search_zones(recovery_system)
    plt.show()
    
    # Step 4: Find candidates
    print(f"\nStep 4: Finding Candidate Vehicles")
    recovery_system.find_candidate_vehicles()
    visualize_step_4_candidate_analysis(recovery_system)
    plt.show()
    
    # Step 5: Attempt insertions
    print(f"\nStep 5: Attempting Insertions")
    recovery_system.attempt_insertions()
    visualize_step_5_insertion_attempts(recovery_system)
    plt.show()
    
    # Step 6: Execute recovery
    print(f"\nStep 6: Executing Recovery")
    recovery_system.resolve_conflicts_and_execute()
    visualize_step_6_final_results(recovery_system)
    plt.show()
    
    # Step 7: Complete summary
    print(f"\nStep 7: Complete Process Summary")
    create_step_by_step_summary(recovery_system)
    plt.show()
    
    # Generate final report
    print(f"\n{'='*60}")
    print("FINAL RECOVERY REPORT")
    print(f"{'='*60}")
    
    total_failed = len(recovery_system.failed_edges)
    recovered = len(recovery_system.successful_insertions)
    recovery_rate = (recovered / total_failed * 100) if total_failed > 0 else 0
    
    print(f"Recovery Performance:")
    print(f"  Total failed edges: {total_failed}")
    print(f"  Successfully recovered: {recovered}")
    print(f"  Recovery rate: {recovery_rate:.1f}%")
    print(f"  Search zones created: {len(recovery_system.search_zones)}")
    print(f"  Candidate vehicles found: {len(recovery_system.candidate_vehicles)}")
    
    print(f"\nRecovery Details:")
    for insertion in recovery_system.successful_insertions:
        edge = insertion['edge']
        vehicle = insertion['vehicle_id']
        score = insertion['insertion_data']['score']
        print(f"  Edge {edge} -> Vehicle {vehicle} (score: {score:.3f})")
    
    print(f"\nComputational Efficiency:")
    total_vehicles = len(VEHICLE_TRIPS) - 1  # Exclude failed vehicle
    candidates_found = len(recovery_system.candidate_vehicles)
    efficiency = (candidates_found / total_vehicles * 100) if total_vehicles > 0 else 0
    print(f"  Search radius eliminated {total_vehicles - candidates_found} vehicles")
    print(f"  Computational focus: {efficiency:.1f}% of vehicles considered")
    
    return recovery_system

if __name__ == "__main__":
    recovery_system = run_simple_failure_recovery_demo()
    
    print(f"\n{'='*60}")
    print("DEMONSTRATION COMPLETE")
    print(f"{'='*60}")
    print("\nKey Features Demonstrated:")
    print("✓ Simple 4-vehicle scenario with clear visualization")
    print("✓ Capacity-based search radius for each failed edge")
    print("✓ Step-by-step process with detailed plots")
    print("✓ Magnetic field scoring for insertion decisions")
    print("✓ Conflict resolution between competing vehicles")
    print("✓ Comprehensive recovery analysis and reporting")