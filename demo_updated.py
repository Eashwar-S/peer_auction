"""
Magnetic Field Vehicle Routing Algorithm Demonstration
====================================================
This demonstrates how the magnetic field approach works with detailed visualizations
showing edge attractions and depot influences, including proper scoring with trip length weighting.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from math import exp, sqrt
import seaborn as sns
from itertools import permutations

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
        (4, 5, 2), (4, 6, 3),
    ])
    return G

# Simple test instance
SIMPLE_GRAPH = create_simple_graph()
START_DEPOT = 0
END_DEPOT = 5
VEHICLE_CAPACITY = 15
REQUIRED_EDGES = [(1, 2), (3, 4)]  # Only 3 required edges for clarity
FAILED_EDGES = [(2, 3), (4, 6), (3,5)]  # Only 2 failed edges to handle

class MagneticFieldRouter:
    """
    Magnetic Field Vehicle Routing Algorithm with detailed visualization and proper scoring
    """
    
    def __init__(self, graph, start_depot, end_depot, capacity, alpha=0.7, gamma=0.5):
        self.graph = graph
        self.start_depot = start_depot
        self.end_depot = end_depot
        self.capacity = capacity
        self.alpha = alpha  # Required edge influence decay
        self.gamma = gamma  # Depot influence decay
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
        """
        Calculate the magnetic field score for an edge based on current trip progress
        This implements the scoring function from the original algorithm
        """
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
        
        # Apply bonus for new required edges
        # if is_new_required:
        #     score = S + 2.0 + w * 2.0
        # else:
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
    
    def find_route_with_magnetic_scoring(self, required_edges, verbose=False):
        """
        Find route using proper magnetic field scoring that considers trip progress
        """
        print(f"\nFinding route with magnetic field scoring...")
        print(f"Required edges: {required_edges}")
        print(f"Vehicle capacity: {self.capacity}")
        
        # Start building route from start depot
        current_route = [self.start_depot]
        current_length = 0
        visited_edges = set()
        required_covered = set()
        
        # Track scoring details for visualization
        scoring_history = []
        
        while len(required_covered) < len(required_edges) or current_route[-1] != self.end_depot:
            current_node = current_route[-1]
            candidates = []
            
            if verbose:
                print(f"\nAt node {current_node}, current length: {current_length:.2f}")
            
            # Get all possible next edges
            for neighbor in self.graph.neighbors(current_node):
                edge = (current_node, neighbor)
                edge_sorted = tuple(sorted(edge))
                
                # Skip if already visited this edge
                if edge_sorted in visited_edges:
                    continue
                
                edge_weight = self.graph[current_node][neighbor]['weight']
                
                # Check capacity constraint
                if current_length + edge_weight > self.capacity:
                    continue
                
                # Check if this is a new required edge
                is_new_required = (edge_sorted in [tuple(sorted(req)) for req in required_edges] 
                                 and edge_sorted not in required_covered)
                
                # Calculate magnetic field score
                score_data = self.calculate_edge_score(edge, required_edges, current_length, is_new_required)
                
                candidates.append({
                    'edge': edge,
                    'neighbor': neighbor,
                    'is_new_required': is_new_required,
                    'score_data': score_data
                })
                
                if verbose:
                    print(f"  Edge {edge}: P={score_data['P']:.3f}, D={score_data['D']:.3f}, "
                          f"w={score_data['w']:.3f}, Score={score_data['final_score']:.3f}")
            
            if not candidates:
                if verbose:
                    print("No valid candidates found!")
                break
            
            # Sort candidates by score (higher is better), then by normalized edge weight (lower is better)
            candidates.sort(key=lambda x: (x['score_data']['final_score'], 
                                         -x['score_data']['normalized_weight']), reverse=True)
            
            # Select best candidate
            best = candidates[0]
            selected_edge = best['edge']
            selected_neighbor = best['neighbor']
            
            # Update route
            current_route.append(selected_neighbor)
            current_length += best['score_data']['edge_weight']
            visited_edges.add(tuple(sorted(selected_edge)))
            
            if best['is_new_required']:
                required_covered.add(tuple(sorted(selected_edge)))
            
            # Record scoring history
            scoring_history.append({
                'step': len(current_route) - 1,
                'edge': selected_edge,
                'score_data': best['score_data'],
                'is_new_required': best['is_new_required'],
                'current_length': current_length
            })
            
            if verbose:
                print(f"  Selected: {selected_edge} (score: {best['score_data']['final_score']:.3f})")
                print(f"  New route: {current_route}")
                print(f"  Required covered: {len(required_covered)}/{len(required_edges)}")
        
        # If we haven't reached the end depot, try to get there
        if current_route[-1] != self.end_depot:
            try:
                path_to_end = nx.shortest_path(self.graph, current_route[-1], self.end_depot, weight='weight')
                additional_length = nx.shortest_path_length(self.graph, current_route[-1], self.end_depot, weight='weight')
                
                if current_length + additional_length <= self.capacity:
                    current_route.extend(path_to_end[1:])  # Exclude the starting node to avoid duplication
                    current_length += additional_length
                else:
                    if verbose:
                        print("Cannot reach end depot within capacity constraint")
                    return None, float('inf'), scoring_history
            except nx.NetworkXNoPath:
                if verbose:
                    print("No path to end depot")
                return None, float('inf'), scoring_history
        
        return current_route, current_length, scoring_history
    
    def calculate_total_attraction(self, required_edges):
        """Calculate total magnetic field attraction for each edge"""
        req_influences = self.calculate_required_edge_influence(required_edges)
        depot_influences = self.calculate_depot_influence()
        
        total_attractions = {}
        
        for edge in self.graph.edges():
            # Sum of required edge influences
            req_sum = sum(req_influences[edge].values())
            
            # Depot influence
            depot_inf = depot_influences[edge]
            
            # Total attraction (weighted combination)
            total_attraction = 0.7 * req_sum + 0.3 * depot_inf
            
            total_attractions[edge] = {
                'required_influence': req_sum,
                'depot_influence': depot_inf,
                'total_attraction': total_attraction
            }
            total_attractions[edge[::-1]] = total_attractions[edge]
        
        return total_attractions, req_influences, depot_influences

def visualize_scoring_progression(router, required_edges, scoring_history):
    """Visualize how the scoring changes as the trip progresses"""
    if not scoring_history:
        print("No scoring history available")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    steps = [s['step'] for s in scoring_history]
    w_values = [s['score_data']['w'] for s in scoring_history]
    P_values = [s['score_data']['P'] for s in scoring_history]
    D_values = [s['score_data']['D'] for s in scoring_history]
    scores = [s['score_data']['final_score'] for s in scoring_history]
    
    # Plot w (trip progress) over time
    ax1.plot(steps, w_values, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Route Step')
    ax1.set_ylabel('w (Trip Progress)')
    ax1.set_title('Trip Progress Weight (w) Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot P and D influences
    ax2.plot(steps, P_values, 'g-o', label='P (Required Edge Influence)', linewidth=2, markersize=6)
    ax2.plot(steps, D_values, 'r-o', label='D (Depot Influence)', linewidth=2, markersize=6)
    ax2.set_xlabel('Route Step')
    ax2.set_ylabel('Influence Value')
    ax2.set_title('Required Edge vs Depot Influences')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot final scores
    ax3.plot(steps, scores, 'm-o', linewidth=2, markersize=6)
    ax3.set_xlabel('Route Step')
    ax3.set_ylabel('Final Score')
    ax3.set_title('Final Magnetic Field Scores')
    ax3.grid(True, alpha=0.3)
    
    # Show the scoring formula progression
    combined_scores = [(1-w)*P + w*D for w, P, D in zip(w_values, P_values, D_values)]
    ax4.plot(steps, combined_scores, 'c-o', label='(1-w)*P + w*D', linewidth=2, markersize=6)
    ax4.plot(steps, scores, 'm-o', label='Final Score', linewidth=2, markersize=6)
    ax4.set_xlabel('Route Step')
    ax4.set_ylabel('Score Value')
    ax4.set_title('Base vs Final Scores')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Magnetic Field Scoring Progression Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

def visualize_route_with_scores(router, route, scoring_history):
    """Visualize the route with scoring information"""
    plt.figure(figsize=(14, 10))
    
    # Draw all edges in light gray
    nx.draw_networkx_edges(router.graph, router.pos, edge_color='lightgray', 
                          width=1, alpha=0.4)
    
    # Draw route edges with scores as colors
    if route and scoring_history:
        # Create colormap for scores
        scores = [s['score_data']['final_score'] for s in scoring_history]
        min_score, max_score = min(scores), max(scores)
        
        for i, score_info in enumerate(scoring_history):
            edge = score_info['edge']
            score = score_info['score_data']['final_score']
            
            # Normalize score for color
            if max_score > min_score:
                norm_score = (score - min_score) / (max_score - min_score)
            else:
                norm_score = 0.5
            
            color = plt.cm.plasma(norm_score)
            width = 3 + 4 * norm_score
            
            # Draw edge
            nx.draw_networkx_edges(router.graph, router.pos, edgelist=[edge],
                                  edge_color=[color], width=width, alpha=0.8)
            
            # Add step number and score
            u, v = edge
            x = (router.pos[u][0] + router.pos[v][0]) / 2
            y = (router.pos[u][1] + router.pos[v][1]) / 2
            plt.text(x, y, f'{i+1}\n{score:.2f}', fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Draw all nodes
    nx.draw_networkx_nodes(router.graph, router.pos, node_color='lightblue',
                          node_size=500, alpha=0.8)
    
    # Highlight route nodes
    if route:
        nx.draw_networkx_nodes(router.graph, router.pos, nodelist=route,
                              node_color='red', node_size=600, alpha=0.9)
    
    # Highlight depots
    depot_nodes = [router.start_depot, router.end_depot]
    nx.draw_networkx_nodes(router.graph, router.pos, nodelist=depot_nodes,
                          node_color='orange', node_size=700, alpha=1.0)
    
    # Highlight required edges
    req_edge_list = [(u, v) for u, v in REQUIRED_EDGES]
    nx.draw_networkx_edges(router.graph, router.pos, edgelist=req_edge_list,
                          edge_color='green', width=4, alpha=0.8, style='dashed')
    
    # Add labels
    nx.draw_networkx_labels(router.graph, router.pos, font_size=12, font_weight='bold')
    
    # Add colorbar for scores
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                              norm=plt.Normalize(vmin=min(scores) if scores else 0, 
                                               vmax=max(scores) if scores else 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, shrink=0.6)
    cbar.set_label('Magnetic Field Score', rotation=270, labelpad=20)
    
    plt.title('Route Construction with Magnetic Field Scores', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

def visualize_graph_structure(router):
    """Visualize the basic graph structure"""
    plt.figure(figsize=(10, 8))
    
    # Draw all edges
    nx.draw_networkx_edges(router.graph, router.pos, edge_color='lightgray', width=2, alpha=0.6)
    
    # Draw nodes
    nx.draw_networkx_nodes(router.graph, router.pos, node_color='lightblue', 
                          node_size=600, alpha=0.8)
    
    # Highlight depots
    depot_nodes = [router.start_depot, router.end_depot]
    nx.draw_networkx_nodes(router.graph, router.pos, nodelist=depot_nodes,
                          node_color='orange', node_size=800, alpha=0.9)
    
    # Draw required edges
    req_edges = [(u, v) for u, v in REQUIRED_EDGES]
    nx.draw_networkx_edges(router.graph, router.pos, edgelist=req_edges,
                          edge_color='green', width=4, alpha=0.8)
    
    # Draw failed edges
    failed_edges = [(u, v) for u, v in FAILED_EDGES]
    nx.draw_networkx_edges(router.graph, router.pos, edgelist=failed_edges,
                          edge_color='red', width=4, alpha=0.8, style='dashed')
    
    # Add labels
    nx.draw_networkx_labels(router.graph, router.pos, font_size=14, font_weight='bold')
    
    # Add edge weights
    edge_labels = nx.get_edge_attributes(router.graph, 'weight')
    nx.draw_networkx_edge_labels(router.graph, router.pos, edge_labels, font_size=10)
    
    plt.title('Problem Instance: Simple Graph for Magnetic Field Demo', 
              fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        patches.Patch(color='orange', label='Depots'),
        patches.Patch(color='green', label='Required Edges'),
        patches.Patch(color='red', label='Failed Edges (to handle)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
    plt.tight_layout()

def run_complete_magnetic_field_demo():
    """Run complete demonstration of magnetic field algorithm with proper scoring"""
    print("Magnetic Field Vehicle Routing Algorithm Demonstration")
    print("=" * 60)
    
    # Create router
    router = MagneticFieldRouter(SIMPLE_GRAPH, START_DEPOT, END_DEPOT, VEHICLE_CAPACITY)
    
    print(f"Graph: {len(SIMPLE_GRAPH.nodes())} nodes, {len(SIMPLE_GRAPH.edges())} edges")
    print(f"Start depot: {START_DEPOT}, End depot: {END_DEPOT}")
    print(f"Vehicle capacity: {VEHICLE_CAPACITY}")
    print(f"Required edges: {REQUIRED_EDGES}")
    print(f"Failed edges: {FAILED_EDGES}")
    print(f"Alpha (required edge decay): {router.alpha}")
    print(f"Gamma (depot influence decay): {router.gamma}")
    print()
    
    # 1. Show problem instance
    print("1. Problem Instance Visualization")
    visualize_graph_structure(router)
    plt.show()
    
    # 2. Find route using magnetic field scoring
    print("2. Route Construction with Magnetic Field Scoring")
    route, cost, scoring_history = router.find_route_with_magnetic_scoring(REQUIRED_EDGES, verbose=False)
    
    if route:
        print(f"\nFound route: {route}")
        print(f"Total cost: {cost:.2f}")
        print(f"Capacity utilization: {cost:.2f}/{VEHICLE_CAPACITY} ({100*cost/VEHICLE_CAPACITY:.1f}%)")
        
        # Show detailed scoring progression
        print("\nScoring Progression:")
        print("-" * 60)
        for i, step in enumerate(scoring_history):
            print(f"Step {step['step']+1}: Edge {step['edge']}")
            print(f"  P={step['score_data']['P']:.3f}, D={step['score_data']['D']:.3f}, "
                  f"w={step['score_data']['w']:.3f}")
            print(f"  Score={(1-step['score_data']['w'])*step['score_data']['P'] + step['score_data']['w']*step['score_data']['D']:.3f}, "
                  f"Final={step['score_data']['final_score']:.3f}")
            print(f"  New Required: {step['is_new_required']}")
            print()
    else:
        print("No feasible route found!")
    
    # 3. Visualize scoring progression
    # if scoring_history:
    #     print("3. Scoring Progression Analysis")
    #     visualize_scoring_progression(router, REQUIRED_EDGES, scoring_history)
    #     plt.show()
        
    #     # 4. Visualize final route with scores
    #     print("4. Final Route with Magnetic Field Scores")
    #     visualize_route_with_scores(router, route, scoring_history)
    #     plt.show()

if __name__ == "__main__":
    run_complete_magnetic_field_demo()