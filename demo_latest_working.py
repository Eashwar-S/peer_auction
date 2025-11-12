"""
Magnetic Field Route Modification Algorithm Demonstration
========================================================
This demonstrates the magnetic field approach for route construction where you start with 
an initial route as input and need to construct a new route that includes new required edges.
Uses the same logic as demo_updated_v2.py but with initial route input and new required edges.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from math import exp, sqrt
import seaborn as sns
from itertools import permutations

# Use the same simple graph from demo_updated_v2.py
def create_simple_graph():
    """Create a simple graph for magnetic field demonstration"""
    G = nx.Graph()
    # Simple 6-node graph
    G.add_weighted_edges_from([
        (0, 1, 3), (0, 2, 4), (0, 3, 5),
        (1, 2, 2), (1, 4, 4), 
        (2, 3, 3), (2, 4, 2),
        (3, 4, 3), (3, 5, 4),
        (4, 5, 2), (4, 6, 2)
    ])
    return G

# Same test instance as demo_updated_v2.py
SIMPLE_GRAPH = create_simple_graph()
START_DEPOT = 0
END_DEPOT = 5
VEHICLE_CAPACITY = 22  # Slightly increased to allow for new edges

# Route modification scenario
INITIAL_REQUIRED_EDGES = [(1, 2), (3, 4)]  # Original required edges from demo_updated_v2.py
INITIAL_ROUTE = [0, 1, 2, 4, 3, 5]  # Initial route provided as input

# New required edges that need to be added (2 new edges instead of 1)
NEW_REQUIRED_EDGES = [ (4, 5), (4, 6), (5, 3)]  # 2 new edges to traverse

# Combined required edges (what we want to achieve)
COMBINED_REQUIRED_EDGES = INITIAL_REQUIRED_EDGES + NEW_REQUIRED_EDGES

class MagneticFieldRouteModifier:
    """
    Magnetic Field Route Construction Algorithm with Initial Route Input
    Constructs route from scratch but considers initial route and adds new required edges sequentially
    """
    
    def __init__(self, graph, start_depot, end_depot, capacity, alpha=0.7, gamma=0.5):
        self.graph = graph
        self.start_depot = start_depot
        self.end_depot = end_depot
        self.capacity = capacity
        self.alpha = alpha  # Required edge influence decay (same as demo_updated_v2.py)
        self.gamma = gamma  # Depot influence decay (same as demo_updated_v2.py)
        self.pos = self._create_layout()
        self.max_edge_weight = max(d['weight'] for u, v, d in graph.edges(data=True))
        
    def _create_layout(self):
        """Create consistent layout for visualizations (same as demo_updated_v2.py)"""
        return nx.spring_layout(self.graph, seed=42, k=2, iterations=50)
    
    def calculate_distances(self):
        """Calculate all shortest path distances"""
        return dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='weight'))
    
    def calculate_required_edge_influence(self, required_edges):
        """Calculate influence of required edges on all other edges (same as demo_updated_v2.py)"""
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
        """Calculate influence of depots on all edges (same as demo_updated_v2.py)"""
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
        Calculate the magnetic field score for an edge (same as demo_updated_v2.py)
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
        if is_new_required:
            score = S + 2.0
        else:
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
    
    def construct_route_with_new_edges(self, initial_route, initial_required, new_required, verbose=False):
        """
        Construct route from scratch using magnetic field approach, considering initial route
        and trying to add new required edges sequentially
        """
        print(f"\nConstructing route with magnetic field approach...")
        print(f"Initial route (for reference): {initial_route}")
        print(f"Initial required edges: {initial_required}")
        print(f"New required edges to add: {new_required}")
        print(f"Vehicle capacity: {self.capacity}")
        
        # Try adding new required edges one by one to see which ones can fit
        successful_new_edges = []
        current_required_set = initial_required.copy()
        
        for new_edge in new_required:
            # Test if we can construct a route with this additional edge
            test_required_set = current_required_set + [new_edge]
            test_route, test_cost, test_history = self.find_route_with_magnetic_scoring(test_required_set, verbose=False)
            
            if test_route and test_cost <= self.capacity:
                # This edge can be added successfully
                successful_new_edges.append(new_edge)
                current_required_set.append(new_edge)
                print(f"✓ New edge {new_edge} can be added (test cost: {test_cost:.2f})")
            else:
                print(f"✗ New edge {new_edge} cannot be added within capacity")
        
        # Now construct the final route with all successful edges
        final_required_edges = initial_required + successful_new_edges
        print(f"Final required edges set: {final_required_edges}")
        
        route, cost, scoring_history = self.find_route_with_magnetic_scoring(final_required_edges, verbose)
        
        return route, cost, scoring_history, successful_new_edges
    
    def find_route_with_magnetic_scoring(self, required_edges, verbose=False):
        """
        Find route using proper magnetic field scoring (same logic as demo_updated_v2.py)
        """
        if verbose:
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
            
            # Record scoring history - ALWAYS record each step
            scoring_history.append({
                'step': len(current_route) - 1,
                'edge': selected_edge,
                'score_data': best['score_data'],
                'is_new_required': best['is_new_required'],
                'current_length': current_length,
                'route_so_far': current_route.copy()  # Keep track of route at each step
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
                    # Add each step in the path to end depot to scoring history
                    for i in range(1, len(path_to_end)):
                        u, v = path_to_end[i-1], path_to_end[i]
                        edge_weight = self.graph[u][v]['weight']
                        current_route.append(v)
                        current_length += edge_weight
                        
                        # Create a dummy score for path to depot
                        dummy_score_data = {
                            'P': 0.0,
                            'D': 1.0,  # High depot influence
                            'w': current_length / self.capacity,
                            'S': 1.0,
                            'final_score': 1.0,
                            'edge_weight': edge_weight,
                            'normalized_weight': edge_weight / self.max_edge_weight
                        }
                        
                        scoring_history.append({
                            'step': len(current_route) - 1,
                            'edge': (u, v),
                            'score_data': dummy_score_data,
                            'is_new_required': False,
                            'current_length': current_length,
                            'route_so_far': current_route.copy(),
                            'is_path_to_depot': True  # Flag to indicate this is path to depot
                        })
                        
                        if verbose:
                            print(f"  Path to depot: {(u, v)} (cost: {edge_weight:.2f})")
                else:
                    if verbose:
                        print("Cannot reach end depot within capacity constraint")
                    return None, float('inf'), scoring_history
            except nx.NetworkXNoPath:
                if verbose:
                    print("No path to end depot")
                return None, float('inf'), scoring_history
        
        return current_route, current_length, scoring_history

def visualize_dynamic_route_construction(modifier, initial_route, final_route, scoring_history, successful_new_edges):
    """
    Dynamic visualization showing route construction step by step with arrows in black
    Now shows ALL steps including path to depot
    """
    if not final_route or not scoring_history:
        print("No route found or no scoring history available")
        return
    
    print(f"Creating dynamic visualization for route construction...")
    print(f"Initial route (reference): {initial_route}")
    print(f"Final route: {final_route}")
    print(f"Successfully added new edges: {successful_new_edges}")
    print(f"Total steps in scoring history: {len(scoring_history)}")
    print(f"Final route length: {len(final_route)} nodes")
    
    # Determine which edges are being used in the required set
    final_required_edges = INITIAL_REQUIRED_EDGES + successful_new_edges
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))
    
    for step_idx in range(len(scoring_history)):
        fig.clear()
        
        # Create subplots for this step
        ax_graph = plt.subplot(1, 2, 1)
        ax_heatmap = plt.subplot(1, 2, 2)
        
        current_step = scoring_history[step_idx]
        
        # Use the route_so_far from scoring history to get exact route at this step
        if 'route_so_far' in current_step:
            current_route_nodes = current_step['route_so_far']
        else:
            # Fallback to the old method
            current_route_nodes = final_route[:step_idx + 2]
        
        current_length = current_step['current_length']
        
        # Check if this is a path-to-depot step
        is_depot_step = current_step.get('is_path_to_depot', False)
        
        # === LEFT SIDE: Route Construction ===
        plt.sca(ax_graph)
        
        # Draw all edges in light gray
        nx.draw_networkx_edges(modifier.graph, modifier.pos, edge_color='lightgray', 
                              width=1, alpha=0.4)
        
        # Draw completed route edges with BLACK ARROWS
        for i in range(len(current_route_nodes) - 1):
            u, v = current_route_nodes[i], current_route_nodes[i + 1]
            x1, y1 = modifier.pos[u]
            x2, y2 = modifier.pos[v]
            
            # Draw arrow from u to v in BLACK
            # Make the current step edge thicker
            if i == len(current_route_nodes) - 2:  # Current (last) edge
                linewidth = 5
                alpha = 1.0
            else:
                linewidth = 3
                alpha = 0.7
            
            ax_graph.annotate('', xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle='->', color='black', lw=linewidth, alpha=alpha))
            
            # Add step numbers
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            plt.text(x, y, str(i+1), fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.8))
        
        # Draw all nodes
        nx.draw_networkx_nodes(modifier.graph, modifier.pos, node_color='lightblue',
                              node_size=500, alpha=0.8)
        
        # Highlight current position
        current_node = current_route_nodes[-1]
        nx.draw_networkx_nodes(modifier.graph, modifier.pos, nodelist=[current_node],
                              node_color='red', node_size=700, alpha=1.0)
        
        # Highlight depots
        depot_nodes = [modifier.start_depot, modifier.end_depot]
        nx.draw_networkx_nodes(modifier.graph, modifier.pos, nodelist=depot_nodes,
                              node_color='orange', node_size=600, alpha=0.9)
        
        # Highlight required edges
        # Initial required edges in RED
        nx.draw_networkx_edges(modifier.graph, modifier.pos, edgelist=INITIAL_REQUIRED_EDGES,
                              edge_color='red', width=3, alpha=0.6, style='dashed')
        
        # New required edges in GREEN
        nx.draw_networkx_edges(modifier.graph, modifier.pos, edgelist=successful_new_edges,
                              edge_color='green', width=3, alpha=0.6, style='dashed')
        
        # Add labels
        nx.draw_networkx_labels(modifier.graph, modifier.pos, font_size=12, font_weight='bold')
        
        # Update title to show if this is a depot step
        if is_depot_step:
            step_type = " (Path to Depot)"
        else:
            step_type = ""
        
        ax_graph.set_title(f'Step {step_idx + 1}/{len(scoring_history)}: Route Construction{step_type}\nCurrent Node: {current_node}, Length: {current_length:.2f}', 
                          fontsize=14, fontweight='bold')
        ax_graph.axis('off')
        
        # === RIGHT SIDE: Score Heatmap ===
        plt.sca(ax_heatmap)
        
        if not is_depot_step:
            # For regular steps, show scores from the DECISION NODE (previous node)
            # The heatmap should show the scores that led to selecting the current edge
            if len(current_route_nodes) >= 2:
                decision_node = current_route_nodes[-2]  # The node we decided from
                selected_edge = current_step['edge']
                selected_neighbor = current_route_nodes[-1]  # Where we went
            else:
                decision_node = modifier.start_depot
                selected_edge = current_step['edge']
                selected_neighbor = current_route_nodes[-1]
            
            edge_scores = {}
            visited_edges = set()
            
            # Get visited edges up to the PREVIOUS step (before this decision)
            for i in range(len(current_route_nodes) - 2):  # Exclude the edge we just selected
                edge_sorted = tuple(sorted([current_route_nodes[i], current_route_nodes[i + 1]]))
                visited_edges.add(edge_sorted)
            
            # Calculate the decision length (before adding the current edge)
            decision_length = current_length - current_step['score_data']['edge_weight']
            
            # Calculate scores for all edges from the DECISION NODE at the time of decision
            for neighbor in modifier.graph.neighbors(decision_node):
                edge = (decision_node, neighbor)
                edge_sorted = tuple(sorted(edge))
                
                # Skip if this edge was already visited before this decision
                if edge_sorted in visited_edges:
                    continue
                    
                edge_weight = modifier.graph[decision_node][neighbor]['weight']
                
                # Check capacity constraint at decision time
                if decision_length + edge_weight > modifier.capacity:
                    continue
                    
                # Check if this was a required edge at decision time
                is_new_required = (edge_sorted in [tuple(sorted(req)) for req in final_required_edges])
                
                # Calculate the score as it was at decision time
                score_data = modifier.calculate_edge_score(edge, final_required_edges, decision_length, is_new_required)
                edge_scores[edge] = score_data['final_score']
            
            # Create adjacency matrix for heatmap
            nodes = list(modifier.graph.nodes())
            n_nodes = len(nodes)
            score_matrix = np.zeros((n_nodes, n_nodes))
            
            # Fill matrix with scores
            for edge, score in edge_scores.items():
                u, v = edge
                u_idx = nodes.index(u)
                v_idx = nodes.index(v)
                score_matrix[u_idx, v_idx] = score
                score_matrix[v_idx, u_idx] = score  # Symmetric
            
            # Create heatmap
            mask = score_matrix == 0
            
            # Create custom annotations that highlight the selected edge
            annot_matrix = np.zeros((n_nodes, n_nodes), dtype=object)
            for edge, score in edge_scores.items():
                u, v = edge
                u_idx = nodes.index(u)
                v_idx = nodes.index(v)
                
                # Check if this is the selected edge
                if (edge == selected_edge or edge == selected_edge[::-1]):
                    annot_matrix[u_idx, v_idx] = f"★{score:.2f}★"  # Highlight selected edge
                    annot_matrix[v_idx, u_idx] = f"★{score:.2f}★"
                else:
                    annot_matrix[u_idx, v_idx] = f"{score:.2f}"
                    annot_matrix[v_idx, u_idx] = f"{score:.2f}"
            
            # Mask zero entries for annotations
            annot_mask = score_matrix == 0
            masked_annot = np.ma.masked_array(annot_matrix, mask=annot_mask)
            
            sns.heatmap(score_matrix, annot=masked_annot, fmt='', cmap='plasma', 
                       xticklabels=nodes, yticklabels=nodes, 
                    #    mask=mask, cbar_kws={'label': 'Magnetic Field Score'},
                       square=True, linewidths=0.5, annot_kws={'fontsize': 9})
            
            ax_heatmap.set_title(f'Edge Scores from Decision Node {decision_node}\n★ = Selected Edge to {selected_neighbor} (Score: {current_step["score_data"]["final_score"]:.3f})', 
                                fontsize=12, fontweight='bold')
            
            # Add scoring formula information
            score_info = current_step['score_data']
            info_text = f"Decision at Node {decision_node}\n"
            info_text += f"Selected: {selected_edge}\n"
            info_text += f"Score: {score_info['final_score']:.3f}\n\n"
            info_text += f"Formula: S = (1-w)*P + w*D\n"
            info_text += f"P: {score_info['P']:.3f}\n"
            info_text += f"D: {score_info['D']:.3f}\n"
            info_text += f"w: {score_info['w']:.3f}"
        else:
            # For depot steps, show a simple message
            ax_heatmap.text(0.5, 0.5, f"Path to End Depot\n\nStep {step_idx + 1} of {len(scoring_history)}\n\nEdge: {current_step['edge']}\nCost: {current_step['score_data']['edge_weight']:.2f}\n\nCompleting route to\nreach end depot at node {modifier.end_depot}", 
                           ha='center', va='center', transform=ax_heatmap.transAxes,
                           fontsize=14, bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            ax_heatmap.set_title('Path to End Depot', fontsize=14, fontweight='bold')
            ax_heatmap.set_xticks([])
            ax_heatmap.set_yticks([])
            
            info_text = f"Completing Route\n\nMoving towards end depot\nto complete the route"
        
        # Add info text
        if not is_depot_step:
            ax_heatmap.text(1.15, 0.5, info_text, transform=ax_heatmap.transAxes,
                           verticalalignment='center', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle(f'Dynamic Magnetic Field Route Construction - Step {step_idx + 1}/{len(scoring_history)}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Pause to show step-by-step construction
        plt.pause(4.0)  # 3 second pause between steps
    
    plt.show()
    
    # Final summary
    print(f"\nRoute Construction Complete!")
    print(f"Final Route: {final_route}")
    print(f"Initial Route (reference): {initial_route}")
    print(f"Successfully added new edges: {successful_new_edges}")
    print(f"Total construction steps: {len(scoring_history)}")

def visualize_scoring_progression(modifier, scoring_history):
    """Visualize how the scoring changes as the trip progresses (same as demo_updated_v2.py)"""
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

def visualize_route_with_scores(modifier, route, scoring_history):
    """Visualize the route with scoring information (same as demo_updated_v2.py)"""
    plt.figure(figsize=(14, 10))
    
    # Draw all edges in light gray
    nx.draw_networkx_edges(modifier.graph, modifier.pos, edge_color='lightgray', 
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
            nx.draw_networkx_edges(modifier.graph, modifier.pos, edgelist=[edge],
                                  edge_color=[color], width=width, alpha=0.8)
            
            # Add step number and score
            u, v = edge
            x = (modifier.pos[u][0] + modifier.pos[v][0]) / 2
            y = (modifier.pos[u][1] + modifier.pos[v][1]) / 2
            plt.text(x, y, f'{i+1}\n{score:.2f}', fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Draw all nodes
    nx.draw_networkx_nodes(modifier.graph, modifier.pos, node_color='lightblue',
                          node_size=500, alpha=0.8)
    
    # Highlight route nodes
    if route:
        nx.draw_networkx_nodes(modifier.graph, modifier.pos, nodelist=route,
                              node_color='red', node_size=600, alpha=0.9)
    
    # Highlight depots
    depot_nodes = [modifier.start_depot, modifier.end_depot]
    nx.draw_networkx_nodes(modifier.graph, modifier.pos, nodelist=depot_nodes,
                          node_color='orange', node_size=700, alpha=1.0)
    
    # Highlight required edges that were successfully included
    final_required_edges = INITIAL_REQUIRED_EDGES + [edge for edge in NEW_REQUIRED_EDGES if edge in COMBINED_REQUIRED_EDGES]
    req_edge_list = [(u, v) for u, v in final_required_edges]
    nx.draw_networkx_edges(modifier.graph, modifier.pos, edgelist=req_edge_list,
                          edge_color='green', width=4, alpha=0.8, style='dashed')
    
    # Add labels
    nx.draw_networkx_labels(modifier.graph, modifier.pos, font_size=12, font_weight='bold')
    
    # Add colorbar for scores
    if scoring_history:
        scores = [s['score_data']['final_score'] for s in scoring_history]
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                                  norm=plt.Normalize(vmin=min(scores), vmax=max(scores)))
        sm.set_array([])
        cbar = plt.colorbar(sm, shrink=0.6)
        cbar.set_label('Magnetic Field Score', rotation=270, labelpad=20)
    
    plt.title('Route Construction with Magnetic Field Scores', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

def visualize_graph_structure(modifier):
    """Visualize the problem instance in 1 row 2 column layout"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # === LEFT SIDE: Problem Instance ===
    plt.sca(ax1)
    
    # Draw all edges in light gray
    nx.draw_networkx_edges(modifier.graph, modifier.pos, edge_color='lightgray', width=2, alpha=0.6)
    
    # Draw nodes
    nx.draw_networkx_nodes(modifier.graph, modifier.pos, node_color='lightblue', 
                          node_size=600, alpha=0.8)
    
    # Highlight depots
    depot_nodes = [modifier.start_depot, modifier.end_depot]
    nx.draw_networkx_nodes(modifier.graph, modifier.pos, nodelist=depot_nodes,
                          node_color='orange', node_size=800, alpha=0.9)
    
    # Draw initial required edges in RED
    nx.draw_networkx_edges(modifier.graph, modifier.pos, edgelist=INITIAL_REQUIRED_EDGES,
                          edge_color='red', width=4, alpha=0.8, style='solid')
    
    # Draw new required edges in GREEN
    nx.draw_networkx_edges(modifier.graph, modifier.pos, edgelist=NEW_REQUIRED_EDGES,
                          edge_color='green', width=4, alpha=0.8, style='solid')
    
    # Add labels
    nx.draw_networkx_labels(modifier.graph, modifier.pos, font_size=14, font_weight='bold')
    
    # Add edge weights
    edge_labels = nx.get_edge_attributes(modifier.graph, 'weight')
    nx.draw_networkx_edge_labels(modifier.graph, modifier.pos, edge_labels, font_size=10)
    
    ax1.set_title('Problem Instance', fontsize=16, fontweight='bold')
    
    # Add legend for left side
    legend_elements_left = [
        patches.Patch(color='orange', label='Depots'),
        patches.Patch(color='red', label='Initial Required Edges'),
        patches.Patch(color='green', label='New Required Edges')
    ]
    ax1.legend(handles=legend_elements_left, loc='upper right')
    ax1.axis('off')
    
    # === RIGHT SIDE: Initial Route ===
    plt.sca(ax2)
    
    # Draw all edges in light gray
    nx.draw_networkx_edges(modifier.graph, modifier.pos, edge_color='lightgray', width=2, alpha=0.6)
    
    # Draw nodes
    nx.draw_networkx_nodes(modifier.graph, modifier.pos, node_color='lightblue', 
                          node_size=600, alpha=0.8)
    
    # Highlight depots
    nx.draw_networkx_nodes(modifier.graph, modifier.pos, nodelist=depot_nodes,
                          node_color='orange', node_size=800, alpha=0.9)
    
    # Draw initial route with arrows in BLACK
    for i in range(len(INITIAL_ROUTE) - 1):
        u, v = INITIAL_ROUTE[i], INITIAL_ROUTE[i + 1]
        x1, y1 = modifier.pos[u]
        x2, y2 = modifier.pos[v]
        
        # Draw arrow from u to v
        ax2.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=3, alpha=0.8))
    
    # Add labels
    nx.draw_networkx_labels(modifier.graph, modifier.pos, font_size=14, font_weight='bold')
    
    ax2.set_title('Initial Route (Reference)', fontsize=16, fontweight='bold')
    
    # Add legend for right side
    legend_elements_right = [
        patches.Patch(color='orange', label='Depots'),
        patches.Patch(color='black', label='Initial Route')
    ]
    ax2.legend(handles=legend_elements_right, loc='upper right')
    ax2.axis('off')
    
    plt.suptitle('Route Construction with New Required Edges', fontsize=18, fontweight='bold')
    plt.tight_layout()

def run_route_modification_demo():
    """Run the complete route modification demonstration"""
    print("MAGNETIC FIELD ROUTE CONSTRUCTION WITH NEW REQUIRED EDGES")
    print("=" * 70)
    print("Using the same simple 6-node graph as demo_updated_v2.py")
    print("Constructs route from scratch but considers initial route and new required edges")
    
    # Create the modifier (same parameters as demo_updated_v2.py)
    modifier = MagneticFieldRouteModifier(SIMPLE_GRAPH, START_DEPOT, END_DEPOT, VEHICLE_CAPACITY)
    
    print(f"Graph: {len(SIMPLE_GRAPH.nodes())} nodes, {len(SIMPLE_GRAPH.edges())} edges")
    print(f"Start depot: {START_DEPOT}, End depot: {END_DEPOT}")
    print(f"Vehicle capacity: {VEHICLE_CAPACITY}")
    print(f"Initial route (reference): {INITIAL_ROUTE}")
    print(f"Initial required edges: {INITIAL_REQUIRED_EDGES}")
    print(f"New required edges to add: {NEW_REQUIRED_EDGES}")
    print(f"Alpha (required edge decay): {modifier.alpha}")
    print(f"Gamma (depot influence decay): {modifier.gamma}")
    
    # 1. Show problem instance
    print("\n1. Problem Instance Visualization")
    visualize_graph_structure(modifier)
    plt.show()
    
    # 2. Construct route using magnetic field approach
    print("\n2. Route Construction with Magnetic Field Approach")
    print("=" * 60)
    print("MAGNETIC FIELD ROUTE CONSTRUCTION")
    print("=" * 60)
    
    route, cost, scoring_history, successful_new_edges = modifier.construct_route_with_new_edges(
        INITIAL_ROUTE, INITIAL_REQUIRED_EDGES, NEW_REQUIRED_EDGES, verbose=True
    )
    
    if route:
        print(f"\nSuccessfully constructed route: {route}")
        print(f"Total cost: {cost:.2f}")
        print(f"Capacity utilization: {cost:.2f}/{VEHICLE_CAPACITY} ({100*cost/VEHICLE_CAPACITY:.1f}%)")
        print(f"Successfully added new edges: {successful_new_edges}")
        print(f"New edges not added: {[edge for edge in NEW_REQUIRED_EDGES if edge not in successful_new_edges]}")
    else:
        print("No feasible route found!")
    
    # 3. Dynamic visualization of route construction (same as demo_updated_v2.py)
    if scoring_history:
        print("\n3. Dynamic Route Construction Visualization")
        print("   (Watch the step-by-step construction with magnetic field scoring)")
        visualize_dynamic_route_construction(modifier, INITIAL_ROUTE, route, scoring_history, successful_new_edges)
        
        # 4. Scoring progression analysis (same as demo_updated_v2.py)
        print("\n4. Scoring Progression Analysis")
        visualize_scoring_progression(modifier, scoring_history)
        plt.show()
        
        # 5. Final route with scores (same as demo_updated_v2.py)
        # print("\n5. Final Route with Magnetic Field Scores")
        # visualize_route_with_scores(modifier, route, scoring_history)
        # plt.show()
    
    # 6. Final summary statistics
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    
    initial_cost = sum(SIMPLE_GRAPH[INITIAL_ROUTE[i]][INITIAL_ROUTE[i+1]]['weight'] 
                      for i in range(len(INITIAL_ROUTE)-1))
    
    print(f"Initial route (reference): {INITIAL_ROUTE}")
    print(f"Initial route cost: {initial_cost:.2f}")
    print(f"Constructed route: {route}")
    print(f"Constructed route cost: {cost:.2f}")
    print(f"Cost difference: {cost - initial_cost:.2f}")
    print(f"Initial required edges: {len(INITIAL_REQUIRED_EDGES)}/{len(INITIAL_REQUIRED_EDGES)} (all required)")
    print(f"New required edges added: {len(successful_new_edges)}/{len(NEW_REQUIRED_EDGES)}")
    print(f"Total required edges covered: {len(INITIAL_REQUIRED_EDGES) + len(successful_new_edges)}/{len(COMBINED_REQUIRED_EDGES)}")
    print(f"Capacity utilization: {cost:.2f}/{VEHICLE_CAPACITY} ({100*cost/VEHICLE_CAPACITY:.1f}%)")
    
    # Success analysis
    if len(successful_new_edges) == len(NEW_REQUIRED_EDGES):
        print("✅ SUCCESS: All new required edges successfully integrated!")
    elif len(successful_new_edges) > 0:
        print(f"⚠️  PARTIAL SUCCESS: {len(successful_new_edges)}/{len(NEW_REQUIRED_EDGES)} new required edges integrated")
        failed_edges = [edge for edge in NEW_REQUIRED_EDGES if edge not in successful_new_edges]
        print(f"   Failed to add: {failed_edges}")
        print("   These edges could not be integrated due to capacity constraints")
    else:
        print("❌ NO SUCCESS: No new required edges could be integrated within capacity constraints")
    
    print(f"\nAlgorithm Details:")
    print(f"- Uses same magnetic field logic as demo_updated_v2.py")
    print(f"- Constructs route from scratch using S = (1-w)*P + w*D scoring")
    print(f"- Tests new required edges sequentially for feasibility")
    print(f"- Adds only those new edges that fit within capacity constraint")
    
    return route, scoring_history, successful_new_edges

if __name__ == "__main__":
    run_route_modification_demo()