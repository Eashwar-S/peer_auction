"""
Enhanced Magnetic Field Route Construction Algorithm
====================================================
This version addresses the edge case where required edges containing depot nodes
are prioritized too early, leading to suboptimal routes with unnecessary detours.

Key improvements:
1. Required Edge Sequencing Priority: Depot-containing required edges get lower priority
2. Required Edge Coverage Maximization: Algorithm tries to maximize number of required edges covered
3. Strategic Depot Avoidance: Temporarily reduces depot influence when other required edges remain
4. Enhanced Scoring: New components for required edge sequencing and coverage optimization
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
        (4, 5, 2)#, (4, 6, 2)
    ])
    return G

# Same test instance as demo_updated_v2.py
SIMPLE_GRAPH = create_simple_graph()
START_DEPOT = 0
END_DEPOT = 5
VEHICLE_CAPACITY = 27  # Slightly increased to allow for new edges

# Edge case scenario - one required edge contains a depot
INITIAL_REQUIRED_EDGES = [(1, 2), (3, 4)]  # Original required edges
INITIAL_ROUTE = [0, 1, 2, 4, 3, 5]  # Initial route provided as input

# New required edges including one that contains the end depot - this creates the edge case
NEW_REQUIRED_EDGES = [(4, 5), (5, 3)]  # Note: (5, 3) contains end depot node 5

# Combined required edges (what we want to achieve)
COMBINED_REQUIRED_EDGES = INITIAL_REQUIRED_EDGES + NEW_REQUIRED_EDGES

class EnhancedMagneticFieldRouteModifier:
    """
    Enhanced Magnetic Field Route Construction Algorithm
    Addresses the depot-containing required edge prioritization issue
    """
    
    def __init__(self, graph, start_depot, end_depot, capacity, alpha=0.7, gamma=0.5, beta=1.0, delta=0.3):
        self.graph = graph
        self.start_depot = start_depot
        self.end_depot = end_depot
        self.capacity = capacity
        self.alpha = alpha  # Required edge influence decay
        self.gamma = gamma  # Depot influence decay
        self.beta = beta    # NEW: Required edge sequencing factor
        self.delta = delta  # NEW: Depot avoidance factor when other required edges remain
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
                influence = 0.5 * (exp(-self.gamma * d_u / self.capacity) + 
                                 exp(-self.gamma * d_v / self.capacity))
            else:
                influence = 0.1
            
            influences[edge] = influence
            influences[edge[::-1]] = influence
        
        return influences
    
    def is_depot_containing_required_edge(self, edge, required_edges):
        """Check if this edge is a required edge that contains a depot node"""
        edge_sorted = tuple(sorted(edge))
        
        for req_edge in required_edges:
            req_sorted = tuple(sorted(req_edge))
            if edge_sorted == req_sorted:
                # Check if this required edge contains a depot
                if self.start_depot in req_edge or self.end_depot in req_edge:
                    return True
        return False
    
    def count_remaining_non_depot_required_edges(self, required_edges, covered_edges):
        """Count how many non-depot-containing required edges remain uncovered"""
        remaining_count = 0
        
        for req_edge in required_edges:
            req_sorted = tuple(sorted(req_edge))
            if req_sorted not in covered_edges:
                # Check if this required edge does NOT contain a depot
                if self.start_depot not in req_edge and self.end_depot not in req_edge:
                    remaining_count += 1
        
        return remaining_count
    
    def calculate_required_edge_coverage_bonus(self, required_edges, covered_edges):
        """
        Calculate bonus for maximizing required edge coverage
        Higher bonus when more required edges can potentially be covered
        """
        total_required = len(required_edges)
        covered_count = len(covered_edges)
        
        if total_required == 0:
            return 0.0
        
        # Coverage ratio - how many we've covered so far
        coverage_ratio = covered_count / total_required
        
        # Bonus increases as we cover more required edges
        # But decreases if we're running out of capacity
        coverage_bonus = (1 - coverage_ratio) * 0.5  # Incentivize covering more
        
        return coverage_bonus
    
    def calculate_enhanced_edge_score(self, edge, required_edges, current_length, covered_required_edges, is_new_required=False):
        """
        Enhanced scoring function that addresses depot-containing required edge prioritization
        
        New Formula: S = (1-w)*P + w*D_modified + β*R + C
        Where:
        - P: Required edge influence (original)
        - D_modified: Modified depot influence (reduced when non-depot required edges remain)
        - R: Required edge sequencing factor (NEW)
        - C: Coverage maximization bonus (NEW)
        - β: Required edge sequencing weight
        """
        req_influences = self.calculate_required_edge_influence(required_edges)
        depot_influences = self.calculate_depot_influence()
        
        # Get maximum required edge influence for this edge (P in the formula)
        P = max(req_influences[edge].values()) if req_influences[edge] else 0.0
        
        # Get depot influence (D in the formula)
        D_original = depot_influences[edge]
        
        # Calculate w - normalized current trip length
        w = current_length / self.capacity if self.capacity > 0 else 0
        
        # NEW: Required edge sequencing factor (R)
        R = 0.0
        if is_new_required:
            # Check if this is a depot-containing required edge
            if self.is_depot_containing_required_edge(edge, required_edges):
                # Count remaining non-depot required edges
                remaining_non_depot = self.count_remaining_non_depot_required_edges(required_edges, covered_required_edges)
                
                if remaining_non_depot > 0:
                    # Penalize depot-containing required edges when other required edges remain
                    R = -0.8  # Strong penalty to defer depot-containing required edges
                else:
                    # No penalty if no other required edges remain
                    R = 0.5  # Small bonus since we should traverse it now
            else:
                # Bonus for non-depot-containing required edges
                R = 0.6  # Encourage non-depot required edges first
        
        # NEW: Modified depot influence (D_modified)
        remaining_non_depot = self.count_remaining_non_depot_required_edges(required_edges, covered_required_edges)
        if remaining_non_depot > 0:
            # Reduce depot influence when other required edges remain
            D_modified = D_original * (1 - self.delta)
        else:
            # Use full depot influence when no other required edges remain
            D_modified = D_original
        
        # NEW: Coverage maximization bonus (C)
        C = self.calculate_required_edge_coverage_bonus(required_edges, covered_required_edges)
        
        # Enhanced formula: S = (1-w)*P + w*D_modified + β*R + C
        S_base = (1 - w) * P + w * D_modified
        S_enhanced = S_base + self.beta * R + C
        
        # Apply bonus for new required edges (original behavior)
        if is_new_required:
            final_score = S_enhanced + 1.5  # Reduced from 2.0 to balance with new factors
        else:
            final_score = S_enhanced
            
        return {
            'P': P,
            'D_original': D_original,
            'D_modified': D_modified,
            'w': w,
            'R': R,
            'C': C,
            'S_base': S_base,
            'S_enhanced': S_enhanced,
            'final_score': final_score,
            'edge_weight': self.graph[edge[0]][edge[1]]['weight'],
            'normalized_weight': self.graph[edge[0]][edge[1]]['weight'] / self.max_edge_weight,
            'remaining_non_depot_required': remaining_non_depot,
            'is_depot_containing': self.is_depot_containing_required_edge(edge, required_edges) if is_new_required else False
        }
    
    def construct_route_with_new_edges(self, initial_route, initial_required, new_required, verbose=False):
        """
        Construct route using enhanced magnetic field approach
        """
        print(f"\nConstructing route with ENHANCED magnetic field approach...")
        print(f"Initial route (for reference): {initial_route}")
        print(f"Initial required edges: {initial_required}")
        print(f"New required edges to add: {new_required}")
        print(f"Vehicle capacity: {self.capacity}")
        print(f"Algorithm parameters: α={self.alpha}, γ={self.gamma}, β={self.beta}, δ={self.delta}")
        
        # Check for depot-containing required edges
        depot_containing_edges = []
        for edge in new_required:
            if self.start_depot in edge or self.end_depot in edge:
                depot_containing_edges.append(edge)
        
        if depot_containing_edges:
            print(f"⚠️  Depot-containing required edges detected: {depot_containing_edges}")
            print("   Enhanced algorithm will defer these until other required edges are covered")
        
        # Try adding new required edges one by one to see which ones can fit
        successful_new_edges = []
        current_required_set = initial_required.copy()
        
        for new_edge in new_required:
            # Test if we can construct a route with this additional edge
            test_required_set = current_required_set + [new_edge]
            test_route, test_cost, test_history = self.find_route_with_enhanced_scoring(test_required_set, verbose=False)
            
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
        
        route, cost, scoring_history = self.find_route_with_enhanced_scoring(final_required_edges, verbose)
        
        return route, cost, scoring_history, successful_new_edges
    
    def find_route_with_enhanced_scoring(self, required_edges, verbose=False):
        """
        Find route using enhanced magnetic field scoring
        """
        if verbose:
            print(f"\nFinding route with ENHANCED magnetic field scoring...")
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
                print(f"Required covered: {len(required_covered)}/{len(required_edges)}")
                remaining_non_depot = self.count_remaining_non_depot_required_edges(required_edges, required_covered)
                print(f"Remaining non-depot required edges: {remaining_non_depot}")
            
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
                
                # Calculate ENHANCED magnetic field score
                score_data = self.calculate_enhanced_edge_score(edge, required_edges, current_length, 
                                                              required_covered, is_new_required)
                
                candidates.append({
                    'edge': edge,
                    'neighbor': neighbor,
                    'is_new_required': is_new_required,
                    'score_data': score_data
                })
                
                if verbose:
                    print(f"  Edge {edge}: P={score_data['P']:.3f}, D_mod={score_data['D_modified']:.3f}, "
                          f"R={score_data['R']:.3f}, C={score_data['C']:.3f}, Score={score_data['final_score']:.3f}")
                    if score_data['is_depot_containing']:
                        print(f"    ⚠️  Depot-containing required edge (remaining non-depot: {score_data['remaining_non_depot_required']})")
            
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
                'current_length': current_length,
                'route_so_far': current_route.copy()
            })
            
            if verbose:
                print(f"  Selected: {selected_edge} (score: {best['score_data']['final_score']:.3f})")
                if best['score_data'].get('is_depot_containing', False):
                    print(f"    ℹ️  Selected depot-containing edge")
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
                            'D_original': 1.0,
                            'D_modified': 1.0,
                            'w': current_length / self.capacity,
                            'R': 0.0,
                            'C': 0.0,
                            'S_base': 1.0,
                            'S_enhanced': 1.0,
                            'final_score': 1.0,
                            'edge_weight': edge_weight,
                            'normalized_weight': edge_weight / self.max_edge_weight,
                            'remaining_non_depot_required': 0,
                            'is_depot_containing': False
                        }
                        
                        scoring_history.append({
                            'step': len(current_route) - 1,
                            'edge': (u, v),
                            'score_data': dummy_score_data,
                            'is_new_required': False,
                            'current_length': current_length,
                            'route_so_far': current_route.copy(),
                            'is_path_to_depot': True
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

def visualize_enhanced_scoring_comparison(modifier_original, modifier_enhanced, required_edges):
    """
    Compare original vs enhanced scoring for all edges to show the difference
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Calculate scores for both approaches at a specific point in the route
    current_length = 50  # Example current length
    covered_edges = set()  # No edges covered yet
    
    # Get all edges and calculate scores
    edges = list(modifier_original.graph.edges())
    original_scores = []
    enhanced_scores = []
    edge_labels = []
    is_depot_containing = []
    
    for edge in edges:
        edge_sorted = tuple(sorted(edge))
        is_required = edge_sorted in [tuple(sorted(req)) for req in required_edges]
        
        # Original scoring
        orig_score = modifier_original.calculate_edge_score(edge, required_edges, current_length, is_required)
        
        # Enhanced scoring
        enh_score = modifier_enhanced.calculate_enhanced_edge_score(edge, required_edges, current_length, 
                                                                   covered_edges, is_required)
        
        original_scores.append(orig_score['final_score'])
        enhanced_scores.append(enh_score['final_score'])
        edge_labels.append(f"{edge[0]}-{edge[1]}")
        
        # Check if depot-containing
        is_depot = modifier_enhanced.is_depot_containing_required_edge(edge, required_edges) and is_required
        is_depot_containing.append(is_depot)
    
    # Create color coding
    colors = ['red' if depot else 'blue' for depot in is_depot_containing]
    
    # Plot 1: Original Scores
    ax1.bar(range(len(edges)), original_scores, color='lightblue', alpha=0.7)
    ax1.set_title('Original Magnetic Field Scores', fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_xticks(range(len(edges)))
    ax1.set_xticklabels(edge_labels, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Enhanced Scores
    bars = ax2.bar(range(len(edges)), enhanced_scores, color=colors, alpha=0.7)
    ax2.set_title('Enhanced Magnetic Field Scores', fontweight='bold')
    ax2.set_ylabel('Score')
    ax2.set_xticks(range(len(edges)))
    ax2.set_xticklabels(edge_labels, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    red_patch = patches.Patch(color='red', label='Depot-containing Required Edges')
    blue_patch = patches.Patch(color='blue', label='Other Edges')
    ax2.legend(handles=[red_patch, blue_patch])
    
    # Plot 3: Score Differences
    score_diffs = [enh - orig for enh, orig in zip(enhanced_scores, original_scores)]
    bars3 = ax3.bar(range(len(edges)), score_diffs, color=colors, alpha=0.7)
    ax3.set_title('Score Differences\n(Enhanced - Original)', fontweight='bold')
    ax3.set_ylabel('Score Difference')
    ax3.set_xticks(range(len(edges)))
    ax3.set_xticklabels(edge_labels, rotation=45)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Original vs Enhanced Magnetic Field Scoring Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

def visualize_problem_with_depot_edge(modifier):
    """Visualize the specific problem instance with depot-containing required edge"""
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
    
    # Draw initial required edges in BLUE
    nx.draw_networkx_edges(modifier.graph, modifier.pos, edgelist=INITIAL_REQUIRED_EDGES,
                          edge_color='blue', width=4, alpha=0.8, style='solid')
    
    # Draw non-depot new required edges in GREEN
    non_depot_new_edges = [edge for edge in NEW_REQUIRED_EDGES 
                          if modifier.start_depot not in edge and modifier.end_depot not in edge]
    if non_depot_new_edges:
        nx.draw_networkx_edges(modifier.graph, modifier.pos, edgelist=non_depot_new_edges,
                              edge_color='green', width=4, alpha=0.8, style='solid')
    
    # Draw depot-containing new required edges in RED
    depot_new_edges = [edge for edge in NEW_REQUIRED_EDGES 
                      if modifier.start_depot in edge or modifier.end_depot in edge]
    if depot_new_edges:
        nx.draw_networkx_edges(modifier.graph, modifier.pos, edgelist=depot_new_edges,
                              edge_color='red', width=5, alpha=0.9, style='solid')
    
    # Add labels
    nx.draw_networkx_labels(modifier.graph, modifier.pos, font_size=14, font_weight='bold')
    
    # Add edge weights
    edge_labels = nx.get_edge_attributes(modifier.graph, 'weight')
    nx.draw_networkx_edge_labels(modifier.graph, modifier.pos, edge_labels, font_size=10)
    
    ax1.set_title('Problem Instance - Depot Edge Case', fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        patches.Patch(color='orange', label='Depots'),
        patches.Patch(color='blue', label='Initial Required Edges'),
        patches.Patch(color='green', label='New Required Edges (Non-Depot)'),
        patches.Patch(color='red', label='New Required Edges (Depot-Containing)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    ax1.axis('off')
    
    # === RIGHT SIDE: Algorithm Strategy ===
    plt.sca(ax2)
    
    # Create a text explanation of the strategy
    strategy_text = """
ENHANCED ALGORITHM STRATEGY

Problem:
• Depot-containing required edges get high priority
• This can cause inefficient detours
• Route may visit depot multiple times

Solution:
1. Required Edge Sequencing (β factor):
   • Penalize depot-containing required edges when
     other required edges remain uncovered
   • Encourage non-depot required edges first

2. Modified Depot Influence (δ factor):
   • Reduce depot attraction when other required
     edges still need to be traversed
   • Restore full depot influence when appropriate

3. Coverage Maximization (C factor):
   • Bonus for maximizing required edge coverage
   • Helps algorithm find better sequences

Enhanced Formula:
S = (1-w)*P + w*D_modified + β*R + C

Where:
• P: Required edge influence (original)
• D_modified: Reduced depot influence when needed
• R: Sequencing factor (negative for depot edges)
• C: Coverage maximization bonus
• β: Sequencing weight parameter
"""
    
    ax2.text(0.05, 0.95, strategy_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    ax2.set_title('Enhanced Algorithm Strategy', fontsize=16, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.suptitle('Depot-Containing Required Edge Challenge', fontsize=18, fontweight='bold')
    plt.tight_layout()

def run_enhanced_comparison_demo():
    """Run comparison between original and enhanced algorithms"""
    print("ENHANCED MAGNETIC FIELD ROUTE CONSTRUCTION COMPARISON")
    print("=" * 70)
    print("Addressing the depot-containing required edge prioritization issue")
    
    # Create both modifiers
    modifier_original = MagneticFieldRouteModifier(SIMPLE_GRAPH, START_DEPOT, END_DEPOT, VEHICLE_CAPACITY)
    modifier_enhanced = EnhancedMagneticFieldRouteModifier(SIMPLE_GRAPH, START_DEPOT, END_DEPOT, VEHICLE_CAPACITY)
    
    print(f"Graph: {len(SIMPLE_GRAPH.nodes())} nodes, {len(SIMPLE_GRAPH.edges())} edges")
    print(f"Start depot: {START_DEPOT}, End depot: {END_DEPOT}")
    print(f"Vehicle capacity: {VEHICLE_CAPACITY}")
    print(f"Initial required edges: {INITIAL_REQUIRED_EDGES}")
    print(f"New required edges: {NEW_REQUIRED_EDGES}")
    
    # Identify depot-containing edges
    depot_edges = []
    for edge in NEW_REQUIRED_EDGES:
        if START_DEPOT in edge or END_DEPOT in edge:
            depot_edges.append(edge)
    
    print(f"Depot-containing required edges: {depot_edges}")
    
    # 1. Show problem visualization
    print("\n1. Problem Instance with Depot Edge Case")
    visualize_problem_with_depot_edge(modifier_enhanced)
    plt.show()
    
    # 2. Show scoring comparison
    print("\n2. Scoring Comparison: Original vs Enhanced")
    all_required = INITIAL_REQUIRED_EDGES + NEW_REQUIRED_EDGES
    visualize_enhanced_scoring_comparison(modifier_original, modifier_enhanced, all_required)
    plt.show()
    
    # 3. Run original algorithm
    print("\n3. ORIGINAL Algorithm Results")
    print("=" * 40)
    
    # Note: We need to create the original algorithm class
    route_orig, cost_orig, history_orig, success_orig = modifier_original.construct_route_with_new_edges(
        INITIAL_ROUTE, INITIAL_REQUIRED_EDGES, NEW_REQUIRED_EDGES, verbose=True
    )
    
    # 4. Run enhanced algorithm
    print("\n4. ENHANCED Algorithm Results")
    print("=" * 40)
    
    route_enh, cost_enh, history_enh, success_enh = modifier_enhanced.construct_route_with_new_edges(
        INITIAL_ROUTE, INITIAL_REQUIRED_EDGES, NEW_REQUIRED_EDGES, verbose=True
    )
    
    # 5. Compare results
    print("\n5. COMPARISON RESULTS")
    print("=" * 50)
    
    print(f"Original Algorithm:")
    print(f"  Route: {route_orig}")
    print(f"  Cost: {cost_orig:.2f}")
    print(f"  Successfully added edges: {success_orig}")
    print(f"  Capacity utilization: {100*cost_orig/VEHICLE_CAPACITY:.1f}%")
    
    print(f"\nEnhanced Algorithm:")
    print(f"  Route: {route_enh}")
    print(f"  Cost: {cost_enh:.2f}")
    print(f"  Successfully added edges: {success_enh}")
    print(f"  Capacity utilization: {100*cost_enh/VEHICLE_CAPACITY:.1f}%")
    
    # Analyze which algorithm handled depot edges better
    print(f"\nDepot Edge Handling Analysis:")
    orig_depot_success = [edge for edge in success_orig if edge in depot_edges]
    enh_depot_success = [edge for edge in success_enh if edge in depot_edges]
    
    print(f"Original - Depot edges added: {orig_depot_success}")
    print(f"Enhanced - Depot edges added: {enh_depot_success}")
    
    # Check route efficiency
    if route_orig and route_enh:
        print(f"\nRoute Efficiency Analysis:")
        print(f"Original route length: {len(route_orig)} nodes")
        print(f"Enhanced route length: {len(route_enh)} nodes")
        
        # Count depot visits
        orig_depot_visits = sum(1 for node in route_orig if node in [START_DEPOT, END_DEPOT])
        enh_depot_visits = sum(1 for node in route_enh if node in [START_DEPOT, END_DEPOT])
        
        print(f"Original depot visits: {orig_depot_visits}")
        print(f"Enhanced depot visits: {enh_depot_visits}")
        
        if cost_enh < cost_orig:
            print("✅ Enhanced algorithm achieved better cost!")
        elif cost_enh == cost_orig:
            print("⚖️  Both algorithms achieved same cost")
        else:
            print("⚠️  Original algorithm achieved better cost")
    
    # 6. Visualize route construction for enhanced algorithm
    if history_enh:
        print("\n6. Enhanced Algorithm Route Construction Visualization")
        visualize_dynamic_route_construction_enhanced(modifier_enhanced, INITIAL_ROUTE, route_enh, 
                                                    history_enh, success_enh)
    
    return route_orig, route_enh, history_orig, history_enh

def visualize_dynamic_route_construction_enhanced(modifier, initial_route, final_route, scoring_history, successful_new_edges):
    """
    Enhanced dynamic visualization showing the improved scoring decisions
    """
    if not final_route or not scoring_history:
        print("No route found or no scoring history available")
        return
    
    print(f"Creating enhanced dynamic visualization...")
    print(f"Final route: {final_route}")
    print(f"Successfully added new edges: {successful_new_edges}")
    
    # Determine which edges are being used in the required set
    final_required_edges = INITIAL_REQUIRED_EDGES + successful_new_edges
    
    # Create figure with subplots
    fig = plt.figure(figsize=(22, 12))
    
    for step_idx in range(len(scoring_history)):
        fig.clear()
        
        # Create subplots for this step
        ax_graph = plt.subplot(2, 2, 1)
        ax_heatmap = plt.subplot(2, 2, 2)
        ax_scoring = plt.subplot(2, 2, 3)
        ax_progress = plt.subplot(2, 2, 4)
        
        current_step = scoring_history[step_idx]
        current_route_nodes = current_step.get('route_so_far', final_route[:step_idx + 2])
        current_length = current_step['current_length']
        is_depot_step = current_step.get('is_path_to_depot', False)
        
        # === TOP LEFT: Route Construction ===
        plt.sca(ax_graph)
        
        # Draw all edges in light gray
        nx.draw_networkx_edges(modifier.graph, modifier.pos, edge_color='lightgray', 
                              width=1, alpha=0.4)
        
        # Draw completed route edges with BLACK ARROWS
        for i in range(len(current_route_nodes) - 1):
            u, v = current_route_nodes[i], current_route_nodes[i + 1]
            x1, y1 = modifier.pos[u]
            x2, y2 = modifier.pos[v]
            
            # Current edge is thicker
            linewidth = 5 if i == len(current_route_nodes) - 2 else 3
            alpha = 1.0 if i == len(current_route_nodes) - 2 else 0.7
            
            ax_graph.annotate('', xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle='->', color='black', lw=linewidth, alpha=alpha))
        
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
        
        # Highlight required edges with different colors
        # Initial required edges in BLUE
        nx.draw_networkx_edges(modifier.graph, modifier.pos, edgelist=INITIAL_REQUIRED_EDGES,
                              edge_color='blue', width=3, alpha=0.6, style='dashed')
        
        # New non-depot required edges in GREEN
        non_depot_new = [edge for edge in successful_new_edges 
                        if modifier.start_depot not in edge and modifier.end_depot not in edge]
        if non_depot_new:
            nx.draw_networkx_edges(modifier.graph, modifier.pos, edgelist=non_depot_new,
                                  edge_color='green', width=3, alpha=0.6, style='dashed')
        
        # New depot-containing required edges in RED
        depot_new = [edge for edge in successful_new_edges 
                    if modifier.start_depot in edge or modifier.end_depot in edge]
        if depot_new:
            nx.draw_networkx_edges(modifier.graph, modifier.pos, edgelist=depot_new,
                                  edge_color='red', width=4, alpha=0.8, style='dashed')
        
        # Add labels
        nx.draw_networkx_labels(modifier.graph, modifier.pos, font_size=12, font_weight='bold')
        
        step_type = " (Path to Depot)" if is_depot_step else ""
        ax_graph.set_title(f'Step {step_idx + 1}: Enhanced Route Construction{step_type}\n'
                          f'Current: {current_node}, Length: {current_length:.2f}', 
                          fontsize=12, fontweight='bold')
        ax_graph.axis('off')
        
        # === TOP RIGHT: Enhanced Scoring Details ===
        plt.sca(ax_scoring)
        
        if not is_depot_step:
            score_data = current_step['score_data']
            
            # Create scoring breakdown visualization
            components = ['P\n(Req Edge)', 'D_mod\n(Depot)', 'R\n(Sequence)', 'C\n(Coverage)']
            values = [score_data.get('P', 0), score_data.get('D_modified', 0), 
                     score_data.get('R', 0), score_data.get('C', 0)]
            colors = ['blue', 'orange', 'purple', 'green']
            
            bars = ax_scoring.bar(components, values, color=colors, alpha=0.7)
            ax_scoring.set_title(f'Enhanced Scoring Components\nSelected Edge: {current_step["edge"]}', 
                               fontweight='bold')
            ax_scoring.set_ylabel('Score Value')
            ax_scoring.grid(True, alpha=0.3)
            
            # Highlight negative values (penalties)
            for bar, val in zip(bars, values):
                if val < 0:
                    bar.set_color('red')
                    bar.set_alpha(0.8)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax_scoring.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.03),
                              f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        else:
            ax_scoring.text(0.5, 0.5, f"Path to Depot\n\nEdge: {current_step['edge']}\nCost: {current_step['score_data']['edge_weight']:.2f}", 
                          ha='center', va='center', transform=ax_scoring.transAxes, fontsize=12,
                          bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            ax_scoring.set_title('Path to End Depot', fontweight='bold')
        
        # === BOTTOM LEFT: Decision Heatmap ===
        plt.sca(ax_heatmap)
        
        if not is_depot_step and len(current_route_nodes) >= 2:
            # Similar to original but show enhanced scores
            decision_node = current_route_nodes[-2]
            selected_edge = current_step['edge']
            
            # Calculate scores for visualization
            edge_scores = {}
            visited_edges = set()
            
            # Get visited edges up to the previous step
            for i in range(len(current_route_nodes) - 2):
                edge_sorted = tuple(sorted([current_route_nodes[i], current_route_nodes[i + 1]]))
                visited_edges.add(edge_sorted)
            
            decision_length = current_length - current_step['score_data']['edge_weight']
            
            # Calculate enhanced scores for all edges from the decision node
            for neighbor in modifier.graph.neighbors(decision_node):
                edge = (decision_node, neighbor)
                edge_sorted = tuple(sorted(edge))
                
                if edge_sorted in visited_edges:
                    continue
                    
                edge_weight = modifier.graph[decision_node][neighbor]['weight']
                if decision_length + edge_weight > modifier.capacity:
                    continue
                    
                is_new_required = (edge_sorted in [tuple(sorted(req)) for req in final_required_edges])
                
                # Get the required covered edges at decision time
                decision_covered = set()
                for i in range(len(current_route_nodes) - 2):
                    edge_sorted = tuple(sorted([current_route_nodes[i], current_route_nodes[i + 1]]))
                    if edge_sorted in [tuple(sorted(req)) for req in final_required_edges]:
                        decision_covered.add(edge_sorted)
                
                score_data = modifier.calculate_enhanced_edge_score(edge, final_required_edges, 
                                                                  decision_length, decision_covered, is_new_required)
                edge_scores[edge] = score_data['final_score']
            
            # Create heatmap matrix
            nodes = list(modifier.graph.nodes())
            n_nodes = len(nodes)
            score_matrix = np.zeros((n_nodes, n_nodes))
            
            for edge, score in edge_scores.items():
                u, v = edge
                u_idx = nodes.index(u)
                v_idx = nodes.index(v)
                score_matrix[u_idx, v_idx] = score
                score_matrix[v_idx, u_idx] = score
            
            mask = score_matrix == 0
            
            # Create annotations highlighting selected edge
            annot_matrix = np.zeros((n_nodes, n_nodes), dtype=object)
            for edge, score in edge_scores.items():
                u, v = edge
                u_idx = nodes.index(u)
                v_idx = nodes.index(v)
                
                if (edge == selected_edge or edge == selected_edge[::-1]):
                    annot_matrix[u_idx, v_idx] = f"★{score:.2f}★"
                    annot_matrix[v_idx, u_idx] = f"★{score:.2f}★"
                else:
                    annot_matrix[u_idx, v_idx] = f"{score:.2f}"
                    annot_matrix[v_idx, u_idx] = f"{score:.2f}"
            
            annot_mask = score_matrix == 0
            masked_annot = np.ma.masked_array(annot_matrix, mask=annot_mask)
            
            sns.heatmap(score_matrix, annot=masked_annot, fmt='', cmap='plasma', 
                       xticklabels=nodes, yticklabels=nodes, 
                       square=True, linewidths=0.5, annot_kws={'fontsize': 8})
            
            ax_heatmap.set_title(f'Enhanced Scores from Node {decision_node}\n'
                               f'Score: {current_step["score_data"]["final_score"]:.3f}', 
                               fontsize=12, fontweight='bold')
        else:
            ax_heatmap.text(0.5, 0.5, "No Decision\nRequired", ha='center', va='center', 
                          transform=ax_heatmap.transAxes, fontsize=14)
            ax_heatmap.set_title('Decision Matrix', fontweight='bold')
        
        # === BOTTOM RIGHT: Algorithm Progress ===
        plt.sca(ax_progress)
        
        # Show algorithm progress and insights
        total_required = len(final_required_edges)
        covered_so_far = sum(1 for i in range(len(current_route_nodes) - 1) 
                           if tuple(sorted([current_route_nodes[i], current_route_nodes[i + 1]])) 
                           in [tuple(sorted(req)) for req in final_required_edges])
        
        remaining_non_depot = current_step['score_data'].get('remaining_non_depot_required', 0)
        
        progress_text = f"Algorithm Progress\n\n"
        progress_text += f"Required Edges: {covered_so_far}/{total_required}\n"
        progress_text += f"Remaining Non-Depot Required: {remaining_non_depot}\n"
        progress_text += f"Capacity Used: {current_length:.1f}/{modifier.capacity}\n\n"
        
        if current_step.get('score_data', {}).get('is_depot_containing', False):
            progress_text += "⚠️ Depot-containing edge selected\n"
            if remaining_non_depot > 0:
                progress_text += "❌ Still have non-depot required edges!\n"
                progress_text += "This may indicate suboptimal sequencing."
            else:
                progress_text += "✅ No non-depot required edges remain\n"
                progress_text += "Good time to select depot edge."
        elif current_step.get('is_new_required', False):
            progress_text += "✅ Non-depot required edge selected\n"
            progress_text += "Good sequencing choice."
        
        ax_progress.text(0.05, 0.95, progress_text, transform=ax_progress.transAxes, fontsize=10,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        ax_progress.set_title('Algorithm Insights', fontweight='bold')
        ax_progress.set_xlim(0, 1)
        ax_progress.set_ylim(0, 1)
        ax_progress.axis('off')
        
        plt.suptitle(f'Enhanced Magnetic Field Algorithm - Step {step_idx + 1}/{len(scoring_history)}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.pause(2.5)
    
    plt.show()
    print(f"\nEnhanced Route Construction Complete!")

# We need to add the original MagneticFieldRouteModifier class for comparison
class MagneticFieldRouteModifier:
    """
    Original Magnetic Field Route Construction Algorithm (for comparison)
    """
    
    def __init__(self, graph, start_depot, end_depot, capacity, alpha=0.7, gamma=0.5):
        self.graph = graph
        self.start_depot = start_depot
        self.end_depot = end_depot
        self.capacity = capacity
        self.alpha = alpha
        self.gamma = gamma
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
            influences[edge[::-1]] = {}
            
            for i, req_edge in enumerate(required_edges):
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
        """Calculate influence of depots on all edges"""
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
    
    def calculate_edge_score(self, edge, required_edges, current_length, is_new_required=False):
        """Original magnetic field scoring"""
        req_influences = self.calculate_required_edge_influence(required_edges)
        depot_influences = self.calculate_depot_influence()
        
        P = max(req_influences[edge].values()) if req_influences[edge] else 0.0
        D = depot_influences[edge]
        w = current_length / self.capacity if self.capacity > 0 else 0
        
        S = (1 - w) * P + w * D
        
        if is_new_required:
            final_score = S + 2.0
        else:
            final_score = S
            
        return {
            'P': P,
            'D': D,
            'w': w,
            'S': S,
            'final_score': final_score,
            'edge_weight': self.graph[edge[0]][edge[1]]['weight'],
            'normalized_weight': self.graph[edge[0]][edge[1]]['weight'] / self.max_edge_weight
        }
    
    def construct_route_with_new_edges(self, initial_route, initial_required, new_required, verbose=False):
        """Original route construction method"""
        print(f"\nConstructing route with ORIGINAL magnetic field approach...")
        
        successful_new_edges = []
        current_required_set = initial_required.copy()
        
        for new_edge in new_required:
            test_required_set = current_required_set + [new_edge]
            test_route, test_cost, test_history = self.find_route_with_magnetic_scoring(test_required_set, verbose=False)
            
            if test_route and test_cost <= self.capacity:
                successful_new_edges.append(new_edge)
                current_required_set.append(new_edge)
                print(f"✓ New edge {new_edge} can be added (test cost: {test_cost:.2f})")
            else:
                print(f"✗ New edge {new_edge} cannot be added within capacity")
        
        final_required_edges = initial_required + successful_new_edges
        route, cost, scoring_history = self.find_route_with_magnetic_scoring(final_required_edges, verbose)
        
        return route, cost, scoring_history, successful_new_edges
    
    def find_route_with_magnetic_scoring(self, required_edges, verbose=False):
        """Original magnetic field route finding"""
        current_route = [self.start_depot]
        current_length = 0
        visited_edges = set()
        required_covered = set()
        scoring_history = []
        
        while len(required_covered) < len(required_edges) or current_route[-1] != self.end_depot:
            current_node = current_route[-1]
            candidates = []
            
            for neighbor in self.graph.neighbors(current_node):
                edge = (current_node, neighbor)
                edge_sorted = tuple(sorted(edge))
                
                if edge_sorted in visited_edges:
                    continue
                
                edge_weight = self.graph[current_node][neighbor]['weight']
                if current_length + edge_weight > self.capacity:
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
            
            if not candidates:
                break
            
            candidates.sort(key=lambda x: (x['score_data']['final_score'], 
                                         -x['score_data']['normalized_weight']), reverse=True)
            
            best = candidates[0]
            selected_edge = best['edge']
            selected_neighbor = best['neighbor']
            
            current_route.append(selected_neighbor)
            current_length += best['score_data']['edge_weight']
            visited_edges.add(tuple(sorted(selected_edge)))
            
            if best['is_new_required']:
                required_covered.add(tuple(sorted(selected_edge)))
            
            scoring_history.append({
                'step': len(current_route) - 1,
                'edge': selected_edge,
                'score_data': best['score_data'],
                'is_new_required': best['is_new_required'],
                'current_length': current_length,
                'route_so_far': current_route.copy()
            })
        
        # Path to end depot if needed
        if current_route[-1] != self.end_depot:
            try:
                path_to_end = nx.shortest_path(self.graph, current_route[-1], self.end_depot, weight='weight')
                additional_length = nx.shortest_path_length(self.graph, current_route[-1], self.end_depot, weight='weight')
                
                if current_length + additional_length <= self.capacity:
                    for i in range(1, len(path_to_end)):
                        u, v = path_to_end[i-1], path_to_end[i]
                        edge_weight = self.graph[u][v]['weight']
                        current_route.append(v)
                        current_length += edge_weight
                        
                        dummy_score_data = {
                            'P': 0.0, 'D': 1.0, 'w': current_length / self.capacity,
                            'S': 1.0, 'final_score': 1.0, 'edge_weight': edge_weight,
                            'normalized_weight': edge_weight / self.max_edge_weight
                        }
                        
                        scoring_history.append({
                            'step': len(current_route) - 1, 'edge': (u, v),
                            'score_data': dummy_score_data, 'is_new_required': False,
                            'current_length': current_length, 'route_so_far': current_route.copy(),
                            'is_path_to_depot': True
                        })
                else:
                    return None, float('inf'), scoring_history
            except nx.NetworkXNoPath:
                return None, float('inf'), scoring_history
        
        return current_route, current_length, scoring_history

if __name__ == "__main__":
    run_enhanced_comparison_demo()