"""
Magnetic Field Vehicle Routing Algorithm Demonstration
====================================================
This demonstrates how the magnetic field approach works with detailed visualizations
showing edge attractions and depot influences.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from math import exp, sqrt
import seaborn as sns

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
VEHICLE_CAPACITY = 15
REQUIRED_EDGES = [(1, 2), (3, 4)]  # Only 2 required edges for clarity
FAILED_EDGES = [(2, 4)]  # Only 1 failed edge to handle

class MagneticFieldRouter:
    """
    Magnetic Field Vehicle Routing Algorithm with detailed visualization
    """
    
    def __init__(self, graph, start_depot, end_depot, capacity, alpha=0.7, gamma=0.5):
        self.graph = graph
        self.start_depot = start_depot
        self.end_depot = end_depot
        self.capacity = capacity
        self.alpha = alpha  # Required edge influence decay
        self.gamma = gamma  # Depot influence decay
        self.pos = self._create_layout()
        
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
                influence = 0.5 * (exp(-self.gamma * d_u) + exp(-self.gamma * d_v))
            else:
                influence = 0
            
            influences[edge] = influence
            influences[edge[::-1]] = influence
        
        return influences
    
    def calculate_total_attraction(self, required_edges):
        """Calculate total magnetic field attraction for each edge"""
        req_influences = self.calculate_required_edge_influence(required_edges)
        print(f"req_influences - {req_influences}")
        depot_influences = self.calculate_depot_influence()
        print(f"depot_influences - {depot_influences}")
        
        total_attractions = {}
        
        for edge in self.graph.edges():
            # Sum of required edge influences
            req_sum = sum(req_influences[edge].values())
            
            # Depot influence
            depot_inf = depot_influences[edge]
            
            # Total attraction (weighted combination)
            total_attraction = req_sum + depot_inf
            
            total_attractions[edge] = {
                'required_influence': req_sum,
                'depot_influence': depot_inf,
                'total_attraction': total_attraction
            }
            total_attractions[edge[::-1]] = total_attractions[edge]
        
        return total_attractions, req_influences, depot_influences
    
    def find_route_with_magnetic_field(self, required_edges):
        """Find route using magnetic field guidance"""
        attractions, req_inf, depot_inf = self.calculate_total_attraction(required_edges)
        
        # Start with required edges
        must_visit_nodes = set()
        for edge in required_edges:
            must_visit_nodes.update(edge)
        must_visit_nodes.update([self.start_depot, self.end_depot])
        
        # Use magnetic field to guide route construction
        best_route = None
        best_cost = float('inf')
        
        # Try different permutations of intermediate nodes
        from itertools import permutations
        intermediate_nodes = list(must_visit_nodes - {self.start_depot, self.end_depot})
        
        for perm in permutations(intermediate_nodes):
            route_nodes = [self.start_depot] + list(perm) + [self.end_depot]
            
            try:
                cost = 0
                valid = True
                full_path = []
                
                for i in range(len(route_nodes) - 1):
                    segment = nx.shortest_path(self.graph, route_nodes[i], route_nodes[i+1], 
                                             weight='weight')
                    if i > 0:
                        segment = segment[1:]  # Avoid duplicating nodes
                    full_path.extend(segment)
                    cost += nx.shortest_path_length(self.graph, route_nodes[i], route_nodes[i+1], 
                                                  weight='weight')
                
                # Check if route covers required edges
                route_edges = set()
                for j in range(len(full_path) - 1):
                    route_edges.add(tuple(sorted([full_path[j], full_path[j+1]])))
                
                req_edges_covered = all(tuple(sorted(edge)) in route_edges for edge in required_edges)
                
                if req_edges_covered and cost < best_cost and cost <= self.capacity:
                    best_cost = cost
                    best_route = full_path
                    
            except nx.NetworkXNoPath:
                continue
        
        return best_route, best_cost, attractions

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

def visualize_required_edge_influences(router, required_edges):
    """Visualize influence of each required edge separately"""
    req_influences = router.calculate_required_edge_influence(required_edges)
    
    n_req = len(required_edges)
    fig, axes = plt.subplots(1, n_req, figsize=(6*n_req, 6))
    if n_req == 1:
        axes = [axes]
    
    for i, req_edge in enumerate(required_edges):
        ax = axes[i]
        
        # Get influences for this required edge
        edge_influences = {}
        for edge in router.graph.edges():
            edge_influences[edge] = req_influences[edge][f'req_{i}']
        
        # Normalize influences for color mapping
        max_inf = max(edge_influences.values()) if edge_influences.values() else 1
        normalized_influences = {k: v/max_inf for k, v in edge_influences.items()}
        
        # Draw edges with color based on influence
        for edge in router.graph.edges():
            influence = normalized_influences[edge]
            color = plt.cm.Reds(influence)
            width = 1 + 4 * influence  # Width proportional to influence
            
            nx.draw_networkx_edges(router.graph, router.pos, edgelist=[edge],
                                  edge_color=[color], width=width, alpha=0.8, ax=ax)
        
        # Draw nodes
        nx.draw_networkx_nodes(router.graph, router.pos, node_color='lightblue',
                              node_size=400, alpha=0.7, ax=ax)
        
        # Highlight the required edge
        nx.draw_networkx_edges(router.graph, router.pos, edgelist=[req_edge],
                              edge_color='green', width=6, alpha=1.0, ax=ax)
        
        # Add labels
        nx.draw_networkx_labels(router.graph, router.pos, font_size=12, 
                               font_weight='bold', ax=ax)
        
        # Add influence values as text
        for edge in router.graph.edges():
            u, v = edge
            x = (router.pos[u][0] + router.pos[v][0]) / 2
            y = (router.pos[u][1] + router.pos[v][1]) / 2
            influence = edge_influences[edge]
            ax.text(x, y, f'{influence:.2f}', fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        ax.set_title(f'Influence of Required Edge {req_edge}', fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Required Edge Influences on All Edges', fontsize=16, fontweight='bold')
    plt.tight_layout()

def visualize_depot_influences(router):
    """Visualize depot influences on all edges"""
    depot_influences = router.calculate_depot_influence()
    
    plt.figure(figsize=(10, 8))
    
    # Normalize influences for color mapping
    max_inf = max(depot_influences.values()) if depot_influences.values() else 1
    normalized_influences = {k: v/max_inf for k, v in depot_influences.items()}
    
    # Draw edges with color based on depot influence
    for edge in router.graph.edges():
        influence = normalized_influences[edge]
        color = plt.cm.Blues(influence)
        width = 1 + 4 * influence
        
        nx.draw_networkx_edges(router.graph, router.pos, edgelist=[edge],
                              edge_color=[color], width=width, alpha=0.8)
    
    # Draw nodes
    nx.draw_networkx_nodes(router.graph, router.pos, node_color='lightblue',
                          node_size=400, alpha=0.7)
    
    # Highlight depots
    depot_nodes = [router.start_depot, router.end_depot]
    nx.draw_networkx_nodes(router.graph, router.pos, nodelist=depot_nodes,
                          node_color='orange', node_size=800, alpha=0.9)
    
    # Add labels
    nx.draw_networkx_labels(router.graph, router.pos, font_size=12, font_weight='bold')
    
    # Add influence values as text
    for edge in router.graph.edges():
        u, v = edge
        x = (router.pos[u][0] + router.pos[v][0]) / 2
        y = (router.pos[u][1] + router.pos[v][1]) / 2
        influence = depot_influences[edge]
        plt.text(x, y, f'{influence:.2f}', fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    plt.title('Depot Influences on All Edges', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

def visualize_total_magnetic_field(router, required_edges):
    """Visualize the combined magnetic field attraction"""
    attractions, req_inf, depot_inf = router.calculate_total_attraction(required_edges)
    
    plt.figure(figsize=(12, 10))
    
    # Get total attractions
    total_attractions = {edge: data['total_attraction'] for edge, data in attractions.items()}
    
    # Normalize for color mapping
    max_attraction = max(total_attractions.values()) if total_attractions.values() else 1
    normalized_attractions = {k: v/max_attraction for k, v in total_attractions.items()}
    
    # Draw edges with color and width based on total attraction
    for edge in router.graph.edges():
        attraction = normalized_attractions[edge]
        color = plt.cm.plasma(attraction)
        width = 1 + 6 * attraction
        
        nx.draw_networkx_edges(router.graph, router.pos, edgelist=[edge],
                              edge_color=[color], width=width, alpha=0.8)
    
    # Draw nodes
    nx.draw_networkx_nodes(router.graph, router.pos, node_color='lightblue',
                          node_size=500, alpha=0.8)
    
    # Highlight depots
    depot_nodes = [router.start_depot, router.end_depot]
    nx.draw_networkx_nodes(router.graph, router.pos, nodelist=depot_nodes,
                          node_color='orange', node_size=700, alpha=0.9)
    
    # Highlight required edges
    req_edge_list = [(u, v) for u, v in required_edges]
    nx.draw_networkx_edges(router.graph, router.pos, edgelist=req_edge_list,
                          edge_color='white', width=3, alpha=1.0, style='dashed')
    
    # Add labels
    nx.draw_networkx_labels(router.graph, router.pos, font_size=12, font_weight='bold')
    
    # Add attraction values as text
    for edge in router.graph.edges():
        u, v = edge
        x = (router.pos[u][0] + router.pos[v][0]) / 2
        y = (router.pos[u][1] + router.pos[v][1]) / 2
        attraction = total_attractions[edge]
        plt.text(x, y, f'{attraction:.2f}', fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.title('Total Magnetic Field Attraction (Combined Required + Depot)', 
              fontsize=16, fontweight='bold')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                              norm=plt.Normalize(vmin=0, vmax=max_attraction))
    sm.set_array([])
    cbar = plt.colorbar(sm, shrink=0.6)
    cbar.set_label('Attraction Strength', rotation=270, labelpad=20)
    
    plt.axis('off')
    plt.tight_layout()

def visualize_route_construction(router, required_edges):
    """Visualize the final route construction"""
    route, cost, attractions = router.find_route_with_magnetic_field(required_edges)
    
    print(f"route - {route}")
    print(f"cost - {cost}")
    print(f"attractions - {attractions}")

    plt.figure(figsize=(12, 8))
    
    # Draw all edges in light gray
    nx.draw_networkx_edges(router.graph, router.pos, edge_color='lightgray', 
                          width=1, alpha=0.4)
    
    # Draw route edges with arrows
    if route:
        for i in range(len(route) - 1):
            start_node = route[i]
            end_node = route[i + 1]
            
            # Get positions
            x1, y1 = router.pos[start_node]
            x2, y2 = router.pos[end_node]
            
            # Draw arrow
            plt.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.8))
            
            # Add step numbers
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            plt.text(mid_x, mid_y, str(i+1), fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
    
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
    req_edge_list = [(u, v) for u, v in required_edges]
    nx.draw_networkx_edges(router.graph, router.pos, edgelist=req_edge_list,
                          edge_color='green', width=4, alpha=0.8, style='dashed')
    
    # Add labels
    nx.draw_networkx_labels(router.graph, router.pos, font_size=12, font_weight='bold')
    
    plt.title(f'Magnetic Field Guided Route (Cost: {cost:.2f})', 
              fontsize=16, fontweight='bold')
    
    # Add route information
    if route:
        route_text = f'Route: {" → ".join(map(str, route))}\n'
        route_text += f'Total Cost: {cost:.2f}\n'
        route_text += f'Capacity Used: {cost:.2f}/{router.capacity}'
        
        plt.text(0.02, 0.98, route_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.axis('off')
    plt.tight_layout()

def create_attraction_heatmap(router, required_edges):
    """Create a heatmap showing attraction values"""
    attractions, req_inf, depot_inf = router.calculate_total_attraction(required_edges)
    
    # Create matrix for heatmap
    nodes = list(router.graph.nodes())
    n_nodes = len(nodes)
    
    # Initialize matrices
    req_matrix = np.zeros((n_nodes, n_nodes))
    depot_matrix = np.zeros((n_nodes, n_nodes))
    total_matrix = np.zeros((n_nodes, n_nodes))
    
    # Fill matrices
    for edge in router.graph.edges():
        u, v = edge
        u_idx = nodes.index(u)
        v_idx = nodes.index(v)
        
        req_val = attractions[edge]['required_influence']
        depot_val = attractions[edge]['depot_influence'] 
        total_val = attractions[edge]['total_attraction']
        
        req_matrix[u_idx, v_idx] = req_val
        req_matrix[v_idx, u_idx] = req_val
        depot_matrix[u_idx, v_idx] = depot_val
        depot_matrix[v_idx, u_idx] = depot_val
        total_matrix[u_idx, v_idx] = total_val
        total_matrix[v_idx, u_idx] = total_val
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Required edge influence heatmap
    sns.heatmap(req_matrix, annot=True, fmt='.2f', cmap='Reds', 
                xticklabels=nodes, yticklabels=nodes, ax=axes[0])
    axes[0].set_title('Required Edge Influences')
    
    # Depot influence heatmap
    sns.heatmap(depot_matrix, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=nodes, yticklabels=nodes, ax=axes[1])
    axes[1].set_title('Depot Influences')
    
    # Total attraction heatmap
    sns.heatmap(total_matrix, annot=True, fmt='.2f', cmap='plasma',
                xticklabels=nodes, yticklabels=nodes, ax=axes[2])
    axes[2].set_title('Total Magnetic Field Attraction')
    
    plt.suptitle('Magnetic Field Attraction Heatmaps', fontsize=16, fontweight='bold')
    plt.tight_layout()

def run_complete_magnetic_field_demo():
    """Run complete demonstration of magnetic field algorithm"""
    print("Magnetic Field Vehicle Routing Algorithm Demonstration")
    print("=" * 60)
    
    # Create router
    router = MagneticFieldRouter(SIMPLE_GRAPH, START_DEPOT, END_DEPOT, VEHICLE_CAPACITY)
    
    print(f"Graph: {len(SIMPLE_GRAPH.nodes())} nodes, {len(SIMPLE_GRAPH.edges())} edges")
    print(f"Start depot: {START_DEPOT}, End depot: {END_DEPOT}")
    print(f"Vehicle capacity: {VEHICLE_CAPACITY}")
    print(f"Required edges: {REQUIRED_EDGES}")
    print(f"Failed edges: {FAILED_EDGES}")
    print()
    
    # 1. Show problem instance
    print("1. Problem Instance Visualization")
    visualize_graph_structure(router)
    plt.show()
    
    # # 2. Show required edge influences
    # print("2. Required Edge Influences")
    # visualize_required_edge_influences(router, REQUIRED_EDGES)
    # plt.show()
    
    # # 3. Show depot influences
    # print("3. Depot Influences")
    # visualize_depot_influences(router)
    # plt.show()
    
    # # 4. Show combined magnetic field
    # print("4. Combined Magnetic Field")
    # visualize_total_magnetic_field(router, REQUIRED_EDGES)
    # plt.show()
    
    # # 5. Show attraction heatmaps
    # print("5. Attraction Heatmaps")
    # create_attraction_heatmap(router, REQUIRED_EDGES)
    # plt.show()
    
    # # 6. Show route construction
    # print("6. Route Construction")
    # visualize_route_construction(router, REQUIRED_EDGES)
    # plt.show()
    
    # Calculate and display results
    route, cost, attractions = router.find_route_with_magnetic_field(REQUIRED_EDGES)
    
    print("Results:")
    print(f"Optimal Route: {route}")
    print(f"Total Cost: {cost:.2f}")
    print(f"Capacity Utilization: {cost:.2f}/{VEHICLE_CAPACITY} ({100*cost/VEHICLE_CAPACITY:.1f}%)")
    
    # Show detailed attraction values
    print("\nDetailed Attraction Values:")
    print("-" * 40)
    for edge in SIMPLE_GRAPH.edges():
        data = attractions[edge]
        print(f"Edge {edge}: Required={data['required_influence']:.3f}, "
              f"Depot={data['depot_influence']:.3f}, "
              f"Total={data['total_attraction']:.3f}")

if __name__ == "__main__":
    run_complete_magnetic_field_demo()