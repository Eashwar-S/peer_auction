"""
Intelligent Parameter Tuning for Magnetic Field Vehicle Routing Algorithm
========================================================================
This extends the original algorithm with intelligent parameter exploration
to find optimal alpha and gamma values that minimize trip time while ensuring
all required edges are traversed.
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
VEHICLE_CAPACITY = 305
REQUIRED_EDGES = [(1, 2), (3, 4), (3, 5)]  # Only 3 required edges for clarity
FAILED_EDGES = [(2, 4), (1, 4)]  # Only 2 failed edges to handle

class MagneticFieldRouter:
    """
    Magnetic Field Vehicle Routing Algorithm with intelligent parameter tuning
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
    #     """Find route using magnetic field scoring"""
    #     # Start building route from start depot
    #     current_route = [self.start_depot]
    #     current_length = 0
    #     visited_edges = set()
    #     required_covered = set()
        
    #     while len(required_covered) < len(required_edges) or current_route[-1] != self.end_depot:
    #         current_node = current_route[-1]
    #         candidates = []
            
    #         # Get all possible next edges
    #         for neighbor in self.graph.neighbors(current_node):
    #             edge = (current_node, neighbor)
    #             edge_sorted = tuple(sorted(edge))
                
    #             # Skip if already visited this edge
    #             if edge_sorted in visited_edges:
    #                 continue
                
    #             edge_weight = self.graph[current_node][neighbor]['weight']
                
    #             # Check capacity constraint
    #             if current_length + edge_weight > self.capacity:
    #                 continue
                
    #             # Check if this is a new required edge
    #             is_new_required = (edge_sorted in [tuple(sorted(req)) for req in required_edges] 
    #                              and edge_sorted not in required_covered)
                
    #             # Calculate magnetic field score
    #             score_data = self.calculate_edge_score(edge, required_edges, current_length, is_new_required)
                
    #             candidates.append({
    #                 'edge': edge,
    #                 'neighbor': neighbor,
    #                 'is_new_required': is_new_required,
    #                 'score_data': score_data
    #             })
            
    #         if not candidates:
    #             break
            
    #         # Sort candidates by score (higher is better), then by normalized edge weight (lower is better)
    #         candidates.sort(key=lambda x: (x['score_data']['final_score'], 
    #                                      -x['score_data']['normalized_weight']), reverse=True)
            
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
        
    #     print(f"Current route: {current_route}, Length: {current_length}, Required covered: {len(required_covered)}")
    #     # If we haven't reached the end depot, try to get there
    #     if current_route[-1] != self.end_depot:
    #         try:
    #             path_to_end = nx.shortest_path(self.graph, current_route[-1], self.end_depot, weight='weight')
    #             additional_length = nx.shortest_path_length(self.graph, current_route[-1], self.end_depot, weight='weight')
                
    #             if current_length + additional_length <= self.capacity:
    #                 current_route.extend(path_to_end[1:])
    #                 current_length += additional_length
    #             else:
    #                 return None, float('inf'), len(required_covered)
    #         except nx.NetworkXNoPath:
    #             return None, float('inf'), len(required_covered)
        
    #     return current_route, current_length, len(required_covered)

    def find_route_with_magnetic_scoring(self, required_edges, verbose=False):
        """Find route using magnetic field scoring, ensuring we can always reach end depot"""
        # Start building route from start depot
        current_route = [self.start_depot]
        current_length = 0
        visited_edges = set()
        required_covered = set()
        
        while True:
            current_node = current_route[-1]
            candidates = []
            
            # If we're at the end depot and have covered some required edges, we can stop
            if current_node == self.end_depot and len(required_covered) > 0:
                break
                
            # If we're at the end depot but haven't covered any required edges yet, continue exploring
            if current_node == self.end_depot and len(required_covered) == 0:
                # Only continue if we have capacity and there are unvisited edges
                pass
            
            # Get all possible next edges
            for neighbor in self.graph.neighbors(current_node):
                edge = (current_node, neighbor)
                edge_sorted = tuple(sorted(edge))
                
                # Skip if already visited this edge
                if edge_sorted in visited_edges:
                    continue
                
                edge_weight = self.graph[current_node][neighbor]['weight']
                
                # Critical check: Can we reach end depot after taking this edge?
                try:
                    # Calculate shortest path from neighbor to end depot
                    if neighbor == self.end_depot:
                        path_to_end_length = 0
                    else:
                        path_to_end_length = nx.shortest_path_length(
                            self.graph, neighbor, self.end_depot, weight='weight'
                        )
                    
                    # Check if we can take this edge AND still reach end depot within capacity
                    total_required_capacity = current_length + edge_weight + path_to_end_length
                    
                    if total_required_capacity > self.capacity:
                        continue  # Skip this edge as it would prevent reaching end depot
                        
                except nx.NetworkXNoPath:
                    # If no path exists from neighbor to end depot, skip this edge
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
                    'score_data': score_data,
                    'path_to_end_length': path_to_end_length
                })
            
            # If no valid candidates, we must go directly to end depot
            if not candidates:
                if current_node != self.end_depot:
                    try:
                        path_to_end = nx.shortest_path(self.graph, current_node, self.end_depot, weight='weight')
                        additional_length = nx.shortest_path_length(self.graph, current_node, self.end_depot, weight='weight')
                        
                        if current_length + additional_length <= self.capacity:
                            current_route.extend(path_to_end[1:])
                            current_length += additional_length
                        else:
                            # This shouldn't happen with our improved logic, but safety check
                            if verbose:
                                print("ERROR: Cannot reach end depot within capacity!")
                            return None, float('inf'), len(required_covered)
                    except nx.NetworkXNoPath:
                        if verbose:
                            print("ERROR: No path to end depot!")
                        return None, float('inf'), len(required_covered)
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
                if verbose:
                    print(f"Covered required edge: {selected_edge}")
            
            if verbose:
                print(f"Step: {selected_edge} -> Node {selected_neighbor}, "
                    f"Length: {current_length:.2f}, Required: {len(required_covered)}/{len(required_edges)}")
        
        # Ensure we end at the end depot (should already be there with improved logic)
        if current_route[-1] != self.end_depot:
            try:
                path_to_end = nx.shortest_path(self.graph, current_route[-1], self.end_depot, weight='weight')
                additional_length = nx.shortest_path_length(self.graph, current_route[-1], self.end_depot, weight='weight')
                current_route.extend(path_to_end[1:])
                current_length += additional_length
            except nx.NetworkXNoPath:
                if verbose:
                    print("Final ERROR: No path to end depot!")
                return None, float('inf'), len(required_covered)
        
        if verbose:
            print(f"Final route: {current_route}, Length: {current_length:.2f}, "
                f"Required covered: {len(required_covered)}/{len(required_edges)}")
        
        return current_route, current_length, len(required_covered)

class IntelligentParameterTuner:
    """
    Intelligent parameter tuner that explores alpha and gamma values
    to optimize route quality
    """
    
    def __init__(self, graph, start_depot, end_depot, capacity, required_edges):
        self.graph = graph
        self.start_depot = start_depot
        self.end_depot = end_depot
        self.capacity = capacity
        self.required_edges = required_edges
        self.results = []
        self.best_params = None
        self.best_score = float('inf')
        
    def evaluate_parameters(self, alpha, gamma):
        """Evaluate a specific parameter combination"""
        router = MagneticFieldRouter(self.graph, self.start_depot, self.end_depot, 
                                   self.capacity, alpha, gamma)
        
        route, cost, required_covered = router.find_route_with_magnetic_scoring(self.required_edges)
        
        # Calculate fitness score
        if route is None:
            fitness = float('inf')
        else:
            # Primary objective: minimize cost
            # Secondary objective: maximize required edges covered
            # required_penalty = 1000 * (len(self.required_edges) - required_covered)
            # fitness = cost + required_penalty

            if required_covered == len(self.required_edges):
                fitness = 1000 * cost
            else:
                missing_edges = len(self.required_edges) - required_covered
                fitness = cost + 1000 * (missing_edges ** 2) + 500
        
        return {
            'alpha': alpha,
            'gamma': gamma,
            'route': route,
            'cost': cost,
            'required_covered': required_covered,
            'fitness': fitness,
            'feasible': route is not None and required_covered == len(self.required_edges)
        }
    
    def random_search(self, n_iterations=500):
        """Random parameter search"""
        print(f"Starting random search with {n_iterations} iterations...")
        
        for i in range(n_iterations):
            # Generate random parameters
            alpha = random.uniform(0.1, 0.9)
            gamma = random.uniform(0.1, 0.9)
            
            result = self.evaluate_parameters(alpha, gamma)
            self.results.append(result)
            
            # Update best if this is better
            if result['fitness'] < self.best_score:
                self.best_score = result['fitness']
                self.best_params = (alpha, gamma)
            
            if (i + 1) % 50 == 0:
                print(f"Iteration {i+1}: Best fitness = {self.best_score:.2f}")
        
        return self.results
    
    def adaptive_search(self, n_iterations=500):
        """Adaptive parameter search that learns from good solutions"""
        print(f"Starting adaptive search with {n_iterations} iterations...")
        
        # Phase 1: Random exploration (first 25% of iterations)
        exploration_phase = n_iterations // 4
        for i in range(exploration_phase):
            alpha = random.uniform(0.1, 0.9)
            gamma = random.uniform(0.1, 0.9)
            
            result = self.evaluate_parameters(alpha, gamma)
            self.results.append(result)
            
            if result['fitness'] < self.best_score:
                self.best_score = result['fitness']
                self.best_params = (alpha, gamma)
        
        # Phase 2: Guided search based on good solutions
        # Find top 20% of feasible solutions
        feasible_results = [r for r in self.results if r['feasible']]
        if feasible_results:
            feasible_results.sort(key=lambda x: x['fitness'])
            top_results = feasible_results[:max(1, len(feasible_results) // 5)]
            
            # Calculate mean and std of good parameters
            alpha_values = [r['alpha'] for r in top_results]
            gamma_values = [r['gamma'] for r in top_results]
            
            alpha_mean, alpha_std = np.mean(alpha_values), np.std(alpha_values)
            gamma_mean, gamma_std = np.mean(gamma_values), np.std(gamma_values)
            
            print(f"Good parameter regions: α={alpha_mean:.3f}±{alpha_std:.3f}, γ={gamma_mean:.3f}±{gamma_std:.3f}")
            
            # Phase 2: Sample around good regions
            for i in range(exploration_phase, n_iterations):
                # Sample from normal distribution around good parameters
                alpha = np.clip(np.random.normal(alpha_mean, max(0.1, alpha_std)), 0.1, 0.9)
                gamma = np.clip(np.random.normal(gamma_mean, max(0.1, gamma_std)), 0.1, 0.9)
                
                result = self.evaluate_parameters(alpha, gamma)
                self.results.append(result)
                
                if result['fitness'] < self.best_score:
                    self.best_score = result['fitness']
                    self.best_params = (alpha, gamma)
                
                if (i + 1) % 50 == 0:
                    print(f"Iteration {i+1}: Best fitness = {self.best_score:.2f}")
        
        return self.results
    
    def grid_search(self, n_points=20):
        """Grid search over parameter space"""
        print(f"Starting grid search with {n_points}x{n_points} grid...")
        
        alpha_values = np.linspace(0.1, 0.9, n_points)
        gamma_values = np.linspace(0.1, 0.9, n_points)
        
        total_evals = len(alpha_values) * len(gamma_values)
        
        for i, alpha in enumerate(alpha_values):
            for j, gamma in enumerate(gamma_values):
                result = self.evaluate_parameters(alpha, gamma)
                self.results.append(result)
                
                if result['fitness'] < self.best_score:
                    self.best_score = result['fitness']
                    self.best_params = (alpha, gamma)
                
                if len(self.results) % 50 == 0:
                    print(f"Evaluation {len(self.results)}/{total_evals}: Best fitness = {self.best_score:.2f}")
        
        return self.results
    
    def bayesian_optimization(self, n_iterations=100):
        """Simple Bayesian optimization using scipy"""
        print(f"Starting Bayesian optimization with {n_iterations} iterations...")
        
        def objective(params):
            alpha, gamma = params
            alpha = np.clip(alpha, 0.1, 0.9)
            gamma = np.clip(gamma, 0.1, 0.9)
            
            result = self.evaluate_parameters(alpha, gamma)
            self.results.append(result)
            
            if result['fitness'] < self.best_score:
                self.best_score = result['fitness']
                self.best_params = (alpha, gamma)
            
            return result['fitness']
        
        # Multiple random starts
        best_result = None
        for start in range(5):
            initial_guess = [random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)]
            
            try:
                result = minimize(objective, initial_guess, 
                                bounds=[(0.1, 0.9), (0.1, 0.9)],
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
        print("PARAMETER TUNING ANALYSIS")
        print("="*60)
        
        # Overall statistics
        feasible_count = sum(1 for r in self.results if r['feasible'])
        print(f"Total evaluations: {len(self.results)}")
        print(f"Feasible solutions: {feasible_count} ({100*feasible_count/len(self.results):.1f}%)")
        
        if feasible_count > 0:
            feasible_df = df[df['feasible']]
            print(f"Best fitness: {self.best_score:.2f}")
            print(f"Best parameters: α={self.best_params[0]:.3f}, γ={self.best_params[1]:.3f}")
            print(f"Average feasible cost: {feasible_df['cost'].mean():.2f}")
            print(f"Cost std: {feasible_df['cost'].std():.2f}")
        
        # Parameter analysis
        if feasible_count > 5:
            feasible_df = df[df['feasible']]
            top_10_percent = feasible_df.nsmallest(max(1, len(feasible_df) // 10), 'fitness')
            
            print(f"\nTop 10% solutions parameter ranges:")
            print(f"α: {top_10_percent['alpha'].min():.3f} - {top_10_percent['alpha'].max():.3f}")
            print(f"γ: {top_10_percent['gamma'].min():.3f} - {top_10_percent['gamma'].max():.3f}")
        
        return df
    
    def visualize_results(self):
        """Create visualizations of the tuning results"""
        if not self.results:
            print("No results to visualize")
            return
        
        df = pd.DataFrame(self.results)
        
        # Create a comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Parameter space heatmap
        if len(df) > 50:
            # Create grid for heatmap
            alpha_bins = np.linspace(0.1, 0.9, 20)
            gamma_bins = np.linspace(0.1, 0.9, 20)
            
            # Bin the data
            alpha_idx = np.digitize(df['alpha'], alpha_bins) - 1
            gamma_idx = np.digitize(df['gamma'], gamma_bins) - 1
            
            # Create fitness grid
            fitness_grid = np.full((len(gamma_bins), len(alpha_bins)), np.nan)
            for i in range(len(df)):
                if 0 <= alpha_idx[i] < len(alpha_bins) and 0 <= gamma_idx[i] < len(gamma_bins):
                    current_fitness = fitness_grid[gamma_idx[i], alpha_idx[i]]
                    new_fitness = df.iloc[i]['fitness']
                    if np.isnan(current_fitness) or new_fitness < current_fitness:
                        fitness_grid[gamma_idx[i], alpha_idx[i]] = new_fitness
            
            # Plot heatmap
            im = axes[0, 0].imshow(fitness_grid, cmap='viridis_r', aspect='auto', 
                                  extent=[0.1, 0.9, 0.1, 0.9], origin='lower')
            axes[0, 0].set_xlabel('Alpha')
            axes[0, 0].set_ylabel('Gamma')
            axes[0, 0].set_title('Parameter Space Fitness Heatmap')
            plt.colorbar(im, ax=axes[0, 0], label='Fitness')
            
            # Mark best point
            if self.best_params:
                axes[0, 0].scatter(self.best_params[0], self.best_params[1], 
                                  color='red', s=100, marker='*', label='Best')
                axes[0, 0].legend()
        
        # 2. Feasible vs infeasible scatter
        feasible_df = df[df['feasible']]
        infeasible_df = df[~df['feasible']]
        
        if len(infeasible_df) > 0:
            axes[0, 1].scatter(infeasible_df['alpha'], infeasible_df['gamma'], 
                              c='red', alpha=0.5, s=20, label='Infeasible')
        if len(feasible_df) > 0:
            axes[0, 1].scatter(feasible_df['alpha'], feasible_df['gamma'], 
                              c='green', alpha=0.7, s=20, label='Feasible')
        
        axes[0, 1].set_xlabel('Alpha')
        axes[0, 1].set_ylabel('Gamma')
        axes[0, 1].set_title('Feasible vs Infeasible Solutions')
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
        
        # 4. Parameter correlation with cost
        if len(feasible_df) > 0:
            axes[1, 0].scatter(feasible_df['alpha'], feasible_df['cost'], alpha=0.6)
            axes[1, 0].set_xlabel('Alpha')
            axes[1, 0].set_ylabel('Cost')
            axes[1, 0].set_title('Alpha vs Cost')
            
            axes[1, 1].scatter(feasible_df['gamma'], feasible_df['cost'], alpha=0.6)
            axes[1, 1].set_xlabel('Gamma')
            axes[1, 1].set_ylabel('Cost')
            axes[1, 1].set_title('Gamma vs Cost')
        
        # 5. Convergence plot
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
        plt.suptitle('Parameter Tuning Analysis', fontsize=16, y=1.02)
        plt.show()

def run_intelligent_tuning_demo():
    """Run the complete intelligent parameter tuning demonstration"""
    print("Intelligent Parameter Tuning for Magnetic Field Routing")
    print("=" * 65)
    
    # Create tuner
    tuner = IntelligentParameterTuner(SIMPLE_GRAPH, START_DEPOT, END_DEPOT, 
                                    VEHICLE_CAPACITY, REQUIRED_EDGES + FAILED_EDGES)
    
    print(f"Graph: {len(SIMPLE_GRAPH.nodes())} nodes, {len(SIMPLE_GRAPH.edges())} edges")
    print(f"Required edges: {REQUIRED_EDGES}")
    print(f"Failed edges: {FAILED_EDGES}")
    print(f"Vehicle capacity: {VEHICLE_CAPACITY}")
    
    # Test different search strategies
    strategies = [
        ("Random Search", lambda: tuner.random_search(200)),
        ("Adaptive Search", lambda: tuner.adaptive_search(200)),
        ("Grid Search", lambda: tuner.grid_search(15)),
        ("Bayesian Optimization", lambda: tuner.bayesian_optimization(100))
    ]
    
    # Run one strategy (you can modify this to run multiple)
    strategy_name, strategy_func = strategies[1]  # Adaptive search

    # strategy_name, strategy_func = strategies[2]  # Grid search

    # strategy_name, strategy_func = strategies[3]  # Bayesian optimization

    # strategy_name, strategy_func = strategies[0]  # Random search

    print(f"\nRunning {strategy_name}...")
    results = strategy_func()
    
    # Analyze results
    df = tuner.analyze_results()
    
    # Test the best parameters
    if tuner.best_params:
        print(f"\nTesting best parameters: α={tuner.best_params[0]:.3f}, γ={tuner.best_params[1]:.3f}")
        
        best_router = MagneticFieldRouter(SIMPLE_GRAPH, START_DEPOT, END_DEPOT, 
                                        VEHICLE_CAPACITY, tuner.best_params[0], tuner.best_params[1])
        
        route, cost, required_covered = best_router.find_route_with_magnetic_scoring(REQUIRED_EDGES + FAILED_EDGES, verbose=True)
        
        if route:
            print(f"Best route: {route}")
            print(f"Cost: {cost:.2f}")
            print(f"Required edges covered: {required_covered}/{len(REQUIRED_EDGES + FAILED_EDGES)}")
        else:
            print("No feasible route found with best parameters")
    
    # Visualize results
    tuner.visualize_results()
    
    return tuner, df

if __name__ == "__main__":
    tuner, results_df = run_intelligent_tuning_demo()