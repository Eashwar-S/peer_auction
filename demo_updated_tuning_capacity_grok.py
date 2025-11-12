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

class IntelligentCapacityTuner:
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
        router = MagneticFieldRouter(self.graph, self.start_depot, self.end_depot, 
                                   capacity, alpha=1.0, gamma=1.0)  # Updated alpha to 1.0
        
        route, cost, required_covered = router.find_route_with_magnetic_scoring(self.required_edges)
        
        if route is None:
            fitness = float('inf')
        else:
            if required_covered == len(self.required_edges):
                fitness = cost
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
    
    # def random_search(self, n_iterations=500):
    #     print(f"Starting random search with {n_iterations} iterations...")
        
    #     for i in range(n_iterations):
    #         capacity = random.uniform(self.max_capacity/2, self.max_capacity)
    #         result = self.evaluate_capacity(capacity)
    #         self.results.append(result)
            
    #         if result['fitness'] < self.best_score:
    #             self.best_score = result['fitness']
    #             self.best_capacity = capacity
            
    #         if (i + 1) % 50 == 0:
    #             print(f"Iteration {i+1}: Best fitness = {self.best_score:.2f}")
        
    #     return self.results
    
    def random_search(self, n_iterations=50):
        print(f"Starting random search with {n_iterations} iterations...")
        
        for i in range(n_iterations):
            # Generate a beta random variable skewed towards higher values
            beta_rv = np.random.beta(2, 1)
            # Scale to the range [self.max_capacity/3, self.max_capacity]
            capacity = self.max_capacity / 3 + beta_rv * (self.max_capacity - self.max_capacity / 3)
            # capacity = self.max_capacity * beta_rv
            result = self.evaluate_capacity(capacity)
            self.results.append(result)
            
            if result['fitness'] < self.best_score:
                self.best_score = result['fitness']
                self.best_capacity = capacity
            
            if (i + 1) % 10 == 0:  # Adjusted for fewer iterations
                print(f"Iteration {i+1}: Best fitness = {self.best_score:.2f}")
        
        return self.results
    
    def adaptive_search(self, n_iterations=500):
        print(f"Starting adaptive search with {n_iterations} iterations...")
        
        exploration_phase = n_iterations // 4
        for i in range(exploration_phase):
            capacity = random.uniform(1, self.max_capacity)
            result = self.evaluate_capacity(capacity)
            self.results.append(result)
            
            if result['fitness'] < self.best_score:
                self.best_score = result['fitness']
                self.best_capacity = capacity
        
        feasible_results = [r for r in self.results if r['feasible']]
        if feasible_results:
            feasible_results.sort(key=lambda x: x['fitness'])
            top_results = feasible_results[:max(1, len(feasible_results) // 5)]
            capacity_values = [r['capacity'] for r in top_results]
            capacity_mean, capacity_std = np.mean(capacity_values), np.std(capacity_values)
            
            print(f"Good capacity region: {capacity_mean:.3f}±{capacity_std:.3f}")
            
            for i in range(exploration_phase, n_iterations):
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
        print(f"Starting binary search for minimum feasible capacity...")
        
        low, high = 1, self.max_capacity
        best_feasible_capacity = None
        
        while high - low > 0.1:
            mid = (low + high) / 2
            result = self.evaluate_capacity(mid)
            self.results.append(result)
            
            if result['feasible']:
                best_feasible_capacity = mid
                high = mid
                print(f"Feasible at capacity {mid:.2f}, trying smaller...")
            else:
                low = mid + 0.1
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
        print(f"Starting Bayesian optimization with {n_iterations} iterations...")
        
        def objective(capacity_array):
            capacity = np.clip(capacity_array[0], 1, self.max_capacity)
            result = self.evaluate_capacity(capacity)
            self.results.append(result)
            
            if result['fitness'] < self.best_score:
                self.best_score = result['fitness']
                self.best_capacity = capacity
            
            return result['fitness']
        
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
        if not self.results:
            print("No results to analyze")
            return
        
        df = pd.DataFrame(self.results)
        print("\n" + "="*60)
        print("CAPACITY TUNING ANALYSIS")
        print("="*60)
        
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
        
        if feasible_count > 5:
            feasible_df = df[df['feasible']]
            top_10_percent = feasible_df.nsmallest(max(1, len(feasible_df) // 10), 'fitness')
            print(f"\nTop 10% solutions capacity range:")
            print(f"Capacity: {top_10_percent['capacity'].min():.3f} - {top_10_percent['capacity'].max():.3f}")
        
        return df
    
    def visualize_results(self):
        if not self.results:
            print("No results to visualize")
            return
        
        df = pd.DataFrame(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
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
            
            if self.best_capacity:
                axes[0, 0].scatter(self.best_capacity, self.best_score, 
                                  color='blue', s=100, marker='*', label='Best', zorder=5)
                axes[0, 0].legend()
        
        feasible_df = df[df['feasible']]
        infeasible_df = df[~df['feasible']]
        
        if len(feasible_df) > 0 and len(infeasible_df) > 0:
            axes[0, 1].hist(infeasible_df['capacity'], bins=20, alpha=0.5, color='red', label='Infeasible')
            axes[0, 1].hist(feasible_df['capacity'], bins=20, alpha=0.7, color='green', label='Feasible')
            axes[0, 1].set_xlabel('Capacity')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Capacity Distribution')
            axes[0, 1].legend()
        
        if len(feasible_df) > 0:
            axes[0, 2].hist(feasible_df['cost'], bins=20, alpha=0.7, color='blue')
            axes[0, 2].axvline(feasible_df['cost'].mean(), color='red', linestyle='--', 
                              label=f'Mean: {feasible_df["cost"].mean():.2f}')
            axes[0, 2].set_xlabel('Route Cost')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Cost Distribution (Feasible Solutions)')
            axes[0, 2].legend()
        
        if len(feasible_df) > 0:
            axes[1, 0].scatter(feasible_df['capacity'], feasible_df['cost'], alpha=0.6, color='blue')
            axes[1, 0].set_xlabel('Capacity')
            axes[1, 0].set_ylabel('Cost')
            axes[1, 0].set_title('Capacity vs Cost (Feasible Solutions)')
            axes[1, 0].grid(True, alpha=0.3)
            
            if len(feasible_df) > 5:
                z = np.polyfit(feasible_df['capacity'], feasible_df['cost'], 1)
                p = np.poly1d(z)
                axes[1, 0].plot(feasible_df['capacity'], p(feasible_df['capacity']), "r--", alpha=0.8)
        
        if len(df) > 0:
            axes[1, 1].scatter(df['capacity'], df['required_covered'], alpha=0.6)
            axes[1, 1].set_xlabel('Capacity')
            axes[1, 1].set_ylabel('Required Edges Covered')
            axes[1, 1].set_title('Capacity vs Required Edge Coverage')
            axes[1, 1].grid(True, alpha=0.3)
        
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
    print("Intelligent Capacity Tuning for Magnetic Field Routing")
    print("=" * 65)
    
    tuner = IntelligentCapacityTuner(SIMPLE_GRAPH, START_DEPOT, END_DEPOT, 
                                   MAX_VEHICLE_CAPACITY, REQUIRED_EDGES + FAILED_EDGES)
    
    print(f"Graph: {len(SIMPLE_GRAPH.nodes())} nodes, {len(SIMPLE_GRAPH.edges())} edges")
    print(f"Required edges: {REQUIRED_EDGES}")
    print(f"Failed edges: {FAILED_EDGES}")
    print(f"Maximum vehicle capacity: {MAX_VEHICLE_CAPACITY}")
    print(f"Alpha fixed at: 1.0")
    print(f"Gamma fixed at: 1.0")
    
    strategies = [
        ("Random Search", lambda: tuner.random_search(50)),
        ("Adaptive Search", lambda: tuner.adaptive_search(50)),
        ("Grid Search", lambda: tuner.grid_search(50)),
        ("Binary Search", lambda: tuner.binary_search()),
        ("Bayesian Optimization", lambda: tuner.bayesian_optimization(50))
    ]
    
    strategy_name, strategy_func = strategies[0]  # Using Random Search
    # strategy_name, strategy_func = strategies[1]  # Adaptive search
    # strategy_name, strategy_func = strategies[2]  # Grid search
    # strategy_name, strategy_func = strategies[3]  # Binary search
    # strategy_name, strategy_func = strategies[4]  # Bayesian optimization
    print(f"\nRunning {strategy_name}...")
    results = strategy_func()
    
    df = tuner.analyze_results()
    
    if tuner.best_capacity:
        print(f"\nTesting best capacity: {tuner.best_capacity:.3f}")
        best_router = MagneticFieldRouter(SIMPLE_GRAPH, START_DEPOT, END_DEPOT, 
                                        tuner.best_capacity, alpha=1.0, gamma=1.0)
        route, cost, required_covered = best_router.find_route_with_magnetic_scoring(REQUIRED_EDGES + FAILED_EDGES, verbose=True)
        
        if route:
            print(f"Best route: {route}")
            print(f"Cost: {cost:.2f}")
            print(f"Required edges covered: {required_covered}/{len(REQUIRED_EDGES + FAILED_EDGES)}")
            print(f"Capacity utilization: {cost:.2f}/{tuner.best_capacity:.2f} ({100*cost/tuner.best_capacity:.1f}%)")
        else:
            print("No feasible route found with best capacity")
    
    tuner.visualize_results()
    return tuner, df

if __name__ == "__main__":
    tuner, results_df = run_intelligent_tuning_demo()