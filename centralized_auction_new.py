import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import operator as op
import time
import os
import pandas as pd
import copy
import re
from demo_updated_tuning_capacity_grok_v10 import MagneticFieldRouter
plt.rcParams["figure.dpi"] = 300


def parse_txt_file(file_path):
    """Parse the .txt failure scenario file to extract graph and parameters"""
    G = nx.Graph()
    required_edges = []
    depot_nodes = []
    vehicle_capacity = None
    recharge_time = None
    num_vehicles = None
    failure_vehicles = []
    vehicle_failure_times = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('NUMBER OF VERTICES'):
            num_vertices = int(line.split(':')[1].strip())
            G.add_nodes_from(range(1, num_vertices + 1))
        elif line.startswith('VEHICLE CAPACITY'):
            vehicle_capacity = float(line.split(':')[1].strip())
        elif line.startswith('RECHARGE TIME'):
            recharge_time = float(line.split(':')[1].strip())
        elif line.startswith('NUMBER OF VEHICLES'):
            num_vehicles = int(line.split(':')[1].strip())
        elif line.startswith('DEPOT:'):
            depot_line = line.split(':', 1)[1].strip()
            depot_nodes = [int(x.strip()) for x in depot_line.split(',')]
        elif line.startswith('LIST_REQUIRED_EDGES:'):
            current_section = 'required_edges'
        elif line.startswith('LIST_NON_REQUIRED_EDGES:'):
            current_section = 'non_required_edges'
        elif line.startswith('FAILURE_SCENARIO:'):
            current_section = 'failure_scenario'
        elif line.startswith('(') and current_section in ['required_edges', 'non_required_edges']:
            # Parse edge: (u,v) edge weight w
            parts = line.split(') ')
            if len(parts) >= 2:
                edge_part = parts[0].strip('(')
                u, v = map(int, edge_part.split(','))
                weight_part = parts[1].strip()
                weight = float(weight_part.split()[-1])
                G.add_edge(u, v, weight=weight)
                if current_section == 'required_edges':
                    required_edges.append([u, v])
        elif current_section == 'failure_scenario' and line.startswith('Vehicle'):
            # Parse: Vehicle X will fail in Y time units.
            match = re.search(r'Vehicle (\d+) will fail in (\d+) time units', line)
            if match:
                vehicle_id = int(match.group(1)) - 1  # Convert to 0-indexed
                failure_time = float(match.group(2))
                failure_vehicles.append(vehicle_id)
                vehicle_failure_times.append(failure_time)
    
    return G, required_edges, depot_nodes, vehicle_capacity, recharge_time, num_vehicles, failure_vehicles, vehicle_failure_times

def calculate_route_times(G, routes):
    """Calculate trip times for each vehicle's routes"""
    vehicle_trip_times = []
    for vehicle_routes in routes:
        trip_times = []
        for trip in vehicle_routes:
            trip_time = 0
            for i in range(len(trip) - 1):
                if G.has_edge(trip[i], trip[i + 1]):
                    trip_time += G[trip[i]][trip[i + 1]]['weight']
                else:
                    print(f"Warning: Edge ({trip[i]}, {trip[i + 1]}) not found in graph")
            trip_times.append(trip_time)
        vehicle_trip_times.append(trip_times)
    return vehicle_trip_times

def vehicle_at_time_t(t, vehicle_routes, vehicle_trip_times, recharge_time, failure_history,
                      idle_time, recent_failure_vehicle):
    vehicle_trip_index = [-1] * len(vehicle_routes)

    for k, routes in enumerate(vehicle_routes):
        sum_distance = 0
        if k not in failure_history or k == recent_failure_vehicle:
            if routes:
                for trip_index, trip in enumerate(routes):
                    if trip_index == len(routes) - 1:
                        sum_distance += vehicle_trip_times[k][trip_index]
                        vehicle_trip_index[k] = trip_index
                        ready_time = sum_distance + recharge_time  # Vehicle is ready after recharging
                        if ready_time < t:  # Only idle if current time exceeds ready time
                            # if trip_index in idle_time[k]:
                            idle_time[k][trip_index] = round(t - ready_time, 1)
                            # else:
                            #     idle_time[k][trip_index] = round(t - ready_time, 1)
                        # if sum_distance < t:
                        #     if trip_index in idle_time[k]:
                        #         idle_time[k][trip_index] += round(t - sum_distance, 1)
                        #     else:
                        #         idle_time[k][trip_index] = round(t - sum_distance, 1)
                    else:
                        if trip_index not in idle_time[k]:
                            idle_time[k][trip_index] = 0
                        sum_distance += vehicle_trip_times[k][trip_index] + idle_time[k][trip_index] + recharge_time
                        if sum_distance >= t:
                            vehicle_trip_index[k] = trip_index
                            break
            else:
                idle_time[k][0] = t
                vehicle_trip_index[k] = 0
        sum_distance = round(sum_distance, 1)
        # print(f'vehicle {k+1} - vehicle trip index - {vehicle_trip_index[k]}, sum_distance - {sum_distance}')
    return vehicle_trip_index, idle_time

# def identifying_failed_trips(vehicle_routes, vehicle_trip_times, vehicle_trip_index, recent_failure_vehicle, required_edges):
#     failed_trips = {}
#     for trip_index, trip in enumerate(vehicle_routes[recent_failure_vehicle]):
#         if trip_index >= vehicle_trip_index[recent_failure_vehicle]:
#             for i in range(len(trip) - 1):
#                 if [trip[i], trip[i+1]] in required_edges or [trip[i+1], trip[i]] in required_edges:
#                     failed_trips[f"{vehicle_trip_times[recent_failure_vehicle][trip_index]}_{trip_index}_{recent_failure_vehicle}"] = trip
#                     break
    
#     return failed_trips

def identifying_failed_trips(vehicle_routes, vehicle_trip_times, vehicle_trip_index, recent_failure_vehicle, required_edges):
    failed_trips = {}
    seen = set()  # canonicalized trips we've already kept

    for trip_index, trip in enumerate(vehicle_routes[recent_failure_vehicle]):
        if trip_index >= vehicle_trip_index[recent_failure_vehicle]:
            # Does this trip cover at least one required edge?
            touches_required = any(
                ([trip[i], trip[i + 1]] in required_edges) or ([trip[i + 1], trip[i]] in required_edges)
                for i in range(len(trip) - 1)
            )
            if not touches_required:
                continue

            # Canonicalize trip to catch duplicates and reverse-duplicates
            t = tuple(trip)
            canon = t if t <= t[::-1] else t[::-1]

            if canon in seen:
                continue  # skip duplicates
            seen.add(canon)

            key = f"{vehicle_trip_times[recent_failure_vehicle][trip_index]}_{trip_index}_{recent_failure_vehicle}"
            failed_trips[key] = trip

    return failed_trips

def calculate_depot_distances(G, depots, C, recharge_time):
    depot_to_depot_distances = {}
    for i, depot1 in enumerate(depots):
        for depot2 in depots[i:]:
            try:
                if depot1 == depot2:
                    depot_to_depot_distances[(depot1, depot2)] = [0, []]
                    depot_to_depot_distances[(depot2, depot1)] = [0, []]
                else:
                    distance, path = nx.single_source_dijkstra(G, depot1, depot2, weight='weight', cutoff=C)
                    distance = round(distance, 1)
                    depot_to_depot_distances[(depot1, depot2)] = [distance, [path]]
                    depot_to_depot_distances[(depot2, depot1)] = [distance, [path[::-1]]]
            except nx.NetworkXNoPath:
                pass
    for k in depots:
        for i in depots:
            for j in depots:
                if (i, k) in depot_to_depot_distances and (k, j) in depot_to_depot_distances:
                    dist_ik, path_ik = depot_to_depot_distances[(i, k)]
                    dist_kj, path_kj = depot_to_depot_distances[(k, j)]
                    total_dist = round(dist_ik + dist_kj + recharge_time, 1)
                    if (i, j) not in depot_to_depot_distances or total_dist < depot_to_depot_distances[(i, j)][0]:
                        depot_to_depot_distances[(i, j)] = [total_dist, path_ik + path_kj]
    return depot_to_depot_distances

def search(G, failure_history, vehicle_routes, trip_indexes, failure_trip_depots, 
           vehicle_capacity, depot_nodes, search_radius, uavLocation):
    vehicle_dict = {k: {} for k in range(len(vehicle_routes))}
    for k in range(len(vehicle_routes)):
        if k not in failure_history:
            n = len(vehicle_routes[k])
            for trip_ind in range(trip_indexes[k], n):
                vehicle_dict[k][vehicle_routes[k][trip_ind][-1]] = trip_ind + 1
            if trip_indexes[k] >= n:  # Vehicle has finished all trips
                last_depot = vehicle_routes[k][-1][-1] if n > 0 else uavLocation[k]
                vehicle_dict[k][last_depot] = n  # Allow insertion at the end

    vehicle_insertion_list = {}
    iteration = 1
    while len(vehicle_insertion_list) == 0:
        iteration += 1
        subgraph_nodes = []
        for depot in failure_trip_depots:
            subgraph_nodes += list(nx.single_source_dijkstra_path_length(G, depot, 
                                                                         cutoff=search_radius, weight='weight').keys())
        subgraph_nodes = list(set(subgraph_nodes))
        subgraph_depots = [node for node in subgraph_nodes if node in depot_nodes]
        for k, depots in vehicle_dict.items():
            insertion_indices = []
            for depot, trip_ins_index in depots.items():
                if depot in subgraph_depots:
                    insertion_indices.append(trip_ins_index)
            if insertion_indices:
                vehicle_insertion_list[k] = list(set(insertion_indices))
        search_radius += vehicle_capacity
    # print(f'vehicle insertion list - {vehicle_insertion_list}')
    return vehicle_insertion_list


def bidding_optimized(t, vehicle_routes, vehicle_trip_times, vehicle_insertion_list,
                      failure_trip_info, depot_to_depot_distances, recharge_time, uavLocation):
    bid_info = {}
    failure_trip_time, failure_trip = failure_trip_info
    # # print(f'failure trip info - {failure_trip}')
    mission_time = max([sum(vehicle_trip_times[k]) + 
                        (len(vehicle_trip_times[k]) - 1) * recharge_time for k in range(len(vehicle_routes))])
    for k, trip_insertions in vehicle_insertion_list.items():
        min_bid = np.inf
        best_route = None
        best_insertion = None
        # # print(f'Vehicle {k+1} bids: ')
        y_k = sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time
        ready_time = y_k + recharge_time if len(vehicle_routes[k]) > 0 else 0  # Time vehicle is ready after recharging
        for trip_ins_index in trip_insertions:
            if trip_ins_index == 0:
                d1 = uavLocation[k]
            else:
                d1 = vehicle_routes[k][trip_ins_index - 1][-1]
            d2 = failure_trip[0][0]
            d3 = failure_trip[0][-1]
            if trip_ins_index == len(vehicle_routes[k]):
                possible_bids = {}
                # if y_k > t:
                #     bid = y_k - t
                if depot_to_depot_distances[(d1,d2)][0] == 0:
                    bid = failure_trip_time
                    route = failure_trip
                    possible_bids[bid] = route
                elif depot_to_depot_distances[(d1,d3)][0] == 0:
                    bid = failure_trip_time
                    route = [failure_trip[0][::-1]]
                    possible_bids[bid] = route
                elif depot_to_depot_distances[(d1,d2)][0] != 0:
                    bid = recharge_time + depot_to_depot_distances[(d1,d2)][0] + failure_trip_time
                    route = depot_to_depot_distances[(d1,d2)][1] + failure_trip
                    possible_bids[bid] = route
                elif depot_to_depot_distances[(d1,d3)][0] != 0:
                    bid = recharge_time + depot_to_depot_distances[(d1,d3)][0] + failure_trip_time
                    route = depot_to_depot_distances[(d1,d3)][1] + [failure_trip[0][::-1]]
                    possible_bids[bid] = route
                bid = np.inf
                route = None
                for p_b, p_r in possible_bids.items():
                    if p_b < bid:
                        bid = p_b
                        route = p_r
            else:
                bid = recharge_time
                if depot_to_depot_distances[(d1,d2)][0] != 0:
                    bid += recharge_time
                bid += failure_trip_time
                if depot_to_depot_distances[(d3,d1)][0] != 0:
                    bid += recharge_time
                route = depot_to_depot_distances[(d1,d2)][1] + failure_trip + depot_to_depot_distances[(d3,d1)][1]
            bid = round(max(y_k, t) + bid - mission_time, 1)
            if bid < min_bid:
                min_bid = bid
                best_route = route
                best_insertion = trip_ins_index
            # # print(f'bid - {bid}, route - {route}, insertion - {trip_ins_index}')
        bid_info[k] = [min_bid, best_route, best_insertion]
        # print()
    return bid_info

def calculate_trip_times(G, routes):
    trip_times = []
    for trip in routes:
        trip_time = 0
        for i in range(len(trip) - 1):
            try:
                trip_time += G.get_edge_data(trip[i], trip[i + 1])[0]["weight"]
            except:
                trip_time += G.get_edge_data(trip[i], trip[i + 1])["weight"]
        trip_times.append(trip_time)
    return trip_times

def update_routes(G, vehicle_routes, vehicle_trip_times, bid):
    for k, (route_time, route, trip_insertion_index) in bid.items():
        trip_times = calculate_trip_times(G, route)
        vehicle_routes[k] = vehicle_routes[k][:trip_insertion_index] + route + vehicle_routes[k][trip_insertion_index:]
        vehicle_trip_times[k] = vehicle_trip_times[k][:trip_insertion_index] + trip_times + vehicle_trip_times[k][trip_insertion_index:]
    return vehicle_routes, vehicle_trip_times

def centralized_auction_optimized(G, t, failure_history, vehicle_routes, vehicle_trip_times, trip_indexes, 
                                    depot_nodes, failure_trips, vehicle_capacity, recharge_time, uavLocation):
    depot_to_depot_distances = calculate_depot_distances(G, depot_nodes, vehicle_capacity, recharge_time)
    # # print(f'Depot to depot distances - {depot_to_depot_distances}')
    # # print('Beginning Auction ')
    # print(vehicle_routes, vehicle_trip_times)
    all_failure_trips = [failure_trip for _, failure_trip in failure_trips.items()]
    # # print(f'all failure trips - {all_failure_trips}')
    while len(all_failure_trips) != 0:
        failure_trip_bids = []
        failure_trip_info_list = []
        # print(f'failure trips - {failure_trips}')
        # print(f'all failure trips - {all_failure_trips}')
        for failure_trip_time_s, failure_trip in failure_trips.items():
            failure_trip_time = float(failure_trip_time_s.split("_")[0])
            # print('Here')
            if failure_trip:
                # print('Here 2')
                failure_trip_info = [failure_trip_time, [failure_trip]]
                failure_trip_depots = list(set([failure_trip[0], failure_trip[-1]]))
                vehicle_insertion_list = search(G, failure_history, vehicle_routes, trip_indexes, failure_trip_depots, 
                                               vehicle_capacity, depot_nodes, search_radius=vehicle_capacity, uavLocation=uavLocation)
                bid_info = bidding_optimized(t, vehicle_routes, vehicle_trip_times, vehicle_insertion_list, failure_trip_info, 
                                             depot_to_depot_distances, recharge_time, uavLocation)
                # # print(f'bid info - {bid_info}')
                for k, [bid, route, insertion_index] in bid_info.items():
                    failure_trip_bids.append(bid)
                    failure_trip_info_list.append({k: [bid, route, insertion_index, failure_trip]})
        # print(f'failure trip info list - {failure_trip_info_list}')
        # print(f'failure trip bids - {failure_trip_bids}')
        
        best_failure_trip = failure_trip_info_list[failure_trip_bids.index(min(failure_trip_bids))]
        # print(f'best failure trip - {best_failure_trip}')
        best_bid = {}
        # # print(f'best_failure trip - {best_failure_trip}')
        for k, [bid, route, insertion_index, failure_trip] in best_failure_trip.items():
            best_bid[k] = [bid, route, insertion_index]
            # # print(f'failure trip - {failure_trip, all_failure_trips}')
            # print(f'all failure trips before removal - {all_failure_trips}')
            all_failure_trips.remove(failure_trip)
            
            # print(f'all failure trips after removal - {all_failure_trips}')
            failure_trips = {trip_time_s: trip for trip_time_s, trip in failure_trips.items() if trip != failure_trip}
            # # print(f'failure trips - {failure_trips}')
        vehicle_routes, vehicle_trip_times = update_routes(G, vehicle_routes, vehicle_trip_times, best_bid)
        # print()
        # # print(f'all failure trips - {all_failure_trips}')
        # # print('After route updation')
        # print(vehicle_routes, vehicle_trip_times)
    return vehicle_routes, vehicle_trip_times

def determine_possible_combinations(vehicle_trip_indexes, vehicle_routes, vehicle_trip_times, mu_veh, failure_history, recharge_time):
    
    if mu_veh == -1:
        print('combinations - []')
        return []
    n = len(vehicle_routes)
    # editable absolute trip indices: strictly > current index; if idx<0 -> all trips editable
    def editable(v):
        idx = vehicle_trip_indexes[v] if v < len(vehicle_trip_indexes) else -1
        start = 0 if (idx is None or idx < 0) else idx + 1
        return list(range(start, len(vehicle_routes[v])))

    # receivers: non-failed, not donor, sorted by usage asc
    usage = []
    for i in range(n):
        t = sum(vehicle_trip_times[i]) + recharge_time * (len(vehicle_trip_times[i]) - 1) if vehicle_trip_times[i] else 0.0
        usage.append(t)
    receivers = [i for i in sorted(range(n), key=lambda x: usage[x]) if i != mu_veh and i not in failure_history]

    d_edit = editable(mu_veh)
    if not d_edit:
        print('combinations - []')
        return []

    combinations = []  # (donor_idx, receiver_idx, donor_trip_abs_indices, receiver_trip_abs_indices)

    for r in receivers:
        r_edit = editable(r)
        m = min(len(d_edit), len(r_edit))
        if m == 0:
            # relocate: try last 1, and last 2 if available
            combinations.append((mu_veh, r, [d_edit[-1]], []))
            if len(d_edit) >= 2:
                combinations.append((mu_veh, r, d_edit[-2:], []))
        else:
            # single swap with the minimum possible bundle size m
            combinations.append((mu_veh, r, d_edit[-m:], r_edit[-m:]))

    print(f'combinations - {combinations}')
    return combinations

def most_used_vehicle(vehicle_routes, vehicle_trip_times, failure_history, recharge_time):
    n = len(vehicle_routes)
    # pick donor = most used non-failed
    mu_veh, mu_time = -1, -1.0
    for i in range(n):
        if i in failure_history: 
            continue
        t = sum(vehicle_trip_times[i]) + recharge_time * (len(vehicle_trip_times[i]) - 1) if vehicle_trip_times[i] else 0.0
        if t > mu_time:
            mu_time, mu_veh = t, i
    print(f'most used vehicle - {mu_veh} and time - {mu_time}')
    return mu_veh

def _norm(a,b):
    return (a,b) if a<=b else (b,a)

def _edges_from_path(path, req_set):
    E = []
    for i in range(len(path)-1):
        e = _norm(path[i], path[i+1])
        if e in req_set:
            E.append(e)
    return E

def _edges_from_routes(routes, req_set):
    out = set()
    for trip in routes:
        out.update(_edges_from_path(trip, req_set))
    return out

def _finish_time(times, R):
    return 0.0 if not times else sum(times) + R*(len(times)-1)

def local_repair(G, depot_nodes, vehicle_capacity, req_edges, dist, start_depot):
    req_set = set(req_edges)
    routes, times = [], []
    rem = set(req_set)
    # trivial guard
    if not rem:
        return routes, times
    # simple multi-trip loop using your MagneticFieldRouter
    while rem:
        router = MagneticFieldRouter(G, dist=dist, start_depot=start_depot, depots=depot_nodes,
                                     capacity=vehicle_capacity, alpha=1.0, gamma=1.0)
        trip, cost, covered = router.find_route_with_magnetic_scoring(list(rem), verbose=False)
        if not trip or cost is None:
            break
        routes.append(trip)
        times.append(float(cost))
        rem.difference_update(_edges_from_path(trip, req_set))
        # next trip starts at the end depot of previous trip
        start_depot = trip[-1]
    return routes, times


# def peer_auction(G, vehicle_routes, vehicle_trip_times, vehicle_trip_indexes,
#                  depot_nodes, req_edges, dist, vehicle_capacity, recharge_time, failure_history):
    
#     mu_veh = most_used_vehicle(vehicle_routes, vehicle_trip_times, failure_history, recharge_time)
#     combinations = determine_possible_combinations(vehicle_trip_indexes, vehicle_routes, vehicle_trip_times, mu_veh, failure_history, recharge_time)
#     req_set = set(tuple(sorted(e)) for e in copy.deepcopy(req_edges))
#     n = len(vehicle_routes)
#     # combinations -> (donor_idx, receiver_idx, donor_trip_abs_indices, receiver_trip_abs_indices)
#     # tasks = {}
#     # for combination in combinations:
#     #     vehicle_routes_c = copy.deepcopy(vehicle_routes)
#     #     donor_idx, receiver_idx, donor_trip_abs_indices, receiver_trip_abs_indices = combination

#     #     task = [vehicle_routes_c[donor_idx][donor_trip_abs_indices[0]][0], ] # start_depot, required_edge_list, vehicle_routes
#     if not combinations:
#         print(f'No improvement')
#         # return vehicle_routes, vehicle_trip_times, vehicle_trip_indexes

#     # edge ownership only on editable suffix; frozen prefix is untouchable
#     def owned_suffix_edges(v):
#         idxs = editable_indices(v)
#         out = set()
#         for ti in idxs:
#             out.update(_edges_from_path(vehicle_routes[v][ti], req_set))
#         return out

#     owned = [owned_suffix_edges(v) for v in range(n)]
#     base_makespan = max(_finish_time(tt, recharge_time) for tt in vehicle_trip_times)
#     best, best_delta = None, 0.0

#     for donor_idx, recv_idx, d_idxs, r_idxs in combinations:
#         vr = copy.deepcopy(vehicle_routes)
#         vt = copy.deepcopy(vehicle_trip_times)

#         # bundles
#         d_edges = set()
#         for ti in d_idxs: d_edges.update(_edges_from_path(vr[donor_idx][ti], req_set))
#         r_edges = set()
#         for ti in r_idxs: r_edges.update(_edges_from_path(vr[recv_idx][ti], req_set))

#         donor_new_edges = (owned[donor_idx] - d_edges) | r_edges
#         recv_new_edges  = (owned[recv_idx]  - r_edges) | d_edges

#         # rebuild suffix only, keep frozen prefix
#         d_freeze = vehicle_trip_indexes[donor_idx] if donor_idx < len(vehicle_trip_indexes) else -1
#         r_freeze = vehicle_trip_indexes[recv_idx]  if recv_idx  < len(vehicle_trip_indexes) else -1

#         d_prefix_routes = vr[donor_idx][:max(d_freeze+1,0)] if d_freeze is not None and d_freeze >= 0 else []
#         r_prefix_routes = vr[recv_idx][:max(r_freeze+1,0)]   if r_freeze is not None and r_freeze >= 0 else []
#         d_prefix_times  = vt[donor_idx][:max(d_freeze+1,0)]  if d_freeze is not None and d_freeze >= 0 else []
#         r_prefix_times  = vt[recv_idx][:max(r_freeze+1,0)]   if r_freeze is not None and r_freeze >= 0 else []

#         # choose start depots for suffix
#         d_start = d_prefix_routes[-1][-1] if d_prefix_routes else (vr[donor_idx][0][0] if vr[donor_idx] else depot_nodes[0])
#         r_start = r_prefix_routes[-1][-1] if r_prefix_routes else (vr[recv_idx][0][0]  if vr[recv_idx]  else depot_nodes[0])

#         # local repair builds a fresh suffix from required edges (you keep your own implementation)
#         # Expect it to return (routes, times) for covering 'donor_new_edges' / 'recv_new_edges'
#         d_suffix_routes, d_suffix_times = local_repair(G, depot_nodes, vehicle_capacity, list(donor_new_edges), dist, d_start)
#         r_suffix_routes, r_suffix_times = local_repair(G, depot_nodes, vehicle_capacity, list(recv_new_edges),  dist, r_start)

#         vr[donor_idx] = d_prefix_routes + d_suffix_routes
#         vt[donor_idx] = d_prefix_times  + d_suffix_times
#         vr[recv_idx]  = r_prefix_routes + r_suffix_routes
#         vt[recv_idx]  = r_prefix_times  + r_suffix_times

#         new_makespan = max(_finish_time(vt[v], recharge_time) for v in range(n))
#         delta = new_makespan - base_makespan
#         if delta < best_delta:
#             best_delta = delta
#             best = (vr, vt)

#     if best is None: 
#         return vehicle_routes, vehicle_trip_times, vehicle_trip_indexes


#     # keep trip indexes unchanged (we never touch current trip), routes/times updated for two vehicles
    # print(f'Improvement obtained')
    # print(f'vehicle routes - {best[0]}')
    # print(f'vehicle route times - {best[1]}')
    # print(f'vehicle trip index - {vehicle_trip_indexes}')
#     pass
    # return best[0], best[1], vehicle_trip_indexes

# def build_expanded_combinations(mu_veh, receivers, vehicle_routes, vehicle_trip_indexes,
#                                 max_b=3, max_per_receiver=32):
#     combos = []  # (donor, recv, donor_trip_idxs, recv_trip_idxs), ABS indices; frozen respected
#     n = len(vehicle_routes)

#     def editable(v):
#         idx = vehicle_trip_indexes[v] if v < len(vehicle_trip_indexes) else -1
#         start = 0 if (idx is None or idx < 0) else idx + 1
#         return list(range(start, len(vehicle_routes[v])))

#     d_edit = editable(mu_veh)
#     if not d_edit:
#         return combos

#     # tail windows from donor/receiver: contiguous suffixes only
#     def tail_windows(idxs, max_b):
#         out = []
#         b = min(max_b, len(idxs))
#         for k in range(1, b + 1):
#             out.append(idxs[-k:])   # last k
#         return out

#     d_ws = tail_windows(d_edit, max_b)

#     for r in receivers:
#         r_edit = editable(r)
#         r_ws = tail_windows(r_edit, max_b) if r_edit else []
#         seen = set()

#         # swaps: all pairings of donor/receiver tail windows (unequal sizes allowed)
#         for d_sub in d_ws:
#             if r_ws:
#                 for r_sub in r_ws:
#                     tup = (mu_veh, r, tuple(d_sub), tuple(r_sub))
#                     if tup not in seen:
#                         combos.append((mu_veh, r, d_sub, r_sub))
#                         seen.add(tup)
#                         if len(seen) >= max_per_receiver: break
#             # relocate variants (receiver empty)
#             tup = (mu_veh, r, tuple(d_sub), ())
#             if tup not in seen:
#                 combos.append((mu_veh, r, d_sub, []))
#                 seen.add(tup)
#                 if len(seen) >= max_per_receiver: pass
#         # optional: also try relocating two-at-once if donor has ≥2 tail windows already included above

#     return combos

# def build_expanded_combinations(mu_veh, receivers, vehicle_routes, vehicle_trip_indexes,
#                                 vehicle_trip_times, max_b=3, max_per_receiver=32):
#     combos = []
#     n = len(vehicle_routes)

#     def editable(v):
#         idx = vehicle_trip_indexes[v] if v < len(vehicle_trip_indexes) else -1
#         start = 0 if (idx is None or idx < 0) else idx + 1
#         return list(range(start, len(vehicle_routes[v])))

#     def tail_windows(idxs, max_b):
#         out = []
#         b = min(max_b, len(idxs))
#         for k in range(1, b + 1):
#             out.append(idxs[-k:])
#         return out

#     totals = [
#         (i, sum(vehicle_trip_times[i]) if i < len(vehicle_trip_times) and vehicle_trip_times[i] else 0.0)
#         for i in range(n)
#     ]
#     totals.sort(key=lambda x: x[1], reverse=True)

#     donors = [mu_veh]
#     for d, _ in totals:
#         if d != mu_veh:# and len(donors) < 3:
#             donors.append(d)

#     for d in donors:
#         d_edit = editable(d)
#         if not d_edit:
#             continue
#         d_ws = tail_windows(d_edit, max_b)

#         for r in receivers:
#             if r == d:
#                 continue
#             r_edit = editable(r)
#             r_ws = tail_windows(r_edit, max_b) if r_edit else []
#             seen = set()

#             for d_sub in d_ws:
#                 if r_ws:
#                     for r_sub in r_ws:
#                         tup = (d, r, tuple(d_sub), tuple(r_sub))
#                         if tup not in seen:
#                             combos.append((d, r, d_sub, r_sub))
#                             seen.add(tup)
#                             if len(seen) >= max_per_receiver:
#                                 break
#                 tup = (d, r, tuple(d_sub), ())
#                 if tup not in seen and len(seen) < max_per_receiver:
#                     combos.append((d, r, d_sub, []))
#                     seen.add(tup)

#     return combos

# version 3
def build_expanded_combinations(mu_veh, receivers, vehicle_routes, vehicle_trip_indexes,
                                vehicle_trip_times, max_b=3, max_per_receiver=32):
    combos = []
    n = len(vehicle_routes)

    def editable(v):
        idx = vehicle_trip_indexes[v] if v < len(vehicle_trip_indexes) else -1
        start = 0 if (idx is None or idx < 0) else idx + 1
        return list(range(start, len(vehicle_routes[v])))

    def all_windows(idxs, max_b):
        out = []
        L = len(idxs)
        b = min(max_b, L)
        for length in range(1, b + 1):
            for start in range(0, L - length + 1):
                out.append(idxs[start:start + length])
        return out

    totals = [
        (i, sum(vehicle_trip_times[i]) if i < len(vehicle_trip_times) and vehicle_trip_times[i] else 0.0)
        for i in range(n)
    ]
    totals.sort(key=lambda x: x[1], reverse=True)

    donors = [mu_veh]
    for d, _ in totals:
        if d != mu_veh and len(donors) < 3:
            donors.append(d)

    for d in donors:
        d_edit = editable(d)
        if not d_edit:
            continue
        d_ws = all_windows(d_edit, max_b)

        for r in receivers:
            if r == d:
                continue
            r_edit = editable(r)
            r_ws = all_windows(r_edit, max_b) if r_edit else []
            seen = set()

            for d_sub in d_ws:
                if r_ws:
                    for r_sub in r_ws:
                        tup = (d, r, tuple(d_sub), tuple(r_sub))
                        if tup not in seen:
                            combos.append((d, r, d_sub, r_sub))
                            seen.add(tup)
                            if len(seen) >= max_per_receiver:
                                break
                tup = (d, r, tuple(d_sub), ())
                if tup not in seen and len(seen) < max_per_receiver:
                    combos.append((d, r, d_sub, []))
                    seen.add(tup)

    return combos


def peer_auction(G, vehicle_routes, vehicle_trip_times, vehicle_trip_indexes,
                 depot_nodes, req_edges, dist, vehicle_capacity, recharge_time, failure_history, eps=1e-6):

    n = len(vehicle_routes)
    req_set = set(tuple(sorted(e)) for e in req_edges)

    # donor = most-used non-failed
    mu_veh, mu_time = -1, -1.0
    for i in range(n):
        if i in failure_history: continue
        t = _finish_time(vehicle_trip_times[i], recharge_time)
        if t > mu_time: mu_time, mu_veh = t, i
    if mu_veh == -1: return vehicle_routes, vehicle_trip_times, vehicle_trip_indexes

    def editable_indices(v):
        idx = vehicle_trip_indexes[v] if v < len(vehicle_trip_indexes) else -1
        start = (idx + 1) if (idx is not None and idx >= 0) else 0
        return list(range(start, len(vehicle_routes[v])))

    # # receivers: non-failed, not donor, least-used first
    usage = [_finish_time(vehicle_trip_times[i], recharge_time) for i in range(n)]
    receivers = [i for i in sorted(range(n), key=lambda x: usage[x]) if i != mu_veh and i not in failure_history]

    d_edit = editable_indices(mu_veh)
    if not d_edit: return vehicle_routes, vehicle_trip_times, vehicle_trip_indexes

    # # combos by min(#tail_d, #tail_r); if 0 => relocate last 1 (and 2 if possible)
    # combinations = []
    # for r in receivers:
    #     r_edit = editable_indices(r)
    #     m = min(len(d_edit), len(r_edit))
    #     if m == 0:
    #         combinations.append((mu_veh, r, [d_edit[-1]], []))
    #         if len(d_edit) >= 2: combinations.append((mu_veh, r, d_edit[-2:], []))
    #     else:
    #         combinations.append((mu_veh, r, d_edit[-m:], r_edit[-m:]))

    # version 1
    # combinations = build_expanded_combinations(mu_veh, receivers, vehicle_routes, vehicle_trip_indexes,
    #                             max_b=3, max_per_receiver=32)
    
    # version 2
    combinations = build_expanded_combinations(mu_veh, receivers, vehicle_routes, vehicle_trip_indexes, vehicle_trip_times,
                                max_b=3, max_per_receiver=32)
    if not combinations: return vehicle_routes, vehicle_trip_times, vehicle_trip_indexes, 0
    
    print(f'combinations - {combinations}')
    # edge ownership only on editable suffix; frozen prefix is untouchable
    def owned_suffix_edges(v):
        idxs = editable_indices(v)
        out = set()
        for ti in idxs:
            out.update(_edges_from_path(vehicle_routes[v][ti], req_set))
        return out

    owned = [owned_suffix_edges(v) for v in range(n)]
    base_makespan = max(_finish_time(tt, recharge_time) for tt in vehicle_trip_times)
    best, best_delta = None, 0.0

    for donor_idx, recv_idx, d_idxs, r_idxs in combinations:
        # print(f'tryin')
        vr = copy.deepcopy(vehicle_routes)
        vt = copy.deepcopy(vehicle_trip_times)

        # bundles
        d_edges = set()
        for ti in d_idxs: d_edges.update(_edges_from_path(vr[donor_idx][ti], req_set))
        r_edges = set()
        for ti in r_idxs: r_edges.update(_edges_from_path(vr[recv_idx][ti], req_set))

        donor_new_edges = (owned[donor_idx] - d_edges) | r_edges
        recv_new_edges  = (owned[recv_idx]  - r_edges) | d_edges

        # rebuild suffix only, keep frozen prefix
        d_freeze = vehicle_trip_indexes[donor_idx] if donor_idx < len(vehicle_trip_indexes) else -1
        r_freeze = vehicle_trip_indexes[recv_idx]  if recv_idx  < len(vehicle_trip_indexes) else -1

        d_prefix_routes = vr[donor_idx][:max(d_freeze+1,0)] if d_freeze is not None and d_freeze >= 0 else []
        r_prefix_routes = vr[recv_idx][:max(r_freeze+1,0)]   if r_freeze is not None and r_freeze >= 0 else []
        d_prefix_times  = vt[donor_idx][:max(d_freeze+1,0)]  if d_freeze is not None and d_freeze >= 0 else []
        r_prefix_times  = vt[recv_idx][:max(r_freeze+1,0)]   if r_freeze is not None and r_freeze >= 0 else []

        # choose start depots for suffix
        d_start = d_prefix_routes[-1][-1] if d_prefix_routes else (vr[donor_idx][0][0] if vr[donor_idx] else depot_nodes[0])
        r_start = r_prefix_routes[-1][-1] if r_prefix_routes else (vr[recv_idx][0][0]  if vr[recv_idx]  else depot_nodes[0])

        # local repair builds a fresh suffix from required edges (you keep your own implementation)
        # Expect it to return (routes, times) for covering 'donor_new_edges' / 'recv_new_edges'
        d_suffix_routes, d_suffix_times = local_repair(G, depot_nodes, vehicle_capacity, list(donor_new_edges), dist, d_start)
        r_suffix_routes, r_suffix_times = local_repair(G, depot_nodes, vehicle_capacity, list(recv_new_edges),  dist, r_start)

        vr[donor_idx] = d_prefix_routes + d_suffix_routes
        vt[donor_idx] = d_prefix_times  + d_suffix_times
        vr[recv_idx]  = r_prefix_routes + r_suffix_routes
        vt[recv_idx]  = r_prefix_times  + r_suffix_times

        new_makespan = max(_finish_time(vt[v], recharge_time) for v in range(n))
        print(f'vehicle routes - {vr}')
        print(f'vehicle route times - {vt}')
        print(f'new makespan - {new_makespan}')
        print()
        delta = new_makespan - base_makespan
        if delta < best_delta:
            best_delta = delta
            best = (vr, vt)

    if best is None: 
        print(f'No improvement')
        return vehicle_routes, vehicle_trip_times, vehicle_trip_indexes, best_delta
    else:
        print(f'Improvement obtained')
        print(f'vehicle routes - {best[0]}')
        print(f'vehicle route times - {best[1]}')
        print(f'vehicle trip index - {vehicle_trip_indexes}')
        mt = round(max([sum(best[1][k]) + (len(best[1][k]) - 1) * recharge_time for k in range(n)]), 1)
        print(f'peer auction mission time - {mt}')
    # pass
    return best[0], best[1], vehicle_trip_indexes, best_delta

def peer_auction_rounds(G, vehicle_routes, vehicle_trip_times, vehicle_trip_indexes,
                        depot_nodes, req_edges, dist, vehicle_capacity, recharge_time, failure_history,
                        max_rounds=5, eps=1e-6):
    routes = copy.deepcopy(vehicle_routes)
    times  = copy.deepcopy(vehicle_trip_times)
    idxes  = list(vehicle_trip_indexes)
    for r in range(max_rounds):
        print(f"------ Round {r+1} ----------")
        print(f'vehicle routes - {routes}')
        print(f'vehicle route times - {times}')
        print(f'vehicle trip index - {idxes}')
        routes2, times2, idxes2, delta = peer_auction(
            G, routes, times, idxes, depot_nodes, req_edges, dist,
            vehicle_capacity, recharge_time, failure_history, eps=eps
        )
        if delta >= -eps:  # no improvement
            break
        print(f'Improvement obtained')
        print(f'vehicle routes - {routes2}')
        print(f'vehicle route times - {times2}')
        print(f'vehicle trip index - {idxes2}')
        mt = round(max(sum(times2[k]) + (len(times2[k]) - 1) * recharge_time for k in range(len(routes2))), 1)
        print(f'peer auction mission time - {mt}')
        print(f'delta - {delta}')
        routes, times, idxes = routes2, times2, idxes2  # commit and continue; donor/combos recomputed next round
    return routes, times, idxes

# GDB
def simulate_1(save_results_to_csv=False):
    p = '.'
    
    # instanceData = {
    #     "Instance Name": [], "Maximum Trip Time": [], "Number of vehicles failed": [],
    #     'Total Number of Vehicles': [], "Maximum Trip Time after auction algorithm": [],
    #     "Execution time for auction algorithm in sec": [], "% increase in maximum trip time": [],
    #     "Number of Function Calls": [], 'idle time': [], "Average time per function call": []
    # }

    instanceData = {
        "Instance Name": [], "Number of Nodes": [], "Number of Edges": [], "Number of Required Edges": [],
        "Capacity": [], "Recharge Time": [], 'Total Number of Vehicles': [], "Number of Depot Nodes": [],
        "Number of vehicles failed": [], "Maximum Trip Time": [], "Number of Function Calls": [],
        "Execution time for auction algorithm in sec": [], "Maximum Trip Time after auction algorithm": [],
        "% increase in maximum trip time": [], 'idle time': [], "Average time per function call": []
    }

    # txt_folder = f"{p}/dataset/new_failure_scenarios/gdb_failure_scenarios"
    txt_folder = f"{p}/gdb_failure_scenarios"
    sol_folder = f"{p}/results/instances_results_without_failure/GDB_Failure_Scenarios_Results/results/gdb_failure_scenarios_solutions"
    
    # Get all scenario files
    scenario_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    
    for file in scenario_files:
        scenario_num = file.split('.')[1]  # Extract scenario number from gdb.X.txt
        
        if int(scenario_num) == 1:  # Process only scenario 1 for testing
            print(f'Running GDB sscenario {scenario_num}')
            # print(f'Running GDB scenario {scenario_num}')
            
            # Parse txt file
            txt_path = os.path.join(txt_folder, file)
            G, required_edges, depot_nodes, vehicle_capacity, recharge_time, num_vehicles, failure_vehicles, vehicle_failure_times = parse_txt_file(txt_path)
            dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
            print(f'required edges - {required_edges}')
            # Load solution routes
            sol_path = os.path.join(sol_folder, f"{scenario_num}.npy")
            if not os.path.exists(sol_path):
                # print(f"Solution file not found: {sol_path}")
                continue
                
            vehicle_routes = np.load(sol_path, allow_pickle=True).tolist()
            vehicle_trip_times = calculate_route_times(G, vehicle_routes)
            
            # Calculate initial mission time
            mission_time = max([sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time 
                               for k in range(len(vehicle_routes)) if vehicle_trip_times[k]])
            
            uavLocation = []
            for k in range(len(vehicle_routes)):
                if vehicle_routes[k]:
                    uavLocation.append(vehicle_routes[k][0][0])
                else:
                    uavLocation.append(depot_nodes[k % len(depot_nodes)])
            
            # print('Failure Scenario information: ')
            # for ij_ in range(len(failure_vehicles)):
            #     # print(f'vehicle {failure_vehicles[ij_]+1} will fail in {vehicle_failure_times[ij_]} time units.')
            # print()
            
            previous_mission_time = mission_time
            instanceData['Instance Name'].append(f"gdb.{scenario_num}")
            instanceData['Number of Nodes'].append(G.number_of_nodes())
            instanceData['Number of Edges'].append(G.number_of_edges())
            instanceData["Number of Required Edges"].append(len(required_edges))
            instanceData['Capacity'].append(vehicle_capacity)
            instanceData['Recharge Time'].append(recharge_time)
            instanceData['Number of Depot Nodes'].append(len(depot_nodes))
            instanceData['Maximum Trip Time'].append(mission_time)
            instanceData['Number of vehicles failed'].append(len(failure_vehicles))
            instanceData['Total Number of Vehicles'].append(len(vehicle_routes))
            
            t = 0
            vehicle_status = [True] * len(vehicle_routes)
            failure_history = []
            detected_failed_vehicles = {}
            vehicle_failure_detection_time = -np.inf
            idle_time = [{} for _ in range(num_vehicles)]
            for k in range(len(vehicle_routes)):
                for trip in range(len(vehicle_routes[k])):
                    idle_time[k][trip] = 0
            
            recent_failure_vehicle = -1
            num_function_calls = 0
            
            start = time.time()
            print()
            print(f'vehicle routes - {vehicle_routes}')
            print(f'vehicle trip times - {vehicle_trip_times}')
            print("Starting Mission")
            print(f"mission time : {mission_time} time units.")
            # print(f'recharge time - {recharge_time} time units')
            # print(f"required edges to be traversed : {required_edges}")
            
            while t <= mission_time:
                for i, k in enumerate(failure_vehicles):
                    if t == vehicle_failure_times[i]:
                        vehicle_status[k] = False
                        recent_failure_vehicle = k
                        vehicle_failure_detection_time = t
                        print(f"Vehicle {k+1} failure detected at {vehicle_failure_detection_time} time units")
                        # print()
                        
                for k, status in enumerate(vehicle_status):
                    if status == False:
                        detected_failed_vehicles[k] = vehicle_failure_detection_time
                        
                if detected_failed_vehicles and t == vehicle_failure_detection_time:
                    for k in list(detected_failed_vehicles.keys()):
                        vehicle_status[k] = None
                    failure_history += list(detected_failed_vehicles.keys())
                    
                    vehicle_trip_index, idle_time = vehicle_at_time_t(t, vehicle_routes, vehicle_trip_times,
                                                                     recharge_time, failure_history, idle_time,
                                                                     recent_failure_vehicle)
                    # print(f"idle time - {idle_time}")
                    
                    failure_trips = identifying_failed_trips(vehicle_routes, vehicle_trip_times, vehicle_trip_index, 
                                                             recent_failure_vehicle, required_edges)
                    # print(f'Vehicle failure trips info - {failure_trips}')
                    
                    vehicle_routes[recent_failure_vehicle] = vehicle_routes[recent_failure_vehicle][:vehicle_trip_index[recent_failure_vehicle]]
                    vehicle_trip_times[recent_failure_vehicle] = vehicle_trip_times[recent_failure_vehicle][:vehicle_trip_index[recent_failure_vehicle]]
                    
                    print('Removing failure trips from vehicle routes and trip times')
                    print('Vehicle routes - ', vehicle_routes)
                    print('Vehicle trip times - ', vehicle_trip_times)
                    print(f'Vehicle trip indexes - {vehicle_trip_index}')
                    print()
                    
                    print('Beginning Auction ')
                    vehicle_routes, vehicle_trip_times = centralized_auction_optimized(G, t, failure_history, vehicle_routes, 
                                                                                         vehicle_trip_times, vehicle_trip_index, 
                                                                                         depot_nodes, failure_trips, vehicle_capacity, 
                                                                                         recharge_time, uavLocation)
                    num_function_calls += 1
                    detected_failed_vehicles = {}
                    
                    print("centralized auction results")
                    print(f"Updated vehicle routes  -  {vehicle_routes}")
                    print(f"Updated Trip Times  -  {vehicle_trip_times}")
                    print(f'After centralized auction Vehicle trip indexes - {vehicle_trip_index}')
                    print(f'failure history - {failure_history}')
                    mission_time = round(max([sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time +
                                             sum(idle_time[k].values()) for k in range(len(vehicle_routes))]), 1)
                    
                    print(f"Mission time extended to {mission_time} time units.")
                    print()

                    print()
                    print('###### Peer auction development ######')
                    # peer_auction(G, vehicle_routes, vehicle_trip_times, vehicle_trip_index, depot_nodes, required_edges, dist, vehicle_capacity, recharge_time, failure_history)
                    vehicle_routes, vehicle_trip_times, _ = peer_auction_rounds(G, vehicle_routes, vehicle_trip_times, vehicle_trip_index,
                        depot_nodes, required_edges, dist, vehicle_capacity, recharge_time, failure_history,
                        max_rounds=10, eps=1e-6)
                    print()

                    mission_time = round(max([sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time +
                                             sum(idle_time[k].values()) for k in range(len(vehicle_routes))]), 1)
                    
                    # print(f"Mission time extended to {mission_time} time units.")
                    # print()
                    
                t += 0.1
                t = round(t, 1)
                
            end = time.time()
            execution_time = round(end - start, 3)
            avg_time_per_call = round(execution_time / num_function_calls, 3) if num_function_calls > 0 else 0
            
            # print(f"Execution time for centralized auction algorithm {execution_time} seconds")
            # print(f"Average time per function call: {avg_time_per_call} seconds")
            # print("Mission Complete")
            print("Final Routes Information")
            print(f"Final Solution Routes  -  {vehicle_routes}")
            print(f"Final Solution Trip Times  -  {vehicle_trip_times}")
            print(f'Final mission time - {mission_time}')
            print(f'required edges - {required_edges}')
            print("----------")
            
            total_idle_time = sum(sum(idle_time[k].values()) for k in range(len(vehicle_routes)))
            
            instanceData['idle time'].append(total_idle_time)
            instanceData['Maximum Trip Time after auction algorithm'].append(mission_time)
            instanceData['% increase in maximum trip time'].append(round(((mission_time - previous_mission_time) / previous_mission_time) * 100, 1))
            instanceData['Execution time for auction algorithm in sec'].append(execution_time)
            instanceData['Number of Function Calls'].append(num_function_calls)
            instanceData['Average time per function call'].append(avg_time_per_call)
            
            if save_results_to_csv:
                df = pd.DataFrame(instanceData)
                df.to_csv(f"{p}/results/instances_results_with_failure/centralized_auction_algorithm_gdb_new.csv", index=True)
            # print()
            # print()


# BCCM 
def simulate_2(save_results_to_csv=False):
    p = '.'
    
    instanceData = {
        "Instance Name": [], "Number of Nodes": [], "Number of Edges": [], "Number of Required Edges": [],
        "Capacity": [], "Recharge Time": [], 'Total Number of Vehicles': [], "Number of Depot Nodes": [],
        "Number of vehicles failed": [], "Maximum Trip Time": [], "Number of Function Calls": [],
        "Execution time for auction algorithm in sec": [], "Maximum Trip Time after auction algorithm": [],
        "% increase in maximum trip time": [], 'idle time': [], "Average time per function call": []
    }

    txt_folder = f"{p}/bccm_failure_scenarios"
    sol_folder = f"{p}/results/instances_results_without_failure/BCCM_Failure_Scenarios_Results/results/bccm_failure_scenarios_solutions"
    
    # Get all scenario files
    scenario_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    
    for file in scenario_files:
        scenario_num = file.split('.')[1]  # Extract scenario number from bccm.X.txt
        
        
        if True:#int(scenario_num) == 5:  # Process only scenario 1 for testing
            print(f'Running BCCM scenario {scenario_num}')
            # print(f'Running BCCM scenario {scenario_num}')
            
            # Parse txt file
            txt_path = os.path.join(txt_folder, file)
            G, required_edges, depot_nodes, vehicle_capacity, recharge_time, num_vehicles, failure_vehicles, vehicle_failure_times = parse_txt_file(txt_path)
            dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
            # Load solution routes
            sol_path = os.path.join(sol_folder, f"{scenario_num}.npy")
            if not os.path.exists(sol_path):
                # print(f"Solution file not found: {sol_path}")
                continue
                
            vehicle_routes = np.load(sol_path, allow_pickle=True).tolist()
            vehicle_trip_times = calculate_route_times(G, vehicle_routes)
            
            # Calculate initial mission time
            mission_time = max([sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time 
                               for k in range(len(vehicle_routes)) if vehicle_trip_times[k]])
            
            uavLocation = []
            for k in range(len(vehicle_routes)):
                if vehicle_routes[k]:
                    uavLocation.append(vehicle_routes[k][0][0])
                else:
                    uavLocation.append(depot_nodes[k % len(depot_nodes)])
            
            # print('Failure Scenario information: ')
            # for ij_ in range(len(failure_vehicles)):
            #     # print(f'vehicle {failure_vehicles[ij_]+1} will fail in {vehicle_failure_times[ij_]} time units.')
            # print()
            
            previous_mission_time = mission_time
            instanceData['Instance Name'].append(f"bccm.{scenario_num}")
            instanceData['Number of Nodes'].append(G.number_of_nodes())
            instanceData['Number of Edges'].append(G.number_of_edges())
            instanceData["Number of Required Edges"].append(len(required_edges))
            instanceData['Capacity'].append(vehicle_capacity)
            instanceData['Recharge Time'].append(recharge_time)
            instanceData['Number of Depot Nodes'].append(len(depot_nodes))
            instanceData['Maximum Trip Time'].append(mission_time)
            instanceData['Number of vehicles failed'].append(len(failure_vehicles))
            instanceData['Total Number of Vehicles'].append(len(vehicle_routes))
            
            t = 0
            vehicle_status = [True] * len(vehicle_routes)
            failure_history = []
            detected_failed_vehicles = {}
            vehicle_failure_detection_time = -np.inf
            idle_time = [{} for _ in range(num_vehicles)]
            for k in range(len(vehicle_routes)):
                for trip in range(len(vehicle_routes[k])):
                    idle_time[k][trip] = 0
            
            recent_failure_vehicle = -1
            num_function_calls = 0
            
            start = time.time()
            print()
            print(f'vehicle routes - {vehicle_routes}')
            print(f'vehicle trip times - {vehicle_trip_times}')
            # print(f'depot Nodes - {depot_nodes}')
            # print("Starting Mission")
            # print(f"mission time : {mission_time} time units.")
            # print(f'recharge time - {recharge_time} time units')
            # print(f"required edges to be traversed : {required_edges}")
            
            while t <= mission_time:
                for i, k in enumerate(failure_vehicles):
                    if t == vehicle_failure_times[i]:
                        vehicle_status[k] = False
                        recent_failure_vehicle = k
                        vehicle_failure_detection_time = t
                        # print(f"Vehicle {k+1} failure detected at {vehicle_failure_detection_time} time units")
                        # print()
                        
                for k, status in enumerate(vehicle_status):
                    if status == False:
                        detected_failed_vehicles[k] = vehicle_failure_detection_time
                        
                if detected_failed_vehicles and t == vehicle_failure_detection_time:
                    for k in list(detected_failed_vehicles.keys()):
                        vehicle_status[k] = None
                    failure_history += list(detected_failed_vehicles.keys())
                    
                    vehicle_trip_index, idle_time = vehicle_at_time_t(t, vehicle_routes, vehicle_trip_times,
                                                                     recharge_time, failure_history, idle_time,
                                                                     recent_failure_vehicle)
                    # print(f"idle time - {idle_time}")
                    
                    failure_trips = identifying_failed_trips(vehicle_routes, vehicle_trip_times, vehicle_trip_index, 
                                                             recent_failure_vehicle, required_edges)
                    # print(f'Vehicle failure trips info - {failure_trips}')
                    
                    vehicle_routes[recent_failure_vehicle] = vehicle_routes[recent_failure_vehicle][:vehicle_trip_index[recent_failure_vehicle]]
                    vehicle_trip_times[recent_failure_vehicle] = vehicle_trip_times[recent_failure_vehicle][:vehicle_trip_index[recent_failure_vehicle]]
                    
                    # print('Removing failure trips from vehicle routes and trip times')
                    # print('Vehicle routes - ', vehicle_routes)
                    # print('Vehicle trip times - ', vehicle_trip_times)
                    # print()
                    
                    # print('Beginning Auction ')
                    vehicle_routes, vehicle_trip_times = centralized_auction_optimized(G, t, failure_history, vehicle_routes, 
                                                                                         vehicle_trip_times, vehicle_trip_index, 
                                                                                         depot_nodes, failure_trips, vehicle_capacity, 
                                                                                         recharge_time, uavLocation)
                    num_function_calls += 1
                    detected_failed_vehicles = {}
                    
                    print("centralized auction results")
                    print(f"Updated vehicle routes  -  {vehicle_routes}")
                    print(f"Updated Trip Times  -  {vehicle_trip_times}")

                    mission_time = round(max([sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time +
                                             sum(idle_time[k].values()) for k in range(len(vehicle_routes))]), 1)
                    print(f'mission time - {mission_time}')

                    print()
                    print('###### Peer auction development ######')
                    # peer_auction(G, vehicle_routes, vehicle_trip_times, vehicle_trip_index, depot_nodes, required_edges, dist, vehicle_capacity, recharge_time, failure_history)
                    vehicle_routes, vehicle_trip_times, _ = peer_auction_rounds(G, vehicle_routes, vehicle_trip_times, vehicle_trip_index,
                        depot_nodes, required_edges, dist, vehicle_capacity, recharge_time, failure_history,
                        max_rounds=5, eps=1e-6)
                    
                    mission_time = round(max([sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time +
                                             sum(idle_time[k].values()) for k in range(len(vehicle_routes))]), 1)
                    print(f'mission time - {mission_time}')
                    print()
                    
                    
                    # print(f"Mission time extended to {mission_time} time units.")
                    # print()
                    
                t += 0.1
                t = round(t, 1)
                
            end = time.time()
            execution_time = round(end - start, 3)
            avg_time_per_call = round(execution_time / num_function_calls, 3) if num_function_calls > 0 else 0
            
            # print(f"Execution time for centralized auction algorithm {execution_time} seconds")
            # print(f"Average time per function call: {avg_time_per_call} seconds")
            # print("Mission Complete")
            print("Final Routes Information")
            print(f"Final Solution Routes  -  {vehicle_routes}")
            print(f"Final Solution Trip Times  -  {vehicle_trip_times}")
            print(f'Final mission time - {mission_time}')
            # print("----------")
            
            total_idle_time = sum(sum(idle_time[k].values()) for k in range(len(vehicle_routes)))
            
            instanceData['idle time'].append(total_idle_time)
            instanceData['Maximum Trip Time after auction algorithm'].append(mission_time)
            instanceData['% increase in maximum trip time'].append(round(((mission_time - previous_mission_time) / previous_mission_time) * 100, 1))
            instanceData['Execution time for auction algorithm in sec'].append(execution_time)
            instanceData['Number of Function Calls'].append(num_function_calls)
            instanceData['Average time per function call'].append(avg_time_per_call)
            
            if save_results_to_csv:
                df = pd.DataFrame(instanceData)
                df.to_csv(f"{p}/results/centralized_auction_algorithm_bccm_new.csv", index=True)
            # print()
            # print()

# EGLESE
def simulate_3(save_results_to_csv=False):
    p = '..'
    
    instanceData = {
        "Instance Name": [], "Number of Nodes": [], "Number of Edges": [], "Number of Required Edges": [],
        "Capacity": [], "Recharge Time": [], 'Total Number of Vehicles': [], "Number of Depot Nodes": [],
        "Number of vehicles failed": [], "Maximum Trip Time": [], "Number of Function Calls": [],
        "Execution time for auction algorithm in sec": [], "Maximum Trip Time after auction algorithm": [],
        "% increase in maximum trip time": [], 'idle time': [], "Average time per function call": []
    }

    txt_folder = f"{p}/dataset/new_failure_scenarios/eglese_failure_scenarios"
    sol_folder = f"{p}/results/instances_results_without_failure/EGLESE_Failure_Scenarios_Results/results/eglese_failure_scenarios_solutions"
    
    # Get all scenario files
    scenario_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    
    for file in scenario_files:
        scenario_num = file.split('.')[1]  # Extract scenario number from eglese.X.txt
        
        print(f'Running EGLESE scenario {scenario_num}')
        if True:#int(scenario_num) == 1:  # Process only scenario 1 for testing
            # print(f'Running EGLESE scenario {scenario_num}')
            
            # Parse txt file
            txt_path = os.path.join(txt_folder, file)
            G, required_edges, depot_nodes, vehicle_capacity, recharge_time, num_vehicles, failure_vehicles, vehicle_failure_times = parse_txt_file(txt_path)
            
            # Load solution routes
            sol_path = os.path.join(sol_folder, f"{scenario_num}.npy")
            if not os.path.exists(sol_path):
                # print(f"Solution file not found: {sol_path}")
                continue
                
            vehicle_routes = np.load(sol_path, allow_pickle=True).tolist()
            vehicle_trip_times = calculate_route_times(G, vehicle_routes)
            
            # Calculate initial mission time
            mission_time = max([sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time 
                               for k in range(len(vehicle_routes)) if vehicle_trip_times[k]])
            
            uavLocation = []
            for k in range(len(vehicle_routes)):
                if vehicle_routes[k]:
                    uavLocation.append(vehicle_routes[k][0][0])
                else:
                    uavLocation.append(depot_nodes[k % len(depot_nodes)])
            
            # print('Failure Scenario information: ')
            # for ij_ in range(len(failure_vehicles)):
            #     # print(f'vehicle {failure_vehicles[ij_]+1} will fail in {vehicle_failure_times[ij_]} time units.')
            # print()
            
            previous_mission_time = mission_time
            instanceData['Instance Name'].append(f"eglese.{scenario_num}")
            instanceData['Number of Nodes'].append(G.number_of_nodes())
            instanceData['Number of Edges'].append(G.number_of_edges())
            instanceData["Number of Required Edges"].append(len(required_edges))
            instanceData['Capacity'].append(vehicle_capacity)
            instanceData['Recharge Time'].append(recharge_time)
            instanceData['Number of Depot Nodes'].append(len(depot_nodes))
            instanceData['Maximum Trip Time'].append(mission_time)
            instanceData['Number of vehicles failed'].append(len(failure_vehicles))
            instanceData['Total Number of Vehicles'].append(len(vehicle_routes))
            
            t = 0
            vehicle_status = [True] * len(vehicle_routes)
            failure_history = []
            detected_failed_vehicles = {}
            vehicle_failure_detection_time = -np.inf
            idle_time = [{} for _ in range(num_vehicles)]
            for k in range(len(vehicle_routes)):
                for trip in range(len(vehicle_routes[k])):
                    idle_time[k][trip] = 0
            
            recent_failure_vehicle = -1
            num_function_calls = 0
            
            start = time.time()
            print()
            print(f'vehicle routes - {vehicle_routes}')
            print(f'vehicle trip times - {vehicle_trip_times}')
            # print("Starting Mission")
            # print(f"mission time : {mission_time} time units.")
            # print(f'recharge time - {recharge_time} time units')
            # print(f"required edges to be traversed : {required_edges}")
            
            while t <= mission_time:
                for i, k in enumerate(failure_vehicles):
                    if t == vehicle_failure_times[i]:
                        vehicle_status[k] = False
                        recent_failure_vehicle = k
                        vehicle_failure_detection_time = t
                        # print(f"Vehicle {k+1} failure detected at {vehicle_failure_detection_time} time units")
                        # print()
                        
                for k, status in enumerate(vehicle_status):
                    if status == False:
                        detected_failed_vehicles[k] = vehicle_failure_detection_time
                        
                if detected_failed_vehicles and t == vehicle_failure_detection_time:
                    for k in list(detected_failed_vehicles.keys()):
                        vehicle_status[k] = None
                    failure_history += list(detected_failed_vehicles.keys())
                    
                    vehicle_trip_index, idle_time = vehicle_at_time_t(t, vehicle_routes, vehicle_trip_times,
                                                                     recharge_time, failure_history, idle_time,
                                                                     recent_failure_vehicle)
                    # print(f"idle time - {idle_time}")
                    
                    failure_trips = identifying_failed_trips(vehicle_routes, vehicle_trip_times, vehicle_trip_index, 
                                                             recent_failure_vehicle, required_edges)
                    # print(f'Vehicle failure trips info - {failure_trips}')
                    
                    vehicle_routes[recent_failure_vehicle] = vehicle_routes[recent_failure_vehicle][:vehicle_trip_index[recent_failure_vehicle]]
                    vehicle_trip_times[recent_failure_vehicle] = vehicle_trip_times[recent_failure_vehicle][:vehicle_trip_index[recent_failure_vehicle]]
                    
                    # print('Removing failure trips from vehicle routes and trip times')
                    # print('Vehicle routes - ', vehicle_routes)
                    # print('Vehicle trip times - ', vehicle_trip_times)
                    # print()
                    
                    # print('Beginning Auction ')
                    vehicle_routes, vehicle_trip_times = centralized_auction_optimized(G, t, failure_history, vehicle_routes, 
                                                                                         vehicle_trip_times, vehicle_trip_index, 
                                                                                         depot_nodes, failure_trips, vehicle_capacity, 
                                                                                         recharge_time, uavLocation)
                    num_function_calls += 1
                    detected_failed_vehicles = {}
                    
                    # print("centralized auction results")
                    # print(f"Updated vehicle routes  -  {vehicle_routes}")
                    # print(f"Updated Trip Times  -  {vehicle_trip_times}")
                    
                    mission_time = round(max([sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time +
                                             sum(idle_time[k].values()) for k in range(len(vehicle_routes))]), 1)
                    # print(f"Mission time extended to {mission_time} time units.")
                    # print()
                    
                t += 0.1
                t = round(t, 1)
                
            end = time.time()
            execution_time = round(end - start, 3)
            avg_time_per_call = round(execution_time / num_function_calls, 3) if num_function_calls > 0 else 0
            
            # print(f"Execution time for centralized auction algorithm {execution_time} seconds")
            # print(f"Average time per function call: {avg_time_per_call} seconds")
            # print("Mission Complete")
            print("Final Routes Information")
            print(f"Final Solution Routes  -  {vehicle_routes}")
            print(f"Final Solution Trip Times  -  {vehicle_trip_times}")
            # print(f'Final mission time - {mission_time}')
            # print("----------")
            
            total_idle_time = sum(sum(idle_time[k].values()) for k in range(len(vehicle_routes)))
            
            instanceData['idle time'].append(total_idle_time)
            instanceData['Maximum Trip Time after auction algorithm'].append(mission_time)
            instanceData['% increase in maximum trip time'].append(round(((mission_time - previous_mission_time) / previous_mission_time) * 100, 1))
            instanceData['Execution time for auction algorithm in sec'].append(execution_time)
            instanceData['Number of Function Calls'].append(num_function_calls)
            instanceData['Average time per function call'].append(avg_time_per_call)
            
            if save_results_to_csv:
                df = pd.DataFrame(instanceData)
                df.to_csv(p + '/results/centralized_auction_algorithm_eglese_new.csv', index=True)
                # print()
                # print()

if __name__ == "__main__":
    # simulate_1(save_results_to_csv=False)
    simulate_2(save_results_to_csv=True)
    # simulate_3(save_results_to_csv=True)