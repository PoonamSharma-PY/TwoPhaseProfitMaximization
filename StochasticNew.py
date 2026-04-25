import pandas as pd
import networkx as nx
import random
import time
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import math
import os

# Configuration
NUM_CPUS = 20  # <<== Change this to set number of processes used
INPUT_FILES = {
    'graph': "euemail.txt",
    'cost': "cost.txt", 
    'benefit': "benefit.txt"
}
GRAPH_VERSIONS = {
    'uniform': "euemail_uniform.txt",
    'trivalency': "euemail_trivalency.txt",
    'weighted': "euemail_weighted.txt"
}
OUTPUT_FILE = "StochasticGreedy_results1_Epsilon0.1.xlsx"
BUDGETS = [500, 1000, 1500, 2000, 2500]
SELECTION_SIMULATIONS = 100
FINAL_SIMULATIONS = 10000
EPSILON = 0.1

def load_graph_version(version, graph_type='directed'):
    if version not in GRAPH_VERSIONS:
        raise ValueError(f"Unknown graph version: {version}")
    
    filepath = GRAPH_VERSIONS[version]
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Graph file not found: {filepath}")
    
    if graph_type.lower() == 'directed':
        G = nx.read_weighted_edgelist(filepath, create_using=nx.DiGraph(), nodetype=int)
        print(f"Loaded {version} as directed graph")
    else:
        G = nx.read_weighted_edgelist(filepath, create_using=nx.Graph(), nodetype=int)
        print(f"Loaded {version} as undirected graph")
    
    return G

def load_data():
    with open(INPUT_FILES['cost'], "r") as f:
        costs = eval(f.read())
        
    with open(INPUT_FILES['benefit'], "r") as f:
        benefits = eval(f.read())
        
    return costs, benefits

def single_simulation(args):
    G, seeds = args
    active = set(seeds)
    frontier = set(seeds)
    
    while frontier:
        new_active = set()
        for node in frontier:
            for neighbor in G.neighbors(node):
                if neighbor not in active and random.random() < G[node][neighbor]['weight']:
                    new_active.add(neighbor)
        frontier = new_active
        active.update(new_active)
    return active

def run_simulations(G, seeds, processes, num_simulations):
    args = [(G, seeds)] * num_simulations
    with mp.Pool(processes) as pool:
        results = pool.map(single_simulation, args)
    return results

def calculate_profit(influenced_sets, benefits, cost):
    benefit_values = [sum(benefits.get(n, 0) for n in s) for s in influenced_sets]
    avg_benefit = sum(benefit_values) / len(benefit_values)
    return avg_benefit - cost

def evaluate_node_normalized(args):
    graph, seeds, total_cost, node_cost, benefit_dict, num_simulations, base_profit = args
    
    influenced = []
    for _ in range(num_simulations):
        active = set(seeds)
        frontier = set(seeds)
        while frontier:
            new_active = set()
            for node in frontier:
                for neighbor in graph.neighbors(node):
                    if neighbor not in active and random.random() < graph[node][neighbor]['weight']:
                        new_active.add(neighbor)
            frontier = new_active
            active.update(new_active)
        influenced.append(active)
    
    new_profit = calculate_profit(influenced, benefit_dict, total_cost)
    normalized_gain = (new_profit - base_profit) / node_cost if node_cost else 0
    return (new_profit, normalized_gain)

# def stochastic_greedy(graph, budget, cost_dict, benefit_dict, processes):
#     print("ENTERING stochastic_greedy()")
#     nodes = list(graph.nodes())
#     seed_set = []
#     total_cost = 0
#     base_profit = 0

#     while total_cost < budget:
#         remaining_budget = budget - total_cost

#         # Filter affordable candidates
#         candidates = [n for n in nodes if n not in seed_set and cost_dict[n] <= remaining_budget]
#         if not candidates:
#             break

#         # Dynamically compute min_cost based on current candidates
#         min_cost = min(cost_dict[n] for n in candidates)

#         # Dynamic update of k and sample size based on remaining budget and min cost
#         k = max(1, int(remaining_budget / min_cost))  # prevent division by zero
#         sample_size = max(1, math.ceil((len(nodes) * math.log(1/EPSILON)) / k))

#         sample = random.sample(candidates, min(sample_size, len(candidates)))

#         args = [(graph, seed_set + [node], total_cost + cost_dict[node], 
#                  cost_dict[node], benefit_dict, SELECTION_SIMULATIONS, base_profit)
#                 for node in sample]

#         with mp.Pool(processes) as pool:
#             results = pool.map(evaluate_node_normalized, args)

#         best_node, best_normalized_gain = None, -float('inf')
#         best_node_profit = base_profit
#         for node, (profit, normalized_gain) in zip(sample, results):
#             if normalized_gain > best_normalized_gain:
#                 best_node = node
#                 best_normalized_gain = normalized_gain
#                 best_node_profit = profit

#         if best_node and (total_cost + cost_dict[best_node]) <= budget:
#             seed_set.append(best_node)
#             total_cost += cost_dict[best_node]
#             base_profit = best_node_profit
#         else:
#             break

#     return seed_set, total_cost


def stochastic_greedy(graph, budget, cost_dict, benefit_dict, processes):
    print("ENTERING stochastic_greedy()")
    nodes = list(graph.nodes())
    seed_set = []
    total_cost = 0
    base_profit = 0

    # Compute min_cost and k once before loop
    min_cost = min(cost_dict.values())
    k = max(1, int(budget / min_cost))  # prevent division by zero
    # sample_size remains dynamic based on fixed k
    sample_size = max(1, math.ceil((len(nodes) * math.log(1 / EPSILON)) / k))

    while total_cost < budget:
        remaining_budget = budget - total_cost

        # # sample_size remains dynamic based on fixed k
        # sample_size = max(1, math.ceil((len(nodes) * math.log(1 / EPSILON)) / k))

        candidates = [n for n in nodes if n not in seed_set and cost_dict[n] <= remaining_budget]
        if not candidates:
            break

        sample = random.sample(candidates, min(sample_size, len(candidates)))

        args = [(graph, seed_set + [node], total_cost + cost_dict[node],
                 cost_dict[node], benefit_dict, SELECTION_SIMULATIONS, base_profit)
                for node in sample]

        with mp.Pool(processes) as pool:
            results = pool.map(evaluate_node_normalized, args)

        best_node, best_normalized_gain = None, -float('inf')
        best_node_profit = base_profit
        for node, (profit, normalized_gain) in zip(sample, results):
            if normalized_gain > best_normalized_gain:
                best_node = node
                best_normalized_gain = normalized_gain
                best_node_profit = profit

        if best_node and (total_cost + cost_dict[best_node]) <= budget:
            seed_set.append(best_node)
            total_cost += cost_dict[best_node]
            base_profit = best_node_profit
        else:
            break

    return seed_set, total_cost


def benchmark():
    costs, benefits = load_data()
    results = []
    
    graphs = {
        "Uniform_0.1": load_graph_version('uniform', 'directed'),
        "Trivalency": load_graph_version('trivalency', 'directed'),
        "Weighted": load_graph_version('weighted', 'directed')
    }
    
    for budget in BUDGETS:
        print(f"\nProcessing budget: {budget}")
        
        for model_name, graph in graphs.items():
            print(f"Running {model_name} model...")
            
            start_sel = time.time()
            seeds, cost = stochastic_greedy(graph, budget, costs, benefits, NUM_CPUS)
            sel_time = time.time() - start_sel
            
            start_sim = time.time()
            influenced = run_simulations(graph, seeds, NUM_CPUS, FINAL_SIMULATIONS)
            sim_time = time.time() - start_sim
            
            profit = calculate_profit(influenced, benefits, cost)
            avg_benefit = sum(sum(benefits.get(n, 0) for n in s) for s in influenced) / len(influenced)
            
            results.append({
                'Budget': budget,
                'Model': model_name,
                'Seed_Set': str(seeds),
                'Seed_Size': len(seeds),
                'Seed_Cost': cost,
                'Remaining_Budget': budget - cost,
                'Avg_Benefit': avg_benefit,
                'Profit': profit,
                'Selection_Time': sel_time,
                'Simulation_Time': sim_time,
                'Total_Time': sel_time + sim_time,
                'Selection_Simulations': SELECTION_SIMULATIONS,
                'Final_Simulations': FINAL_SIMULATIONS,
                'Epsilon': EPSILON,
                'Used_CPUs': NUM_CPUS
            })

    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    benchmark()
