import pandas as pd
import networkx as nx
import random
import time
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import math
import os

# Configuration
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
OUTPUT_FILE = "StochasticGreedy_results.xlsx"
BUDGETS = [500, 1000, 1500, 2000, 2500]
SELECTION_SIMULATIONS = 100   # Reduced for faster seed selection
FINAL_SIMULATIONS = 10000     # For final evaluation
EPSILON = 0.2                 # Exploration parameter

def load_graph_version(version, graph_type='directed'):
    """Load one of the pre-generated graph versions with specified type"""
    if version not in GRAPH_VERSIONS:
        raise ValueError(f"Unknown graph version: {version}")
    
    filepath = GRAPH_VERSIONS[version]
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Graph file not found: {filepath}")
    
    # Load based on specified graph type
    if graph_type.lower() == 'directed':
        G = nx.read_weighted_edgelist(filepath, create_using=nx.DiGraph(), nodetype=int)
        print(f"Loaded {version} as directed graph")
    else:
        G = nx.read_weighted_edgelist(filepath, create_using=nx.Graph(), nodetype=int)
        print(f"Loaded {version} as undirected graph")
    
    return G

def load_data():
    """Load graph and attribute files"""
    # Load attributes
    with open(INPUT_FILES['cost'], "r") as f:
        costs = eval(f.read())
        
    with open(INPUT_FILES['benefit'], "r") as f:
        benefits = eval(f.read())
        
    return costs, benefits

def single_simulation(args):
    """Single simulation without nested multiprocessing"""
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
    """Run parallel simulations"""
    args = [(G, seeds)] * num_simulations
    with mp.Pool(processes) as pool:
        results = pool.map(single_simulation, args)
    return results

def calculate_profit(influenced_sets, benefits, cost):
    """Calculate profit from influenced sets"""
    benefit_values = [sum(benefits.get(n, 0) for n in s) for s in influenced_sets]
    avg_benefit = sum(benefit_values) / len(benefit_values)
    return avg_benefit - cost

def evaluate_node_normalized(args):
    """
    Evaluate node with normalized marginal gain
    (No nested multiprocessing)
    """
    graph, seeds, total_cost, node_cost, benefit_dict, num_simulations = args
    
    # Run simulations sequentially
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
    
    # Calculate current profit without this node
    if len(seeds) > 1:
        current_influenced = []
        for _ in range(num_simulations):
            active = set(seeds[:-1])
            frontier = set(seeds[:-1])
            
            while frontier:
                new_active = set()
                for node in frontier:
                    for neighbor in graph.neighbors(node):
                        if neighbor not in active and random.random() < graph[node][neighbor]['weight']:
                            new_active.add(neighbor)
                frontier = new_active
                active.update(new_active)
            current_influenced.append(active)
        current_profit = calculate_profit(current_influenced, benefit_dict, total_cost - node_cost)
    else:
        current_profit = -total_cost
    
    normalized_gain = (new_profit - current_profit) / node_cost
    return (new_profit, normalized_gain)

def stochastic_greedy(graph, budget, cost_dict, benefit_dict, processes):
    print("ENTERING stochastic_greedy()")
    print(f"Parameters: budget={budget}, processes={processes}")
    print(f"Graph nodes: {len(graph.nodes())}, edges: {len(graph.edges())}")
    print(f"Cost dict size: {len(cost_dict)}, Benefit dict size: {len(benefit_dict)}")
    """
    Stochastic Greedy with cost-normalized marginal gains
    """
    nodes = list(graph.nodes())
    seed_set = []
    total_cost = 0
    n = len(nodes)
    min_cost = min(cost_dict.values())  # Moved outside loop
    k = int(budget/min_cost)
    print("The value of k is ", k)
    iteration = 0
    
    while total_cost < budget:
        remaining_budget = budget - total_cost
        
        # Adaptive sample size
        sample_size = max(1, math.ceil((n * math.log(1/EPSILON)) / k))
        print("The value of sample_size is ", sample_size)
        # Get affordable candidates
        candidates = [n for n in nodes 
                     if n not in seed_set 
                     and cost_dict[n] <= remaining_budget]
        if not candidates:
            iteration = iteration+1
            break
            
        sample = random.sample(candidates, min(sample_size, len(candidates)))
        
        # Prepare arguments for parallel processing
        args = [(graph, seed_set + [node], total_cost + cost_dict[node], 
                cost_dict[node], benefit_dict, SELECTION_SIMULATIONS)
               for node in sample]
        
        # Evaluate nodes in parallel
        with mp.Pool(processes) as pool:
            results = pool.map(evaluate_node_normalized, args)
        
        # Select node with best normalized gain
        best_node, best_normalized_gain = None, -float('inf')
        for node, (profit, normalized_gain) in zip(sample, results):
            if normalized_gain > best_normalized_gain:
                best_node = node
                best_normalized_gain = normalized_gain
        
        if best_node and (total_cost + cost_dict[best_node]) <= budget:
            seed_set.append(best_node)
            print("The best node added is ", best_node)
            total_cost += cost_dict[best_node]
        else:
            break
    
    return seed_set, total_cost

def benchmark():
    """Main benchmarking function"""
    costs, benefits = load_data()
    results = []
    processes = cpu_count()
    
  # Load all graph versions with specified types
    graphs = {
        "Uniform_0.1": load_graph_version('uniform', 'directed'),
        "Trivalency": load_graph_version('trivalency', 'directed'),
        "Weighted": load_graph_version('weighted', 'directed')  # Example: weighted as directed
    }
    
    for budget in BUDGETS:
        print(f"\nProcessing budget: {budget}")
        
        for model_name, graph in graphs.items():
            print(f"Running {model_name} model...")
            
            # Seed selection (using stochastic greedy)
            start_sel = time.time()
            seeds, cost = stochastic_greedy(graph, budget, costs, benefits, processes)
            sel_time = time.time() - start_sel
            
            # Final evaluation (using FINAL_SIMULATIONS)
            start_sim = time.time()
            influenced = run_simulations(graph, seeds, processes, FINAL_SIMULATIONS)
            sim_time = time.time() - start_sim
            
            # Calculate metrics
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
                'Epsilon': EPSILON
            })

    # Save results
    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    benchmark()