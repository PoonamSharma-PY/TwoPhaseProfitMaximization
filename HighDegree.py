import pandas as pd
import numpy as np
import statistics 
import ast
import time
import random
import copy
import math
import networkx as nx
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
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
OUTPUT_FILE = "HighDegree_Results.xlsx"
BUDGETS = [500, 1000, 1500, 2000, 2500]
SIMULATIONS = 10000
STEPS = 0

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
        costs = ast.literal_eval(f.read())
        
    with open(INPUT_FILES['benefit'], "r") as f:
        benefits = ast.literal_eval(f.read())
        
    return costs, benefits

def parallel_independent_cascade(args):
    """Helper function to run a single independent cascade simulation"""
    G, seeds, steps = args
    active_nodes = set(seeds)
    newly_active = set(seeds)
    
    while newly_active:
        new_activations = set()
        for node in newly_active:
            for neighbor in G.neighbors(node):
                if neighbor not in active_nodes:
                    prob = G[node][neighbor].get('weight', 0.1)
                    if random.random() < prob:
                        new_activations.add(neighbor)
        newly_active = new_activations
        active_nodes.update(new_activations)
    return active_nodes

def independent_cascade_parallel(G, seeds, steps, num_simulations, processes):
    """Perform multiple runs of ICM using parallel processing"""
    args = [(G, seeds, steps) for _ in range(num_simulations)]
    with mp.Pool(processes=processes) as pool:
        results = pool.map(parallel_independent_cascade, args)
    return results

def high_degree_algorithm(graph, budget, cost_dict):
    """
    Selects seed set based on node degrees in descending order within budget
    """
    degree_dict = {node: graph.degree(node) for node in graph.nodes()}
    sorted_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
    
    seed_set = []
    total_cost = 0
    for node, _ in sorted_nodes:
        node_cost = cost_dict[node]
        if total_cost + node_cost <= budget:
            seed_set.append(node)
            total_cost += node_cost
        if total_cost >= budget:
            break
    return seed_set, total_cost

def sum_benefits(influenced_nodes, benefit_dict):
    """Calculate total benefits from influenced nodes"""
    return [sum(benefit_dict.get(node, 0) for node in s) for s in influenced_nodes]

def benchmark():
    """Main benchmarking function"""
    costs, benefits = load_data()
    results = []
    processes = cpu_count()
    
    # Load all graph versions with specified types
    graphs = {
        "Uniform_0.1": load_graph_version('uniform', 'directed'),
        "Trivalency": load_graph_version('trivalency', 'directed'),
        "Weighted": load_graph_version('weighted', 'directed')
    }
    
    for budget in BUDGETS:
        print(f"\nProcessing budget: {budget}")
        
        for model_name, graph in graphs.items():
            print(f"Running {model_name} model...")
            
            # Generate seed set using HighDegree Algorithm
            start_sel = time.time()
            seed_set, cost_seed_set = high_degree_algorithm(graph, budget, costs)
            sel_time = time.time() - start_sel
            
            # Run simulations
            start_sim = time.time()
            influenced_nodes = independent_cascade_parallel(
                graph, seed_set, STEPS, SIMULATIONS, processes
            )
            sim_time = time.time() - start_sim
            
            # Calculate metrics
            total_benefits = sum_benefits(influenced_nodes, benefits)
            avg_benefit = sum(total_benefits) / len(total_benefits)
            profit = avg_benefit - cost_seed_set
            
            results.append({
                'Budget': budget,
                'Model': model_name,
                'Seed_Set': str(seed_set),
                'Seed_Size': len(seed_set),
                'Seed_Cost': cost_seed_set,
                'Remaining_Budget': budget - cost_seed_set,
                'Avg_Benefit': avg_benefit,
                'Profit': profit,
                'Selection_Time': sel_time,
                'Simulation_Time': sim_time,
                'Total_Time': sel_time + sim_time,
                'Simulations': SIMULATIONS
            })

    # Save results
    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    start_time = time.time()
    benchmark()
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")