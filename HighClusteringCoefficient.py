import pandas as pd
import networkx as nx
import random
import time
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
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
OUTPUT_FILE = "HighClustering_Results.xlsx"
BUDGETS = [500, 1000, 1500, 2000, 2500]
SIMULATIONS = 10000
STEPS = 0
RANDOM_SEED = 42

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
        print(f"Loaded {version} as directed graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    else:
        G = nx.read_weighted_edgelist(filepath, create_using=nx.Graph(), nodetype=int)
        print(f"Loaded {version} as undirected graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    return G

def load_data():
    """Load graph and attribute files"""
    # Load attributes
    with open(INPUT_FILES['cost'], "r") as f:
        costs = eval(f.read())
        print(f"Loaded costs for {len(costs)} nodes")
        
    with open(INPUT_FILES['benefit'], "r") as f:
        benefits = eval(f.read())
        print(f"Loaded benefits for {len(benefits)} nodes")
        
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

def high_clustering_algorithm(graph, budget, cost_dict):
    """
    Selects seed nodes based on clustering coefficient in descending order
    Args:
        graph: NetworkX graph
        budget: Total budget for seed nodes
        cost_dict: Dictionary of {node: cost}
    Returns:
        seed_set: Selected nodes within budget
        total_cost: Total cost of seed set
    """
    seed_set = []
    total_cost = 0
    
    # Calculate clustering coefficients for all nodes
    clustering_coeffs = nx.clustering(graph)
    
    # Sort nodes by clustering coefficient (descending order)
    sorted_nodes = sorted(clustering_coeffs.items(), key=lambda x: x[1], reverse=True)
    
    # Select nodes until budget is exhausted
    for node, _ in sorted_nodes:
        node_cost = cost_dict.get(node, float('inf'))  # Use get() with default to handle missing nodes
        if total_cost + node_cost <= budget:
            seed_set.append(node)
            total_cost += node_cost
        else:
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
    random.seed(RANDOM_SEED)
    print(f"Using {processes} CPU cores for parallel processing")
    
    # Load all graph versions with specified types
    graphs = {
        "Uniform_0.1": load_graph_version('uniform', 'directed'),
        "Trivalency": load_graph_version('trivalency', 'directed'),
        "Weighted": load_graph_version('weighted', 'directed')
    }
    
    for budget in BUDGETS:
        print(f"\n{'='*40}\nProcessing budget: {budget}\n{'='*40}")
        
        for model_name, graph in graphs.items():
            print(f"\n{'='*20}\nRunning {model_name} model\n{'='*20}")
            
            # Verify graph and costs alignment
            graph_nodes = set(graph.nodes())
            cost_nodes = set(costs.keys())
            missing_in_costs = graph_nodes - cost_nodes
            if missing_in_costs:
                print(f"Warning: {len(missing_in_costs)} nodes in graph missing from cost dict")
            
            # Seed selection using High Clustering algorithm
            print("Starting seed selection with High Clustering...")
            start_sel = time.time()
            seed_set, cost_seed_set = high_clustering_algorithm(graph, budget, costs)
            sel_time = time.time() - start_sel
            print(f"Selected {len(seed_set)} seeds with cost {cost_seed_set}/{budget}")
            print(f"Seed selection completed in {sel_time:.2f} seconds")
            
            # Run simulations
            print(f"Running {SIMULATIONS} simulations...")
            start_sim = time.time()
            influenced_nodes = independent_cascade_parallel(
                graph, seed_set, STEPS, SIMULATIONS, processes
            )
            sim_time = time.time() - start_sim
            print(f"Simulations completed in {sim_time:.2f} seconds")
            
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
                'Simulations': SIMULATIONS,
                'Algorithm': 'HighClustering'
            })

    # Save results
    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    start_time = time.time()
    mp.freeze_support()  # For Windows multiprocessing support
    benchmark()
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")