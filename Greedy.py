import networkx as nx
import random
import time
import ast
from multiprocessing import Pool, cpu_count
import pandas as pd
import os
from collections import defaultdict

# Global variables for multiprocessing workers
adjacency_global = None
benefits_global = None

# Configuration
GRAPH_VERSIONS = {
    'uniform': "euemail_uniform.txt",
    'trivalency': "euemail_trivalency.txt",
    'weighted': "euemail_weighted.txt"
}

def init_worker(adjacency, benefits):
    global adjacency_global, benefits_global
    adjacency_global = adjacency
    benefits_global = benefits

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

def simulate_single_icm(seed_set):
    activated = set(seed_set)
    total_benefit = sum(benefits_global.get(node, 0) for node in activated)
    newly_activated = set(activated)
    while newly_activated:
        next_activated = set()
        for node in newly_activated:
            for neighbor, prob in adjacency_global.get(node, []):
                if neighbor not in activated and random.random() < prob:
                    activated.add(neighbor)
                    next_activated.add(neighbor)
                    total_benefit += benefits_global.get(neighbor, 0)
        newly_activated = next_activated
    return total_benefit

def evaluate_candidate(node, current_seed_set, cost_node, current_profit, total_current_cost, num_simulations):
    """Evaluate a candidate node's marginal gain."""
    new_seed_set = current_seed_set.union({node})
    benefits_sum = 0
    for _ in range(num_simulations):
        benefits_sum += simulate_single_icm(new_seed_set)
    avg_benefit = benefits_sum / num_simulations
    new_total_cost = total_current_cost + cost_node
    profit_with_node = avg_benefit - new_total_cost
    marginal_gain = (profit_with_node - current_profit) / cost_node
    return (node, marginal_gain)

def greedy_maximization_with_profit(graph, costs, benefits, budget, num_simulations, processes, probability_model="uniform"):
    start_time = time.time()

    # Precompute adjacency list
    adjacency = {u: [] for u in graph.nodes}
    for u, v in graph.edges():
        adjacency[u].append((v, graph[u][v].get('weight', 0.1)))

    with Pool(processes=processes, initializer=init_worker, initargs=(adjacency, benefits)) as pool:
        seed_set = set()
        remaining_budget = budget
        total_cost_of_seed_set = 0  # This is the correct variable name
        iteration = 0

        while remaining_budget > 0:
            iteration += 1
            candidates = [node for node in adjacency if node not in seed_set and costs.get(node, float('inf')) <= remaining_budget]
            if not candidates:
                print("No more valid candidates.")
                break

            # Calculate current profit
            current_benefit = sum(pool.map(simulate_single_icm, [seed_set] * num_simulations)) / num_simulations
            current_profit = current_benefit - total_cost_of_seed_set

            # Evaluate candidates - FIXED THE VARIABLE NAME HERE
            args_list = [
                (node, seed_set, costs[node], current_profit, total_cost_of_seed_set, num_simulations)
                for node in candidates
            ]
            results = pool.starmap(evaluate_candidate, args_list)

            # Find best candidate
            best_node, best_gain = None, 0
            for node, gain in results:
                if gain > best_gain:
                    best_gain = gain 
                    best_node = node

            if best_gain <= 0:
                print("No positive gain candidates.")
                break

            # Update seed set
            seed_set.add(best_node)
            total_cost_of_seed_set += costs[best_node]
            remaining_budget -= costs[best_node]
            print(f"Iteration {iteration}: Added {best_node}, Gain: {best_gain:.4f}, Remaining Budget: {remaining_budget}")

        # Final profit calculation
        final_benefit = sum(pool.map(simulate_single_icm, [seed_set] * num_simulations)) / num_simulations
        profit_earned = final_benefit - total_cost_of_seed_set

    execution_time = time.time() - start_time
    return {
        "seed_set": seed_set,
        "execution_time": execution_time,
        "profit_earned": profit_earned,
        "remaining_budget": remaining_budget,
        "seed_set_length": len(seed_set),
        "total_cost_of_seed_set": total_cost_of_seed_set,
        "probability_model": probability_model
    }

def simulate_single_icm_nodes(seed_set):
    """Simulate ICM and return activated nodes."""
    activated = set(seed_set)
    newly_activated = set(activated)
    while newly_activated:
        next_activated = set()
        for node in newly_activated:
            for neighbor, prob in adjacency_global.get(node, []):
                if neighbor not in activated and random.random() < prob:
                    activated.add(neighbor)
                    next_activated.add(neighbor)
        newly_activated = next_activated
    return activated

def simulate_icm_nodes_parallel(graph, seed_set, benefits, num_simulations, processes):
    """Parallel simulation returning influenced nodes."""
    # Compute adjacency list from graph
    adjacency = {u: [] for u in graph.nodes}
    for u, v in graph.edges():
        adjacency[u].append((v, graph[u][v].get('weight', 0.1)))

    args = [seed_set] * num_simulations
    with Pool(processes=processes, initializer=init_worker, initargs=(adjacency, benefits)) as pool:
        results = pool.map(simulate_single_icm_nodes, args)
    return results

def sum_benefits_from_influenced_nodes(influenced_nodes, benefits):
    """Calculate the total benefit from the influenced nodes."""
    benefits_list = []
    for node_set in influenced_nodes:
        total_benefit = sum(benefits.get(node, 0) for node in node_set)
        benefits_list.append(total_benefit)
    return benefits_list

def cost_seed_set(S_star, costs):
    """Calculate the total cost of the seed set."""
    cost_seed = sum(costs.get(c, 0) for c in S_star)
    return cost_seed

def main():
    # Read cost and benefit data
    with open("cost.txt", "r") as file:
        costs = ast.literal_eval(file.read())
    
    with open("benefit.txt", "r") as file:
        benefits = ast.literal_eval(file.read())

    # Budget values to iterate over
    budgets = [500, 1000, 1500, 2000, 2500]
    num_simulations_greedy = 100  # Number of simulations for greedy algorithm
    num_simulations_final = 10000  # Number of simulations for final profit calculation
    # processes = cpu_count()
    processes = 20

    # List to hold results for all models and budgets
    all_results_data = []

    for probability_model, graph_file in GRAPH_VERSIONS.items():
        print(f"\nRunning with probability model: {probability_model}")
        
        # Load the pre-generated graph version using the specified loader
        graph = load_graph_version(probability_model, 'directed')  # Default to directed
        is_directed = nx.is_directed(graph)
        
        for budget in budgets:
            print(f"\nProcessing Budget: {budget} with {probability_model} model")
            start_time_budget = time.time()
            
            # Run greedy algorithm with current probability model
            results = greedy_maximization_with_profit(
                graph,  # Use the pre-loaded graph
                costs, 
                benefits, 
                budget, 
                num_simulations_greedy, 
                processes,
                probability_model
            )
            
            # Run large-scale simulation for accurate profit calculation
            final_seed_set = results["seed_set"]
            influenced_nodes = simulate_icm_nodes_parallel(
                graph, final_seed_set, benefits, num_simulations_final, processes
            )
            
            # Calculate benefits directly from influenced nodes
            total_benefits = sum_benefits_from_influenced_nodes(influenced_nodes, benefits)
            average_benefit = sum(total_benefits) / len(total_benefits) if total_benefits else 0
            cost_seed = cost_seed_set(final_seed_set, costs)
            final_profit = average_benefit - cost_seed

            # Collect results
            row = {
                "Probability Model": probability_model,
                "Seed Set": str(results["seed_set"]),
                "Seed Set Size": results["seed_set_length"],
                "Budget": budget,
                "Remaining Budget": results["remaining_budget"],
                "Execution Time": results["execution_time"],
                "Profit Earned (Greedy)": results["profit_earned"],
                "Cost of Seed Set": results["total_cost_of_seed_set"],
                "Final Profit (Influenced Nodes)": final_profit,
                "Graph Type": "Directed" if is_directed else "Undirected"
            }
            all_results_data.append(row)
            print(f"Budget {budget} with {probability_model} completed in {time.time() - start_time_budget:.2f} seconds")

    # Save to Excel
    df = pd.DataFrame(all_results_data)
    excel_filename = "Greedy_Results.xlsx"
    df.to_excel(excel_filename, index=False)
    print(f"\nResults saved to {excel_filename}")

if __name__ == "__main__":
    main()