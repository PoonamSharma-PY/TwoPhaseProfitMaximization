# FINAL FULL OPTIMIZED SCRIPT: Greedy Influence Maximization with Numba and Multiprocessing

import networkx as nx
import numpy as np
import time
import ast
import pandas as pd
from multiprocessing import Pool
from numba import njit
import os

# ------------- Graph Conversion and Numba-Accelerated ICM -------------

def convert_graph_to_matrices(graph, benefits):
    n = max(graph.nodes()) + 1
    adjacency_list = [[] for _ in range(n)]
    probability_matrix = [[] for _ in range(n)]
    benefit_array = np.zeros(n)

    for node, value in benefits.items():
        benefit_array[node] = value

    for u, v, data in graph.edges(data=True):
        adjacency_list[u].append(v)
        probability_matrix[u].append(data.get('weight', 0.1))

    max_len = max(len(l) for l in adjacency_list)
    adj_matrix = -np.ones((n, max_len), dtype=np.int32)
    prob_matrix = np.zeros((n, max_len), dtype=np.float32)

    for i in range(n):
        for j, v in enumerate(adjacency_list[i]):
            adj_matrix[i, j] = v
            prob_matrix[i, j] = probability_matrix[i][j]

    return adj_matrix, prob_matrix, benefit_array

@njit
def simulate_icm_numba(seed_set, adj_matrix, prob_matrix, benefit_array):
    n = len(benefit_array)
    activated = np.zeros(n, dtype=np.bool_)
    newly_activated = np.zeros(n, dtype=np.bool_)

    for node in seed_set:
        activated[node] = True
        newly_activated[node] = True

    total_benefit = 0.0
    for node in seed_set:
        total_benefit += benefit_array[node]

    while np.any(newly_activated):
        next_newly_activated = np.zeros(n, dtype=np.bool_)
        for u in range(n):
            if newly_activated[u]:
                for j in range(adj_matrix.shape[1]):
                    v = adj_matrix[u, j]
                    if v == -1:
                        break
                    if not activated[v] and np.random.rand() < prob_matrix[u, j]:
                        activated[v] = True
                        next_newly_activated[v] = True
                        total_benefit += benefit_array[v]
        newly_activated = next_newly_activated

    return total_benefit

# ------------------- Greedy Maximization -------------------

def evaluate_candidate(args):
    node, current_seed_set, cost_node, current_profit, total_current_cost, num_simulations, adj_matrix, prob_matrix, benefit_array = args
    new_seed_set = list(current_seed_set) + [node]
    benefits_sum = 0.0
    for _ in range(num_simulations):
        benefits_sum += simulate_icm_numba(new_seed_set, adj_matrix, prob_matrix, benefit_array)
    avg_benefit = benefits_sum / num_simulations
    new_total_cost = total_current_cost + cost_node
    profit_with_node = avg_benefit - new_total_cost
    marginal_gain = (profit_with_node - current_profit) / cost_node
    return (node, marginal_gain)

# FINAL FIXED SCRIPT with Empty Seed Set Check (Numba-safe)

def greedy_maximization(graph, costs, benefits, budget, num_simulations, processes, probability_model):
    adj_matrix, prob_matrix, benefit_array = convert_graph_to_matrices(graph, benefits)
    seed_set = set()
    remaining_budget = budget
    total_cost = 0
    iteration = 0

    while remaining_budget > 0:
        iteration += 1
        candidates = [node for node in range(len(benefit_array)) if node not in seed_set and costs.get(node, float('inf')) <= remaining_budget]
        if not candidates:
            break

        seed_set_list = list(seed_set)
        if seed_set_list:
            current_benefit = 0.0
            for _ in range(num_simulations):
                current_benefit += simulate_icm_numba(seed_set_list, adj_matrix, prob_matrix, benefit_array)
            current_benefit /= num_simulations
        else:
            current_benefit = 0.0

        current_profit = current_benefit - total_cost

        args = [
            (node, seed_set, costs[node], current_profit, total_cost, num_simulations, adj_matrix, prob_matrix, benefit_array)
            for node in candidates
        ]
        with Pool(processes=processes) as pool:
            results = pool.map(evaluate_candidate, args)

        best_node, best_gain = max(results, key=lambda x: x[1])
        if best_gain <= 0:
            break

        seed_set.add(best_node)
        total_cost += costs[best_node]
        remaining_budget -= costs[best_node]
        print(f"Iteration {iteration}: Added node {best_node}, Gain: {best_gain:.4f}, Remaining Budget: {remaining_budget}")

    # Final profit calculation
    seed_set_list = list(seed_set)
    if seed_set_list:
        final_benefit = 0.0
        for _ in range(10000):
            final_benefit += simulate_icm_numba(seed_set_list, adj_matrix, prob_matrix, benefit_array)
        final_benefit /= 10000
    else:
        final_benefit = 0.0

    profit_earned = final_benefit - total_cost

    return {
        "seed_set": seed_set,
        "profit_earned": profit_earned,
        "remaining_budget": remaining_budget,
        "seed_set_length": len(seed_set),
        "total_cost_of_seed_set": total_cost,
        "probability_model": probability_model
    }


# ------------------- Main Execution -------------------

def main():
    GRAPH_VERSIONS = {
        'uniform': "euemail_uniform.txt",
        'trivalency': "euemail_trivalency.txt",
        'weighted': "euemail_weighted.txt"
    }

    with open("cost.txt", "r") as file:
        costs = ast.literal_eval(file.read())

    with open("benefit.txt", "r") as file:
        benefits = ast.literal_eval(file.read())

    budgets = [500, 1000, 1500, 2000, 2500]
    num_simulations_greedy = 100
    processes = 20

    all_results = []

    for model, path in GRAPH_VERSIONS.items():
        if not os.path.exists(path):
            continue
        graph = nx.read_weighted_edgelist(path, create_using=nx.DiGraph(), nodetype=int)
        is_directed = nx.is_directed(graph)
        print(f"\nLoaded {model} with {len(graph.nodes)} nodes and {len(graph.edges)} edges")

        for budget in budgets:
            print(f"\nProcessing Budget: {budget} for model {model}")
            start_time = time.time()
            result = greedy_maximization(graph, costs, benefits, budget, num_simulations_greedy, processes, model)
            elapsed = time.time() - start_time

            result.update({
                "Budget": budget,
                "Execution Time": elapsed,
                "Graph Type": "Directed" if is_directed else "Undirected"
            })
            all_results.append(result)

    df = pd.DataFrame(all_results)
    df.to_excel("Greedy_Results_Optimized.xlsx", index=False)
    print("\nResults saved to Greedy_Results_Optimized.xlsx")

if __name__ == "__main__":
    main()
