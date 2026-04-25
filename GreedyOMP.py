import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import networkx as nx
import ast
import time
import math
from joblib import Parallel, delayed
from numba import njit
from tqdm import tqdm

# Config
NUM_CPUS = 28
SIMULATIONS = 10000
BUDGETS = [500, 1000, 1500, 2000, 2500]
GRAPH_VERSIONS = {
    'Uniform': "lesmis_uniform.txt",
    'Trivalency': "lesmis_trivalency.txt",
    'Weighted': "lesmis_weighted.txt"
}

@njit
def simulate_icm_numba(seed_set, adj_matrix, prob_matrix):
    n = len(adj_matrix)
    activated = np.zeros(n, dtype=np.bool_)
    newly_activated = np.zeros(n, dtype=np.bool_)
    for node in seed_set:
        if 0 <= node < n:
            activated[node] = True
            newly_activated[node] = True
    steps = 0
    while np.any(newly_activated):
        next_new = np.zeros(n, dtype=np.bool_)
        for u in range(n):
            if newly_activated[u]:
                for j in range(adj_matrix.shape[1]):
                    v = adj_matrix[u, j]
                    if v == -1:
                        break
                    if 0 <= v < n and not activated[v] and np.random.rand() < prob_matrix[u, j]:
                        activated[v] = True
                        next_new[v] = True
        newly_activated = next_new
        steps += 1
    return activated, steps

def simulate_parallel(seed_set, adj_matrix, prob_matrix, sims=10000):
    results = Parallel(n_jobs=NUM_CPUS)(
        delayed(simulate_icm_numba)(np.array(seed_set, dtype=np.int32), adj_matrix, prob_matrix)
        for _ in range(sims)
    )
    return results

# def to_matrix(graph):
#     n = max(graph.nodes()) + 1
#     max_deg = max(dict(graph.out_degree()).values())
#     adj = -np.ones((n, max_deg), dtype=np.int32)
#     prob = np.zeros((n, max_deg), dtype=np.float32)
#     deg = np.zeros(n, dtype=np.int32)
#     for u, v, data in graph.edges(data=True):
#         idx = deg[u]
#         adj[u, idx] = v
#         prob[u, idx] = data.get("weight", 0.1)
#         deg[u] += 1
#     return adj, prob

def to_matrix(graph):
    n = max(graph.nodes()) + 1
    if nx.is_directed(graph):
        max_deg = max(dict(graph.out_degree()).values())
    else:
        max_deg = max(dict(graph.degree()).values())

    adj = -np.ones((n, max_deg), dtype=np.int32)
    prob = np.zeros((n, max_deg), dtype=np.float32)
    deg = np.zeros(n, dtype=np.int32)

    for u, v, data in graph.edges(data=True):
        idx = deg[u]
        adj[u, idx] = v
        prob[u, idx] = data.get("weight", 0.1)
        deg[u] += 1

    return adj, prob


def evaluate_candidate(node, current_seed_set, cost_node, current_profit, total_current_cost, num_simulations, adj_matrix, prob_matrix, benefits):
    new_seed_set = list(current_seed_set) + [node]

    def simulate_once():
        activated, _ = simulate_icm_numba(new_seed_set, adj_matrix, prob_matrix)
        return sum(benefits.get(i, 0) for i in range(len(activated)) if activated[i])

    benefits_list = Parallel(n_jobs=NUM_CPUS)(delayed(simulate_once)() for _ in range(num_simulations))
    avg_benefit = np.mean(benefits_list)
    new_total_cost = total_current_cost + cost_node
    profit_with_node = avg_benefit - new_total_cost
    marginal_gain = (profit_with_node - current_profit) / cost_node if cost_node > 0 else float('inf')
    return (node, marginal_gain)

def greedy_maximization(graph, costs, benefits, budget, num_simulations=100):
    adj_matrix, prob_matrix = to_matrix(graph)
    seed_set = set()
    total_cost = 0
    remaining_budget = budget

    while remaining_budget > 0:
        candidates = [node for node in graph.nodes() if node not in seed_set and costs.get(node, float('inf')) <= remaining_budget]
        if not candidates:
            break

        seed_set_list = list(seed_set)
        current_benefit = 0
        if seed_set_list:
            sims = simulate_parallel(seed_set_list, adj_matrix, prob_matrix, num_simulations)
            current_benefit = np.mean([sum(benefits.get(i, 0) for i in range(len(activated)) if activated[i]) for activated, _ in sims])
        current_profit = current_benefit - total_cost

        results = Parallel(n_jobs=NUM_CPUS)(
            delayed(evaluate_candidate)(
                node, seed_set, costs.get(node, float('inf')), current_profit, total_cost,
                num_simulations, adj_matrix, prob_matrix, benefits
            )
            for node in tqdm(candidates, desc=f"Candidates (budget={budget})", leave=False)
        )

        best_node, best_gain = max(results, key=lambda x: x[1])
        if best_gain <= 0:
            break

        seed_set.add(best_node)
        total_cost += costs[best_node]
        remaining_budget -= costs[best_node]

    return list(seed_set), total_cost

def load_data():
    with open("cost.txt") as f:
        costs = ast.literal_eval(f.read())
    with open("benefit.txt") as f:
        benefits = ast.literal_eval(f.read())
    return costs, benefits

def benchmark_greedy():
    costs, benefits = load_data()

    for model_name, graph_path in GRAPH_VERSIONS.items():
        print(f"\n🌐 Model: {model_name}")
        if not os.path.exists(graph_path):
            print(f"❌ Missing file: {graph_path}. Skipping.")
            continue

        # graph = nx.read_weighted_edgelist(graph_path, create_using=nx.DiGraph(), nodetype=int)
        graph = nx.read_weighted_edgelist(graph_path, create_using=nx.Graph(), nodetype=int)
        adj_matrix, prob_matrix = to_matrix(graph)
        results = []

        for budget in BUDGETS:
            print(f"🚀 Running: Budget={budget}")
            start = time.time()
            seed_set, seed_cost = greedy_maximization(graph, costs, benefits, budget, num_simulations=100)
            sims = simulate_parallel(seed_set, adj_matrix, prob_matrix, SIMULATIONS)

            benefits_list = []
            steps_list = []
            activated_counts = []
            for activated, steps in sims:
                total = sum(benefits.get(i, 0) for i in range(len(activated)) if activated[i])
                benefits_list.append(total)
                steps_list.append(steps)
                activated_counts.append(np.sum(activated))

            avg_benefit = np.mean(benefits_list)
            avg_steps = np.mean(steps_list)
            avg_activated = np.mean(activated_counts)
            profit = avg_benefit - seed_cost

            results.append({
                "Model": "Greedy",
                "Graph_Version": model_name,
                "Total_Budget": budget,
                "Seed_Size": len(seed_set),
                "Seed_Cost": seed_cost,
                "Remaining_Budget": budget - seed_cost,
                "Avg_Benefit": avg_benefit,
                "Profit": profit,
                "Avg_Activated_Count": avg_activated,
                "Avg_Timestep": math.ceil(avg_steps),
                "Execution_Time": time.time() - start,
                "Simulations": SIMULATIONS,
                "Seed_Set": ','.join(map(str, sorted(seed_set)))
            })

        output_file = f"Greedy_Numba_Results_{model_name}.xlsx"
        pd.DataFrame(results).to_excel(output_file, index=False)
        print(f"✅ Saved to {output_file}")

if __name__ == "__main__":
    benchmark_greedy()