import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import networkx as nx
import ast
import time
import math
import random
from joblib import Parallel, delayed
from numba import njit

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

def to_matrix(graph):
    n = max(graph.nodes()) + 1
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

def random_algorithm(graph, budget, cost_dict):
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    seed_set = []
    total_cost = 0
    for node in nodes:
        node_cost = cost_dict.get(node, float('inf'))
        if total_cost + node_cost <= budget:
            seed_set.append(node)
            total_cost += node_cost
        if total_cost >= budget:
            break
    return seed_set, total_cost

def load_data():
    with open("cost.txt") as f:
        costs = ast.literal_eval(f.read())
    with open("benefit.txt") as f:
        benefits = ast.literal_eval(f.read())
    return costs, benefits

def benchmark_random():
    costs, benefits = load_data()

    for model_name, graph_path in GRAPH_VERSIONS.items():
        print(f"\n🌐 Model: {model_name}")
        # graph = nx.read_weighted_edgelist(graph_path, create_using=nx.DiGraph(), nodetype=int)
        graph = nx.read_weighted_edgelist(graph_path, create_using=nx.Graph(), nodetype=int)
        adj_matrix, prob_matrix = to_matrix(graph)
        results = []

        for budget in BUDGETS:
            print(f"🚀 Running: Budget={budget}")
            start = time.time()
            seed_set, seed_cost = random_algorithm(graph, budget, costs)
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
                "Model": "Random",
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

        output_file = f"Random_Numba_Results_{model_name}.xlsx"
        pd.DataFrame(results).to_excel(output_file, index=False)
        print(f"✅ Saved to {output_file}")

if __name__ == "__main__":
    benchmark_random()