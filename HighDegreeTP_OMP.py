import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import ast
import random
import math
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from numba import njit

# Configuration
GRAPH_VERSIONS = {
    'Uniform': "lesmis_uniform.txt",
    'Trivalency': "lesmis_trivalency.txt",
    'Weighted': "lesmis_weighted.txt"
}
NUM_CPUS = 28
FINAL_SIMULATIONS = 100
BUDGETS = [500, 1000, 1500, 2000, 2500]
SPLIT_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
TIMESTEPS = [2, 4, 6, 8, 10]

@njit
def simulate_icm(seed_set, adj_matrix, prob_matrix, blocked=None, timelimit=-1):
    n = len(adj_matrix)
    activated = np.zeros(n, dtype=np.bool_)
    newly_activated = np.zeros(n, dtype=np.bool_)
    recently_activated = np.zeros(n, dtype=np.bool_)
    for node in seed_set:
        if node >= 0 and node < n and (blocked is None or not blocked[node]):
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
                    if not activated[v] and (blocked is None or not blocked[v]):
                        if np.random.rand() < prob_matrix[u, j]:
                            activated[v] = True
                            next_new[v] = True
                            if timelimit > 0 and steps + 1 == timelimit:
                                recently_activated[v] = True
        newly_activated = next_new
        steps += 1
        if timelimit > 0 and steps >= timelimit:
            break
    return activated, recently_activated, steps

def to_matrix(graph):
    n = max(graph.nodes()) + 1
    max_deg = max(dict(graph.degree()).values())
    adj = -np.ones((n, max_deg), dtype=np.int32)
    prob = np.zeros((n, max_deg), dtype=np.float32)
    deg = np.zeros(n, dtype=np.int32)
    for u, v, data in graph.edges(data=True):
        idx = deg[u]
        adj[u, idx] = v
        prob[u, idx] = data.get('weight', 0.1)
        deg[u] += 1
    return adj, prob

def high_degree_selection(nodes, graph, budget, cost_dict):
    degree_list = sorted(nodes, key=lambda x: graph.in_degree(x) if graph.is_directed() else graph.degree(x), reverse=True)
    seed_set = []
    total_cost = 0
    for node in degree_list:
        cost = cost_dict[node]
        if total_cost + cost <= budget:
            seed_set.append(node)
            total_cost += cost
        if total_cost >= budget:
            break
    return seed_set, total_cost

def simulate_parallel(seeds, adj_matrix, prob_matrix, blocked, timelimit):
    results = Parallel(n_jobs=NUM_CPUS)(
        delayed(simulate_icm)(np.array(seeds, dtype=np.int32), adj_matrix, prob_matrix, blocked, timelimit)
        for _ in range(FINAL_SIMULATIONS)
    )
    return results

def run_high_degree_two_phase(G, costs, benefits, budget, split, timestep):
    nodes = list(G.nodes())
    adj_matrix, prob_matrix = to_matrix(G)
    blocked = np.zeros(len(adj_matrix), dtype=np.bool_)

    budget1 = int(budget * split)
    budget2 = budget - budget1

    phase1_seeds, cost1 = high_degree_selection(nodes, G, budget1, costs)
    sims_phase1 = simulate_parallel(phase1_seeds, adj_matrix, prob_matrix, None, timestep)
    budget2 = budget - cost1
    phase1_activated_sets = [set(np.where(s[0])[0]) for s in sims_phase1]
    phase1_recent = [set(np.where(s[1])[0]) for s in sims_phase1]
    phase1_avg_activated = np.mean([len(s) for s in phase1_activated_sets])
    phase1_profit_vals = [sum(benefits.get(i, 0) for i in s) - cost1 for s in phase1_activated_sets]

    best_result = None
    best_profit = -float('inf')
    phase2_avg_activated_count = 0

    for idx, (act_set, rec) in enumerate(zip(phase1_activated_sets, phase1_recent)):
        for node in act_set:
            blocked[node] = True
        candidates = list(set(nodes) - act_set)
        phase2_seeds, cost2 = high_degree_selection(candidates, G, budget2, costs)
        diffusion_seeds = list(set(phase2_seeds).union(rec))
        combined_seeds = list(set(phase1_seeds).union(set(phase2_seeds)))

        sims_phase2 = simulate_parallel(diffusion_seeds, adj_matrix, prob_matrix, blocked, -1)
        profits = [sum(benefits.get(i, 0) for i in range(len(a)) if a[i] and i not in act_set) - cost2 for a, _, _ in sims_phase2]
        steps = [s for _, _, s in sims_phase2]
        avg_profit = np.mean(profits)
        avg_activated_phase2 = np.mean([np.sum(s[0]) for s in sims_phase2])

        if avg_profit + phase1_profit_vals[idx] > best_profit:
            best_profit = avg_profit + phase1_profit_vals[idx]
            best_result = {
                "Phase1_Seeds": phase1_seeds,
                "Phase2_Seeds": phase2_seeds,
                "Combined_Seeds": combined_seeds,
                "Total_Profit": best_profit,
                "Phase1_Profit": phase1_profit_vals[idx],
                "Phase2_Profit": avg_profit,
                "Phase1_Activated_Count": phase1_avg_activated,
                "Phase2_Activated_Count": avg_activated_phase2,
                "Phase2_Timestep": math.ceil(np.mean(steps)),
                "Total_Timestep": timestep + math.ceil(np.mean(steps)),
                "Phase1_Cost": cost1,
                "Phase2_Cost": cost2,
                "Total_Cost": cost1 + cost2,
                "Remaining_Budget": budget - (cost1 + cost2)
            }
            phase2_avg_activated_count = avg_activated_phase2

    return best_result, budget1, budget2

def main():
    with open("cost.txt") as f:
        costs = ast.literal_eval(f.read())
    with open("benefit.txt") as f:
        benefits = ast.literal_eval(f.read())
    
    for name, path in GRAPH_VERSIONS.items():
        # G = nx.read_weighted_edgelist(path, create_using=nx.DiGraph(), nodetype=int)
        G = nx.read_weighted_edgelist(path, create_using=nx.Graph(), nodetype=int)
        results = []
        for budget in BUDGETS:
            for split in SPLIT_RATIOS:
                for t in TIMESTEPS:
                    print(f"\n🚀 {name} | Budget={budget}, Split={split}, Timestep={t}")
                    start = time.time()
                    result, b1, b2 = run_high_degree_two_phase(G, costs, benefits, budget, split, t)
                    elapsed = time.time() - start
                    results.append({
                        "Model": name,
                        "Total_Budget": budget,
                        "Split_Ratio": split,
                        "Timestep": t,

                        "Phase1_Budget": b1,
                        "Phase2_Budget": b2,
                        "Phase1_Cost": result["Phase1_Cost"],
                        "Phase2_Cost": result["Phase2_Cost"],
                        "Phase1_Remaining_Budget": b1 - result["Phase1_Cost"],
                        "Phase2_Remaining_Budget": b2 - result["Phase2_Cost"],
                        "Total_Remaining_Budget": budget - result["Total_Cost"],
                        "Remaining_Budget": result["Remaining_Budget"],

                        "Phase1_SeedSize": len(result["Phase1_Seeds"]),
                        "Phase2_SeedSize": len(result["Phase2_Seeds"]),
                        "Total_SeedSize": len(result["Combined_Seeds"]),
                        "Phase1_Seeds": ','.join(map(str, result["Phase1_Seeds"])),
                        "Phase2_Seeds": ','.join(map(str, result["Phase2_Seeds"])),

                        "Phase1_Activated_Count": result["Phase1_Activated_Count"],
                        "Phase2_Activated_Count": result["Phase2_Activated_Count"],

                        "Phase1_Profit": result["Phase1_Profit"],
                        "Phase2_Profit": result["Phase2_Profit"],
                        "Total_Profit": result["Total_Profit"],

                        "Total_Timestep": result["Total_Timestep"],
                        "Execution_Time": elapsed
                    })
        output_file = f"TwoPhase_HighDegree_Numba_Results_{name}.xlsx"
        pd.DataFrame(results).to_excel(output_file, index=False)
        print(f"\n✅ Results for {name} saved to {output_file}")

if __name__ == "__main__":
    main()