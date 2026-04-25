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
SIMULATIONS_PHASE1 = 100
SIMULATIONS_PHASE2 = 100
BUDGETS = [500, 1000, 1500, 2000, 2500]
SPLIT_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
TIMESTEPS = [2, 4, 6, 8, 10]
GRAPH_VERSIONS = {
    'Uniform': "lesmis_uniform.txt",
    'Trivalency': "lesmis_trivalency.txt",
    'Weighted': "lesmis_weighted.txt"
}

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

def simulate_parallel(seed_set, adj_matrix, prob_matrix, sims, blocked=None, timelimit=-1):
    results = Parallel(n_jobs=NUM_CPUS)(
        delayed(simulate_icm)(np.array(seed_set, dtype=np.int32), adj_matrix, prob_matrix, blocked, timelimit)
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

def select_random_seeds(nodes, budget, cost_dict):
    candidates = nodes.copy()
    random.shuffle(candidates)
    selected = []
    total = 0
    for node in candidates:
        cost = cost_dict[node]
        if total + cost <= budget:
            selected.append(node)
            total += cost
        if total >= budget:
            break
    return selected, total

def benchmark():
    with open("cost.txt") as f:
        costs = ast.literal_eval(f.read())
    with open("benefit.txt") as f:
        benefits = ast.literal_eval(f.read())
    graphs = {
        # name: nx.read_weighted_edgelist(path, create_using=nx.DiGraph(), nodetype=int)
        name: nx.read_weighted_edgelist(path, create_using=nx.Graph(), nodetype=int)
        for name, path in GRAPH_VERSIONS.items()
    }
    for name, G in graphs.items():
        nodes = list(G.nodes())
        adj_matrix, prob_matrix = to_matrix(G)
        results = []
        for split in SPLIT_RATIOS:
            for budget in BUDGETS:
                for timestep in TIMESTEPS:
                    print(f"🔁 {name} | Budget: {budget} | Split: {split} | Timestep: {timestep}")
                    start = time.time()

                    b1 = int(budget * split)
                    b2 = budget - b1
                    phase1_seeds, cost1 = select_random_seeds(nodes, b1, costs)
                    b2 = budget - cost1
                    sims_phase1 = simulate_parallel(phase1_seeds, adj_matrix, prob_matrix, SIMULATIONS_PHASE1, timelimit=timestep)
                    phase1_activated = [set(np.where(s[0])[0]) for s in sims_phase1]
                    phase1_recent = [set(np.where(s[1])[0]) for s in sims_phase1]
                    phase1_profit = [sum(benefits.get(i, 0) for i in act) - cost1 for act in phase1_activated]
                    phase1_avg_profit = np.mean(phase1_profit)
                    phase1_avg_activated_count = np.mean([len(a) for a in phase1_activated])

                    best_result = None
                    best_profit = -float('inf')
                    phase2_avg_activated_count = 0

                    for idx, (act1, rec) in enumerate(zip(phase1_activated, phase1_recent)):
                        blocked = np.zeros(len(costs), dtype=np.bool_)
                        for a in act1:
                            blocked[a] = True
                        candidates = list(set(nodes) - act1)
                        phase2_seeds, cost2 = select_random_seeds(candidates, b2, costs)
                        diffusion_seeds = list(set(phase2_seeds).union(rec))
                        combined_seeds = list(set(phase1_seeds).union(set(phase2_seeds)))
                        sims = simulate_parallel(diffusion_seeds, adj_matrix, prob_matrix, SIMULATIONS_PHASE2, blocked)
                        prof = np.mean([sum(benefits.get(i, 0) for i in range(len(s[0])) if s[0][i] and i not in act1) - cost2 for s in sims])
                        avg_count = np.mean([np.sum(s[0]) for s in sims])
                        total_profit = prof + phase1_profit[idx]
                        if total_profit > best_profit:
                            best_profit = total_profit
                            best_result = {
                                "Phase1_Seeds": phase1_seeds,
                                "Phase2_Seeds": phase2_seeds,
                                "Combined_Seeds": combined_seeds,
                                "Total_Profit": total_profit,
                                "Phase1_Profit": phase1_profit[idx],
                                "Phase2_Profit": prof,
                                "Total_Cost": cost1 + cost2,
                                "Total_Timestep": timestep + math.ceil(np.mean([s[2] for s in sims])),
                                "Remaining_Budget": budget - (cost1 + cost2)
                            }
                            phase2_avg_activated_count = avg_count

                    elapsed = time.time() - start
                    results.append({
                        "Model": name,
                        "Total_Budget": budget,
                        "Split_Ratio": split,
                        "Timestep": timestep,

                        "Phase1_Budget": b1,
                        "Phase2_Budget": b2,
                        "Phase1_Cost": cost1,
                        "Phase2_Cost": cost2,
                        "Phase1_Remaining_Budget": b1 - cost1,
                        "Phase2_Remaining_Budget": b2 - cost2,
                        "Total_Remaining_Budget": budget - (cost1 + cost2),
                        "Remaining_Budget": best_result["Remaining_Budget"],

                        "Phase1_SeedSize": len(best_result["Phase1_Seeds"]),
                        "Phase2_SeedSize": len(best_result["Phase2_Seeds"]),
                        "Total_SeedSize": len(best_result["Combined_Seeds"]),
                        "Phase1_Seeds": ','.join(map(str, best_result["Phase1_Seeds"])),
                        "Phase2_Seeds": ','.join(map(str, best_result["Phase2_Seeds"])),

                        "Phase1_Activated_Count": phase1_avg_activated_count,
                        "Phase2_Activated_Count": phase2_avg_activated_count,

                        "Total_Cost": best_result["Total_Cost"],
                        "Phase1_Profit": best_result["Phase1_Profit"],
                        "Phase2_Profit": best_result["Phase2_Profit"],
                        "Total_Profit": best_result["Total_Profit"],

                        "Total_Timestep": best_result["Total_Timestep"],
                        "Execution_Time": elapsed
                    })
        # Save results for each model to a separate file
        output_file = f"TwoPhase_Random_Numba_Results_{name}.xlsx"
        pd.DataFrame(results).to_excel(output_file, index=False)
        print(f"✅ Results for {name} saved to {output_file}")

if __name__ == '__main__':
    benchmark()