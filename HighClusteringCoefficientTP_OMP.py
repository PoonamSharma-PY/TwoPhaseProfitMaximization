import os
import time
import ast
import random
import pandas as pd
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict
from numba import njit

# Configuration
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

NUM_CPUS = 28
SIMULATIONS_PHASE1 = 100
SIMULATIONS_PHASE2 = 100
BUDGETS = [500, 1000, 1500, 2000, 2500]
SPLIT_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
TIMESTEPS = [2, 4, 6, 8, 10]

GRAPH_VERSIONS = {
    "Uniform": "lesmis_uniform.txt",
    "Trivalency": "lesmis_trivalency.txt",
    "Weighted": "lesmis_weighted.txt"
}

@njit
def simulate_icm(seed_set, adj_matrix, prob_matrix, blocked=None, timelimit=-1):
    n = len(adj_matrix)
    activated = np.zeros(n, dtype=np.bool_)
    newly_activated = np.zeros(n, dtype=np.bool_)
    recently_activated = np.zeros(n, dtype=np.bool_)
    for node in seed_set:
        if 0 <= node < n and (blocked is None or not blocked[node]):
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
    return Parallel(n_jobs=NUM_CPUS)(
        delayed(simulate_icm)(np.array(seed_set, dtype=np.int32), adj_matrix, prob_matrix, blocked, timelimit)
        for _ in range(sims)
    )

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

def select_seeds_by_clustering(G, clustering, costs, budget):
    nodes = list(G.nodes())
    node_data = [(n, clustering.get(n, 0.0), costs[n]) for n in nodes if costs[n] <= budget]
    node_data.sort(key=lambda x: (-x[1], x[2]))
    selected = []
    total = 0
    for n, _, c in node_data:
        if total + c <= budget:
            selected.append(n)
            total += c
        if total >= budget:
            break
    return selected, total

def run_benchmark():
    with open("cost.txt") as f:
        costs = ast.literal_eval(f.read())
    with open("benefit.txt") as f:
        benefits = ast.literal_eval(f.read())

    for model_name, path in GRAPH_VERSIONS.items():
        if not os.path.exists(path):
            print(f"❌ Missing file: {path}. Skipping {model_name}.")
            continue

        print(f"\n📊 Running model: {model_name}")
        # G = nx.read_weighted_edgelist(path, create_using=nx.DiGraph(), nodetype=int)
        G = nx.read_weighted_edgelist(path, create_using=nx.Graph(), nodetype=int)
        clustering = nx.clustering(G.to_undirected())
        nodes = list(G.nodes())
        adj_matrix, prob_matrix = to_matrix(G)
        results = []

        for split in SPLIT_RATIOS:
            for budget in BUDGETS:
                for timestep in TIMESTEPS:
                    print(f"▶ Model={model_name}, Budget={budget}, Split={split}, Timestep={timestep}")
                    start = time.time()
                    b1 = int(budget * split)
                    b2 = budget - b1

                    seeds1, cost1 = select_seeds_by_clustering(G, clustering, costs, b1)
                    b2 = budget - cost1
                    sims_phase1 = simulate_parallel(seeds1, adj_matrix, prob_matrix, SIMULATIONS_PHASE1, timelimit=timestep)
                    phase1_activated = [set(np.where(s[0])[0]) for s in sims_phase1]
                    phase1_recent = [set(np.where(s[1])[0]) for s in sims_phase1]
                    phase1_profit = [sum(benefits.get(i, 0) for i in act) - cost1 for act in phase1_activated]
                    phase1_avg_profit = np.mean(phase1_profit)
                    phase1_avg_activated = np.mean([len(a) for a in phase1_activated])

                    best_result = None
                    best_profit = -float('inf')

                    for idx, (act1, rec) in enumerate(zip(phase1_activated, phase1_recent)):
                        blocked = np.zeros(len(costs), dtype=np.bool_)
                        for a in act1:
                            blocked[a] = True
                        candidates = list(set(nodes) - act1)
                        seeds2, cost2 = select_seeds_by_clustering(G.subgraph(candidates), clustering, costs, b2)
                        diffusion_seeds = list(set(seeds2).union(rec))
                        combined_seeds = list(set(seeds1).union(set(seeds2)))
                        sims = simulate_parallel(diffusion_seeds, adj_matrix, prob_matrix, SIMULATIONS_PHASE2, blocked)
                        prof = np.mean([sum(benefits.get(i, 0) for i in range(len(s[0])) if s[0][i] and i not in act1) - cost2 for s in sims])
                        avg_timestep = np.mean([s[2] for s in sims])
                        avg_activated2 = np.mean([np.sum(s[0]) for s in sims])
                        total_profit = prof + phase1_profit[idx]
                        if total_profit > best_profit:
                            best_result = {
                                "Phase1_Seeds": seeds1,
                                "Phase2_Seeds": seeds2,
                                "Combined_Seeds": combined_seeds,
                                "Phase1_Cost": cost1,
                                "Phase2_Cost": cost2,
                                "Total_Cost": cost1 + cost2,
                                "Remaining_Budget": budget - (cost1 + cost2),
                                "Phase1_Profit": phase1_profit[idx],
                                "Phase2_Profit": prof,
                                "Total_Profit": total_profit,
                                "Phase1_Activated_Count": phase1_avg_activated,
                                "Phase2_Activated_Count": avg_activated2,
                                "Total_Timestep": timestep + int(round(avg_timestep))
                            }
                            best_profit = total_profit

                    elapsed = time.time() - start
                    results.append({
                        "Model": "HighClustering",
                        "Graph_Version": model_name,
                        "Total_Budget": budget,
                        "Split_Ratio": split,
                        "Timestep": timestep,

                        "Phase1_Budget": b1,
                        "Phase2_Budget": b2,
                        "Phase1_Cost": best_result["Phase1_Cost"],
                        "Phase2_Cost": best_result["Phase2_Cost"],
                        "Phase1_Remaining_Budget": b1 - best_result["Phase1_Cost"],
                        "Phase2_Remaining_Budget": b2 - best_result["Phase2_Cost"],
                        "Total_Remaining_Budget": budget - best_result["Total_Cost"],
                        "Remaining_Budget": best_result["Remaining_Budget"],

                        "Phase1_SeedSize": len(best_result["Phase1_Seeds"]),
                        "Phase2_SeedSize": len(best_result["Phase2_Seeds"]),
                        "Total_SeedSize": len(best_result["Combined_Seeds"]),
                        "Phase1_Seeds": ','.join(map(str, best_result["Phase1_Seeds"])),
                        "Phase2_Seeds": ','.join(map(str, best_result["Phase2_Seeds"])),

                        "Phase1_Activated_Count": best_result["Phase1_Activated_Count"],
                        "Phase2_Activated_Count": best_result["Phase2_Activated_Count"],

                        "Phase1_Profit": best_result["Phase1_Profit"],
                        "Phase2_Profit": best_result["Phase2_Profit"],
                        "Total_Profit": best_result["Total_Profit"],

                        "Total_Timestep": best_result["Total_Timestep"],
                        "Execution_Time": elapsed
                    })

        df_model = pd.DataFrame(results)
        out_file = f"TwoPhase_HighClustering_Numba_Results_{model_name}.xlsx"
        df_model.to_excel(out_file, index=False)
        print(f"✅ Results saved to {out_file}")

if __name__ == "__main__":
    run_benchmark()