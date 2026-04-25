import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import random
import ast
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import set_start_method
from joblib import Parallel, delayed
from numba import njit

# Config
NUM_CPUS = 28
NUM_SIM_PHASE1 = 100
NUM_SIM_PHASE2 = 100

GRAPH_VERSIONS = {
    'Uniform': "lesmis_uniform.txt",
    'Trivalency': "lesmis_trivalency.txt",
    'Weighted': "lesmis_weighted.txt"
}

BUDGETS = [500, 1000, 1500, 2000, 2500]
SPLIT_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
TIMESTEPS = [2, 4, 6, 8, 10]

def load_graph(version):
    path = GRAPH_VERSIONS[version]
    return nx.read_weighted_edgelist(path, create_using=nx.Graph(), nodetype=int)
    # return nx.read_weighted_edgelist(path, create_using=nx.DiGraph(), nodetype=int)


def convert_graph_to_matrix(graph, benefits):
    n = max(graph.nodes()) + 1
    max_deg = max(dict(graph.degree()).values())
    adj_matrix = -np.ones((n, max_deg), dtype=np.int32)
    prob_matrix = np.zeros((n, max_deg), dtype=np.float32)
    benefit_array = np.zeros(n)
    deg = [0] * n
    for node, val in benefits.items():
        benefit_array[node] = val
    for u, v, data in graph.edges(data=True):
        idx = deg[u]
        adj_matrix[u, idx] = v
        prob_matrix[u, idx] = data.get('weight', 0.1)
        deg[u] += 1
    return adj_matrix, prob_matrix, benefit_array

@njit
def simulate_icm_matrix(seed_set, adj_matrix, prob_matrix, benefit_array, blocked, max_steps):
    n = len(benefit_array)
    activated = np.zeros(n, dtype=np.bool_)
    newly_activated = np.zeros(n, dtype=np.bool_)
    recently_activated = np.zeros(n, dtype=np.bool_)
    for node in seed_set:
        if node >= 0 and node < n and blocked[node] == 0:
            activated[node] = True
            newly_activated[node] = True

    steps = 0
    total_benefit = benefit_array[activated].sum()

    while np.any(newly_activated):
        next_new = np.zeros(n, dtype=np.bool_)
        for u in range(n):
            if newly_activated[u]:
                for j in range(adj_matrix.shape[1]):
                    v = adj_matrix[u, j]
                    if v == -1:
                        break
                    if not activated[v] and blocked[v] == 0:
                        if np.random.rand() < prob_matrix[u, j]:
                            activated[v] = True
                            next_new[v] = True
                            total_benefit += benefit_array[v]
                            if max_steps > 0 and steps + 1 == max_steps:
                                recently_activated[v] = True
        newly_activated = next_new
        steps += 1
        if max_steps > 0 and steps >= max_steps:
            break

    return activated, recently_activated, steps, total_benefit

def run_phase1(adj_matrix, prob_matrix, benefit_array, cost_array, seeds, timestep):
    blocked = np.zeros(len(benefit_array), dtype=np.int32)
    activated, recently_activated, steps, benefit = simulate_icm_matrix(np.array(list(seeds), dtype=np.int32), adj_matrix, prob_matrix, benefit_array, blocked, timestep)
    profit = benefit - cost_array[list(seeds)].sum()
    return set(np.where(activated)[0]), set(np.where(recently_activated)[0]), profit, activated.sum()

def run_phase2(adj_matrix, prob_matrix, benefit_array, cost_array, phase2_seeds, blocked, phase1_activated):
    activated, _, steps, benefit = simulate_icm_matrix(np.array(list(phase2_seeds), dtype=np.int32), adj_matrix, prob_matrix, benefit_array, blocked, -1)
    # Exclude phase 1 activated nodes from profit calculation
    profit = sum(benefit_array[i] for i in range(len(activated)) if activated[i] and i not in phase1_activated) - cost_array[list(phase2_seeds)].sum()
    return activated, profit, steps, activated.sum()

def single_discount_select(adj_matrix, cost_array, budget, blocked):
    degrees = np.array([np.count_nonzero(adj_matrix[u] != -1) if not blocked[u] else -1e9 for u in range(len(adj_matrix))])
    selected = set()
    remaining = budget
    while True:
        node = np.argmax(degrees)
        if degrees[node] <= -1e8 or cost_array[node] > remaining:
            break
        selected.add(node)
        remaining -= cost_array[node]
        for j in range(adj_matrix.shape[1]):
            v = adj_matrix[node, j]
            if v == -1:
                break
            degrees[v] -= 1
        degrees[node] = -1e9
    return selected, remaining

def run_experiment(model, graph, costs, benefits, budgets, split_ratios, timesteps):
    adj_matrix, prob_matrix, benefit_array = convert_graph_to_matrix(graph, benefits)
    cost_array = np.zeros(len(benefit_array))
    for node, val in costs.items():
        cost_array[node] = val

    results = []
    for budget in budgets:
        for split in split_ratios:
            for t in timesteps:
                print(f"\n🚀 {model} | Budget={budget} | Split={split} | Timestep={t}")
                start = time.time()
                budget1 = int(budget * split)
                budget2 = budget - budget1

                blocked = np.zeros(len(benefit_array), dtype=np.int32)
                seeds_phase1, rem1 = single_discount_select(adj_matrix, cost_array, budget1, blocked)
                cost1 = cost_array[list(seeds_phase1)].sum()
                budget2 += rem1

                phase1_results = Parallel(n_jobs=NUM_CPUS)(
                    delayed(run_phase1)(adj_matrix, prob_matrix, benefit_array, cost_array, seeds_phase1, t)
                    for _ in range(NUM_SIM_PHASE1)
                )

                best = None
                best_profit = -float('inf')
                for act1, recently, profit1, count1 in phase1_results:
                    blocked = np.zeros(len(benefit_array), dtype=np.int32)
                    for n in act1:
                        blocked[n] = 1

                    seeds2, rem2 = single_discount_select(adj_matrix, cost_array, budget2, blocked)
                    cost2 = cost_array[list(seeds2)].sum()
                    diffusion_seeds = seeds2 | recently
                    combined_seeds = seeds_phase1 | seeds2

                    sims = Parallel(n_jobs=NUM_CPUS)(
                        delayed(run_phase2)(adj_matrix, prob_matrix, benefit_array, cost_array, diffusion_seeds, blocked, act1)
                        for _ in range(NUM_SIM_PHASE2)
                    )

                    avg_profit2 = sum(s[1] for s in sims) / NUM_SIM_PHASE2
                    avg_steps2 = sum(s[2] for s in sims) / NUM_SIM_PHASE2
                    avg_activated2 = sum(s[3] for s in sims) / NUM_SIM_PHASE2
                    total_profit = profit1 + avg_profit2

                    if total_profit > best_profit:
                        best_profit = total_profit
                        best = {
                            "Model": model,
                            "Total_Budget": budget,
                            "Split_Ratio": split,
                            "Timestep": t,
                            "Phase1_Budget": budget1,
                            "Phase2_Budget": budget2,
                            "Phase1_Cost": cost1,
                            "Phase2_Cost": cost2,
                            "Phase1_Remaining_Budget": budget1 - cost1,
                            "Phase2_Remaining_Budget": budget2 - cost2,
                            "Total_Remaining_Budget": budget - (cost1 + cost2),
                            "Phase1_SeedSize": len(seeds_phase1),
                            "Phase2_SeedSize": len(seeds2),
                            "Total_SeedSize": len(combined_seeds),
                            "Phase1_Seeds": ','.join(map(str, seeds_phase1)),
                            "Phase2_Seeds": ','.join(map(str, seeds2)),
                            "Phase1_Activated_Count": count1,
                            "Phase2_Activated_Count": avg_activated2,
                            "Phase1_Profit": profit1,
                            "Phase2_Profit": avg_profit2,
                            "Total_Profit": total_profit,
                            "Total_Timestep": int(round(avg_steps2)) + t,
                            "Execution_Time": time.time() - start
                        }
                if best:
                    results.append(best)
        output_file = f"TwoPhase_SingleDiscount_Numba_Results_{model}.xlsx"
        pd.DataFrame(results).to_excel(output_file, index=False)
        print(f"✅ Results for {model} saved to {output_file}")
    return pd.DataFrame(results)

def main():
    with open("cost.txt") as f:
        costs = ast.literal_eval(f.read())
    with open("benefit.txt") as f:
        benefits = ast.literal_eval(f.read())

    for model in GRAPH_VERSIONS.keys():
        G = load_graph(model)
        run_experiment(model, G, costs, benefits, BUDGETS, SPLIT_RATIOS, TIMESTEPS)

if __name__ == '__main__':
    set_start_method("spawn", force=True)
    main()