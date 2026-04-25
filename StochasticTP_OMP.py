import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import networkx as nx
import ast
import random
import math
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from numba import njit

NUM_CPUS = 28
FINAL_SIMULATIONS = 100
PHASE2_SIMULATIONS = 1
EPSILON = 0.01
BUDGETS = [500, 1000, 1500, 2000, 2500]
SPLIT_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
TIMESTEPS = [2, 4, 6, 8, 10]
GRAPH_VERSIONS = {
    'Uniform': "lesmis_uniform.txt",
    'Trivalency': "lesmis_trivalency.txt",
    'Weighted': "lesmis_weighted.txt"
}

@njit
def simulate_icm_matrix(seed_set, adj_matrix, prob_matrix, benefit_array, blocked=None, timelimit=-1):
    n = len(benefit_array)
    activated = np.zeros(n, dtype=np.bool_)
    newly_activated = np.zeros(n, dtype=np.bool_)
    for node in seed_set:
        if 0 <= node < n and (blocked is None or not blocked[node]):
            activated[node] = True
            newly_activated[node] = True

    total_benefit = 0.0
    for i in range(n):
        if activated[i]:
            total_benefit += benefit_array[i]

    timestep = 0
    all_newly = []

    while np.any(newly_activated):
        all_newly.append(np.where(newly_activated)[0])
        next_new = np.zeros(n, dtype=np.bool_)
        for u in range(n):
            if newly_activated[u]:
                for j in range(adj_matrix.shape[1]):
                    v = adj_matrix[u, j]
                    if v == -1:
                        break
                    if 0 <= v < n and not activated[v] and (blocked is None or not blocked[v]):
                        if np.random.rand() < prob_matrix[u, j]:
                            activated[v] = True
                            next_new[v] = True
                            total_benefit += benefit_array[v]
        newly_activated = next_new
        timestep += 1
        if timelimit > 0 and timestep >= timelimit:
            break

    return activated, total_benefit, timestep, all_newly

def simulate_wrapper(seed_set, adj_matrix, prob_matrix, benefit_array, blocked=None, t=-1):
    activated, benefit, _, all_newly = simulate_icm_matrix(
        np.array(seed_set, dtype=np.int32), adj_matrix, prob_matrix, benefit_array, blocked=blocked, timelimit=t
    )
    recently_activated = set(all_newly[t - 1]) if t > 0 and len(all_newly) >= t else set()
    return set(np.where(activated)[0]), benefit, recently_activated

def graph_to_matrix(graph, benefits):
    n = max(graph.nodes()) + 1
    max_deg = max(dict(graph.degree()).values())
    adj_matrix = -np.ones((n, max_deg), dtype=np.int32)
    prob_matrix = np.zeros((n, max_deg), dtype=np.float32)
    benefit_array = np.zeros(n)
    for node, val in benefits.items():
        benefit_array[node] = val
    deg = [0] * n
    for u, v, data in graph.edges(data=True):
        p = data.get('weight', 0.1)
        idx = deg[u]
        adj_matrix[u, idx] = v
        prob_matrix[u, idx] = p
        deg[u] += 1
    return adj_matrix, prob_matrix, benefit_array

memo_cache = {}

def stochastic_greedy_round(nodes, budget, cost_dict, adj_matrix, prob_matrix, benefit_array, sims_per_eval, num_cpus, blocked=None):
    seed_set = []
    total_cost = 0
    base_profit = 0
    min_cost = min(cost_dict.values())
    k_initial = max(1, int(budget / min_cost))

    while total_cost < budget:
        remaining_budget = budget - total_cost
        candidates = [n for n in nodes if n not in seed_set and cost_dict[n] <= remaining_budget]
        if not candidates:
            break
        base_sample_size = max(1, math.ceil((len(nodes) * math.log(1 / EPSILON)) / k_initial))
        budget_ratio = remaining_budget / budget
        dynamic_sample_size = int(max(1, base_sample_size * (budget_ratio ** 1.5)))
        sample = random.sample(candidates, min(dynamic_sample_size, len(candidates)))
        results = []
        for node in sample:
            key = tuple(seed_set + [node])
            if key in memo_cache:
                results.append(memo_cache[key])
            else:
                result = simulate_icm_matrix(np.array(seed_set + [node], dtype=np.int32), adj_matrix, prob_matrix, benefit_array, blocked)
                memo_cache[key] = result
                results.append(result)
        best_node, best_gain, best_profit = None, -float('inf'), base_profit
        for i, node in enumerate(sample):
            profit = results[i][1]
            gain = (profit - base_profit) / cost_dict[node]
            if gain > best_gain and gain > 0:
                best_gain = gain
                best_node = node
                best_profit = profit
        if best_node is not None:
            seed_set.append(best_node)
            total_cost += cost_dict[best_node]
            base_profit = best_profit
        else:
            break
    return set(seed_set)

def two_phase_stochastic_greedy(graph, costs, benefits, budget, split_ratio, timestep):
    nodes = list(graph.nodes())
    adj_matrix, prob_matrix, benefit_array = graph_to_matrix(graph, benefits)
    simulate_icm_matrix(np.array([0], dtype=np.int32), adj_matrix, prob_matrix, benefit_array)

    budget1 = int(budget * split_ratio)
    phase1_seeds = stochastic_greedy_round(
        nodes, budget1, costs, adj_matrix, prob_matrix, benefit_array, PHASE2_SIMULATIONS, NUM_CPUS
    )
    cost1 = sum(costs[n] for n in phase1_seeds)
    budget2 = budget - cost1

    phase1_results = Parallel(n_jobs=NUM_CPUS)(
        delayed(simulate_wrapper)(list(phase1_seeds), adj_matrix, prob_matrix, benefit_array, None, timestep)
        for _ in range(FINAL_SIMULATIONS)
    )

    best_result = None
    best_total_profit = -float('inf')

    for activated1, benefit1, recently_activated in tqdm(phase1_results, desc="Phase 2"):
        blocked = np.zeros(len(benefit_array), dtype=np.bool_)
        for node in activated1:
            blocked[node] = True

        candidates = list(set(nodes) - activated1)
        phase2_seeds = stochastic_greedy_round(
            candidates, budget2, costs, adj_matrix, prob_matrix, benefit_array, PHASE2_SIMULATIONS, NUM_CPUS, blocked
        )
        cost2 = sum(costs[n] for n in phase2_seeds)

        diffusion_seeds = list(phase2_seeds | recently_activated)
        combined_seeds = set(phase1_seeds) | set(phase2_seeds)

        results = Parallel(n_jobs=NUM_CPUS)(
            delayed(simulate_icm_matrix)(
                np.array(diffusion_seeds, dtype=np.int32), adj_matrix, prob_matrix, benefit_array, blocked
            ) for _ in range(FINAL_SIMULATIONS)
        )
        # Exclude phase 1 activated nodes from phase 2 profit
        avg_profit2 = np.mean([sum(benefit_array[i] for i in range(len(r[0])) if r[0][i] and i not in activated1) - cost2 for r in results])
        avg_steps2 = np.mean([r[2] for r in results])
        avg_activated2 = np.mean([np.sum(r[0]) for r in results])
        total_profit = (benefit1 - cost1) + avg_profit2

        if total_profit > best_total_profit:
            best_total_profit = total_profit
            best_result = {
                'Phase1_Seeds': phase1_seeds,
                'Phase2_Seeds': phase2_seeds,
                'Combined_Seeds': combined_seeds,
                'Phase1_Cost': cost1,
                'Phase2_Cost': cost2,
                'Phase1_Budget': budget1,
                'Phase2_Budget': budget2,
                'Phase1_Remaining_Budget': budget1 - cost1,
                'Phase2_Remaining_Budget': budget2 - cost2,
                'Total_Remaining_Budget': budget - (cost1 + cost2),
                'Phase1_Profit': benefit1 - cost1,
                'Phase2_Profit': avg_profit2,
                'Total_Profit': total_profit,
                'Phase1_Activated_Count': len(activated1),
                'Phase2_Activated_Count': avg_activated2,
                'Phase2_Duration': int(np.ceil(avg_steps2)),
                'Total_Timestep': timestep + int(np.ceil(avg_steps2))
            }

    return best_result

def benchmark_two_phase_stochastic():
    with open("cost.txt") as f:
        costs = ast.literal_eval(f.read())
    with open("benefit.txt") as f:
        benefits = ast.literal_eval(f.read())

    for model_name, path in GRAPH_VERSIONS.items():
        print(f"\n\U0001f310 Model: {model_name}")
        # graph = nx.read_weighted_edgelist(path, create_using=nx.DiGraph(), nodetype=int)
        graph = nx.read_weighted_edgelist(path, create_using=nx.Graph(), nodetype=int)

        live_csv = f"TwoPhaseStochasticGreedy_Numba_Results_{model_name}_{EPSILON}.csv"
        output_xlsx = f"TwoPhaseStochasticGreedy_Numba_Results_{model_name}_{EPSILON}.xlsx"

        if os.path.exists(live_csv):
            df_existing = pd.read_csv(live_csv)
            completed = set(zip(df_existing.Total_Budget, df_existing.Split_Ratio, df_existing.Timestep))
        else:
            completed = set()

        results = []

        for split in SPLIT_RATIOS:
            for budget in BUDGETS:
                for timestep in TIMESTEPS:
                    key = (budget, split, timestep)
                    if key in completed:
                        print(f"⏩ Skipping {key}")
                        continue

                    print(f"🚀 Running: Budget={budget}, Split={split}, Timestep={timestep}")
                    try:
                        start = time.time()
                        result = two_phase_stochastic_greedy(graph, costs, benefits, budget, split, timestep)
                        elapsed = time.time() - start

                        row = {
                            "Model": "StochasticGreedy",
                            "Graph_Version": model_name,
                            "Total_Budget": budget,
                            "Split_Ratio": split,
                            "Timestep": timestep,
                            "Phase1_Budget": result['Phase1_Budget'],
                            "Phase2_Budget": result['Phase2_Budget'],
                            "Phase1_Cost": result['Phase1_Cost'],
                            "Phase2_Cost": result['Phase2_Cost'],
                            "Phase1_Remaining_Budget": result['Phase1_Remaining_Budget'],
                            "Phase2_Remaining_Budget": result['Phase2_Remaining_Budget'],
                            "Total_Remaining_Budget": result['Total_Remaining_Budget'],
                            "Phase1_SeedSize": len(result['Phase1_Seeds']),
                            "Phase2_SeedSize": len(result['Phase2_Seeds']),
                            "Total_SeedSize": len(result['Combined_Seeds']),
                            "Phase1_Seeds": ','.join(map(str, sorted(result['Phase1_Seeds']))),
                            "Phase2_Seeds": ','.join(map(str, sorted(result['Phase2_Seeds']))),
                            "Phase1_Activated_Count": result['Phase1_Activated_Count'],
                            "Phase2_Activated_Count": result['Phase2_Activated_Count'],
                            "Phase1_Profit": result['Phase1_Profit'],
                            "Phase2_Profit": result['Phase2_Profit'],
                            "Total_Profit": result['Total_Profit'],
                            "Phase2_Duration": result['Phase2_Duration'],
                            "Total_Timestep": result['Total_Timestep'],
                            "Execution_Time": elapsed,
                            "Epsilon": EPSILON
                        }

                        results.append(row)
                        pd.DataFrame([row]).to_csv(live_csv, mode='a', index=False, header=not os.path.exists(live_csv))
                        print(f"✅ Logged: {key}")

                    except Exception as e:
                        print(f"❌ Error in {key}: {e}")

        if results:
            pd.DataFrame(results).to_excel(output_xlsx, index=False)
            print(f"📁 Final XLSX saved: {output_xlsx}")

if __name__ == "__main__":
    benchmark_two_phase_stochastic()