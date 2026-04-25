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

# Configuration
NUM_CPUS = 28
SELECTION_SIMULATIONS = 100
FINAL_SIMULATIONS = 10000
# SELECTION_SIMULATIONS = 5
# FINAL_SIMULATIONS = 100
EPSILON = 0.01
BUDGETS = [500, 1000, 1500, 2000, 2500]
GRAPH_VERSIONS = {
    'trivalency': "lesmis_trivalency.txt",
    'uniform': "lesmis_uniform.txt",
    'weighted': "lesmis_weighted.txt"
}

profit_cache = {}

@njit
def simulate_icm_matrix(seed_set, adj_matrix, prob_matrix, benefit_array):
    n = len(benefit_array)
    activated = np.zeros(n, dtype=np.bool_)
    newly_activated = np.zeros(n, dtype=np.bool_)

    for node in seed_set:
        activated[node] = True
        newly_activated[node] = True

    total_benefit = 0.0
    for node in seed_set:
        total_benefit += benefit_array[node]

    steps = 0
    while np.any(newly_activated):
        steps += 1
        next_new = np.zeros(n, dtype=np.bool_)
        for u in range(n):
            if newly_activated[u]:
                for j in range(adj_matrix.shape[1]):
                    v = adj_matrix[u, j]
                    if v == -1:
                        break
                    if not activated[v] and np.random.rand() < prob_matrix[u, j]:
                        activated[v] = True
                        next_new[v] = True
                        total_benefit += benefit_array[v]
        newly_activated = next_new

    return total_benefit, steps

def simulate_avg_profit_cached(seed_set, adj_matrix, prob_matrix, benefit_array, sims):
    key = tuple(sorted(seed_set))
    if key in profit_cache:
        return profit_cache[key], None
    results = Parallel(n_jobs=NUM_CPUS)(
        delayed(simulate_icm_matrix)(np.array(seed_set, dtype=np.int32), adj_matrix, prob_matrix, benefit_array)
        for _ in range(sims)
    )
    profits, steps = zip(*results)
    avg_profit = sum(profits) / sims
    avg_steps = sum(steps) / sims
    profit_cache[key] = avg_profit
    return avg_profit, avg_steps

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
        prob = data.get('weight', 0.1)
        idx = deg[u]
        adj_matrix[u, idx] = v
        prob_matrix[u, idx] = prob
        deg[u] += 1

    return adj_matrix, prob_matrix, benefit_array

def evaluate_candidate(node, seed_set, total_cost, cost, base_profit, adj_matrix, prob_matrix, benefit_array):
    trial_seeds = seed_set + [node]
    profit, _ = simulate_avg_profit_cached(trial_seeds, adj_matrix, prob_matrix, benefit_array, SELECTION_SIMULATIONS)
    gain = (profit - base_profit) / cost
    return node, gain, profit

def stochastic_greedy_numba(graph, budget, cost_dict, benefit_dict):
    nodes = list(graph.nodes())
    seed_set = []
    total_cost = 0
    base_profit = 0

    adj_matrix, prob_matrix, benefit_array = graph_to_matrix(graph, benefit_dict)
    _ = simulate_icm_matrix(np.array([0], dtype=np.int32), adj_matrix, prob_matrix, benefit_array)

    min_cost = min(cost_dict.values())
    k = max(1, int(budget / min_cost))
    sample_size = max(1, math.ceil((len(nodes) * math.log(1 / EPSILON)) / k))

    while total_cost < budget:
        remaining_budget = budget - total_cost
        candidates = [n for n in nodes if n not in seed_set and cost_dict[n] <= remaining_budget]
        if not candidates:
            break

        sample = random.sample(candidates, min(sample_size, len(candidates)))

        args = [(node, seed_set, total_cost, cost_dict[node], base_profit,
                 adj_matrix, prob_matrix, benefit_array) for node in sample]

        results = Parallel(n_jobs=NUM_CPUS)(
            delayed(evaluate_candidate)(*arg) for arg in args
        )

        best_node, best_gain, best_profit = None, -float('inf'), base_profit
        for node, gain, profit in results:
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

    return seed_set, total_cost, adj_matrix, prob_matrix, benefit_array

def run_final_simulation(seeds, adj_matrix, prob_matrix, benefit_array):
    results = Parallel(n_jobs=NUM_CPUS)(
        delayed(simulate_icm_matrix)(np.array(seeds, dtype=np.int32), adj_matrix, prob_matrix, benefit_array)
        for _ in range(FINAL_SIMULATIONS)
    )
    profits, steps = zip(*results)
    avg_profit = sum(profits) / len(profits)  # Fixed: Compute average correctly
    avg_steps = sum(steps) / len(steps)       # Fixed: Compute average correctly
    return avg_profit, avg_steps

def load_data():
    with open("cost.txt") as f:
        costs = ast.literal_eval(f.read())
    with open("benefit.txt") as f:
        benefits = ast.literal_eval(f.read())
    return costs, benefits

def load_graph_version(version):
    path = GRAPH_VERSIONS[version]
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    return nx.read_weighted_edgelist(path, create_using=nx.Graph(), nodetype=int)
    # return nx.read_weighted_edgelist(path, create_using=nx.DiGraph(), nodetype=int)

def benchmark_stochastic_greedy_with_logging():
    costs, benefits = load_data()

    for model_name in GRAPH_VERSIONS.keys():
        graph = load_graph_version(model_name)
        live_csv = f"StochasticGreedy_{model_name}_Numba_Live_{EPSILON}.csv"
        output_xlsx = f"StochasticGreedy_{model_name}_Numba_Result_{EPSILON}.xlsx"

        if os.path.exists(live_csv):
            df_existing = pd.read_csv(live_csv)
            completed = set(zip(df_existing["Budget"], df_existing["Model"]))
        else:
            completed = set()

        results = []
        for budget in BUDGETS:
            if (budget, model_name) in completed:
                print(f"⏩ Skipping: {budget}, {model_name}")
                continue

            try:
                print(f"\n🚀 Running {model_name} — Budget {budget}")
                start_sel = time.time()
                seeds, cost, adj_matrix, prob_matrix, benefit_array = stochastic_greedy_numba(graph, budget, costs, benefits)
                sel_time = time.time() - start_sel

                start_sim = time.time()
                avg_benefit, avg_timestep = run_final_simulation(seeds, adj_matrix, prob_matrix, benefit_array)
                sim_time = time.time() - start_sim
                profit = avg_benefit - cost

                row = {
                    'Budget': budget,
                    'Model': model_name,
                    'Seed_Set': str(seeds),
                    'Seed_Size': len(seeds),
                    'Seed_Cost': cost,
                    'Remaining_Budget': budget - cost,
                    'Avg_Benefit': avg_benefit,
                    'Profit': profit,
                    'Avg_Timestep': math.ceil(avg_timestep),
                    'Selection_Time': sel_time,
                    'Simulation_Time': sim_time,
                    'Total_Time': sel_time + sim_time,
                    'Epsilon': EPSILON,
                    'Selection_Simulations': SELECTION_SIMULATIONS,
                    'Final_Simulations': FINAL_SIMULATIONS
                }

                results.append(row)
                pd.DataFrame([row]).to_csv(live_csv, mode='a', index=False, header=not os.path.exists(live_csv))
                print(f"✅ Logged: {budget}, {model_name}")
            except Exception as e:
                print(f"❌ Error for {(budget, model_name)}: {e}")

        if results:
            pd.DataFrame(results).to_excel(output_xlsx, index=False)
            print(f"📁 Saved XLSX: {output_xlsx}")

if __name__ == "__main__":
    benchmark_stochastic_greedy_with_logging()