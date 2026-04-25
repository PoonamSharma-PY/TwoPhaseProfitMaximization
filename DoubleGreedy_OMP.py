import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import networkx as nx
import numpy as np
import pandas as pd
import random
import time
import math
import ast
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm

# Config
GRAPH_VERSIONS = {
    'trivalency': "lesmis_trivalency.txt",
    'uniform': "lesmis_uniform.txt",
    'weighted': "lesmis_weighted.txt"
}
INPUT_FILES = {
    'cost': "cost.txt",
    'benefit': "benefit.txt"
}
BUDGETS = [500, 1000, 1500, 2000, 2500]
SELECTION_SIMULATIONS = 100
FINAL_SIMULATIONS = 10000
# SELECTION_SIMULATIONS = 5
# FINAL_SIMULATIONS = 100
PROCESSES = 28

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
        steps += 1

    return total_benefit, steps

def simulate_wrapper(seed_set, adj_matrix, prob_matrix, benefit_array):
    return simulate_icm_matrix(seed_set, adj_matrix, prob_matrix, benefit_array)

def run_parallel_simulations(seed_set, adj_matrix, prob_matrix, benefit_array, num_simulations, processes):
    results = Parallel(n_jobs=processes)(
        delayed(simulate_wrapper)(np.array(seed_set, dtype=np.int32), adj_matrix, prob_matrix, benefit_array)
        for _ in range(num_simulations)
    )
    benefits, steps = zip(*results)
    return sum(benefits) / len(benefits), sum(steps) / len(steps)

def graph_to_matrix(graph, benefits):
    n = max(graph.nodes()) + 1
    max_deg = max(dict(graph.degree()).values())
    adj_matrix = -np.ones((n, max_deg), dtype=np.int32)
    prob_matrix = np.zeros((n, max_deg), dtype=np.float32)
    benefit_array = np.zeros(n)

    for node, val in benefits.items():
        benefit_array[node] = val

    count = np.zeros(n, dtype=np.int32)
    for u, v, data in graph.edges(data=True):
        prob = data.get('weight', 0.1)
        idx = count[u]
        adj_matrix[u, idx] = v
        prob_matrix[u, idx] = prob
        count[u] += 1

    return adj_matrix, prob_matrix, benefit_array

def double_greedy_matrix(graph, budget, cost_dict, benefit_dict, processes, num_simulations=SELECTION_SIMULATIONS):
    nodes = list(graph.nodes())
    random.shuffle(nodes)

    adj_matrix, prob_matrix, benefit_array = graph_to_matrix(graph, benefit_dict)
    simulate_icm_matrix(np.array([0], dtype=np.int32), adj_matrix, prob_matrix, benefit_array)  # warm-up

    X = set()
    Y = set(nodes)
    total_cost = 0

    for node in tqdm(nodes, desc=f"Double Greedy for Budget={budget}"):
        if total_cost >= budget or cost_dict.get(node, float('inf')) > budget:
            Y.discard(node)
            continue

        cost_node = cost_dict[node]

        if total_cost + cost_node > budget:
            Y.discard(node)
            continue

        profit_X, _ = run_parallel_simulations(list(X), adj_matrix, prob_matrix, benefit_array, num_simulations, processes)
        X_add = X | {node}
        profit_add, _ = run_parallel_simulations(list(X_add), adj_matrix, prob_matrix, benefit_array, num_simulations, processes)

        gain = (profit_add - profit_X) / cost_node

        Y_remove = Y - {node}
        profit_Y, _ = run_parallel_simulations(list(Y), adj_matrix, prob_matrix, benefit_array, num_simulations, processes)
        profit_Y_remove, _ = run_parallel_simulations(list(Y_remove), adj_matrix, prob_matrix, benefit_array, num_simulations, processes)

        loss = (profit_Y - profit_Y_remove) / cost_node

        prob = 0 if (gain <= 0 or gain + loss <= 0) else gain / (gain + loss)

        if random.random() < prob:
            X.add(node)
            total_cost += cost_node
        else:
            Y.discard(node)

    return list(X), total_cost, adj_matrix, prob_matrix, benefit_array

def load_graph_version(version):
    path = GRAPH_VERSIONS[version]
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    return nx.read_weighted_edgelist(path, create_using=nx.Graph(), nodetype=int)
    # return nx.read_weighted_edgelist(path, create_using=nx.DiGraph(), nodetype=int)

def load_data():
    with open(INPUT_FILES['cost']) as f:
        costs = ast.literal_eval(f.read())
    with open(INPUT_FILES['benefit']) as f:
        benefits = ast.literal_eval(f.read())
    return costs, benefits

def benchmark():
    costs, benefits = load_data()

    for model_name in GRAPH_VERSIONS.keys():
        print(f"\n🌐 Running model: {model_name}")
        log_file = f"DoubleGreedy_{model_name}_Numba_Live.csv"
        excel_file = f"DoubleGreedy_{model_name}_Numba_Result.xlsx"
        if os.path.exists(log_file):
            df_existing = pd.read_csv(log_file)
            completed_jobs = set(zip(df_existing["Budget"], df_existing["Model"]))
        else:
            completed_jobs = set()

        graph = load_graph_version(model_name)
        results = []

        for budget in BUDGETS:
            if (budget, model_name) in completed_jobs:
                print(f"⏭️ Already done: Budget={budget}, Model={model_name}")
                continue

            start_sel = time.time()
            seeds, cost, adj_matrix, prob_matrix, benefit_array = double_greedy_matrix(graph, budget, costs, benefits, PROCESSES)
            sel_time = time.time() - start_sel

            start_sim = time.time()
            avg_benefit, avg_timestep = run_parallel_simulations(seeds, adj_matrix, prob_matrix, benefit_array, FINAL_SIMULATIONS, PROCESSES)
            sim_time = time.time() - start_sim

            profit = avg_benefit - cost

            result = {
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
                'Selection_Simulations': SELECTION_SIMULATIONS,
                'Final_Simulations': FINAL_SIMULATIONS
            }

            results.append(result)
            pd.DataFrame([result]).to_csv(log_file, mode='a', index=False, header=not os.path.exists(log_file))

        if results:
            pd.DataFrame(results).to_excel(excel_file, index=False)
            print(f"✅ Saved XLSX: {excel_file}")

if __name__ == "__main__":
    benchmark()