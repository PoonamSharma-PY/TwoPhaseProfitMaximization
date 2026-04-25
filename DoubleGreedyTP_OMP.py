import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import random
import ast
import time
import networkx as nx
from joblib import Parallel, delayed
from tqdm import tqdm
from numba import njit

@njit
def simulate_icm_matrix(seed_set, adj_matrix, prob_matrix, benefit_array, blocked=None, timelimit=-1):
    n = len(benefit_array)
    activated = np.zeros(n, dtype=np.bool_)
    newly_activated = np.zeros(n, dtype=np.bool_)
    activation_log = -np.ones(n, dtype=np.int32)

    for node in seed_set:
        if node >= 0 and node < n and (blocked is None or not blocked[node]):
            activated[node] = True
            newly_activated[node] = True
            activation_log[node] = 0

    total_benefit = 0.0
    for node in range(n):
        if activated[node]:
            total_benefit += benefit_array[node]

    time_step = 0
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
                            activation_log[v] = time_step + 1
                            total_benefit += benefit_array[v]
        newly_activated = next_new
        time_step += 1
        if timelimit > 0 and time_step >= timelimit:
            break

    return activated, total_benefit, time_step, activation_log

def run_parallel(seed_set, adj_matrix, prob_matrix, benefit_array, sims, processes, blocked=None, t=-1):
    return Parallel(n_jobs=processes)(
        delayed(simulate_icm_matrix)(np.array(seed_set, dtype=np.int32), adj_matrix, prob_matrix, benefit_array, blocked, t)
        for _ in range(sims)
    )

def double_greedy_matrix(nodes, budget, costs, adj_matrix, prob_matrix, benefit_array, processes, simulations, blocked=None):
    random.shuffle(nodes)
    X = set()
    Y = set(nodes)
    total_cost = 0

    for node in tqdm(nodes, desc="Double Greedy Phase", leave=False):
        cost_node = costs.get(node, float('inf'))
        if cost_node > budget - total_cost:
            Y.discard(node)
            continue
        if blocked is not None and blocked[node]:
            Y.discard(node)
            continue

        X_add = X | {node}
        profit_add = sum(simulate_icm_matrix(np.array(list(X_add), dtype=np.int32), adj_matrix, prob_matrix, benefit_array, blocked)[1]
                         for _ in range(simulations)) / simulations
        profit_X = sum(simulate_icm_matrix(np.array(list(X), dtype=np.int32), adj_matrix, prob_matrix, benefit_array, blocked)[1]
                       for _ in range(simulations)) / simulations
        gain = (profit_add - profit_X) / cost_node

        Y_remove = Y - {node}
        profit_Y = sum(simulate_icm_matrix(np.array(list(Y), dtype=np.int32), adj_matrix, prob_matrix, benefit_array, blocked)[1]
                       for _ in range(simulations)) / simulations
        profit_Y_rm = sum(simulate_icm_matrix(np.array(list(Y_remove), dtype=np.int32), adj_matrix, prob_matrix, benefit_array, blocked)[1]
                          for _ in range(simulations)) / simulations
        loss = (profit_Y - profit_Y_rm) / cost_node

        prob = 0 if (gain <= 0 or gain + loss <= 0) else gain / (gain + loss)

        if random.random() < prob:
            X.add(node)
            total_cost += cost_node
        else:
            Y.discard(node)

    return X, total_cost

def two_phase_double_greedy(graph, costs, benefits, budget, split_ratio, timestep, sims_phase1, sims_phase2, processes):
    n = max(graph.nodes()) + 1
    nodes = list(graph.nodes())
    max_deg = max(dict(graph.degree()).values())
    adj_matrix = -np.ones((n, max_deg), dtype=np.int32)
    prob_matrix = np.zeros((n, max_deg), dtype=np.float32)
    benefit_array = np.zeros(n)

    for node, val in benefits.items():
        benefit_array[node] = val

    deg = [0] * n
    for u, v, data in graph.edges(data=True):
        idx = deg[u]
        adj_matrix[u, idx] = v
        prob_matrix[u, idx] = data.get('weight', 0.1)
        deg[u] += 1

    budget1 = int(budget * split_ratio)
    budget2 = budget - budget1

    phase1_seeds, cost1 = double_greedy_matrix(nodes, budget1, costs, adj_matrix, prob_matrix, benefit_array, processes, 1)
    budget2 = budget - cost1
    phase1_results = run_parallel(list(phase1_seeds), adj_matrix, prob_matrix, benefit_array, sims_phase1, processes, t=timestep)

    best_total_profit = float('-inf')
    best_result = None

    for activated1, benefit1, _, log in tqdm(phase1_results, desc="Phase 2 per Simulation", leave=False):
        recently_activated = {i for i in range(n) if log[i] == timestep}
        blocked = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if log[i] >= 0:
                blocked[i] = True

        rem_nodes = list(set(nodes) - set(np.where(blocked)[0]))
        phase2_seeds, cost2 = double_greedy_matrix(rem_nodes, budget2, costs, adj_matrix, prob_matrix, benefit_array, processes, 1, blocked)
        remaining_budget = budget2 - cost2
        diffusion_seeds = list(phase2_seeds | recently_activated)
        combined_seeds = list(set(phase1_seeds).union(phase2_seeds))

        sims = run_parallel(diffusion_seeds, adj_matrix, prob_matrix, benefit_array, sims_phase2, processes, blocked=blocked)
        # Exclude phase 1 activated nodes from phase 2 profit
        avg_profit2 = np.mean([sum(benefit_array[i] for i in range(len(s[0])) if s[0][i] and i not in np.where(activated1)[0]) - cost2 for s in sims])
        avg_steps2 = np.mean([s[2] for s in sims])
        avg_activated2 = np.mean([np.sum(s[0]) for s in sims])
        total_profit = (benefit1 - cost1) + avg_profit2

        if total_profit > best_total_profit:
            best_total_profit = total_profit
            best_result = {
                "Phase1_Seeds": phase1_seeds,
                "Phase2_Seeds": phase2_seeds,
                "Combined_Seeds": combined_seeds,
                "Phase1_Cost": cost1,
                "Phase2_Cost": cost2,
                "Phase1_Profit": benefit1 - cost1,
                "Phase2_Profit": avg_profit2,
                "Total_Profit": total_profit,
                "Phase1_Activated_Count": np.sum(activated1),
                "Phase2_Activated_Count": avg_activated2,
                "Phase1_Budget": budget1,
                "Phase2_Budget": budget2,
                "Phase1_Remaining_Budget": budget1 - cost1,
                "Phase2_Remaining_Budget": budget2 - cost2,
                "Total_Remaining_Budget": budget - (cost1 + cost2),
                "Total_Timestep": timestep + int(np.ceil(avg_steps2))
            }

    return best_result

def load_graph_version(version):
    GRAPH_VERSIONS = {
        'Uniform': "lesmis_uniform.txt",
        'Trivalency': "lesmis_trivalency.txt",
        'Weighted': "lesmis_weighted.txt"
    }
    path = GRAPH_VERSIONS[version]
    return nx.read_weighted_edgelist(path, create_using=nx.Graph(), nodetype=int)
    # return nx.read_weighted_edgelist(path, create_using=nx.DiGraph(), nodetype=int)

def run_benchmark(graph_versions, budgets, split_ratios, timesteps, costs, benefits,
                  processes=28, sims_phase1=100, sims_phase2=100):
    for name, graph in graph_versions.items():
        log_csv = f"TwoPhaseDoubleGreedy_Numba_Results_{name}.csv"
        final_excel = f"TwoPhaseDoubleGreedy_Numba_Results_{name}.xlsx"
        results = []

        if os.path.exists(log_csv):
            df_exist = pd.read_csv(log_csv)
            completed = set(zip(df_exist["Graph_Version"], df_exist["Total_Budget"], df_exist["Split_Ratio"], df_exist["Timestep"]))
        else:
            completed = set()

        for split in split_ratios:
            for budget in budgets:
                for timestep in timesteps:
                    key = (name, budget, split, timestep)
                    if key in completed:
                        print(f"⏩ Skipping {key}")
                        continue

                    print(f"\n🚀 Running: {key}")
                    try:
                        start = time.time()
                        result = two_phase_double_greedy(graph, costs, benefits, budget, split, timestep,
                                                         sims_phase1, sims_phase2, processes)
                        elapsed = time.time() - start

                        row = {
                            "Model": "DoubleGreedy",
                            "Graph_Version": name,
                            "Total_Budget": budget,
                            "Split_Ratio": split,
                            "Timestep": timestep,
                            "Phase1_Budget": result["Phase1_Budget"],
                            "Phase2_Budget": result["Phase2_Budget"],
                            "Phase1_Cost": result["Phase1_Cost"],
                            "Phase2_Cost": result["Phase2_Cost"],
                            "Phase1_Remaining_Budget": result["Phase1_Remaining_Budget"],
                            "Phase2_Remaining_Budget": result["Phase2_Remaining_Budget"],
                            "Total_Remaining_Budget": result["Total_Remaining_Budget"],
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
                        }
                        results.append(row)
                        pd.DataFrame([row]).to_csv(log_csv, mode='a', index=False, header=not os.path.exists(log_csv))
                        print(f"✅ Logged: {key}")
                    except Exception as e:
                        print(f"❌ Error in {key}: {e}")

        if results:
            pd.DataFrame(results).to_excel(final_excel, index=False)
            print(f"📄 Final results saved: {final_excel}")

if __name__ == "__main__":
    with open("cost.txt") as f:
        costs = ast.literal_eval(f.read())
    with open("benefit.txt") as f:
        benefits = ast.literal_eval(f.read())

    graphs = {
        "Uniform": load_graph_version("Uniform"),
        "Trivalency": load_graph_version("Trivalency"),
        "Weighted": load_graph_version("Weighted")
    }

    run_benchmark(
        graph_versions=graphs,
        budgets=[500, 1000, 1500, 2000, 2500],
        split_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
        timesteps=[2, 4, 6, 8, 10],
        costs=costs,
        benefits=benefits,
        processes=28
    )