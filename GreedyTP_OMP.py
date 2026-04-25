import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import networkx as nx
import ast
from tqdm import tqdm
from joblib import Parallel, delayed
from numba import njit
import time

@njit
def simulate_icm_with_tracking(seed_set, adj_matrix, prob_matrix, benefit_array, blocked=None, timelimit=-1):
    n = len(benefit_array)
    activated = np.zeros(n, dtype=np.bool_)
    newly_activated = np.zeros(n, dtype=np.bool_)
    recently_activated = np.zeros(n, dtype=np.bool_)

    for node in seed_set:
        if node >= 0 and node < n and (blocked is None or not blocked[node]):
            activated[node] = True
            newly_activated[node] = True

    total_benefit = benefit_array[activated].sum()
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
                            total_benefit += benefit_array[v]
                            if timelimit > 0 and time_step + 1 == timelimit:
                                recently_activated[v] = True
        time_step += 1
        newly_activated = next_new
        if timelimit > 0 and time_step >= timelimit:
            break

    return activated, recently_activated, total_benefit, time_step

def simulate_wrapper(seed_set, adj_matrix, prob_matrix, benefit_array, blocked=None, t=-1):
    activated, recent, benefit, steps = simulate_icm_with_tracking(np.array(seed_set, dtype=np.int32),
                                                                   adj_matrix, prob_matrix, benefit_array,
                                                                   blocked=blocked, timelimit=t)
    return set(np.where(activated)[0]), set(np.where(recent)[0]), benefit, steps

def run_parallel(seed_set, adj_matrix, prob_matrix, benefit_array, sims, processes, blocked=None, t=-1):
    return Parallel(n_jobs=processes)(
        delayed(simulate_wrapper)(seed_set, adj_matrix, prob_matrix, benefit_array, blocked, t)
        for _ in range(sims)
    )

def greedy_select_seeds(candidates, budget, costs, adj_matrix, prob_matrix, benefit_array, sims, processes, blocked=None):
    selected = set()
    remaining_budget = budget
    for _ in range(len(candidates)):
        best_gain = 0
        best_node = None
        base_profit = sum(simulate_icm_with_tracking(np.array(list(selected), dtype=np.int32),
                                                     adj_matrix, prob_matrix, benefit_array, blocked)[2]
                          for _ in range(sims)) / sims if selected else 0.0
        for node in candidates:
            if node in selected or costs[node] > remaining_budget:
                continue
            trial = selected | {node}
            profit = sum(simulate_icm_with_tracking(np.array(list(trial), dtype=np.int32),
                                                    adj_matrix, prob_matrix, benefit_array, blocked)[2]
                         for _ in range(sims)) / sims
            gain = (profit - base_profit) / costs[node]
            if gain > best_gain:
                best_gain = gain
                best_node = node
        if best_node is None:
            break
        selected.add(best_node)
        remaining_budget -= costs[best_node]
    return selected

def two_phase_greedy_icm(graph, costs, benefits, budget, split_ratio, timestep, sims_phase1, sims_phase2, processes):
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
        prob_matrix[u, idx] = data.get("weight", 0.1)
        deg[u] += 1

    budget1 = int(budget * split_ratio)
    budget2 = budget - budget1
    phase1_seeds = greedy_select_seeds(nodes, budget1, costs, adj_matrix, prob_matrix, benefit_array, 1, processes)
    cost1 = sum(costs[n] for n in phase1_seeds)
    budget2 += (budget1 - cost1)

    phase1_results = run_parallel(list(phase1_seeds), adj_matrix, prob_matrix, benefit_array, sims_phase1, processes, t=timestep)

    best_result = None
    best_total_profit = -float('inf')

    for activated1, recently_activated, profit1, _ in tqdm(phase1_results, desc="Phase 2 for each Phase 1 sim"):
        blocked = np.zeros(n, dtype=np.bool_)
        for node in activated1:
            blocked[node] = True

        candidates = list(set(nodes) - activated1)
        phase2_seeds = greedy_select_seeds(candidates, budget2, costs, adj_matrix, prob_matrix, benefit_array, 1, processes, blocked)
        cost2 = sum(costs[n] for n in phase2_seeds)
        remaining_budget = budget2 - cost2

        diffusion_seeds = list(phase2_seeds | recently_activated)
        combined_seeds = list(set(phase1_seeds).union(phase2_seeds))
        sims = run_parallel(diffusion_seeds, adj_matrix, prob_matrix, benefit_array, sims_phase2, processes, blocked=blocked)

        # Exclude phase 1 activated nodes from phase 2 profit
        avg_profit2 = np.mean([sum(benefit_array[i] for i in range(len(s[0])) if i in s[0] and i not in activated1) - cost2 for s in sims])
        avg_steps2 = np.mean([s[3] for s in sims])
        avg_activated2 = np.mean([len(s[0]) for s in sims])
        total_profit = profit1 + avg_profit2

        if total_profit > best_total_profit:
            best_result = {
                "Phase1_Seeds": phase1_seeds,
                "Phase2_Seeds": phase2_seeds,
                "Combined_Seeds": combined_seeds,
                "Phase1_Cost": cost1,
                "Phase2_Cost": cost2,
                "Total_Cost": cost1 + cost2,
                "Remaining_Budget": remaining_budget,
                "Phase1_Profit": profit1,
                "Phase2_Profit": avg_profit2,
                "Total_Profit": total_profit,
                "Phase1_Activated_Count": len(activated1),
                "Phase2_Activated_Count": avg_activated2,
                "Total_Timestep": int(timestep + np.ceil(avg_steps2))
            }
            best_total_profit = total_profit
    return best_result

def load_graph_version(version):
    GRAPH_VERSIONS = {
        'Uniform': "lesmis_uniform.txt",
        'Trivalency': "lesmis_trivalency.txt",
        'Weighted': "lesmis_weighted.txt"
    }
    return nx.read_weighted_edgelist(GRAPH_VERSIONS[version], create_using=nx.Graph(), nodetype=int)
    # return nx.read_weighted_edgelist(GRAPH_VERSIONS[version], create_using=nx.DiGraph(), nodetype=int)

def benchmark_two_phase_greedy_icm(graphs, costs, benefits,
                                   budgets, split_ratios, timesteps,
                                   processes=28,
                                   sims_phase1=100, sims_phase2=100,
                                   base_filename="TwoPhaseGreedy_Numba_Results"):
    for graph_name, graph in graphs.items():
        results = []
        csv_file = f"{base_filename}_{graph_name}.csv"
        xlsx_file = f"{base_filename}_{graph_name}.xlsx"

        if os.path.exists(csv_file):
            existing = pd.read_csv(csv_file)
            completed = set(zip(existing["Total_Budget"], existing["Split_Ratio"], existing["Timestep"]))
        else:
            completed = set()

        for split in split_ratios:
            for budget in budgets:
                for timestep in timesteps:
                    key = (budget, split, timestep)
                    if key in completed:
                        print(f"⏩ Skipping: {key}")
                        continue

                    print(f"\n🚀 {graph_name} | Budget={budget}, Split={split}, Timestep={timestep}")
                    try:
                        start_time = time.time()
                        result = two_phase_greedy_icm(
                            graph=graph,
                            costs=costs,
                            benefits=benefits,
                            budget=budget,
                            split_ratio=split,
                            timestep=timestep,
                            sims_phase1=sims_phase1,
                            sims_phase2=sims_phase2,
                            processes=processes
                        )
                        elapsed = time.time() - start_time

                        row = {
                            "Model": "Greedy",
                            "Graph_Version": graph_name,
                            "Total_Budget": budget,
                            "Split_Ratio": split,
                            "Timestep": timestep,
                            "Phase1_Budget": int(budget * split),
                            "Phase2_Budget": budget - int(budget * split),
                            "Phase1_Cost": result["Phase1_Cost"],
                            "Phase2_Cost": result["Phase2_Cost"],
                            "Phase1_Remaining_Budget": int(budget * split) - result["Phase1_Cost"],
                            "Phase2_Remaining_Budget": (budget - int(budget * split)) - result["Phase2_Cost"],
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
                            "Execution_Time": round(elapsed, 2)
                        }
                        results.append(row)
                        pd.DataFrame([row]).to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file))
                    except Exception as e:
                        print(f"❌ Error at {key}: {str(e)}")

        if results:
            pd.DataFrame(results).to_excel(xlsx_file, index=False)
            print(f"\n📄 Results saved to {xlsx_file}")

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

    benchmark_two_phase_greedy_icm(
        graphs=graphs,
        costs=costs,
        benefits=benefits,
        budgets=[500, 1000, 1500, 2000, 2500],
        split_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
        timesteps=[2, 4, 6, 8, 10],
        processes=28
    )