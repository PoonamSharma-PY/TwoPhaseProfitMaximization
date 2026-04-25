import pandas as pd
import networkx as nx
import random
import time
import math
import ast
import os
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Configuration
INPUT_FILES = {
    'graph': "euemail.txt",
    'cost': "cost.txt",
    'benefit': "benefit.txt"
}
GRAPH_VERSIONS = {
    'uniform': "euemail_uniform.txt",
    'trivalency': "euemail_trivalency.txt",
    'weighted': "euemail_weighted.txt"
}
BUDGETS = [500, 1000, 1500, 2000, 2500]
# BUDGETS = [1500]

# SELECTED_SPLIT_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]  # Choose your desired split ratios
SELECTED_SPLIT_RATIOS = [0.1]  # Choose your desired split ratios
SELECTED_TIMESTEPS = [2, 4, 6, 8, 10]      # Choose your desired timesteps
# SELECTED_TIMESTEPS = [6]      # Choose your desired timesteps
SELECTION_SIMULATIONS = 5
FINAL_SIMULATIONS = 100
EPSILON = 0.1
NUM_CPUS = 20


def load_graph_version(version, graph_type='directed'):
    if version not in GRAPH_VERSIONS:
        raise ValueError(f"Unknown graph version: {version}")
    filepath = GRAPH_VERSIONS[version]
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Graph file not found: {filepath}")
    if graph_type.lower() == 'directed':
        G = nx.read_weighted_edgelist(filepath, create_using=nx.DiGraph(), nodetype=int)
    else:
        G = nx.read_weighted_edgelist(filepath, create_using=nx.Graph(), nodetype=int)
    return G


def load_data():
    with open(INPUT_FILES['cost'], "r") as f:
        costs = ast.literal_eval(f.read())
    with open(INPUT_FILES['benefit'], "r") as f:
        benefits = ast.literal_eval(f.read())
    return costs, benefits


def run_phase1_diffusion(adjacency, benefits, costs, seed_set, timestep):
    active_nodes = set(seed_set)
    newly_active_nodes = set(seed_set)
    recently_activated = set()
    for t in range(timestep):
        current_newly_active = set()
        for node in newly_active_nodes:
            for neighbor, prob in adjacency.get(node, []):
                if neighbor not in active_nodes and random.random() < prob:
                    current_newly_active.add(neighbor)
        newly_active_nodes = current_newly_active
        if t == timestep - 1:
            recently_activated = newly_active_nodes.copy()
        active_nodes.update(newly_active_nodes)
    total_benefit = sum(benefits.get(n, 0) for n in active_nodes)
    total_cost = sum(costs.get(n, 0) for n in seed_set)
    profit = total_benefit - total_cost
    return {
        'already_activated': active_nodes,
        'recently_activated': recently_activated,
        'profit': profit
    }


def run_phase2_diffusion(adjacency, benefits, costs, seed_set, already_activated, phase1_seeds):
    active_nodes = set(seed_set)
    newly_active_nodes = set(seed_set)
    while newly_active_nodes:
        current_newly_active = set()
        for node in newly_active_nodes:
            for neighbor, prob in adjacency.get(node, []):
                if neighbor not in active_nodes and neighbor not in already_activated and neighbor not in phase1_seeds:
                    if random.random() < prob:
                        current_newly_active.add(neighbor)
        newly_active_nodes = current_newly_active
        active_nodes.update(newly_active_nodes)
    total_benefit = sum(benefits.get(n, 0) for n in active_nodes)
    total_cost = sum(costs.get(n, 0) for n in seed_set)
    profit = total_benefit - total_cost
    return active_nodes, profit


def run_phase1_simulation(args):
    adjacency, benefits, costs, seed_set, timestep, _ = args
    return run_phase1_diffusion(adjacency, benefits, costs, seed_set, timestep)


def run_phase2_simulation(args):
    adjacency, benefits, costs, seed_set, already_activated, phase1_seeds, _ = args
    return run_phase2_diffusion(adjacency, benefits, costs, seed_set, already_activated, phase1_seeds)


class TwoPhaseStochasticGreedy:
    def __init__(self, graph, costs, benefits):
        self.graph = graph
        self.costs = costs
        self.benefits = benefits
        self.nodes = list(graph.nodes())
        self.adjacency = self._create_adjacency()

    def _create_adjacency(self):
        adj = {u: [] for u in self.graph.nodes()}
        for u, v in self.graph.edges():
            adj[u].append((v, self.graph[u][v]['weight']))
        return adj

    # def stochastic_greedy_select_seeds(self, budget, excluded_nodes=None):
    #     if excluded_nodes is None:
    #         excluded_nodes = set()
    #     nodes = [n for n in self.nodes if n not in excluded_nodes]
    #     seed_set = []
    #     total_cost = 0
    #     if not self.costs:
    #         return [], 0
    #     while total_cost < budget:
    #         rem_budget = budget - total_cost
    #         candidates = [
    #             n for n in nodes
    #             if n not in seed_set and self.costs.get(n, float('inf')) <= rem_budget
    #         ]
    #         if not candidates:
    #             break
    #         # Calculate min_cost from *candidates* here:
    #         min_cost = min((self.costs[c] for c in candidates), default=0)
    #         k = int(rem_budget / min_cost) if min_cost > 0 else len(candidates)
    #         sample_size = max(1, math.ceil((len(candidates) * math.log(1 / EPSILON)) / k)) if k > 0 else len(candidates)
    #         sample = random.sample(candidates, min(sample_size, len(candidates)))

    #         # Calculate base_profit for the current seed_set
    #         base_profit = 0
    #         if seed_set:
    #             args_base = [(self.graph, seed_set, total_cost, 0, self.benefits, SELECTION_SIMULATIONS, 0)]  # Pass 0 for base_profit in base case
    #             with mp.Pool(NUM_CPUS) as pool_base:
    #                 results_base = pool_base.map(evaluate_node_normalized, args_base)
    #             base_profit = results_base[0][0]  # Get the profit from the first result

    #         args = [(self.graph, seed_set + [node], total_cost + self.costs.get(node, 0), self.costs.get(node, 0), self.benefits, SELECTION_SIMULATIONS, base_profit) for node in sample] # Pass base_profit
    #         with mp.Pool(NUM_CPUS) as pool:
    #             results = pool.map(evaluate_node_normalized, args)
    #         best_node, best_gain = None, -float('inf')
    #         for node, (profit, norm_gain) in zip(sample, results):
    #             if norm_gain > best_gain and (total_cost + self.costs.get(node, 0)) <= budget:
    #                 best_node, best_gain = node, norm_gain
    #         if best_node is not None and best_gain > 0: # Check if the marginal gain is positive
    #             seed_set.append(best_node)
    #             total_cost += self.costs.get(best_node, 0)
    #         else:
    #             break
    #     return seed_set, total_cost
    def stochastic_greedy_select_seeds(self, budget, excluded_nodes=None):
        if excluded_nodes is None:
            excluded_nodes = set()
        nodes = [n for n in self.nodes if n not in excluded_nodes]
        seed_set = []
        total_cost = 0
        current_profit = 0  # Initialize current profit
        if not self.costs:
            return [], 0

        while total_cost < budget:
            rem_budget = budget - total_cost
            candidates = [
                n for n in nodes
                if n not in seed_set and self.costs.get(n, float('inf')) <= rem_budget
            ]
            if not candidates:
                break

            min_cost = min((self.costs[c] for c in candidates), default=0)
            k = int(rem_budget / min_cost) if min_cost > 0 else len(candidates)
            sample_size = max(1, math.ceil((len(candidates) * math.log(1 / EPSILON)) / k)) if k > 0 else len(candidates)
            sample = random.sample(candidates, min(sample_size, len(candidates)))

            best_node, best_gain = None, -float('inf')

            args = [(self.graph, seed_set + [node], total_cost + self.costs.get(node, 0), self.costs.get(node, 0), self.benefits, SELECTION_SIMULATIONS, current_profit) for node in sample] # Pass current_profit
            with mp.Pool(NUM_CPUS) as pool:
                results = pool.map(evaluate_node_normalized, args)

            for i, (profit, norm_gain) in enumerate(results):
                if norm_gain > best_gain and (total_cost + self.costs.get(sample[i], 0)) <= budget:
                    best_node, best_gain = sample[i], norm_gain
                    potential_profit = profit # Store the potential profit if this node is selected

            if best_node is not None and best_gain > 0:
                seed_set.append(best_node)
                total_cost += self.costs.get(best_node, 0)
                current_profit = potential_profit # Update the current profit
            else:
                break

        return seed_set, total_cost


def evaluate_node_normalized(args):
    graph, seeds, total_cost, node_cost, benefit_dict, num_sim, base_profit = args # Get base_profit
    influenced = []
    for _ in range(num_sim):
        active = set(seeds)
        frontier = set(seeds)
        while frontier:
            new = set()
            for node in frontier:
                for neighbor in graph.neighbors(node):
                    if neighbor not in active and random.random() < graph[node][neighbor]['weight']:
                        new.add(neighbor)
            frontier = new
            active.update(new)
        influenced.append(active)
    new_profit = calc_profit(influenced, benefit_dict, total_cost)
    return (new_profit, (new_profit - base_profit) / node_cost if node_cost > 0 else 0)


def calc_profit(influenced_sets, benefits, cost):
    avg_benefit = sum(sum(benefits.get(n, 0) for n in s) for s in influenced_sets) / len(influenced_sets) if influenced_sets else 0
    return avg_benefit - cost


def save_results_to_excel(results, graph_version, split_ratio, timestep):
    filename = f"Results_{graph_version}_split_{split_ratio}_timestep_{timestep}.xlsx"
    try:
        pd.DataFrame(results).to_excel(filename, index=False)
        print(f"\nResults for graph {graph_version}, split {split_ratio}, timestep {timestep} saved to {filename}")
    except Exception as e:
        print(f"\nError saving results: {str(e)}")


def main():
    print("Loading data...")
    costs, benefits = load_data()

    for graph_version in GRAPH_VERSIONS.keys():
        print(f"\nLoading graph version: {graph_version}")
        graph = load_graph_version(graph_version, graph_type='directed')
        icm = TwoPhaseStochasticGreedy(graph, costs, benefits)
        print(f"Using {NUM_CPUS} CPU cores for graph: {graph_version}")

        for split_ratio in SELECTED_SPLIT_RATIOS:
            print(f"\nProcessing for split ratio: {split_ratio}\n" + "="*50)
            for timestep in SELECTED_TIMESTEPS:
                print(f"\n  Processing for timestep: {timestep}\n" + "-"*40)
                all_results = []
                for budget in BUDGETS:
                    iter_start = time.time()
                    budget1 = int(budget * split_ratio)
                    seeds1, cost1 = icm.stochastic_greedy_select_seeds(budget1)
                    budget2 = budget - cost1
                    args_list_phase1 = [(icm.adjacency, icm.benefits, icm.costs, seeds1, timestep, _) for _ in range(FINAL_SIMULATIONS)]
                    with mp.Pool(NUM_CPUS) as pool:
                        phase1_results = list(tqdm(pool.imap_unordered(run_phase1_simulation, args_list_phase1), total=FINAL_SIMULATIONS, desc=f"Phase 1 (Budget {budget})"))

                    best = None
                    best_profit2 = -float('inf')
                    eligible_nodes_for_phase2 = 0  # Initialize counter
                    phase2_progress = tqdm(phase1_results, desc=f"Phase 2 (Budget {budget})", leave=False)
                    for r in phase2_progress:
                        excluded_nodes_phase2 = r['already_activated'].union(set(seeds1))
                        eligible_nodes_for_phase2 = len([n for n in icm.nodes if n not in excluded_nodes_phase2]) # Calculate eligible nodes
                        print("Eligible nodes for phase 2 in this iteration", eligible_nodes_for_phase2)
                        seeds2, cost2_iter = icm.stochastic_greedy_select_seeds(budget2, excluded_nodes=excluded_nodes_phase2)
                        seed_union = set(seeds2).union(r['recently_activated'])
                        args2 = [(icm.adjacency, icm.benefits, icm.costs, seed_union, r['already_activated'], set(seeds1), _) for _ in range(5)] # Reduced simulations for selection
                        with mp.Pool(NUM_CPUS) as pool:
                            outputs2 = list(pool.imap_unordered(run_phase2_simulation, args2))
                        profits2 = [o[1] for o in outputs2]
                        avg_profit2 = sum(profits2) / len(outputs2) if outputs2 else 0

                        if avg_profit2 > best_profit2:
                            best_profit2 = avg_profit2
                            best = (r, seeds2, seed_union, outputs2, cost2_iter)

                    if best:
                        final_activated = best[0]['already_activated'].union(max(best[3], key=lambda x: len(x[0]))[0]) if best[3] else best[0]['already_activated']
                        total_profit = best[0]['profit'] + best_profit2 # Using the profit from the best Phase 1 outcome
                        total_cost = cost1 + best[4]
                        phase2_seed_set_size = len(best[1])
                        phase2_seed_set_cost = best[4]
                    else:
                        final_activated = phase1_results[0]['already_activated'] if phase1_results else set(seeds1)
                        total_profit = sum(r['profit'] for r in phase1_results) / len(phase1_results) if phase1_results else 0
                        total_cost = cost1
                        phase2_seed_set_size = 0
                        phase2_seed_set_cost = 0

                    remaining_budget = budget - total_cost
                    exec_time = time.time() - iter_start

                    all_results.append({
                        'Graph_Version': graph_version,
                        'Split_Ratio': split_ratio,
                        'Budget': budget,
                        'Timestep': timestep,
                        'Phase1_SeedSetSize': len(seeds1),
                        'Phase1_SeedSetCost': cost1,
                        'Phase1_Profit_Avg': sum(r['profit'] for r in phase1_results) / len(phase1_results) if phase1_results else 0,
                        'Phase2_SeedSetSize': phase2_seed_set_size,
                        'Phase2_SeedSetCost': phase2_seed_set_cost,
                        'Phase2_Profit_Avg': best_profit2,
                        'Final_SeedSetSize': len(seeds1) + phase2_seed_set_size,
                        'Final_SeedSetCost': total_cost,
                        'Total_Activated': len(final_activated),
                        'Total_Profit': total_profit,
                        'Remaining_Budget': remaining_budget,
                        'Execution_Time(s)': exec_time,
                        'Eligible_Nodes_Phase2': eligible_nodes_for_phase2  # Add eligible nodes count
                    })
                save_results_to_excel(all_results, graph_version, split_ratio, timestep)

    print("\nAll experiments completed. Results saved to Excel files.")

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()
