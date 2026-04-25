import networkx as nx
import random
import time
import pandas as pd
from tqdm import tqdm
import ast
import multiprocessing
from functools import partial
import os
from collections import defaultdict
from multiprocessing import cpu_count

# Configuration
GRAPH_VERSIONS = {
    'uniform': "euemail_uniform.txt",
    'trivalency': "euemail_trivalency.txt",
    'weighted_cascade': "euemail_weighted.txt"
}

# Set number of CPU cores to use
# NUM_CPUS = multiprocessing.cpu_count()  # Added parentheses to call the function
NUM_CPUS = 20
profit_cache_global = {}

def load_graph_version(version, graph_type='directed'):
    if version not in GRAPH_VERSIONS:
        raise ValueError(f"Unknown graph version: {version}")
    filepath = GRAPH_VERSIONS[version]
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Graph file not found: {filepath}")
    if graph_type.lower() == 'directed':
        G = nx.read_weighted_edgelist(filepath, create_using=nx.DiGraph(), nodetype=int)
        print(f"Loaded {version} as directed graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    else:
        G = nx.read_weighted_edgelist(filepath, create_using=nx.Graph(), nodetype=int)
        print(f"Loaded {version} as undirected graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    return G

class TwoPhaseGreedyICM:
    def __init__(self, graph, model_type, is_directed, costs, benefits):
        self.graph = graph.copy()
        self.model_type = model_type
        self.is_directed = is_directed
        self.nodes = list(graph.nodes())
        self.costs = costs
        self.benefits = benefits
        self._set_activation_probabilities()
        self.adjacency = self._create_adjacency_list()

    def _set_activation_probabilities(self):
        if self.model_type == 'uniform':
            for u, v in self.graph.edges():
                self.graph[u][v]['weight'] = 0.1

    def _create_adjacency_list(self):
        adjacency = {u: [] for u in self.graph.nodes()}
        for u, v in self.graph.edges():
            adjacency[u].append((v, self.graph[u][v]['weight']))
        return adjacency

    def phase1_diffusion(self, seed_set, timestep):
        active_nodes = set(seed_set)
        newly_active_nodes = set(seed_set)
        recently_activated = set()
        for t in range(timestep):
            current_newly_active = set()
            for node in newly_active_nodes:
                for neighbor, prob in self.adjacency.get(node, []):
                    if neighbor not in active_nodes and random.random() < prob:
                        current_newly_active.add(neighbor)
            newly_active_nodes = current_newly_active
            if t == timestep - 1:
                recently_activated = newly_active_nodes.copy()
            active_nodes.update(newly_active_nodes)
        return active_nodes, recently_activated

    def phase2_diffusion(self, seed_set, already_activated_nodes):
        active_nodes = set(seed_set)
        newly_active_nodes = set(seed_set)
        while newly_active_nodes:
            current_newly_active = set()
            for node in newly_active_nodes:
                for neighbor, prob in self.adjacency.get(node, []):
                    if neighbor not in active_nodes and neighbor not in already_activated_nodes:
                        if random.random() < prob:
                            current_newly_active.add(neighbor)
            newly_active_nodes = current_newly_active
            active_nodes.update(newly_active_nodes)
        return active_nodes

    def calculate_profit(self, activated_nodes, seed_set):
        total_benefit = sum(self.benefits.get(node, 0) for node in activated_nodes)
        total_cost = sum(self.costs.get(node, 0) for node in seed_set)
        return total_benefit - total_cost

    def greedy_select_seeds(self, candidate_nodes, budget, num_simulations=5):
        selected = set()
        remaining_budget = budget
        cached_benefit = None
        cached_cost = 0
        with multiprocessing.Pool(NUM_CPUS, initializer=init_worker, initargs=(self.adjacency, self.benefits)) as pool:
            while remaining_budget > 0:
                candidates = [n for n in candidate_nodes if n not in selected and self.costs.get(n, float('inf')) <= remaining_budget]
                if not candidates:
                    break
                if cached_benefit is None:
                    cached_benefit = get_cached_profit(selected, cached_cost, pool, num_simulations)
                current_profit = cached_benefit - cached_cost
                args_list = [(node, selected, self.costs[node], current_profit, cached_cost, num_simulations) for node in candidates]
                results = list(tqdm(pool.imap_unordered(evaluate_candidate, args_list), total=len(args_list), desc="Evaluating candidates", leave=False))
                best_node, best_gain = max(results, key=lambda x: x[1], default=(None, 0))
                if best_node and best_gain > 0 and self.costs[best_node] <= remaining_budget:
                    selected.add(best_node)
                    remaining_budget -= self.costs[best_node]
                    cached_benefit = None
                else:
                    break
        return selected, remaining_budget

def get_cached_profit(seed_set, cost, pool, num_simulations):
    key = (frozenset(seed_set), cost)
    if key in profit_cache_global:
        return profit_cache_global[key] + cost
    total_benefit = sum(pool.map(simulate_single_icm, [seed_set] * num_simulations)) / num_simulations
    profit_cache_global[key] = total_benefit - cost
    return total_benefit

def init_worker(adjacency, benefits):
    global adjacency_global, benefits_global
    adjacency_global = adjacency
    benefits_global = benefits

def simulate_single_icm(seed_set):
    activated = set(seed_set)
    total_benefit = sum(benefits_global.get(node, 0) for node in activated)
    newly_activated = set(activated)
    while newly_activated:
        next_activated = set()
        for node in newly_activated:
            for neighbor, prob in adjacency_global.get(node, []):
                if neighbor not in activated and random.random() < prob:
                    activated.add(neighbor)
                    next_activated.add(neighbor)
                    total_benefit += benefits_global.get(neighbor, 0)
        newly_activated = next_activated
    return total_benefit

def evaluate_candidate(args):
    node, current_seed_set, cost_node, current_profit, total_current_cost, num_simulations = args
    new_seed_set = current_seed_set.union({node})
    key = (frozenset(new_seed_set), total_current_cost + cost_node)
    if key in profit_cache_global:
        profit_with_node = profit_cache_global[key]
    else:
        benefits_sum = sum(simulate_single_icm(new_seed_set) for _ in range(num_simulations))
        avg_benefit = benefits_sum / num_simulations
        profit_with_node = avg_benefit - (total_current_cost + cost_node)
        profit_cache_global[key] = profit_with_node
    marginal_gain = (profit_with_node - current_profit) / cost_node
    return (node, marginal_gain)


# [imports and class definition remain the same — see previous message for full class code]
# Continuing with additional experiment setup:

def run_phase1_simulation(icm, phase1_seeds, timestep, _):
    already_activated, recently_activated = icm.phase1_diffusion(phase1_seeds, timestep)
    profit = icm.calculate_profit(already_activated, phase1_seeds)
    return {
        'already_activated': already_activated,
        'recently_activated': recently_activated,
        'profit': profit
    }

def run_phase2_simulation(icm, diffusion_seeds, already_activated, _):
    activated = icm.phase2_diffusion(diffusion_seeds, already_activated)
    profit = icm.calculate_profit(activated, diffusion_seeds)
    return activated, profit

def save_split_ratio_results(results, split_ratio, base_filename="Greedy_Two_Phase_Results"):
    filename = f"{base_filename}_split_{split_ratio}.xlsx"
    try:
        pd.DataFrame(results).to_excel(filename, index=False)
        print(f"\nResults for split ratio {split_ratio} saved to {filename}")
    except Exception as e:
        print(f"\nError saving results for split ratio {split_ratio}: {str(e)}")

def run_experiment_for_split_ratio(graph_versions, budgets, split_ratio, timesteps, costs, benefits, num_simulations=100):
    results = []
    for version, graph_type in graph_versions.items():
        graph = load_graph_version(version, graph_type)
        is_directed = nx.is_directed(graph)
        for budget in tqdm(budgets, desc=f"Budgets (split={split_ratio})"):
            for timestep in tqdm(timesteps, desc="Timesteps", leave=False):
                icm = TwoPhaseGreedyICM(graph, version, is_directed, costs, benefits)
                budget_phase1 = int(budget * split_ratio)
                budget_phase2 = budget - budget_phase1
                start_time = time.time()

                # Phase 1 seed selection
                phase1_seeds, remaining_budget = icm.greedy_select_seeds(icm.nodes, budget_phase1, num_simulations=5)
                budget_phase2 += remaining_budget

                with multiprocessing.Pool(NUM_CPUS) as pool:
                    phase1_partial = partial(run_phase1_simulation, icm, phase1_seeds, timestep)
                    phase1_simulation_results = list(tqdm(pool.imap_unordered(phase1_partial, range(num_simulations)),
                                                   total=num_simulations, desc="Phase 1 simulations", leave=False))

                avg_phase1_profit = sum(res['profit'] for res in phase1_simulation_results) / num_simulations

                phase2_results = []
                for i, phase1_data in enumerate(phase1_simulation_results):
                    candidate_nodes = [n for n in icm.nodes 
                                       if n not in phase1_data['already_activated'] and n not in phase1_seeds]
                    phase2_new_seeds, _ = icm.greedy_select_seeds(candidate_nodes, budget_phase2, num_simulations=5)
                    phase2_diffusion_seeds = phase2_new_seeds.union(phase1_data['recently_activated'])

                    with multiprocessing.Pool(NUM_CPUS) as pool:
                        phase2_partial = partial(run_phase2_simulation, icm, phase2_diffusion_seeds,
                                                 phase1_data['already_activated'])
                        phase2_outputs = list(tqdm(pool.imap_unordered(phase2_partial, range(5)),
                                                   total=5, desc="Phase 2 simulations", leave=False))

                    avg_phase2_profit = sum(p[1] for p in phase2_outputs) / 5
                    max_activation = max(phase2_outputs, key=lambda x: len(x[0]))[0]
                    phase2_results.append({
                        'phase2_new_seeds': phase2_new_seeds,
                        'phase1_data': phase1_data,
                        'avg_phase2_profit': avg_phase2_profit,
                        'max_activation': max_activation,
                        'phase1_index': i
                    })

                best_result = max(phase2_results, key=lambda x: x['avg_phase2_profit'])

                final_seed_set = phase1_seeds.union(best_result['phase2_new_seeds'])
                total_activated = best_result['phase1_data']['already_activated'].union(best_result['max_activation'])
                total_profit = avg_phase1_profit + best_result['avg_phase2_profit']

                result = {
                    'Graph_Type': 'Directed' if is_directed else 'Undirected',
                    'Model': version,
                    'Total_Budget': budget,
                    'Split_Ratio': split_ratio,
                    'Timestep': timestep,
                    'Phase1_SeedSetSize': len(phase1_seeds),
                    'Phase1_SeedSetCost': sum(costs.get(n, 0) for n in phase1_seeds),
                    'Phase1_Activated_Avg': sum(len(res['already_activated']) for res in phase1_simulation_results) / num_simulations,
                    'Phase1_Profit_Avg': avg_phase1_profit,
                    'Phase2_NewSeedSetSize': len(best_result['phase2_new_seeds']),
                    'Phase2_NewSeedSetCost': sum(costs.get(n, 0) for n in best_result['phase2_new_seeds']),
                    'Phase2_Activated': len(best_result['max_activation']),
                    'Phase2_Profit_Avg': best_result['avg_phase2_profit'],
                    'Final_SeedSet': str(final_seed_set),
                    'Final_SeedSetSize': len(final_seed_set),
                    'Final_SeedSetCost': sum(costs.get(n, 0) for n in final_seed_set),
                    'Total_Activated': len(total_activated),
                    'Total_Profit': total_profit,
                    'Execution_Time': time.time() - start_time,
                    'Best_Phase1_Simulation_Index': best_result['phase1_index'],
                    'Remaining_Budget': budget - sum(costs.get(n, 0) for n in final_seed_set)
                }
                results.append(result)

    return results

def main():
    print("Loading data...")
    with open("cost.txt", "r") as file:
        costs = ast.literal_eval(file.read())
    with open("benefit.txt", "r") as file:
        benefits = ast.literal_eval(file.read())

    print(f"Using {NUM_CPUS} CPU cores for multiprocessing")

    graph_versions = {
        'uniform': 'directed',
        'trivalency': 'directed',
        'weighted_cascade': 'directed'
    }


    budgets = [500, 1000, 1500, 2000, 2500]
    # budgets = [500]
    # split_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    split_ratios = [0.9]
    timesteps = [2, 4, 6, 8, 10]
    # timesteps = [2]
    num_simulations = 100

    print("Running experiments with Greedy algorithm...")
    for split_ratio in split_ratios:
        print(f"\n{'='*50}\nProcessing split ratio: {split_ratio}\n{'='*50}")
        results = run_experiment_for_split_ratio(
            graph_versions, budgets, split_ratio, timesteps, costs, benefits, num_simulations
        )
        save_split_ratio_results(results, split_ratio)
    print("\nAll experiments completed. Results saved by split ratio.")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
