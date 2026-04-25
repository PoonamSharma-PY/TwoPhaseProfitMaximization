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
from datetime import datetime

# Configuration
GRAPH_VERSIONS = {
    'uniform': "euemail_uniform.txt",
    'trivalency': "euemail_trivalency.txt",
    'weighted_cascade': "euemail_weighted.txt"
}

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

class TwoPhaseDoubleGreedyICM:
    def __init__(self, graph, model_type, is_directed, costs, benefits):
        self.graph = graph.copy()
        self.model_type = model_type
        self.is_directed = is_directed
        self.nodes = list(graph.nodes())
        self.costs = costs
        self.benefits = benefits
        self.profit_cache = {}
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

    def double_greedy_select_seeds(self, candidate_nodes, budget, num_simulations=100):
        nodes = [n for n in candidate_nodes if self.costs.get(n, float('inf')) <= budget]
        random.shuffle(nodes)
        X = set()
        Y = set(nodes)
        total_cost = 0
        for node in nodes:
            if total_cost >= budget:
                break
            current_cost = self.costs[node]
            if total_cost + current_cost > budget:
                Y.remove(node)
                continue
            X_add = X.copy(); X_add.add(node)
            gain = self._calculate_marginal_gain(X, X_add, total_cost, current_cost, num_simulations)
            Y_remove = Y.copy(); Y_remove.discard(node)
            loss = self._calculate_marginal_loss(Y, Y_remove, num_simulations)
            prob = gain / (gain + loss) if (gain + loss) > 0 else 0
            if random.random() < prob:
                X.add(node); total_cost += current_cost
            else:
                Y.remove(node)
        return X, budget - total_cost

    def _calculate_marginal_gain(self, current_set, new_set, current_cost, node_cost, num_simulations):
        key1 = (frozenset(current_set), current_cost)
        key2 = (frozenset(new_set), current_cost + node_cost)
        if key1 not in self.profit_cache:
            self.profit_cache[key1] = sum(self._simulate_profit((current_set, current_cost)) for _ in range(num_simulations)) / num_simulations
        if key2 not in self.profit_cache:
            self.profit_cache[key2] = sum(self._simulate_profit((new_set, current_cost + node_cost)) for _ in range(num_simulations)) / num_simulations
        return (self.profit_cache[key2] - self.profit_cache[key1]) / node_cost

    def _calculate_marginal_loss(self, current_set, new_set, num_simulations):
        cost_current = sum(self.costs.get(n, 0) for n in current_set)
        cost_new = sum(self.costs.get(n, 0) for n in new_set)
        key1 = (frozenset(current_set), cost_current)
        key2 = (frozenset(new_set), cost_new)
        if key1 not in self.profit_cache:
            self.profit_cache[key1] = sum(self._simulate_profit((current_set, cost_current)) for _ in range(num_simulations)) / num_simulations
        if key2 not in self.profit_cache:
            self.profit_cache[key2] = sum(self._simulate_profit((new_set, cost_new)) for _ in range(num_simulations)) / num_simulations
        return (self.profit_cache[key1] - self.profit_cache[key2]) / (cost_current - cost_new)

    def _simulate_profit(self, args):
        seed_set, cost = args
        activated = set(seed_set)
        newly_activated = set(seed_set)
        while newly_activated:
            current_newly_active = set()
            for node in newly_activated:
                for neighbor, prob in self.adjacency.get(node, []):
                    if neighbor not in activated and random.random() < prob:
                        activated.add(neighbor)
                        current_newly_active.add(neighbor)
            newly_activated = current_newly_active
        total_benefit = sum(self.benefits.get(node, 0) for node in activated)
        return total_benefit - cost

def run_phase1_simulation(sim_id, icm=None, phase1_seeds=None, timestep=None):
    already_activated, recently_activated = icm.phase1_diffusion(phase1_seeds, timestep)
    profit = icm.calculate_profit(already_activated, phase1_seeds)
    return {
        'already_activated': already_activated,
        'recently_activated': recently_activated,
        'profit': profit
    }

def run_phase2_simulation(sim_id, icm=None, diffusion_seeds=None, already_activated=None):
    activated = icm.phase2_diffusion(diffusion_seeds, already_activated)
    profit = icm.calculate_profit(activated, diffusion_seeds)
    return activated, profit

def save_split_ratio_results(results, split_ratio, base_filename="DoubleGreedy_Two_Phase_Results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_filename}_split_{split_ratio}_{timestamp}.xlsx"
    pd.DataFrame(results).to_excel(filename, index=False)

def run_experiment_for_split_ratio(graph_versions, budgets, split_ratio, timesteps, costs, benefits, selection_simulations, final_simulations):
    results = []
    cpu_count = multiprocessing.cpu_count()
    for version, graph_type in graph_versions.items():
        graph = load_graph_version(version, graph_type)
        is_directed = nx.is_directed(graph)
        icm = TwoPhaseDoubleGreedyICM(graph, version, is_directed, costs, benefits)

        with multiprocessing.Pool(cpu_count) as pool:
            for timestep in timesteps:
                for budget in tqdm(budgets, desc=f"Split {split_ratio}, Time {timestep}, Graph {version}"):
                    budget_phase1 = int(budget * split_ratio)
                    budget_phase2 = budget - budget_phase1
                    start_time = time.time()

                    phase1_seeds, remaining_budget = icm.double_greedy_select_seeds(icm.nodes, budget_phase1, selection_simulations)
                    budget_phase2 += remaining_budget

                    phase1_partial = partial(run_phase1_simulation, icm=icm, phase1_seeds=phase1_seeds, timestep=timestep)
                    phase1_simulation_results = list(pool.imap(phase1_partial, range(final_simulations)))

                    phase1_profits = [res['profit'] for res in phase1_simulation_results]
                    avg_phase1_profit = sum(phase1_profits) / final_simulations

                    phase2_results = []
                    final_simulations_second = 5

                    for i, phase1_data in enumerate(phase1_simulation_results):
                        candidate_nodes = [n for n in icm.nodes if n not in phase1_data['already_activated'] and n not in phase1_seeds]
                        phase2_new_seeds, _ = icm.double_greedy_select_seeds(candidate_nodes, budget_phase2, selection_simulations)
                        phase2_diffusion_seeds = phase2_new_seeds.union(phase1_data['recently_activated'])

                        phase2_partial = partial(run_phase2_simulation, icm=icm, diffusion_seeds=phase2_diffusion_seeds, already_activated=phase1_data['already_activated'])
                        phase2_outputs = list(pool.imap(phase2_partial, range(final_simulations_second)))

                        phase2_profits = [p[1] for p in phase2_outputs]
                        avg_phase2_profit = sum(phase2_profits) / final_simulations_second
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
                        'Phase1_Activated_Avg': sum(len(res['already_activated']) for res in phase1_simulation_results) / final_simulations,
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
    with open("cost.txt", "r") as file:
        costs = ast.literal_eval(file.read())
    with open("benefit.txt", "r") as file:
        benefits = ast.literal_eval(file.read())

    graph_versions = {
        'uniform': 'directed',
        'trivalency': 'directed',
        'weighted_cascade': 'directed'
    }

    budgets = [500, 1000, 1500, 2000, 2500]
    # split_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    timesteps = [2, 4, 6, 8, 10]
    # budgets = [500]
    split_ratios = [0.5]
    # timesteps = [2]

    selection_simulations = 5
    final_simulations = 100

    for split_ratio in tqdm(split_ratios, desc="Overall Progress: Split Ratios"):
        for version, _ in graph_versions.items():
            for timestep in timesteps:
                results = run_experiment_for_split_ratio(
                    {version: 'directed'}, budgets, split_ratio, [timestep], costs, benefits,
                    selection_simulations, final_simulations
                )
                save_split_ratio_results(results, split_ratio, base_filename=f"DoubleGreedy_Two_Phase_ResultsTry_{version}_time_{timestep}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()