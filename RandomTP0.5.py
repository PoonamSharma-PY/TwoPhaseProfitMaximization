import networkx as nx
import random
import time
import pandas as pd
from tqdm import tqdm
import ast
import multiprocessing
from functools import partial
import os

# Configuration
GRAPH_VERSIONS = {
    'uniform': "euemail_uniform.txt",
    'trivalency': "euemail_trivalency.txt",
    'weighted': "euemail_weighted.txt"
}

def load_graph_version(version, graph_type='directed'):
    """Load one of the pre-generated graph versions with specified type"""
    if version not in GRAPH_VERSIONS:
        raise ValueError(f"Unknown graph version: {version}")
    
    filepath = GRAPH_VERSIONS[version]
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Graph file not found: {filepath}")
    
    # Load based on specified graph type
    if graph_type.lower() == 'directed':
        G = nx.read_weighted_edgelist(filepath, create_using=nx.DiGraph(), nodetype=int)
        print(f"Loaded {version} as directed graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    else:
        G = nx.read_weighted_edgelist(filepath, create_using=nx.Graph(), nodetype=int)
        print(f"Loaded {version} as undirected graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    return G



class TwoPhaseICM:
    def __init__(self, graph, model_type, is_directed, costs, benefits):
        self.graph = graph.copy()
        self.model_type = model_type
        self.is_directed = is_directed
        self.nodes = list(graph.nodes())
        self.costs = costs
        self.benefits = benefits
        self._set_activation_probabilities()

    def _set_activation_probabilities(self):
        if self.model_type == 'uniform':
            for u, v in self.graph.edges():
                self.graph[u][v]['weight'] = 0.1
        elif self.model_type == 'trivalency':
            for u, v in self.graph.edges():
                self.graph[u][v]['weight'] = random.choice([0.1, 0.01, 0.001])
        elif self.model_type == 'weighted':
            # Weighted Cascade model - handles both directed and undirected graphs
            if self.is_directed:
                degrees = dict(self.graph.in_degree())
            else:
                degrees = dict(self.graph.degree())
                
            for u, v in self.graph.edges():
                degree = degrees[v]
                self.graph[u][v]['weight'] = 1.0 / degree if degree != 0 else 0.0

    def phase1_diffusion(self, seed_set, timestep):
        active_nodes = set(seed_set)
        newly_active_nodes = set(seed_set)
        recently_activated = set()

        for t in range(timestep):
            current_newly_active = set()
            for node in newly_active_nodes:
                # Handle both directed and undirected cases
                neighbors = self.graph.successors(node) if self.is_directed else self.graph.neighbors(node)
                for neighbor in neighbors:
                    if neighbor not in active_nodes:
                        prob = self.graph[node][neighbor]['weight']
                        if random.random() < prob:
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
                # Handle both directed and undirected cases
                neighbors = self.graph.successors(node) if self.is_directed else self.graph.neighbors(node)
                for neighbor in neighbors:
                    if neighbor not in active_nodes and neighbor not in already_activated_nodes:
                        prob = self.graph[node][neighbor]['weight']
                        if random.random() < prob:
                            current_newly_active.add(neighbor)
            newly_active_nodes = current_newly_active
            active_nodes.update(newly_active_nodes)

        return active_nodes

    def calculate_profit(self, activated_nodes, seed_set):
        total_benefit = sum(self.benefits.get(node, 0) for node in activated_nodes)
        total_cost = sum(self.costs.get(node, 0) for node in seed_set)
        return total_benefit - total_cost

    def select_seeds_with_budget(self, candidate_nodes, budget):
        selected = set()
        remaining_budget = budget
        candidates = candidate_nodes.copy()
        random.shuffle(candidates)

        for node in candidates:
            node_cost = self.costs.get(node, 0)
            if node_cost <= remaining_budget:
                selected.add(node)
                remaining_budget -= node_cost

        return selected, remaining_budget


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


def save_intermediate_results(results, filename="Random_Two_Phase_Results.xlsx"):
    """Save intermediate results to Excel file."""
    temp_filename = filename.replace('.xlsx', '_temp.xlsx')
    try:
        pd.DataFrame(results).to_excel(temp_filename, index=False)
        if os.path.exists(filename):
            os.remove(filename)
        os.rename(temp_filename, filename)
        print(f"\nIntermediate results saved to {filename}")
    except Exception as e:
        print(f"\nError saving intermediate results: {str(e)}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)



def save_split_ratio_results(results, split_ratio, base_filename="Random_Two_Phase_Results"):
    """Save results for a specific split ratio to Excel file."""
    filename = f"{base_filename}_split_{split_ratio}.xlsx"
    try:
        pd.DataFrame(results).to_excel(filename, index=False)
        print(f"\nResults for split ratio {split_ratio} saved to {filename}")
    except Exception as e:
        print(f"\nError saving results for split ratio {split_ratio}: {str(e)}")


def run_experiment_for_split_ratio(graph_versions, budgets, split_ratio, timesteps, costs, benefits, 
                                 num_simulations):
    """Run complete experiment for a single split ratio"""
    results = []
    cpu_count = multiprocessing.cpu_count()
    
    for version, graph_type in graph_versions.items():
        graph = load_graph_version(version, graph_type)
        is_directed = nx.is_directed(graph)
        icm = TwoPhaseICM(graph, version, is_directed, costs, benefits)
        for budget in tqdm(budgets, desc=f"Budgets (split={split_ratio})"):
            for timestep in tqdm(timesteps, desc="Timesteps", leave=False):
                # icm = TwoPhaseICM(graph, version, is_directed, costs, benefits)
                budget_phase1 = int(budget * split_ratio)
                budget_phase2 = budget - budget_phase1

                start_time = time.time()
                phase1_seeds, remaining_budget = icm.select_seeds_with_budget(icm.nodes, budget_phase1)
                budget_phase2 += remaining_budget

                with multiprocessing.Pool(cpu_count) as pool:
                    phase1_partial = partial(run_phase1_simulation, icm, phase1_seeds, timestep)
                    phase1_simulation_results = pool.map(phase1_partial, range(num_simulations))

                phase1_profits = [res['profit'] for res in phase1_simulation_results]
                avg_phase1_profit = sum(phase1_profits) / num_simulations

                phase2_results = []
                for i, phase1_data in enumerate(phase1_simulation_results):
                    candidate_nodes = [n for n in icm.nodes
                                    if n not in phase1_data['already_activated']
                                    and n not in phase1_seeds]
                    phase2_new_seeds, _ = icm.select_seeds_with_budget(candidate_nodes, budget_phase2)

                    phase2_diffusion_seeds = phase2_new_seeds.union(phase1_data['recently_activated'])
                    num_simulations_second = 5
                    with multiprocessing.Pool(cpu_count) as pool:
                        phase2_partial = partial(run_phase2_simulation, icm, phase2_diffusion_seeds,
                                                phase1_data['already_activated'])
                        
                        phase2_outputs = pool.map(phase2_partial, range(num_simulations_second))

                    total_profit = sum(p[1] for p in phase2_outputs)
                    last_activation = max(phase2_outputs, key=lambda x: len(x[0]))[0]

                    phase2_results.append({
                        'avg_profit': total_profit / num_simulations_second,
                        'phase2_new_seeds': phase2_new_seeds,
                        'phase1_data': phase1_data,
                        'last_activation': last_activation,
                        'phase1_index': i
                    })

                best_result = max(phase2_results, key=lambda x: x['avg_profit'])
                final_seed_set = phase1_seeds.union(best_result['phase2_new_seeds'])
                final_seed_cost = sum(costs.get(n, 0) for n in final_seed_set)
                total_activated = best_result['phase1_data']['already_activated'].union(
                    best_result['last_activation'])

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
                    'Phase2_Activated': len(best_result['last_activation']),
                    'Phase2_Profit_Avg': best_result['avg_profit'],
                    'Final Seed Set': str(final_seed_set),
                    'Final_SeedSetSize': len(final_seed_set),
                    'Final_SeedSetCost': final_seed_cost,
                    'Total_Activated': len(total_activated),
                    'Total_Benefit': sum(benefits.get(n, 0) for n in total_activated),
                    'Total_Profit': avg_phase1_profit + best_result['avg_profit'],
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
    print("The code is started")
    cpu_count = multiprocessing.cpu_count()
    print(f"The CPU count is {cpu_count}")
    
    # Define graph versions and their types
    graph_versions = {
        'uniform': 'directed',
        'trivalency': 'directed',
        'weighted': 'directed'
    }
    
    budgets = [500, 1000, 1500, 2000, 2500]
    # budgets = [500]
    # split_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    split_ratios = [0.5]
    timesteps = [2, 4, 6, 8, 10]
    num_simulations = 100

    print("Running experiments with parallel simulations...")
    
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