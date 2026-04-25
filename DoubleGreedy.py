import pandas as pd
import networkx as nx
import random
import time
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import os

# Configuration
INPUT_FILES = {
    'cost': "cost.txt",
    'benefit': "benefit.txt"
}
GRAPH_VERSIONS = {
    'uniform': "euemail_uniform.txt",
    'trivalency': "euemail_trivalency.txt",
    'weighted': "euemail_weighted.txt"
}
BUDGETS = [500, 1000, 1500, 2000, 2500]
SELECTION_SIMULATIONS = 100
FINAL_SIMULATIONS = 10000
PROCESSES = 20

def load_graph_version(version, graph_type='directed'):
    path = GRAPH_VERSIONS[version]
    if graph_type.lower() == 'directed':
        G = nx.read_weighted_edgelist(path, create_using=nx.DiGraph(), nodetype=int)
    else:
        G = nx.read_weighted_edgelist(path, create_using=nx.Graph(), nodetype=int)
    print(f"Loaded {version} graph with {len(G.nodes())} nodes")
    return G

def load_data():
    with open(INPUT_FILES['cost'], "r") as f:
        costs = eval(f.read())
    with open(INPUT_FILES['benefit'], "r") as f:
        benefits = eval(f.read())
    return costs, benefits

def independent_cascade_simulation(args):
    G, seeds = args
    active = set(seeds)
    frontier = set(seeds)
    while frontier:
        new_active = set()
        for node in frontier:
            for neighbor in G.neighbors(node):
                if neighbor not in active and random.random() < G[node][neighbor].get('weight', 0.1):
                    new_active.add(neighbor)
        frontier = new_active
        active.update(new_active)
    return active

def run_simulations(G, seeds, num_simulations):
    with mp.Pool(PROCESSES) as pool:
        results = pool.map(independent_cascade_simulation, [(G, seeds)] * num_simulations)
    return results

def calculate_profit(influenced_sets, benefits, cost):
    total_benefits = sum(sum(benefits.get(n, 0) for n in s) for s in influenced_sets)
    avg_benefit = total_benefits / len(influenced_sets)
    return avg_benefit, avg_benefit - cost

def double_greedy(G, budget, costs, benefits):
    nodes = list(G.nodes())
    random.shuffle(nodes)
    X = set()
    Y = set(nodes)
    total_cost = 0

    for node in nodes:
        cost_node = costs.get(node, float('inf'))
        if total_cost + cost_node > budget:
            Y.discard(node)
            continue

        X_add = X | {node}
        profit_X_add = calculate_profit(run_simulations(G, list(X_add), SELECTION_SIMULATIONS), benefits, total_cost + cost_node)[1]
        profit_X = calculate_profit(run_simulations(G, list(X), SELECTION_SIMULATIONS), benefits, total_cost)[1]

        gain = (profit_X_add - profit_X) / cost_node

        Y_rm = Y - {node}
        profit_Y = calculate_profit(run_simulations(G, list(Y), SELECTION_SIMULATIONS), benefits,
                                    sum(costs[n] for n in Y))[1]
        profit_Y_rm = calculate_profit(run_simulations(G, list(Y_rm), SELECTION_SIMULATIONS), benefits,
                                       sum(costs[n] for n in Y_rm))[1]

        loss = (profit_Y - profit_Y_rm) / cost_node

        prob = gain / (gain + loss) if gain + loss > 0 else 0
        if random.random() < prob:
            X.add(node)
            total_cost += cost_node
        else:
            Y.discard(node)
    return list(X), total_cost

def benchmark():
    costs, benefits = load_data()

    for model, path in GRAPH_VERSIONS.items():
        G = load_graph_version(model)
        results = []
        live_csv = f"DoubleGreedy_{model}.csv"
        final_excel = f"DoubleGreedy_{model}.xlsx"

        if os.path.exists(live_csv):
            completed = pd.read_csv(live_csv)[["Budget"]].drop_duplicates()["Budget"].tolist()
        else:
            completed = []

        for budget in BUDGETS:
            if budget in completed:
                print(f"⏩ Skipping {model} budget {budget}")
                continue

            print(f"\n🚀 Running {model} budget {budget}")
            try:
                start_time = time.time()

                sel_start = time.time()
                seeds, cost = double_greedy(G, budget, costs, benefits)
                sel_time = time.time() - sel_start

                sim_start = time.time()
                influenced_sets = run_simulations(G, seeds, FINAL_SIMULATIONS)
                sim_time = time.time() - sim_start

                avg_benefit, profit = calculate_profit(influenced_sets, benefits, cost)
                elapsed = time.time() - start_time

                row = {
                    'Model': model,
                    'Budget': budget,
                    'Seed_Set': str(seeds),
                    'Seed_Size': len(seeds),
                    'Seed_Cost': cost,
                    'Remaining_Budget': budget - cost,
                    'Avg_Benefit': avg_benefit,
                    'Profit': profit,
                    'Selection_Time': sel_time,
                    'Simulation_Time': sim_time,
                    'Total_Time': elapsed,
                    'Selection_Simulations': SELECTION_SIMULATIONS,
                    'Final_Simulations': FINAL_SIMULATIONS
                }

                results.append(row)
                pd.DataFrame([row]).to_csv(live_csv, mode='a', index=False, header=not os.path.exists(live_csv))
                print(f"✅ Completed {model} budget {budget}")
            except Exception as e:
                print(f"❌ Error in {model} budget {budget}: {e}")

        if results:
            pd.DataFrame(results).to_excel(final_excel, index=False)
            print(f"📄 Saved results to {final_excel}")

if __name__ == '__main__':
    mp.freeze_support()
    benchmark()
