import networkx as nx
import random
import ast

# Step 1: Load the GML graph and relabel nodes to integers
graph = nx.read_gml("lesmiserables.gml")
graph = nx.relabel_nodes(graph, {node: idx for idx, node in enumerate(graph.nodes())})

# Step 2: Assign cost and benefit
cost = {node: random.randint(50, 100) for node in graph.nodes()}
benefit = {node: random.randint(800, 1000) for node in graph.nodes()}

# Step 3: Save to files
with open('cost.txt', 'w') as file:
    file.write(str(cost))

with open('benefit.txt', 'w') as file:
    file.write(str(benefit))

# Step 4: Read and reload the dictionaries
with open('cost.txt', 'r') as file:
    cost = ast.literal_eval(file.read())

with open('benefit.txt', 'r') as file:
    benefit = ast.literal_eval(file.read())

# Step 5: Sanity checks
print(f"Number of nodes: {len(graph)}")
print(f"Sum of cost values: {sum(cost.values())}")
print(f"Sum of benefit values: {sum(benefit.values())}")
