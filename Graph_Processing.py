import networkx as nx
import random

# Configuration
INPUT_FILE = "lesmiserables.gml"  # Original Les Misérables file (GML)
OUTPUT_FILES = {
    'uniform': "lesmis_uniform.txt",
    'trivalency': "lesmis_trivalency.txt",
    'weighted': "lesmis_weighted.txt"
}

def load_and_convert_lesmis():
    """Load the Les Misérables graph and relabel nodes with integers."""
    G = nx.read_gml(INPUT_FILE)
    print(f"Original graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Map node names to integers
    mapping = {node: idx for idx, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    print("Nodes relabeled to integers.")

    # Ensure graph is undirected
    if nx.is_directed(G):
        G = G.to_undirected()
    return G

def create_graph_versions(G):
    """Create 3 versions of the graph with different probability assignments"""
    # Version 1: Uniform 0.1 probability
    uniform_G = G.copy()
    for u, v in uniform_G.edges():
        uniform_G[u][v]['weight'] = 0.1

    # Version 2: Trivalency (random choice from [0.1, 0.01, 0.001])
    trivalency_G = G.copy()
    trivalency_values = [0.1, 0.01, 0.001]
    for u, v in trivalency_G.edges():
        trivalency_G[u][v]['weight'] = random.choice(trivalency_values)

    # Version 3: Weighted probability (1 / degree of target)
    weighted_G = G.copy()
    for u, v in weighted_G.edges():
        degree = weighted_G.degree(v)
        weighted_G[u][v]['weight'] = 1.0 / degree if degree > 0 else 0
    return uniform_G, trivalency_G, weighted_G

def save_graph(G, filename):
    """Save graph in edge list format with weights"""
    with open(filename, 'w') as f:
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1.0)
            f.write(f"{u} {v} {weight:.6f}\n")

def main():
    G = load_and_convert_lesmis()
    uniform_G, trivalency_G, weighted_G = create_graph_versions(G)

    save_graph(uniform_G, OUTPUT_FILES['uniform'])
    save_graph(trivalency_G, OUTPUT_FILES['trivalency'])
    save_graph(weighted_G, OUTPUT_FILES['weighted'])

    print("✅ Les Misérables graph converted and saved in all formats.")

    # Quick sanity check
    sample = next(iter(weighted_G.edges(data=True)), None)
    if sample:
        u, v, data = sample
        print(f"Sample weighted edge: {u} - {v}, weight = {data['weight']:.4f}")
    else:
        print("Graph has no edges.")

if __name__ == "__main__":
    main()
