import networkx as nx
from data.structual_equations import rand_steq



def simulate_samples(num_samples=1000, type='causal'):
    """
    Simulate samples based on structural equations.

    Args:
        equations: Dictionary of structural equations (output of rand_steq).
        num_samples: Number of samples to generate.

    Returns:
        A dictionary where each key is a node, and the value is a NumPy array of samples.
    """
    # Generate random structural equations
    if type == 'causal':
        sdag = nx.DiGraph()
        # 添加边（U -> W, U -> X, U -> Y, X -> Y）
        sdag.add_edges_from([('U', 'W'), ('U', 'X'), ('U', 'Y'), ('X', 'Y')])
    elif type == 'independent':
        sdag = nx.DiGraph()
        # 添加边（U -> W, U -> X, U -> Y）
        sdag.add_edges_from([('U', 'W'), ('U', 'X'), ('U', 'Y')])
    
    equations = rand_steq(sdag)
    samples = {}
    # Generate samples in topological order
    for node, eq in equations.items():
        if not eq['parents']:  # Head node: No parents, just noise
            samples[node] = eq['dist'](size=num_samples)
        else:  # Node with parents: fV(parents) + noise
            parent_samples = {parent: samples[parent] for parent in eq['parents']}
            parent_effect = sum(eq['fV'](parent_samples[parent]) for parent in eq['parents'])
        
            noise = eq['dist'](size=num_samples)
            samples[node] = parent_effect + noise
    return samples
