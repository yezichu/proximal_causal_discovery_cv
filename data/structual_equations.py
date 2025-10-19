import numpy as np
from functools import partial



def distribution(name, std=1, beta=0.5, k=2, theta=0.3):  # std=1, beta=0.5, k=2, theta=0.3
    # Returns a random variable generator based on the specified distribution
    if name == 'gaussian':
        funct = partial(np.random.normal, loc=0, scale=std)
    elif name == 'uniform':
        funct = partial(np.random.uniform, low=-std, high=std)
    elif name == 'exponential':
        funct = lambda size=None: np.random.choice([-1, 1], size=size)*np.random.exponential(scale=beta, size=size)
    elif name == 'gamma':
        funct = lambda size=None: np.random.choice([-1, 1], size=size) * np.random.gamma(shape=k, scale=theta, size=size) 
    else:
        raise ValueError("Supported distributions are: gaussian, uniform, exponential, gamma")
    return funct



def function(name):
    # Returns a function based on the specified name
    if name == 'linear':
        return lambda x: x  
    elif name == 'tanh':
        return lambda x: np.tanh(x) 
    elif name == 'sin':
        return lambda x: np.sin(np.pi/2*x)
    elif name == 'sqrt':
        return lambda x: np.sign(x)*np.sqrt(abs(x)/2)
    elif name == 'sigmoid':
        return lambda x: 1.0/(1.0+np.exp(-x))
    else:
        raise ValueError("Supported functions are: linear, tanh, sin, sigmoid, sqrt")
    

#  dists=['gaussian', 'uniform', 'exponential', 'gamma']
# funcs=['linear', 'tanh', 'sin', 'exp']
def rand_steq(sdag, funcs=['linear', 'tanh', 'sin', 'sqrt'], dists=['gaussian', 'uniform', 'exponential', 'gamma']):
    """
    Generate random structural equations for a given DAG.
    
    Args:
        sdag: A DAG where sdag.edges gives parent-child relationships.
        funcs: List of functions to choose from for fV.
        dists: List of distributions to choose from for NV.

    Returns:
        A dictionary mapping each variable to its structural equation components.
        Each entry is structured as:
            {
                'parents': [list of parent nodes],
                'fV': <function> or None for head nodes,
                'dist': <distribution generator>
            }
    """
    equations = {}
    
    for node in sdag.nodes:
        # Get parent nodes (PAV)
        parents = [p for p, c in sdag.edges if c == node]

        # Randomly choose a noise distribution NV
        dist = distribution(np.random.choice(dists))

        if not parents:  # 如果是头节点
            equations[node] = {
                'parents': [],
                'fV': None,  # 头节点无函数，直接由噪声生成
                'dist': dist
            }
        else:  # 如果有父节点
            # Randomly choose a function fV
            fV = function(np.random.choice(funcs))
            equations[node] = {
                'parents': parents,
                'fV': fV,
                'dist': dist
            }

    return equations