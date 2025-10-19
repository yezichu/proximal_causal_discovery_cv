import numpy as np

def generate_observed(
    num_samples,
    p,
    gamma_x,
    type='causal', return_dict=True
):
    # Gaussian noise
    U = np.random.normal(0, 1, num_samples)
    V = np.random.normal(0, 1, (num_samples,p))
    
    epsilon_X = np.random.normal(0, 1, num_samples)
    epsilon_W = np.random.normal(0, 1, num_samples)
    epsilon_Y = np.random.normal(0, 1, num_samples)

    # Structural equations

    X = 0.5 + U + 0.3 * U**2 + V @ (0.5 * np.ones(p)) + epsilon_X   # 修正
    W = 1 + U + V @ (0.5 * np.ones(p) ) + epsilon_W
    if type == 'causal':
        Y = -1 + U + 0.4 * U**2 + V @ np.ones(p) + gamma_x * X + epsilon_Y
    else:
        Y = -1 + U + 0.4 * U**2 + V @ np.ones(p) + epsilon_Y
    
    # U = np.random.randn(num_samples)
    # V = np.random.randn(num_samples, p)
    # epsilons = np.random.randn(num_samples, 4)
    # eps1, eps2, eps3, eps4 = epsilons.T


    # # 构造变量
    # X = 0.5 + U + 0.1 * U**2 + V @ (0.5 * np.ones(p) )+ eps1
    
    # W = 1 + U + V @ (0.5 * np.ones(p) ) + eps4
    # if type == 'causal':
    #     Y = -1 + U + 0.2 * U**2 + V @ np.ones(p) + gamma_x * X + eps2
    # else:
    #     Y = -1 + U + 0.2 * U**2 + V @ np.ones(p) + + eps2
        

    if return_dict:
        return {'X': X, 'U': U, 'W': W, 'V': V, 'Y': Y}
    else:
        return X, U, W, V, Y
