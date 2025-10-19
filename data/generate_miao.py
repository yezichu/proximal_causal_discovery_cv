import numpy as np

def sample_discrete_variables(n_samples, scenario = 'causal'):

    # Define probability distributions
    # P(X) = [3, 3, 4] / 10
    p_x = np.array([3, 3, 4]) / 10
    
    # P(U|X) - rows correspond to u values, columns to x values
    # P(U|X=x1) = [3, 7]/10, P(U|X=x2) = [6, 4]/10, P(U|X=x3) = [5, 5]/10
    p_u_given_x = np.array([[3, 6, 5],
                            [7, 4, 5]]) / 10
    
    # P(W|U) - rows correspond to w values, columns to u values  
    # P(W|U=u1) = [8, 2]/10, P(W|U=u2) = [3, 7]/10
    p_w_given_u = np.array([[8, 3],
                            [2, 7]]) / 10
    
    # P(Y|U,X) depends on scenario
    if scenario == 'causal':
        # Different P(Y|U,x_i) for each x_i
        p_y_given_u_x1 = np.array([[5, 4],
                                   [3, 2], 
                                   [2, 4]]) / 10
        
        p_y_given_u_x2 = np.array([[4, 6],
                                   [2, 3],
                                   [4, 1]]) / 10
        
        p_y_given_u_x3 = np.array([[3, 2],
                                   [4, 5],
                                   [3, 3]]) / 10
        
        p_y_given_u_x = [p_y_given_u_x1, p_y_given_u_x2, p_y_given_u_x3]
        
    elif scenario == 'independent':
        # Same P(Y|U,x_i) for all x_i
        p_y_given_u_independent = np.array([[5, 4],
                                        [3, 5],
                                        [2, 1]]) / 10
        
        p_y_given_u_x = [p_y_given_u_independent] * 3
    
    else:
        raise ValueError("scenario must be 'original' or 'uniform'")
    
    # Initialize sample arrays
    X_samples = np.zeros(n_samples, dtype=int)
    U_samples = np.zeros(n_samples, dtype=int)
    W_samples = np.zeros(n_samples, dtype=int)
    Y_samples = np.zeros(n_samples, dtype=int)
    
    # Generate samples
    for i in range(n_samples):
        # Sample X from P(X)
        x = np.random.choice(3, p=p_x) + 1
        X_samples[i] = x
        
        # Sample U from P(U|X=x)
        u = np.random.choice(2, p=p_u_given_x[:, x-1]) + 1
        U_samples[i] = u
        
        # Sample W from P(W|U=u)
        w = np.random.choice(2, p=p_w_given_u[:, u-1]) + 1
        W_samples[i] = w
        
        # Sample Y from P(Y|U=u, X=x)
        y = np.random.choice(3, p=p_y_given_u_x[x-1][:, u-1]) + 1
        Y_samples[i] = y
        
    samples = {"U": U_samples, "W": W_samples, "X": X_samples, "Y": Y_samples}

    return samples