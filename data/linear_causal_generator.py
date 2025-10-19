import numpy as np

def generate_samples(
    num_samples,
    gamma_x,
    alpha_x=0.5,
    alpha_0=1.0,
    mu_w=0.8,
    mu_0=2.0,
    gamma_u=0.7,
    gamma_0=0.5,
    return_dict=True
):
    """
    Simulate samples from a linear structural causal model:
        X = ε_X
        U = α_x * X + α_0 + ε_U
        W = μ_w * U + μ_0 + ε_W
        Y = γ_u * U + γ_x * X + γ_0 + ε_Y

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    alpha_x, alpha_0, mu_w, mu_0, gamma_u, gamma_x, gamma_0 : float
        Structural coefficients of the causal model.
    return_dict : bool
        If True, returns dict with keys 'X', 'U', 'W', 'Y'; else returns tuple.

    Returns
    -------
    dict or tuple of np.ndarrays
    """
    # Gaussian noise
    epsilon_X = np.random.normal(0, 1, num_samples)
    epsilon_U = np.random.normal(0, 1, num_samples)
    epsilon_W = np.random.normal(0, 1, num_samples)
    epsilon_Y = np.random.normal(0, 1, num_samples)

    # Structural equations
    X = epsilon_X
    U = alpha_x * X + alpha_0 + epsilon_U
    W = mu_w * U + mu_0 + epsilon_W
    Y = gamma_u * U + gamma_x * X + gamma_0 + epsilon_Y

    if return_dict:
        return {'X': X, 'U': U, 'W': W, 'Y': Y}
    else:
        return X, U, W, Y
