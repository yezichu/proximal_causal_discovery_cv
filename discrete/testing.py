import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import chi2


def monte_carlo_test(samples, Sigma, B, d, t_grid):
    m = len(t_grid)
    mean = np.zeros(m*d)    
    Sigma_total_reg = Sigma
    samples = multivariate_normal.rvs(mean=mean, cov=Sigma_total_reg, size=B)
    vectors = samples.reshape(B, m, d)
    norms_sq = np.sum(vectors**2, axis=2)
    S_sim = np.mean(norms_sq, axis=1)
    return S_sim


def chi2_test(Sigma, t_grid, d, T):
    T = np.asarray(T).reshape(-1, order='F')
    p = T.size
    K = len(t_grid)
    assert p == K * d, f"T 长度应为 K*d = {K*d}, 实际 {p}"
    Q_obs = T.T @ ( np.linalg.pinv(Sigma) @ T)
    p_value = chi2.sf(Q_obs.item(),np.linalg.matrix_rank(Sigma))
    return Q_obs, p_value
