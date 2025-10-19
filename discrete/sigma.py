import numpy as np
from data.generate_random_dis import sample_from_probs
from discrete.estimate_H import estimate_H

def compute_Sigma_ttprime(probs, t, t_prime, phi_type="sin"):
    """
    计算任意 t, t' 的 Sigma(t,t')，每个 Σ'_kk 是标量，严格对应公式

    输入:
        probs: dict, 包含 P_U, P_X_given_U, P_W_given_U, P_Y_given_U
        t, t_prime: 标量 t 点
        phi_type: "sin" 或 "cos"

    输出:
        Sigma_ttprime: 投影后的协方差矩阵 n_X x n_X
    """
    P_U = probs["P_U"]
    P_X_given_U = probs["P_X_given_U"]  # (n_X, n_U)
    P_W_given_U = probs["P_W_given_U"]  # (n_W, n_U)
    P_Y_given_U = probs["P_Y_given_U"]  # (n_Y, n_U)

    n_X, n_U = P_X_given_U.shape
    n_W = P_W_given_U.shape[0]
    n_Y = P_Y_given_U.shape[0]

    # --- 1) P(X) 和 P(U|X) ---
    P_X = (P_X_given_U * P_U).sum(axis=1)                  # (n_X,)
    P_U_given_X_all = (P_X_given_U * P_U) / P_X[:, None]  # (n_X, n_U)

    # --- 2) P(W|X) ---
    P_W_given_X = P_W_given_U @ P_U_given_X_all.T         # (n_W, n_X)

    # --- 3) 投影矩阵 ---
    Q = P_W_given_X.T                                      # n_X x n_W
    P_mat = Q @ np.linalg.inv(Q.T @ Q) @ Q.T
    D = np.diag(P_X)
    I = np.eye(n_X)

    # --- 4) phi(Y,t) ---
    Y_vals = np.arange(1, n_Y+1)
    if phi_type == "sin":
        phi_t = np.sin(Y_vals * t)
        phi_t_prime = np.sin(Y_vals * t_prime)
    elif phi_type == "cos":
        phi_t = np.cos(Y_vals * t)
        phi_t_prime = np.cos(Y_vals * t_prime)
    else:
        raise ValueError("phi_type must be 'sin' or 'cos'")

    # --- 5) 条件均值 ---
    P_Y_given_X_all = P_Y_given_U @ P_U_given_X_all.T  # (n_Y, n_X)
    q_t = P_Y_given_X_all.T @ phi_t                   # (n_X,)
    q_t_prime = P_Y_given_X_all.T @ phi_t_prime       # (n_X,)

    # --- 6) 条件协方差块 Σ'_kk (标量) ---
    Sigma_prime_blocks = []
    for x in range(n_X):
        P_Y_given_X = P_Y_given_X_all[:, x]  # (n_Y,)
        cov_scalar = np.sum(P_Y_given_X * phi_t * phi_t_prime) - q_t[x] * q_t_prime[x]
        Sigma_prime_blocks.append(cov_scalar / P_X[x])  # 严格按照公式

    # --- 7) block-diagonal Σ' ---
    Sigma_prime = np.diag(Sigma_prime_blocks)  # n_X x n_X

    # --- 8) 投影 ---
    Sigma_ttprime = D @ (I - P_mat) @ Sigma_prime @ (I - P_mat).T @ D.T

    return Sigma_ttprime


def estimate_Sigma_from_samples(samples, t, t_prime, phi_type="sin"):
    X, W, Y = samples["X"], samples["W"], samples["Y"]
    n_X, n_W = X.max(), W.max()
    n_Y = Y.max()
    N = len(X)

    # ----------- Step 1. 构造经验条件概率矩阵 Q_hat ------------
    n_x = np.bincount(X-1, minlength=n_X)
    n_xw = np.zeros((n_X, n_W), dtype=int)
    np.add.at(n_xw, (X-1, W-1), 1)   # 累加计数

    with np.errstate(divide='ignore', invalid='ignore'):
        P_w_given_x = n_xw / n_x[:, None]
    Q_hat = P_w_given_x
    P_hat = Q_hat @ np.linalg.pinv(Q_hat.T @ Q_hat) @ Q_hat.T

    # 经验 P(X)
    P_X = n_x / N
    D = np.diag(P_X)

    # --- phi ---
    Y_vals = np.arange(1, n_Y+1)
    if phi_type == "sin":
        phi_t = np.sin(Y_vals * t)
        phi_t_prime = np.sin(Y_vals * t_prime)
    elif phi_type == "cos":
        phi_t = np.cos(Y_vals * t)
        phi_t_prime = np.cos(Y_vals * t_prime)
    else:
        raise ValueError("phi_type must be 'sin' or 'cos'")

    # --- 样本条件协方差 Σ'_kk ---
    Sigma_prime_blocks = []
    for k in range(1, n_X+1):
        idx_k = np.where(X == k)[0]
        Nk = len(idx_k)
        P_Xk = Nk / N

        # 样本条件概率 P(Y|X=k)
        Y_k = Y[idx_k]
        P_Y_given_X = np.array([np.sum(Y_k == y)/Nk for y in Y_vals])

        # 条件均值
        q_t = np.sum(P_Y_given_X * phi_t)
        q_t_prime = np.sum(P_Y_given_X * phi_t_prime)

        # 条件协方差
        cov_scalar = np.sum(P_Y_given_X * phi_t * phi_t_prime) - q_t * q_t_prime
        Sigma_prime_blocks.append(cov_scalar / P_Xk)

    Sigma_prime = np.diag(Sigma_prime_blocks)

    # --- 投影得到最终 Sigma ---
    Sigma_ttprime = D @ (np.eye(n_X) - P_hat) @ Sigma_prime @ (np.eye(n_X) - P_hat) @ D

    return Sigma_ttprime, Sigma_prime


def compute_Tn(samples, H_hat, t_grid, phi_type="sin"):
    X = samples["X"]
    W = samples["W"]
    Y = samples["Y"]
    n = len(X)
    n_X = X.max() 
    t_grid = np.asarray(t_grid)

    # --- 1) Phi(Y,t): (n, n_t)
    Y_expand = Y[:, None]  # (n,1)
    if phi_type == "sin":
        Phi = np.sin(t_grid * Y_expand)
    elif phi_type == "cos":
        Phi = np.cos(t_grid * Y_expand)
    else:
        raise ValueError("phi_type must be 'sin' or 'cos'")

    # --- 2) H_sample: H(W_i,t_j) shape (n, n_t)
    H_sample = H_hat.T[W-1, :]   # (n, n_t)
    # --- 3) U_hat
    U_hat = Phi - H_sample      # (n, n_t)
    # --- 4) X one-hot encoding
    X_onehot = np.eye(n_X)[X-1]  # (n, n_X)
    # --- 5) Tn = (1/sqrt(n)) * sum_i U_hat * e(x_i)
    Tn = (X_onehot.T @ U_hat) / np.sqrt(n)  # (n_X, n_t)
    delta_phi = np.mean(np.sum(Tn**2, axis=0))

    return Tn, delta_phi

def estimate_Sigma_experiment(probs, t, t_prime, n_repeats=100, phi_type="sin"):
    """
    多次采样估计经验协方差，并返回理论协方差
    """
    # --- 假设 samples_func(seed) 每次返回 dict {X,W,Y} ---
    Tn_list = []
    for i in range(n_repeats):
        samples = sample_from_probs(probs, n_samples=1000)
        t_grid = [t, t_prime]
        H_hat = estimate_H(samples, t_grid, phi_type=phi_type)
        Tn, _ = compute_Tn(samples, H_hat, t_grid=t_grid, phi_type=phi_type)
        Tn_list.append(Tn)  # n_X x 2

    Tn_all = np.stack(Tn_list, axis=0)  # n_repeats x n_X x 2

    n_X = Tn_all.shape[1]
    Sigma_hat = np.zeros((n_X, n_X))
    for i in range(n_X):
        for j in range(n_X):
            Sigma_hat[i,j] = np.cov(Tn_all[:,i,0], Tn_all[:,j,1], bias=True)[0,1]
    return Sigma_hat

