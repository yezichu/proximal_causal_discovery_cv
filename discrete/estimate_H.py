import numpy as np

def estimate_H(samples, t_grid, phi_type="sin",return_all=None):
    X, W, Y = samples["X"], samples["W"], samples["Y"]
    n_X, n_W = X.max(), W.max()
    t_grid = np.asarray(t_grid)
    n_t = len(t_grid)
    
    # ----------- Step 1. 构造经验条件概率矩阵 Q_hat ------------
    # n(x) 和 n(x,w)
    n_x = np.bincount(X-1, minlength=n_X)
    n_xw = np.zeros((n_X, n_W), dtype=int)
    np.add.at(n_xw, (X-1, W-1), 1)   # 累加计数
    
    # P(W|X) 矩阵
    with np.errstate(divide='ignore', invalid='ignore'):
        P_w_given_x = n_xw / n_x[:, None]
    Q_hat = P_w_given_x    # shape (n_X, n_W)
    # ----------- Step 2. 构造 φ(Y,t) 并计算 q_hat(x,t) ------------
    # φ(Y,t): shape (n_samples, n_t)
    Y_expand = Y[:, None]  # (n,1)
    if phi_type == "sin":
        Phi = np.sin(t_grid * Y_expand)
    elif phi_type == "cos":
        Phi = np.cos(t_grid * Y_expand)
    else:
        raise ValueError("phi_type must be 'sin' or 'cos'")
    
    # 按 x 分组求均值 => q_hat(x,t)，shape (n_X, n_t)
    # one-hot 编码 (n, n_X)
    one_hot = np.eye(n_X)[X - 1]  
    counts = one_hot.sum(axis=0)  # (n_X,)
    # 矩阵乘法实现分组求和
    q_hat_sum = one_hot.T @ Phi   # (n_X, n_t)
    # 除以计数 -> 分组均值
    q_hat = q_hat_sum / counts[:, None]
    
    # ----------- Step 3. 最小二乘解（一次性解所有 t） ------------
    # H_hat = (Q'Q)^(-1) Q' q_hat ；对所有 t 一起算

    Q_pinv = np.linalg.inv(Q_hat.T @ Q_hat) @ Q_hat.T   # (n_W, n_X)
    # 一次性计算所有 j
    H_hat = (Q_pinv @ q_hat).T   # (n_t, n_W)
    
    if return_all:
    
        return {
            "Q": Q_hat,
            "q_t": q_hat,
        }
    else:
        return H_hat
