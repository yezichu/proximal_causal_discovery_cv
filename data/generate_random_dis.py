import numpy as np

def generate_samples_dag_with_probs(
    n_samples=1000, n_U=3, n_W=4, n_X=6, n_Y=3,
    edge_X_to_Y=False, seed=None, max_tries=1000
):
    rng = np.random.default_rng(seed)

    if n_U > n_W:
        raise ValueError(f"要求 n_U <= n_W, 但 n_U={n_U}, n_W={n_W}")
    if not (n_X > n_W):
        raise ValueError(f"要求 n_X > n_W, 但 n_X={n_X}, n_W={n_W}")

    # 1) P(U)
    P_U = rng.random(n_U)
    P_U /= P_U.sum()

    # 2) P(W|U) 满列秩
    tries = 0
    while True:
        tries += 1
        P_W_given_U = rng.random((n_W, n_U))
        P_W_given_U /= P_W_given_U.sum(axis=0, keepdims=True)
        cond = np.linalg.cond(P_W_given_U)
        if np.linalg.matrix_rank(P_W_given_U) == n_U and cond < 1e3:
            break
        if tries >= max_tries:
            raise RuntimeError(f"无法在 {max_tries} 次尝试内生成满列秩的 P(W|U)")
        

    # 3) P(X|U)
    P_X_given_U = rng.random((n_X, n_U))
    P_X_given_U /= P_X_given_U.sum(axis=0, keepdims=True)

    # 4) P(Y|U) 或 P(Y|U,X)
    if edge_X_to_Y:
        P_Y_given_UX = rng.random((n_Y, n_U, n_X))
        P_Y_given_UX /= P_Y_given_UX.sum(axis=0, keepdims=True)
        P_Y_given_U = None
    else:
        P_Y_given_U = rng.random((n_Y, n_U))
        P_Y_given_U /= P_Y_given_U.sum(axis=0, keepdims=True)
        P_Y_given_UX = None

    # ----- 采样 -----
    U = rng.choice(n_U, size=n_samples, p=P_U) + 1
    W = np.empty(n_samples, dtype=int)
    X = np.empty(n_samples, dtype=int)
    Y = np.empty(n_samples, dtype=int)
    
    if edge_X_to_Y:
        for i, u in enumerate(U):
            W[i] = rng.choice(n_W, p=P_W_given_U[:, u-1]) + 1
            X[i] = rng.choice(n_X, p=P_X_given_U[:, u-1]) + 1
            Y[i] = rng.choice(n_Y, p=P_Y_given_UX[:, u-1, X[i]-1]) + 1
    else:
        for i, u in enumerate(U):
            W[i] = rng.choice(n_W, p=P_W_given_U[:, u-1]) + 1
            X[i] = rng.choice(n_X, p=P_X_given_U[:, u-1]) + 1
            Y[i] = rng.choice(n_Y, p=P_Y_given_U[:, u-1]) + 1

    samples = {"U": U, "W": W, "X": X, "Y": Y}
    probs = {
        "P_U": P_U,
        "P_W_given_U": P_W_given_U,
        "P_X_given_U": P_X_given_U,
        "P_Y_given_U": P_Y_given_U,
        "P_Y_given_UX": P_Y_given_UX
    }

    return samples, probs

import numpy as np

def sample_from_probs(probs, n_samples=1000, seed=None):
    """
    根据给定的 probs 采样 U, W, X, Y
    probs: dict 包含
        P_U: (n_U,)
        P_W_given_U: (n_W, n_U)
        P_X_given_U: (n_X, n_U)
        P_Y_given_U: (n_Y, n_U) 或 None
        P_Y_given_UX: (n_Y, n_U, n_X) 或 None
    n_samples: 样本量
    seed: 随机种子
    """
    rng = np.random.default_rng(seed)

    P_U = probs["P_U"]
    P_W_given_U = probs["P_W_given_U"]
    P_X_given_U = probs["P_X_given_U"]
    P_Y_given_U = probs["P_Y_given_U"]
    P_Y_given_UX = probs["P_Y_given_UX"]

    n_U = P_U.size
    n_W = P_W_given_U.shape[0]
    n_X = P_X_given_U.shape[0]
    n_Y = P_Y_given_U.shape[0] if P_Y_given_U is not None else P_Y_given_UX.shape[0]

    # --- 采样 U ---
    U = rng.choice(n_U, size=n_samples, p=P_U) + 1  # +1 保持从1开始

    # --- 初始化 W, X, Y ---
    W = np.empty(n_samples, dtype=int)
    X = np.empty(n_samples, dtype=int)
    Y = np.empty(n_samples, dtype=int)

    if P_Y_given_UX is not None:
        # Y|U,X
        for i, u in enumerate(U):
            W[i] = rng.choice(n_W, p=P_W_given_U[:, u-1]) + 1
            X[i] = rng.choice(n_X, p=P_X_given_U[:, u-1]) + 1
            Y[i] = rng.choice(n_Y, p=P_Y_given_UX[:, u-1, X[i]-1]) + 1
    else:
        # Y|U
        for i, u in enumerate(U):
            W[i] = rng.choice(n_W, p=P_W_given_U[:, u-1]) + 1
            X[i] = rng.choice(n_X, p=P_X_given_U[:, u-1]) + 1
            Y[i] = rng.choice(n_Y, p=P_Y_given_U[:, u-1]) + 1

    samples = {"U": U, "W": W, "X": X, "Y": Y}
    return samples
