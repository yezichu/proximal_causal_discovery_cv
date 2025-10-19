import numpy as np
from sklearn.model_selection import KFold
from utils.kernel_fun import ColumnWiseGaussianKernel, GaussianKernel


class MMR_Testing:
    def __init__(self, lam=None, cv=2, lam_range=(4.9e-6, 0.25), lam_num=50, scale=1.0, func_name = 'sin'):
        self.lam = lam
        self.scale = scale
        self.cv = cv
        if self.cv > 1:
            self.lam = None
            self.lam_candidates = np.logspace(np.log10(
                lam_range[0]), np.log10(lam_range[1]), lam_num)
        else:
            if lam is None:
                raise ValueError("Please provide a value for 'lam' when cv=1.")
            self.lam = lam
            self.lam_candidates = None
        self.func_name = func_name
        
    def choose_kernel(self,input_data):
        """选择适当的核函数"""
        if input_data.shape[1] == 1:
            return GaussianKernel()
        else:
            return ColumnWiseGaussianKernel()
        
    def phi(self, X,):
        """选择适当的权重函数"""
        if self.func_name == 'sin':
            return np.sin(X)
        elif self.func_name == 'cos':
            return np.cos(X)
        elif self.func_name == 'exp':
            return np.exp(X)
        elif self.func_name == 'indentity':
            return X
        else:
            raise ValueError("Unsupported function name. Use 'sin' or 'cos'.")

    def fit(self, input, Y, condition, T):
        self.input_kernel_func = self.choose_kernel(input)
        self.condition_kernel_func = self.choose_kernel(condition)
        self.input_kernel_func.fit(input, scale=self.scale)
        self.condition_kernel_func.fit(condition, scale=self.scale)
        
        W = self.condition_kernel_func.cal_kernel_mat(condition, condition)
        L = self.input_kernel_func.cal_kernel_mat(input, input)
        
        n_train = Y.shape[0]

        if self.cv == 1:
            self._fit_cv_1(input, Y, T, n_train, L, W)
        else:
            self._fit_cv_gt_1(input, condition, Y, T, n_train, L, W)

    def _fit_cv_1(self, input, Y, T, n_train, L, W):
        if self.lam is None:
            raise ValueError("Please provide a value for 'lam' when cv=1.")
        else:
            self.lam = self.lam

        K_lam = L @ W @ L + self.lam * n_train ** 2 * L
        eigvals = np.linalg.eigvalsh(K_lam)  # 对称正定矩阵用 eigvalsh 更快更准
        if min(eigvals) < 1e-6:
            K_lam += 1e-6 * np.eye(K_lam.shape[0])  # 避免除以0

        H = L @ W
        self.alpha = np.empty((n_train, len(T)))
        for i, t in enumerate(T):
            Y_new = self.phi(t * Y)
            self.alpha[:, i] = (np.linalg.solve(K_lam, H @ Y_new)).flatten()

        self.input = input

    def _fit_cv_gt_1(self, input, condition, Y, T, n_train, L, W):
        all_scores = np.full((self.cv, len(self.lam_candidates), len(T)), np.nan)
        for it1, (train, test) in enumerate(KFold(n_splits=self.cv).split(input)):
            input_train, input_test = input[train], input[test]

            W_train = W[np.ix_(train, train)]
            L_train = L[np.ix_(train, train)]

            K_train = L_train @ W_train @ L_train
            H_train = L_train @ W_train

            for lam_idx, lam in enumerate(self.lam_candidates):
                LWL_lambda_L = K_train + lam * len(train) ** 2 * L_train
                for t_idx, t in enumerate(T):
                    Y_new = self.phi(t * Y)
                    alpha = np.linalg.solve(LWL_lambda_L, H_train @ Y_new[train])                    
                    res = Y_new[test] - self.input_kernel_func.cal_kernel_mat(
                        input_test, input_train) @ alpha
                
                    all_scores[it1, lam_idx, t_idx] = (res.T @ W[np.ix_(test, test)] @ res)[0] / (len(test) ** 2)


        avg_scores_per_t = np.mean(all_scores, axis=0)
        best_lam_indices = np.argmin(avg_scores_per_t, axis=0)
        best_lam_per_t = [self.lam_candidates[idx].item()
                          for idx in best_lam_indices]
        K_lam = L @ W @ L
        H = L @ W
        self.alpha = np.empty((n_train, len(T)))

        for i, t in enumerate(T):
            Y_new = self.phi(t * Y)
            self.alpha[:, i] = (np.linalg.solve(K_lam + best_lam_per_t[i] * n_train ** 2 * L,
                                                   H @ Y_new)).flatten()
        self.input = input

    def predict_bridge(self, input):
        test_kernel = self.input_kernel_func.cal_kernel_mat(self.input, input)
        pred = test_kernel.T @ self.alpha
        return pred
    
    def residual(self, input, Y, condition, T):
        input, Y, condition = map(lambda arr: arr.reshape(-1, 1) if arr.ndim == 1 else arr, [input, Y, condition])
        self.fit(input, Y, condition, T)
        pred = self.predict_bridge(input)
        res = self.phi(Y * T) - pred
        return res
