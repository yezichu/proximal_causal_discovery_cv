import numpy as np
import pandas as pd
from scipy.stats import norm
from model.kcm_complex import kcm_complex
from model.kcm import kcm
from discrete.liu.binn import quantile_bin
from discrete.liu.proxy import ProxyTest
from model.MMR_test import MMR_Testing
from KCI.KCI import KCI_CInd
from joblib import Parallel, delayed

class CausalTester:
    def __init__(self, data):
        if len(data) == 3:
            self.X, self.W, self.Y = data
            self.V = None
        elif len(data) == 4:
            self.X, self.W, self.Y, self.V = data
        self.data = data  
        self.df = pd.DataFrame({'X': self.X, 'W': self.W, 'Y': self.Y})

    def kernel_proxy_test(self,lam=None,cv=2):
        quantile_indices = np.linspace(0, 1, 50 + 1)[1:-1]
        T = norm.ppf(quantile_indices)
        rng = np.random.default_rng(42)  # 创建随机数生成器，种子为 42
        T = rng.standard_normal(50) 
        
        RKHS_Re = MMR_Testing(lam=lam, cv=cv, lam_num=50, func_name='cos')
        RKHS_Im = MMR_Testing(lam=lam, cv=cv, lam_num=50, func_name='sin')
        if self.V is None:
            res_re = RKHS_Re.residual(self.W, self.Y, self.X, T)
            res_im = RKHS_Im.residual(self.W, self.Y, self.X, T)
            mh, p, dec, bvals = kcm_complex(self.X, res_re, res_im, 2000, 0.05)
        else:
            input = np.column_stack((self.W, self.V))
            condition = np.column_stack((self.X, self.V))
            res_re = RKHS_Re.residual(input, self.Y, condition, T)
            res_im = RKHS_Im.residual(input, self.Y, condition, T)
            mh, p, dec, bvals = kcm_complex(condition, res_re, res_im, 2000, 0.05)
        return dec,p
    
    def moment_proxy_test(self):
        RKHS = MMR_Testing(cv=2, lam_num=50, func_name='indentity')
        if self.V is None:
            res = RKHS.residual(self.W, self.Y, self.X, np.array([1]))
            mh, p, dec, bvals = kcm(self.X, res, 2000, 0.05)
        else:
            input = np.column_stack((self.W, self.V))
            condition = np.column_stack((self.X, self.V))
            res = RKHS.residual(input, self.Y, input, np.array([1]))
            mh, p, dec, bvals = kcm(condition, res, 2000, 0.05)
        return dec,p
    
    
    def discretize(self, levelx, levelu, levely):
        tx = quantile_bin(self.df.X, levelx)
        tw = quantile_bin(self.df.W, levelu)
        ty = quantile_bin(self.df.Y, levely)
        bindata = np.stack([tx, tw, ty], axis=1)
        return bindata

    def proxy_test(self, levelx=14, levelu=12, levely=4):
        bindata = self.discretize(levelx, levelu, levely)
        bindf = pd.DataFrame(bindata, columns=['X', 'W', 'Y'])
        tester = ProxyTest(bindf, levelx=levelx, levelws=[levelu], levely=levely)
        pproxy = tester.test()
        return int(pproxy < 0.05),pproxy

    def kernel_test(self):
        kci = KCI_CInd()
        if self.V is None:
            pvalue, _ = kci.compute_pvalue(self.X.reshape(-1, 1),
                                        self.Y.reshape(-1, 1),
                                        self.W.reshape(-1, 1))
        else:
            condition = np.column_stack((self.W, self.V))
            pvalue, _ = kci.compute_pvalue(self.X.reshape(-1, 1),
                                        self.Y.reshape(-1, 1),
                                        condition)
        return int(pvalue < 0.05),pvalue
    
    
class TwoProxiesTester:
    def __init__(self, data):
        if len(data) == 4:
            self.X, self.W, self.Y, self.Z = data
            self.V = None
        elif len(data) == 5:
            self.X, self.W, self.Y, self.Z, self.V = data
        self.data = data  
        
    def kernel_two_proxies_test(self,lam=None,cv=2):
        quantile_indices = np.linspace(0, 1, 50 + 1)[1:-1]
        T = norm.ppf(quantile_indices)
        RKHS_Re = MMR_Testing(lam=lam, cv=cv, lam_num=50, func_name='cos')
        RKHS_Im = MMR_Testing(lam=lam, cv=cv, lam_num=50, func_name='sin')
        if self.V is None:
            res_re = RKHS_Re.residual(self.W, self.Y, self.X, T)
            res_im = RKHS_Im.residual(self.W, self.Y, self.X, T)
            _, _, dec1, _ = kcm_complex(self.X, res_re,res_im, 2000, 0.05)
            XZ = np.stack([self.X,self.Z],axis = 1)
            _, _, dec2, _ = kcm_complex(XZ, res_re,res_im, 2000, 0.05)
        else:
            input = np.column_stack((self.W, self.V))
            condition = np.column_stack((self.X, self.V))
            XZV = np.column_stack((self.X, self.Z, self.V))
            print("input shape:", input.shape, "condition shape:", condition.shape, "XZV shape:", XZV.shape)
            res_re = RKHS_Re.residual(input, self.Y, condition, T)
            res_im = RKHS_Im.residual(input, self.Y, condition, T)
            
            _, p1, dec1, _ = kcm_complex(condition, res_re,res_im, 2000, 0.05)
            _, p2, dec2, _ = kcm_complex(XZV, res_re, res_im, 2000, 0.05)
        return dec1, dec2, p1, p2
    


def run_tests(datas, method='kernel_proxy_test', use_parallel=True, n_jobs=50):
    """
    datas: list of (X,W,Y) or (X,W,Y,V) tuples
    method: one of 'kernel_proxy_test', 'proxy_test', 'proxy_test_exp', 'kernel_test'
    use_parallel: whether to run in parallel
    """
    
    def run_one(data):
        tester = CausalTester(data)
        if method == 'kernel_proxy_test':
            dec,_ = tester.kernel_proxy_test()
            return dec
        elif method == 'moment_proxy_test':
            dec,_ = tester.moment_proxy_test()
            return dec
        elif method == 'proxy_test':
            dec, _ = tester.proxy_test()
            return dec
        elif method == 'kernel_test':
            dec, _ = tester.kernel_test()
            return dec
        else:
            raise ValueError(f"Unknown method '{method}'")

    if use_parallel:
        results = Parallel(n_jobs=n_jobs)(delayed(run_one)(data) for data in datas)
    else:
        results = [run_one(data) for data in datas]

    return results


def run_two_tests(datas, method='kernel_two_proxies_test', use_parallel=True, n_jobs=50):
    """
    datas: list of (X,W,Y) or (X,W,Y,V) tuples
    method: one of 'kernel_proxy_test', 'proxy_test', 'proxy_test_exp', 'kernel_test'
    use_parallel: whether to run in parallel
    """
    
    def run_one(data):
        tester = TwoProxiesTester(data)
        if method == 'kernel_two_proxies_test':
            dec1, dec2, p1, p2 = tester.kernel_two_proxies_test()
            return (dec1,dec2)
    
        else:
            raise ValueError(f"Unknown method '{method}'")

    if use_parallel:
        results = Parallel(n_jobs=n_jobs)(delayed(run_one)(data) for data in datas)
    else:
        results = [run_one(data) for data in datas]
    dec1_list = [dec1 for dec1, _ in results]
    dec2_list = [dec2 for _, dec2 in results]
    return dec1_list, dec2_list


