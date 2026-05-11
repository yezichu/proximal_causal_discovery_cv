import numpy as np
from data.generate_miao import sample_discrete_variables
from discrete.estimate_H import estimate_H
from discrete.sigma import estimate_Sigma_from_samples,compute_Tn
from discrete.testing import monte_carlo_test,chi2_test
from discrete.liu.proxy import ProxyTest
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


def single_experiment(samples,Sigma,delta_phi,t_grid,
                      bvals=2000, alpha=0.05):
    n_X = samples["X"].max()
    delta_b = monte_carlo_test(samples, Sigma,B=bvals, d=n_X, t_grid=t_grid)
    p_value = np.mean(delta_b >= delta_phi)
    dec = int(p_value < alpha)    
    return dec

def single_experiment_prime(samples,Sigma, Tn,t_grid,
                        alpha=0.05):
    n_X = int(samples["X"].max())
    p_value = chi2_test(Sigma, t_grid, d=n_X, T=Tn)[1]
    dec = int(p_value < alpha)  
    return dec

def single_experiment_proxy(samples,alpha=0.05):
    n_X = samples["X"].max()
    n_W = samples["W"].max()
    n_Y = samples["Y"].max()
    bindf = pd.DataFrame(samples)
    tester = ProxyTest(bindf,int(n_X),[int(n_W)],int(n_Y))
    pproxy = tester.test()
    dec = int(pproxy < alpha)
    return dec

def run_three_methods(samples, n_t=100, bvals=2000,phi_type='sin', alpha=0.05):
    t_grid = np.random.randn(n_t)
    n_samples = samples["X"].shape[0]
    n_X = int(samples["X"].max())
    Sigma = np.zeros((n_t*n_X, n_t*n_X))
    for i in range(n_t):
        for j in range(n_t):
            block, Sigma_prime = estimate_Sigma_from_samples(samples, t_grid[i], t_grid[j])  # d x d
            Sigma[i*n_X:(i+1)*n_X, j*n_X:(j+1)*n_X] = block
    H_hat = estimate_H(samples, t_grid, phi_type=phi_type)  # (n_t, n_W)
    Tn, delta_phi = compute_Tn(samples, H_hat, t_grid, phi_type=phi_type)
    
    dec1 = single_experiment(
        samples=samples,Sigma=Sigma, delta_phi=delta_phi,t_grid=t_grid, 
        bvals=bvals, alpha=alpha
    )
    dec2 = single_experiment_prime(
        samples=samples, Sigma=Sigma, Tn=Tn,t_grid=t_grid,
         alpha=alpha
    )
    # 方法2（proxy）
    dec3 = single_experiment_proxy(
        samples=samples,alpha=alpha
    )
    return {
        "n_samples": n_samples,
        "dec_method1": dec1,
        "dec_method2": dec2,
        "dec_method3": dec3,
    }
    

def run_power_experiments(sample_sizes=[400,600,800,1000,1200],
                          n_experiments=20, n_sim=100, n_jobs=50,
                          scenario="causal",
                          n_t=100, bvals=2000, phi_type='sin', alpha=0.05):
    all_results = []

    for n_samples in sample_sizes:
        # 每个样本量下重复 n_experiments 次实验
        for exp_id in range(n_experiments):
            # 内层：用 n_sim 次随机 seed 估计 power
            sim_results = Parallel(n_jobs=n_jobs)(
                delayed(run_three_methods)(
                    samples= sample_discrete_variables(n_samples=n_samples, scenario=scenario),  # 每次重复内部也可以重新采样
                    n_t=n_t, bvals=bvals, 
                    phi_type=phi_type, alpha=alpha
                )
                for i in np.arange(n_sim, dtype=int)
            )
            df_sim = pd.DataFrame(sim_results)

            # 单次实验 power = 平均拒绝率
            power_method1 = df_sim["dec_method1"].mean()
            power_method2 = df_sim["dec_method2"].mean()
            power_method3 = df_sim["dec_method3"].mean()

            all_results.append({
                "n_samples": n_samples,
                "experiment_id": exp_id,
                "power_method1": power_method1,
                "power_method2": power_method2,
                "power_method3": power_method3,
            })

    final_df = pd.DataFrame(all_results)
    return final_df


if __name__ == "__main__":
    results_df = run_power_experiments(n_experiments=20,scenario='independent')
    results_df.to_csv('independent_miao.csv', index=False)
    results_df = run_power_experiments(n_experiments=20,scenario='causal')
    results_df.to_csv('causal_miao.csv', index=False)
   