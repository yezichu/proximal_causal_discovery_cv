from data.two_proxies import generate_samples_sin
from tqdm import tqdm
import numpy as np
from CausalTestBench import run_tests,run_two_tests
import json
import logging
from model.MMR_test import MMR_Testing
from joblib import Parallel, delayed
from model.kcm_complex import kcm_complex
from scipy.stats import norm



def kernel_proxy_test(data):
    X,W,Y,Z = data
    quantile_indices = np.linspace(0, 1, 50 + 1)[1:-1]
    T = norm.ppf(quantile_indices)
    RKHS_Re = MMR_Testing(cv=2,lam_num=50,func_name = 'cos') 
    res_re = RKHS_Re.residual(W,Y,X,T)
    RKHS_Im = MMR_Testing(cv=2,lam_num=50,func_name = 'sin')
    res_im = RKHS_Im.residual(W,Y,X,T)
    mh, p, dec1, bvals = kcm_complex(X, res_re,res_im, 2000, 0.05)
    XZ = np.stack([X,Z],axis = 1)
    mh, p, dec2, bvals = kcm_complex(XZ, res_re,res_im, 2000, 0.05)
    return dec1, dec2




def run_experiment(
    sample_sizes,
    num_runs=20,
    num_samples_per_run=100,
    methods=['kernel_proxy_test', 'proxy_test', 'kernel_test'],
    output_path='decs_results.json',
    use_parallel=True,
    n_jobs=50,
    log_time=True):

    decs_results = {}
    for causal in [False, True]:
        causal_key = 'causal' if causal else 'independent'

        decs_results[causal_key] = {}
        for i in tqdm(range(num_runs), desc=f"{causal_key} runs"):
            i_decs = {}
            for sample_size in sample_sizes:
                # Simulate data
                datas = []
                datas_two = []
                for _ in range(num_samples_per_run):
                    samples = generate_samples_sin(gamma_w=1,num_samples=sample_size, type=causal_key)
                    X, W, Z, Y = samples['X'], samples['W'], samples['Z'], samples['Y']
                    datas.append((X, W, Y))
                    datas_two.append((X, W, Y, Z))
                    
                results_dict = {}
                for method in methods:
                    if method == 'kernel_two_proxies_test':
                        results_1, results_2 = run_two_tests(datas_two, method=method, use_parallel=use_parallel, n_jobs=n_jobs)
                        results_dict['kernel_proxies_test'] = results_1
                        results_dict['kernel_two_proxies_test'] = results_2
                    else:
                        results = run_tests(datas, method=method, use_parallel=use_parallel, n_jobs=n_jobs)
                        results_dict[method] = results

                # Summarize results
                decs_summary = {
                    f'{method}_sum': sum(results)
                    for method, results in results_dict.items()
                }
                # Log info
                if log_time:
                    log_str = f"Sample size: {sample_size}, causal: {causal_key}"
                else:
                    log_str = f"Sample size: {sample_size}, causal: {causal_key}"

                for method_name, sum_val in decs_summary.items():
                    log_str += f", {method_name}: {sum_val}"
                logging.info(log_str)

                # Store results
                i_decs[sample_size] = decs_summary

            decs_results[causal_key][i] = i_decs

    # Save results
    with open(output_path, 'w') as json_file:
        json.dump(decs_results, json_file, indent=4)

    return decs_results

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(filename='experiment7.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    sample_sizes = [200, 400, 600, 800, 1000, 1200]
    decs_results = run_experiment(
    sample_sizes=sample_sizes,
    num_runs=20,
    num_samples_per_run=100,
    methods=['kernel_two_proxies_test', 'proxy_test', 'kernel_test',],
    output_path='decs_results_V.json',
    use_parallel=True,
    n_jobs=50
)
