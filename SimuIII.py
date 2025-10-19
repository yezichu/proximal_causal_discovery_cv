from data.two_proxies import generate_samples,generate_samples_nonlinear
from tqdm import tqdm
from CausalTestBench import run_tests
import json
import logging


def run_experiment(
    sample_sizes,
    num_samples_per_run=100,
    methods=['kernel_proxy_test', 'proxy_test', 'kernel_test'],
    output_path='decs_results.json',
    use_parallel=True,
    n_jobs=50,
    log_time=True):

    decs_results = {}
    gamma_ws = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    for i,gamma_w in tqdm(enumerate(gamma_ws)):
        i_decs = {}
        for sample_size in sample_sizes:
            # Simulate data
            datas = []
            for _ in range(num_samples_per_run):
                samples = generate_samples(gamma_w=gamma_w,num_samples=sample_size, type='causal')
                X, W, Y = samples['X'], samples['W'], samples['Y']
                datas.append((X, W, Y))
                
            results_dict = {}
            for method in methods:
                results = run_tests(datas, method=method, use_parallel=use_parallel, n_jobs=n_jobs)
                results_dict[method] = results

            # Summarize results
            decs_summary = {
                f'{method}_sum': sum(results)
                for method, results in results_dict.items()
            }
            # Log info
            if log_time:
                log_str = f"Sample size: {sample_size}"
            else:
                log_str = f"Sample size: {sample_size}"

            for method in methods:
                log_str += f", {method}_sum: {decs_summary[f'{method}_sum']}"

            logging.info(log_str)

            # Store results
            i_decs[sample_size] = decs_summary

        decs_results[i] = i_decs

    # Save results
    with open(output_path, 'w') as json_file:
        json.dump(decs_results, json_file, indent=4)

    return decs_results

def run_experiment_nonlinear(
    sample_sizes,
    num_samples_per_run=100,
    methods=['kernel_proxy_test', 'proxy_test', 'kernel_test'],
    output_path='decs_results.json',
    use_parallel=True,
    n_jobs=50,
    log_time=True):

    decs_results = {}
    gamma_ws = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8]
    for i,gamma_w in tqdm(enumerate(gamma_ws)):
        i_decs = {}
        for sample_size in sample_sizes:
            # Simulate data
            datas = []
            for _ in range(num_samples_per_run):
                samples = generate_samples_nonlinear(gamma_w=gamma_w,num_samples=sample_size)
                X, W, Y = samples['X'], samples['W'], samples['Y']
                datas.append((X, W, Y))
                
            results_dict = {}
            for method in methods:
                results = run_tests(datas, method=method, use_parallel=use_parallel, n_jobs=n_jobs)
                results_dict[method] = results

            # Summarize results
            decs_summary = {
                f'{method}_sum': sum(results)
                for method, results in results_dict.items()
            }
            # Log info
            if log_time:
                log_str = f"Sample size: {sample_size}"
            else:
                log_str = f"Sample size: {sample_size}"

            for method in methods:
                log_str += f", {method}_sum: {decs_summary[f'{method}_sum']}"

            logging.info(log_str)

            # Store results
            i_decs[sample_size] = decs_summary

        decs_results[i] = i_decs

    # Save results
    with open(output_path, 'w') as json_file:
        json.dump(decs_results, json_file, indent=4)

    return decs_results

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(filename='experiment7.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    sample_sizes = [800]
#     decs_results = run_experiment(
#     sample_sizes=sample_sizes,
#     num_samples_per_run=100,
#     methods=['kernel_proxy_test', 'proxy_test'],
#     output_path='decs_results_III.json',
#     use_parallel=True,
#     n_jobs=50
# )
    decs_results = run_experiment_nonlinear(
    sample_sizes=sample_sizes,
    num_samples_per_run=100,
    methods=['kernel_proxy_test', 'proxy_test'],
    output_path='decs_results_III_nonlinear.json',
    use_parallel=True,
    n_jobs=50
)
