from data.generate_obersved import generate_observed
from tqdm import tqdm
from CausalTestBench import run_tests
import json
import logging


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
        if causal_key == 'causal':
            gamma_x = 1
        else:
            gamma_x = 0
        decs_results[causal_key] = {}
        for i in tqdm(range(num_runs), desc=f"{causal_key} runs"):
            i_decs = {}
            for sample_size in sample_sizes:
                # Simulate data
                datas = []
                for _ in range(num_samples_per_run):
                    samples = generate_observed(num_samples=sample_size, p = 1, gamma_x=gamma_x)
                    X, W, Y, V = samples['X'], samples['W'], samples['Y'], samples['V']
                    datas.append((X, W, Y, V))
                    
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
                    log_str = f"Sample size: {sample_size}, causal: {causal_key}"
                else:
                    log_str = f"Sample size: {sample_size}, causal: {causal_key}"

                for method in methods:
                    log_str += f", {method}_sum: {decs_summary[f'{method}_sum']}"

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
    logging.basicConfig(filename='experiment2.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    sample_sizes = [200, 400, 600, 800, 1000, 1200]
    decs_results = run_experiment(
    sample_sizes=sample_sizes,
    num_runs=20,
    num_samples_per_run=100,
    methods=['kernel_proxy_test', 'kernel_test'],
    output_path='decs_results_VII.json',
    use_parallel=True,
    n_jobs=50
)
