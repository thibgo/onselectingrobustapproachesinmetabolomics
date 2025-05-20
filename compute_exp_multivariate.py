import os
import sys
import subprocess
import pandas as pd
import pickle as pkl
from joblib import Parallel, delayed

from utils.utils_learning import run_exp
from config.config_parameters_grids import parameters_grid
from config.config_local import n_jobs, trials_steps, repetition_steps, dataset_bulk_dir

results_dir = 'results/experiments_multivariate'

print('##########################################################################')

algos_to_run = []
algos_to_run = algos_to_run + ['DT']
algos_to_run = algos_to_run + ['SCM']
algos_to_run = algos_to_run + ['RF']
algos_to_run = algos_to_run + ['rSCM']
algos_to_run = algos_to_run + ['PLSDA']
algos_to_run = algos_to_run + ['ElasticNet']
algos_to_run = algos_to_run + ['SVMlinear']
algos_to_run = algos_to_run + ['SVMrbf']
algos_to_run = algos_to_run + ['AdaBoost']
algos_to_run = algos_to_run + ['GBtree']
algos_to_run = algos_to_run + ['SCMBoost'] 
algos_to_run = algos_to_run + ['XGBoost']

##metric_name = 'f1score'
metric_name = 'balancedaccuracy'
##metric_name = 'mcc'

dataset_bulk_dir = dataset_bulk_dir
print("looking for all datasets in ", dataset_bulk_dir)
# list all datasets to run as the files in MTBLSdatasets
datasets_to_run = os.listdir(dataset_bulk_dir)
datasets_to_run = [x.replace('.pkl', '') for x in datasets_to_run]
datasets_to_run.sort(reverse=True)
print("-> datasets_to_run=\n", datasets_to_run)

full_repetitions_range = list(range(8)) # nb of splits to run
bayesian_optimization_trials = 50

if '-dataset' in sys.argv:
    dataset_name = sys.argv[sys.argv.index('-dataset') + 1]
    datasets_to_run = [dataset_name]
if '-algo' in sys.argv:
    algo = sys.argv[sys.argv.index('-algo') + 1]
    algos_to_run = [algo]

if not os.path.exists(os.path.abspath(os.path.join(results_dir, os.pardir))):
    subprocess.run(["mkdir", os.path.abspath(os.path.join(results_dir, os.pardir))])
if not os.path.exists(results_dir):
    subprocess.run(["mkdir", results_dir])

# experiments settings where the algorithm was not able to build a model, we will skip them
patterns_to_skip = []
with open('config/list_of_experiments_to_skip.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        patterns_to_skip.append(line.strip())
# to try them again, comment the above 4 lines

for bo_trials_to_do in range(trials_steps, bayesian_optimization_trials + trials_steps, trials_steps):
    bo_trials_to_do = min(bo_trials_to_do, bayesian_optimization_trials)
    for repetition_step in range(repetition_steps, len(full_repetitions_range) + repetition_steps, repetition_steps):
        repetitions_range = full_repetitions_range[:min(repetition_step, len(full_repetitions_range))]
        print('repetitions_range', repetitions_range)
        for dataset_name in datasets_to_run:
            for algo in algos_to_run:
                pattern = f'metric_{metric_name}_{dataset_name}_{algo}'
                print('pattern=', pattern)
                if pattern in patterns_to_skip:
                    print('    skipping', pattern)
                    continue
                exp_done_split_trials_dict = {}
                exp_snapshot_dict = {i: None for i in repetitions_range}
                for file in os.listdir(results_dir):
                    if pattern in file:
                        split_id = file.split('_')[-1].split('.')[0]
                        n_trials = file.split('trials')[1].split('_')[1]
                        try:
                            exp_done_split_trials_dict[int(split_id)] = int(n_trials)
                        except:
                            print('    error with', file)
                            print(file.split('_')[-1].split('.')[0])
                            print(file.split('trials')[1].split('_')[1])
                            print('split_id', split_id, 'n_trials', n_trials)
                        exp_snapshot_dict[int(split_id)] = file
                print('trials step=', bo_trials_to_do)
                print('dataset=', dataset_name)
                print('algo=', algo)
                done_repetition_range = [k for k, v in exp_done_split_trials_dict.items() if v == bo_trials_to_do]
                to_be_done_repetition_range = [r for r in repetitions_range if r not in done_repetition_range]
                print('already done repetitions_range=', done_repetition_range)
                print('expected repetitions_range=', repetitions_range)
                print('to_be_done_repetition_range=', to_be_done_repetition_range)
                loc_param_grid = parameters_grid[algo].copy()
                if len(to_be_done_repetition_range) > 0:
                    Parallel(n_jobs=n_jobs, verbose=0)(delayed(run_exp)
                        (algo=algo,
                        dataset_name=dataset_name,
                        metric=metric_name,
                        split_id=split_id, 
                        bayesian_optimization_trials=bo_trials_to_do,
                        param_grid=loc_param_grid, 
                        results_dir=results_dir,
                        exp_snapshot_to_start=exp_snapshot_dict[split_id],
                        ) for split_id in to_be_done_repetition_range)
