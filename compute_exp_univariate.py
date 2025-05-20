
import os
import subprocess

from utils.utils_learning import run_univariate_exp
from config.config_local import dataset_bulk_dir

results_dir = 'results/experiments_univariate'

# silent UndefinedMetricWarning
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)

dataset_bulk_dir = dataset_bulk_dir
print("looking for all datasets in ", dataset_bulk_dir)
# list all datasets to run as the files in MTBLSdatasets
datasets_to_run = os.listdir(dataset_bulk_dir)
datasets_to_run.sort()
datasets_to_run = [x.replace('.pkl', '') for x in datasets_to_run]
#print("-> datasets_to_run=\n", datasets_to_run)

nsplits = 8

if not os.path.exists(os.path.abspath(os.path.join(results_dir, os.pardir))):
    subprocess.run(["mkdir", os.path.abspath(os.path.join(results_dir, os.pardir))])
if not os.path.exists(results_dir):
    subprocess.run(["mkdir", results_dir])

for dataset_name in datasets_to_run:
    pattern = f'univariate_df_{dataset_name}'
    print('pattern=', pattern)
    already_done = os.path.exists(os.path.join(results_dir, f'{pattern}.csv'))
    if not already_done:
        res_df = run_univariate_exp(nsplits, dataset_name, results_dir)
    else:
        print(f'{pattern} already done, skipping')
        