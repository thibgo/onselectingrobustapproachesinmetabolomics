runtime_multiplicator_dict = {
    'PLSDA':        0.3,
    'DT':           1,  
    'SCM':          1,
    'RF':           2.5,
    'rSCM':         2,
    'SVMlinear':    1,
    'SVMrbf':       1,
    'AdaBoost':     4,
    'GBtree':       4,
    'SCMBoost':     4,
    'XGBoost':      2.5,
    'ElasticNet':   1,
}

dataset_bulk_dir = 'data'

results_dir_multivariate =      'results/experiments_multivariate'
results_dir_univariate =        'results/experiments_univariate'

# data splitting
test_size = 0.2

n_jobs = 1
trials_steps = 50
repetition_steps = 100