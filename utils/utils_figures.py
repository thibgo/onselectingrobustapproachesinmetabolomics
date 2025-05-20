import os
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, balanced_accuracy_score

from utils.utils_learning import locate_dataset
from config.config_local import results_dir_multivariate, results_dir_univariate

tendency_to_col_dict = {
    'multivariate models performs better than univariate models':   "#1E88E5",
    'multivariate and univariate models are equivalent':            "#278F17",
    'univariate models performs better than multivariate models':   "#FFC107",
    'no models performs better than the baseline':                  "#AFAFAF",
}

structure_to_col_dict = {
    'rule-based multivariate':   '#2ed1ff',
    'linear-based multivariate': '#e817ff',
    'univariate':                '#cccccc',
    'baseline':                  '#cccccc',
}

# https://davidmathlogic.com/colorblind/#%237B24FF-%23658BFF-%2365DCFF-%2319A900-%2354BD24-%2382FF48-%23585858-%23A0A0A0-%23E65959-%23FF8080-%23BD4A4A-%2367E62C-%23F3D636
organism_to_col_dict = {
    'Homo sapiens':             '#7B24FF',
    'Mus musculus':             '#658BFF',
    'multiple':                 '#585858',
    'Plants':                   '#19A900',
    'Bacteria':                 '#DC4E4E',
    'other mammals':            '#55C7F9',
    'other or unknown':         '#cccccc',
}

technique_to_col_dict = {
    #'LCMS' :    '#66c2a5',
    #'GCMS' :    '#ffd92f',
    #'other' :   '#cccccc',
    #'NMR' :     '#fc8d62',
    #'DIMS' :    '#8da0cb',
    'LCMS' :    '#56B4E9',
    'GCMS' :    '#0072B2',
    'other' :   '#cccccc',
    'NMR' :     '#E69F00',
    'DIMS' :    '#CC79A7',
}

keyword_to_col_dict = { 
    'cancer':            '#CA054D', 
    'cultivar':          '#4EB300', 
    'location':          '#0000ff', 
    'disease':           '#ff00ff', 
    'mutation':          '#FFD700', 
    'treatment':         '#EF8300', 
    'diet':              '#06FBB1',
    'age':               '#72C9FD',
    'birth':             '#7810B7',
    'season':            '#1071B7',
    'other':             '#cccccc',
}


algo_display_short_name_dict = {
    'dummy': 'Dummy',
    'ElasticNet': 'Lin',
    'PLSDA': 'PLSDA',
    'SVMlinear': 'SVMl',
    'SVMrbf': 'SVMr',
    'AdaBoost': 'AdaB',
    'SCMBoost': 'SCMB',
    'GBtree': 'GBtree',
    'XGBoost': 'XGB',
    'DT': 'DT',
    'RF': 'RF',
    'SCM': 'SCM',
    'rSCM': 'rSCM'
    }

algo_to_col_dict = {
    'ElasticNet':      '#f346ff',
    'PLSDA':            '#f00fff',
    'SVMlinear':        '#e11eff',
    'SVMrbf':           '#d22dff',
    'AdaBoost':         '#1ee1ff',
    'SCMBoost':         '#2dd2ff',
    'GBtree':           '#3cc3ff',
    'XGBoost':          '#4bb4ff',
    'DT':               '#5aa5ff',
    'RF':               '#6996ff',
    'SCM':              '#7887ff',
    'rSCM':             '#8778ff',
}

year_to_col_dict = {
    '2012': '#eb9e76',
    '2013': '#e98d6b',
    '2014': '#e77b62',
    '2015': '#e3685c',
    '2016': '#dc575c',
    '2017': '#d14a61',
    '2018': '#c14168',
    '2019': '#b13c6c',
    '2020': '#a0376f',
    '2021': '#8f3371',
    '2022': '#7d2e70',
    '2023': '#6c2b6d',
    '2024': '#5b2867',
    '?'   : '#cccccc',
}

#eb9e76
#e98d6b
#e77b62
#e3685c
#dc575c
#d14a61
#c14168
#b13c6c
#a0376f
#8f3371
#7d2e70
#6c2b6d
#5b2867

def get_experiments_results_big_df(result_dir, metric, nb_of_splits):
    df_results_list = []
    bayesian_optimization_trials = 50

    #list files in directory:
    for file in os.listdir(result_dir):
        if 'perf_df' in file and f'metric_{metric}' and f'trials_{bayesian_optimization_trials}' in file:
            try:
                df_loc = pd.read_csv(os.path.join(result_dir, file), dtype={'dataset': str, 'metric': str, 'score': np.float64})
                # keep only metric == metric
            except:
                print('error reading file', file)
                continue
            df_loc = df_loc[df_loc['metric'] == metric]
            if df_loc.shape[0] == 0:
                print('no metric', metric, 'in file', file)
                continue
            df_results_list.append(df_loc)
    big_perf_df = pd.concat(df_results_list)
    print(big_perf_df.shape)

    # reindex
    big_perf_df.reset_index(inplace=True)
    del big_perf_df['index']
    print(big_perf_df.index)

    # remove all split values high than n_splits
    big_perf_df = big_perf_df[big_perf_df['split'] < nb_of_splits]

    # check the number of splits for each experiment
    for dataset in big_perf_df['dataset'].unique():
        for algo in big_perf_df['algo'].unique():
            splits_done = big_perf_df[(big_perf_df["dataset"] == dataset) & (big_perf_df["algo"] == algo)]["split"].unique()
            if set(splits_done) != set(range(nb_of_splits)):
                print(f'{dataset} {algo} {big_perf_df[(big_perf_df["dataset"] == dataset) & (big_perf_df["algo"] == algo)]["split"].unique()}')
                # remove from the dataframe
                big_perf_df = big_perf_df[~((big_perf_df["dataset"] == dataset) & (big_perf_df["algo"] == algo))]
    
    # drop fit time since there are errors
    big_perf_df = big_perf_df[big_perf_df['metric'] != 't2-t1']

    # turn into numeric
    big_perf_df['score'] = pd.to_numeric(big_perf_df['score'])

    return big_perf_df

def get_best_univariate_dict(datasets_names, metric):
    best_univariate_dict = {}
    for dataset_name in datasets_names:
        univariate_file = 'univariate_df_{}.csv'.format(dataset_name)
        try:
            # load also univariate if it exists
            res_df_univariate = pd.read_csv(os.path.join(results_dir_univariate, univariate_file), index_col=0, nrows=1)
        except:
            print('error reading file', univariate_file)
            continue
        res_df_univariate = pd.read_csv(os.path.join(results_dir_univariate, univariate_file), header=0, usecols=[metric], dtype=np.float64)
        best_univariate_score = res_df_univariate[metric].max()
        best_univariate_dict[dataset_name] = best_univariate_score
    return best_univariate_dict

def get_baseline_score(dataset_name, metric):
    data_path, dataset_filename = locate_dataset(dataset_name)
    with open(os.path.join(data_path, dataset_filename), 'rb') as fo:
        dataset = pkl.load(fo)
    y = dataset['y']
    majoritary_class_in_y = np.argmax(np.bincount(y))
    #print(f"majoritary_class_in_y: {majoritary_class_in_y}", y)
    if metric == 'f1_score':
        return f1_score(y, [majoritary_class_in_y] * len(y))
    elif metric == 'accuracy':
        return accuracy_score(y, [majoritary_class_in_y] * len(y))
    elif metric == 'mcc':
        return matthews_corrcoef(y, [majoritary_class_in_y] * len(y))
    elif metric == 'balancedaccuracy':
        return balanced_accuracy_score(y, [majoritary_class_in_y] * len(y))
    else:
        raise ValueError('metric not recognized', metric)
        
def get_labels_counts(dataset_name):
    data_path, dataset_filename = locate_dataset(dataset_name)
    with open(os.path.join(data_path, dataset_filename), 'rb') as fo:
        dataset = pkl.load(fo)
    y = dataset['y']
    n_zeros = len(y) - np.sum(y)
    n_ones = np.sum(y)
    assert n_zeros + n_ones == len(y)
    return n_zeros, n_ones

def define_tendency(best_algo_score, best_univariate_score, baseline_majoritary_score, best_algo_name):
    if max(best_algo_score, best_univariate_score) < baseline_majoritary_score + 0.05 :
        tendency = 'no models performs better than the baseline'
    elif abs(best_algo_score - best_univariate_score) < 0.02:
        tendency = 'multivariate and univariate models are equivalent'
    elif best_algo_score > best_univariate_score:
        tendency = 'multivariate models performs better than univariate models'
    elif best_algo_score < best_univariate_score:
        tendency = 'univariate models performs better than multivariate models'
    else:
        raise ValueError('case not treated', best_algo_score, best_univariate_score)
    return tendency

def generate_meta_df(big_perf_df, best_univariate_dict, datasets_df, metric):
    if 'optimethod' in big_perf_df.columns:
        assert len(big_perf_df['optimethod'].unique()) == 1
        big_perf_df = big_perf_df.drop(columns=['optimethod'])
    meta_df = datasets_df.copy()
    meta_df['baseline majoritary score'] = '?'
    meta_df['best univariate score'] = '?'
    meta_df['best method score'] = '?'
    meta_df['best algo name'] = '?'
    meta_df['best algo score'] = '?'
    meta_df['tendency'] = '?'
    meta_df['number of algos runned'] = '?'
    
    for dataset_name in list(big_perf_df['dataset'].unique()):
        #print('dataset_name', dataset_name)
        #print('best_univariate', best_univariate_dict[dataset_name])
        meta_df.loc[dataset_name, 'best univariate score'] = best_univariate_dict[dataset_name]
        # get the best algo and its accuracy
        loc_df = big_perf_df[big_perf_df['dataset'] == dataset_name]
        loc_df = loc_df[loc_df['type'] == 'test']
        loc_df = loc_df.drop(columns=['metric', 'type', 'dataset'])
        #std_df = loc_df.groupby(['algo']).std()
        mean_df = loc_df.groupby(['algo']).mean()
        mean_df['algo'] = mean_df.index
        mean_df = mean_df.sort_values(by='score', ascending=False)
        best_algo_score = mean_df['score'].iloc[0]
        best_algo_name = mean_df['algo'].iloc[0]
        meta_df.loc[dataset_name, 'best algo name'] = best_algo_name
        meta_df.loc[dataset_name, 'best algo score'] = best_algo_score
        baseline_score = get_baseline_score(dataset_name, metric)
        meta_df.loc[dataset_name, 'baseline majoritary score'] = baseline_score
        meta_df.loc[dataset_name, 'best method score'] = max(best_algo_score, best_univariate_dict[dataset_name])
        tendency = define_tendency(meta_df.loc[dataset_name, 'best algo score'], meta_df.loc[dataset_name, 'best univariate score'], meta_df.loc[dataset_name, 'baseline majoritary score'], meta_df.loc[dataset_name, 'best algo name'])
        meta_df.loc[dataset_name, 'tendency'] = tendency
        #
        n_zeros, n_ones = get_labels_counts(dataset_name)
        meta_df.loc[dataset_name, 'n zeros in y'] = n_zeros
        meta_df.loc[dataset_name, 'n ones in y'] = n_ones
        # 
        algos_order_lin = ['dummy', 'NaiveBayes', 'ElsasticNet', 'PLSDA', 'SVMlinear', 'SVMrbf']
        algos_order_rul = ['AdaBoost', 'SCMBoost', 'GBtree', 'XGBoost', 'DT', 'RF', 'SCM', 'rSCM']
        mean_df_lin = mean_df[mean_df['algo'].isin(algos_order_lin)]
        mean_df_rul = mean_df[mean_df['algo'].isin(algos_order_rul)]
        if mean_df_lin.shape[0] == 0 or mean_df_rul.shape[0] == 0:
            print('not enough algos to compare linear-based vs rule-based')
            continue
        mean_df_lin = mean_df_lin.sort_values(by='score', ascending=False)
        mean_df_rul = mean_df_rul.sort_values(by='score', ascending=False)
        meta_df.loc[dataset_name, 'best linear-based algo name'] = mean_df_lin['algo'].iloc[0]
        meta_df.loc[dataset_name, 'best linear-based algo score'] = mean_df_lin['score'].iloc[0]
        meta_df.loc[dataset_name, 'best rule-based algo name'] = mean_df_rul['algo'].iloc[0]
        meta_df.loc[dataset_name, 'best rule-based algo score'] = mean_df_rul['score'].iloc[0]
    return meta_df