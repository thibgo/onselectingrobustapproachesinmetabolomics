import os
import random
import numpy as np
import pandas as pd
import pickle as pkl
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score, matthews_corrcoef

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

from utils.pls import PLSClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.dummy import DummyClassifier
from randomscm import RandomScmClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from pyscm import SetCoveringMachineClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from config.config_local import dataset_bulk_dir, test_size


def locate_dataset(dataset_name):
    data_path = dataset_bulk_dir
    if 'featuredeletion' in dataset_name:
        data_path = dataset_featuredeletion_bulk_dir
    dataset_filename = f'{dataset_name}.pkl'
    return data_path, dataset_filename

def generate_train_test_split(dataset_name, split_id):
    data_path, dataset_filename = locate_dataset(dataset_name)
    with open(os.path.join(data_path, dataset_filename), 'rb') as fo:
        dataset = pkl.load(fo)
    X, y, features_names = dataset['X'], dataset['y'], dataset['features_names']
    assert len(X) == len(y)
    assert X.shape[1] == len(features_names)
    #print("X.shape", X.shape)
    #print("len(y)", len(y))
    if split_id == 'full_dataset':
        return X, X, y, y
    if 'subject_idx' in dataset:
        # presence of a subject_idx field means that we have to do a subject-wise split
        # the multiple samples of a same subject must be in the same set
        subjects_ids = dataset['subject_idx']
        assert len(subjects_ids) == len(X)
        unique_subjects_ids = list(set(subjects_ids))
        print('unique_subjects_ids', unique_subjects_ids)
        # set random seed
        random.seed(11+split_id)
        # get a random sample of subjects
        subjects_ids_test_set = random.sample(unique_subjects_ids, int(len(unique_subjects_ids)*test_size))
        # generate the test set
        sample_idx_test_set, sample_idx_train_set = [], []
        for i in range(len(X)):
            if subjects_ids[i] in subjects_ids_test_set:
                sample_idx_test_set.append(i)
            else:
                sample_idx_train_set.append(i)
        X_test = X[sample_idx_test_set]
        y_test = [y[i] for i in sample_idx_test_set]
        X_train_and_val = X[sample_idx_train_set]
        y_train_and_val = [y[i] for i in sample_idx_train_set]
        print('X_test.shape', X_test.shape)
        print('X_train_and_val.shape', X_train_and_val.shape)
        print('len(y_test)', len(y_test))
        print('len(y_train_and_val)', len(y_train_and_val))
    else:
        X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(X, y, test_size=test_size, random_state=11+split_id)
    return X_train_and_val, X_test, y_train_and_val, y_test

def model_selector(algo, parameters_to_evaluate):
    if algo == 'RF':
        classifier = RandomForestClassifier(**parameters_to_evaluate, n_jobs=1)
    elif algo == 'rSCM':
        classifier = RandomScmClassifier(**parameters_to_evaluate, n_jobs=1)
    elif algo == 'DT':
        classifier = DecisionTreeClassifier(**parameters_to_evaluate)
    elif algo == 'SCM':
        classifier = SetCoveringMachineClassifier(**parameters_to_evaluate)
    elif algo == 'SVMlinear':
        classifier = Pipeline([("scaler", StandardScaler()), ("svm", LinearSVC(**parameters_to_evaluate))])
    elif algo == 'SVMrbf':
        classifier = Pipeline([("scaler", StandardScaler()), ("svm", SVC(**parameters_to_evaluate))])
    elif algo == 'AdaBoost':
        classifier = AdaBoostClassifier(**parameters_to_evaluate)
    elif algo == 'SCMBoost':
        # dict of parameters retricted to p, max_rules, model_type
        weak_classifier_parameters = {k: parameters_to_evaluate[k] for k in ['p', 'max_rules', 'model_type']}
        ensemble_parameters = {k: parameters_to_evaluate[k] for k in ['n_estimators', 'learning_rate', 'algorithm']}
        classifier = AdaBoostClassifier(estimator=SetCoveringMachineClassifier(**weak_classifier_parameters), **ensemble_parameters)
    elif algo == 'dummy':
        classifier = DummyClassifier()
    elif algo == 'GBtree':
        classifier = GradientBoostingClassifier(**parameters_to_evaluate)
    elif algo == 'PLSDA':
        classifier = PLSClassifier(**parameters_to_evaluate)
    elif algo == 'XGBoost':
        classifier = XGBClassifier(**parameters_to_evaluate)
    elif algo == 'NaiveBayes':
        classifier = GaussianNB(**parameters_to_evaluate)
    elif algo == 'ElasticNet':
        classifier = LogisticRegression(**parameters_to_evaluate)
    return classifier

def compute_runtime_for_a_dataset(dataset_name):
    x_train, x_test, y_train, y_test = generate_train_test_split(dataset_name, split_id=0)
    n_features = x_train.shape[1]
    n_samples = x_train.shape[0] + x_test.shape[0]
    #print('computing running time for dataset {}, n_features={}, n_samples={}'.format(dataset_name, n_features, n_samples))
    divider = max(2, (80-(np.log(n_features) * np.log(max(100, n_samples))))//10)
    runtime = (np.log(n_features) * np.log(n_samples))//divider
    #print('computed runtime in minutes', runtime)
    return runtime

def compute_model_sparsity(model):
    if isinstance(model, RandomForestClassifier):
        n_nonzero_features = sum(model.feature_importances_ > 0)
        n_nodes = 0
        for estimator in model.estimators_:
            n_nodes += estimator.tree_.node_count
        best_feature_id = model.feature_importances_.argmax()
    elif isinstance(model, DecisionTreeClassifier):
        n_nonzero_features = sum(model.feature_importances_ > 0)
        n_nodes = model.tree_.node_count
        best_feature_id = model.feature_importances_.argmax()
    elif isinstance(model, SetCoveringMachineClassifier):
        rule_imp = model.rule_importances_
        # non zero rule imp
        rule_imp_nonzero = rule_imp[rule_imp > 0]
        n_nonzero_features = len(rule_imp_nonzero)
        n_nodes = len(rule_imp)
        best_feature_id = model.model_.rules[0].feature_idx
    elif isinstance(model, RandomScmClassifier):
        n_nonzero_features = sum(model.feature_importances_ > 0)
        n_nodes = 0
        for estimator in model.estimators:
            if hasattr(estimator, 'rule_importances_'):
                n_nodes += sum(estimator.rule_importances_ > 0)
        best_feature_id = model.feature_importances_.argmax()
    elif isinstance(model, Pipeline) and isinstance(model['svm'], LinearSVC):
        n_nonzero_features = sum(abs(model['svm'].coef_[0]) > 0)
        n_nodes = None
        best_feature_id = abs(model['svm'].coef_[0]).argmax()
    elif isinstance(model, Pipeline) and isinstance(model['svm'], SVC):
        #print(model['svm'].support_.shape[0])
        n_nonzero_features = model['svm'].support_.shape[0]
        n_nodes = None
        best_feature_id = None
    elif isinstance(model, AdaBoostClassifier) and isinstance(model.estimators_[0], DecisionTreeClassifier):
        n_nonzero_features = sum(model.feature_importances_ > 0)
        n_nodes = 0
        for estim in model.estimators_:
            n_nodes += estim.tree_.node_count
        best_feature_id = model.feature_importances_.argmax()
    elif isinstance(model, AdaBoostClassifier) and isinstance(model.estimators_[0], SetCoveringMachineClassifier):
        feat_usage_dict = {}
        n_nodes = 0
        for estim in model.estimators_:
            rule_imp = estim.rule_importances_
            # non zero rule imp
            rule_imp_nonzero = rule_imp[rule_imp > 0]
            n_nodes += len(rule_imp_nonzero)
            for i in range(len(rule_imp_nonzero)):
                feat_id_in_rule = estim.model_.rules[i].feature_idx
                if feat_id_in_rule not in feat_usage_dict:
                    feat_usage_dict[feat_id_in_rule] = 1
                else:
                    feat_usage_dict[feat_id_in_rule] += 1
        n_nonzero_features = len(feat_usage_dict)
        # key of max value
        best_feature_id = max(feat_usage_dict, key=feat_usage_dict.get)
    # (len(model.estimators_) > 0) and 
    #elif isinstance(model, AdaBoostClassifier) and (len(model.estimators_) == 0):
    #    n_nonzero_features = 1
    #    n_nodes = 1
    #    best_feature_id = None
    elif isinstance(model, GradientBoostingClassifier):
        n_nonzero_features = sum(model.feature_importances_ > 0)
        n_nodes = 0
        for estim in model.estimators_.flatten():
            n_nodes += estim.tree_.node_count
        best_feature_id = model.feature_importances_.argmax()
    elif isinstance(model, DummyClassifier):
        n_nonzero_features = 1
        n_nodes = 1
        best_feature_id = None
    elif isinstance(model, PLSClassifier):
        n_nonzero_features = sum(sum(model.model.coef_ > 0))
        n_nodes = sum(sum(model.model.coef_ > 0))
        best_feature_id = model.model.coef_.argmax()
    elif isinstance(model, XGBClassifier):
        # TODO
        n_nonzero_features = len(model.get_booster().get_score())
        n_nodes = 0
        modeljson = model.get_booster().get_dump(dump_format='json')
        for tree_id in range(len(modeljson)):
            nodes_idx = []
            for e in modeljson[tree_id].split('nodeid": ')[1:]:
                nodes_idx.append(int(e.split(',')[0]))
            assert len(nodes_idx) == len(set(nodes_idx))
            n_nodes += len(nodes_idx)
        most_important_feature = max(model.get_booster().get_score(), key=model.get_booster().get_score().get)
        best_feature_id = int(most_important_feature[1:])
    elif isinstance(model, GaussianNB):
        n_nonzero_features = model.theta_.shape[1]
        n_nodes = model.theta_.shape[1]
        best_feature_id = None
    elif isinstance(model, LogisticRegression):
        coeffs = abs(model.coef_)
        assert coeffs.shape[0] == 1
        coeffs = coeffs[0]
        n_nonzero_features = sum(coeffs > 0)
        n_nodes = sum(coeffs > 0)
        best_feature_id = coeffs.argmax()
    else:
        print('model', model)
        raise ValueError('model not recognized for sparsity computation')
    if n_nonzero_features is not None:
        n_nonzero_features = int(n_nonzero_features)
    if n_nodes is not None:
        n_nodes = int(n_nodes)
    if best_feature_id is not None:
        best_feature_id = int(best_feature_id)
    return n_nonzero_features, n_nodes, best_feature_id
        
def run_exp(algo, dataset_name, metric, split_id, bayesian_optimization_trials, param_grid, results_dir, exp_snapshot_to_start, exp_specifics={}):
    if 'noisefat' in exp_specifics:
        dataset_name_for_splits = f'{dataset_name}_noisefat_{exp_specifics["noisefat"]}'
    else:
        dataset_name_for_splits = dataset_name

    X_train_and_val, X_test, y_train_and_val, y_test = generate_train_test_split(dataset_name_for_splits, split_id=split_id)

    if 'sparsityconstraint' in exp_specifics:
        objective_name = metric + '+sparsityconstraint'
    else:
        objective_name = metric

    if 'optimethod' in exp_specifics:
        optimethod = exp_specifics['optimethod']
    else:
        optimethod = 'bayesianoptimization'

    metric_sklearn_name_dict = {'f1score': 'f1_macro', 'balancedaccuracy': 'balanced_accuracy', 'mcc': 'matthews_corrcoef'}
    metric_sklearn_name = metric_sklearn_name_dict[metric]

    def evaluation_function(parameters_to_evaluate):
        classifier = model_selector(algo, parameters_to_evaluate)
        n_total_features = X_train_and_val.shape[1]
        # if it is a tuple
        if 'sparsityconstraint' in exp_specifics:
            try:
                cv_res_df = cross_validate(classifier, X_train_and_val, y_train_and_val, cv=5, scoring=metric_sklearn_name, n_jobs=1, verbose=1, return_estimator=True)
                print('cv_res_df', cv_res_df)
                perf_score = cv_res_df['test_score'].mean()
                n_nodes_list = []
                for sub_classif in list(cv_res_df['estimator']):
                    _, loc_n_nodes, _ = compute_model_sparsity(sub_classif)
                    n_nodes_list.append(loc_n_nodes)
                n_nodes = sum(n_nodes_list)/len(n_nodes_list)
                # if the number of nodes is above the constraint, penalize the score : the more nodes, the less the score
                score = perf_score - (max(n_nodes - exp_specifics['sparsityconstraint'], 0)/exp_specifics['sparsityconstraint'])
            except:
                print("  ||  failed to fit a model, the score is set to 0  ||  ")
                score = 0
        else:
            score = cross_val_score(classifier, X_train_and_val, y_train_and_val, cv=5, scoring=metric_sklearn_name, n_jobs=1, verbose=1).mean()
            if np.isnan(score):
                # if the model cannot be fitted, return a very bad score
                # this case is raised for some instance of SCMBoost
                score = 0
                print("  ||  failed to fit a model, the score is set to 0  ||  ")
        return score
    
    t1 = datetime.now()

    if optimethod == 'bayesianoptimization':
        if exp_snapshot_to_start is not None:
            #if False: ## TODO : remettre la ligne d'au dessus
            ax_client = AxClient.load_from_json_file(filepath=os.path.join(results_dir, exp_snapshot_to_start))
            number_of_already_computed_trials = len(ax_client.experiment.trials)
            print('-------------------------------------------')
            print('exp_snapshot_to_start', exp_snapshot_to_start)
            print('number_of_already_computed_trials', number_of_already_computed_trials)
            print('-------------------------------------------')
            if number_of_already_computed_trials >= bayesian_optimization_trials:
                print('-------------------------------------------')
                print('already done more trials than asked')
                print('-------------------------------------------')
                return None
        else:
            ax_client = AxClient(random_seed=11)
            number_of_already_computed_trials = 0
            ax_client.create_experiment(
                name=f"{dataset_name}_{algo}_{bayesian_optimization_trials}_{split_id}",
                parameters=param_grid,
                objectives={objective_name: ObjectiveProperties(minimize=False)},
                overwrite_existing_experiment=True,
            )
        for k in range(number_of_already_computed_trials, bayesian_optimization_trials):
            parameters_to_evaluate, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(trial_index=trial_index, raw_data=evaluation_function(parameters_to_evaluate))
            if algo == 'dummy' or algo == 'NaiveBayes' :
                break
            elif algo == 'PLSDA' and k>=10:
                break
        best_parameters, metrics = ax_client.get_best_parameters()
        experiment_df = ax_client.get_trials_data_frame()
    elif optimethod == 'gridsearch':
        gridsearch = GridSearchCV(model_selector(algo, {}), param_grid, cv=5, scoring=metric_sklearn_name, n_jobs=2, verbose=1)
        gridsearch.fit(X_train_and_val, y_train_and_val)
        best_parameters = gridsearch.best_params_
        experiment_df = pd.DataFrame(data=gridsearch.cv_results_)
        metrics = gridsearch.best_score_
    elif optimethod.startswith('gridsearchfavorsparse'):
        tie_threshold = 0
        if optimethod.endswith('percent'):
            tie_threshold = int(optimethod.split('tie')[1].split('percent')[0])/100
        print('tie_threshold', tie_threshold)
        gridsearch = GridSearchCV(model_selector(algo, {}), param_grid, cv=5, scoring=metric_sklearn_name, n_jobs=2, verbose=1)
        gridsearch.fit(X_train_and_val, y_train_and_val)
        experiment_df = pd.DataFrame(data=gridsearch.cv_results_)
        max_test_score = experiment_df['mean_test_score'].max()
        print('max_test_score', max_test_score)
        tie_exp_df = experiment_df[experiment_df['mean_test_score'] >= max_test_score - tie_threshold].copy()
        print('tie_exp_df.shape', tie_exp_df.shape)
        sparsity_col = []
        for loc_params in tie_exp_df['params']:
            sub_classif = model_selector(algo, loc_params)
            sub_classif.fit(X_train_and_val, y_train_and_val)
            _, loc_n_nodes, _ = compute_model_sparsity(sub_classif)
            sparsity_col.append(loc_n_nodes)
            print(loc_params, loc_n_nodes)
        tie_exp_df['nnodes'] = sparsity_col
        # sort by score
        tie_exp_df = tie_exp_df.sort_values(by='mean_test_score', ascending=False)
        tie_exp_df = tie_exp_df.sort_values(by='nnodes', ascending=True)
        best_parameters = tie_exp_df['params'].values[0]
        metrics = {'score of best solution' : tie_exp_df['mean_test_score'].values[0],
                     'nnodes of best solution' : tie_exp_df['nnodes'].values[0]}

    print('************************************')
    print("best_parameters", best_parameters)
    print('metrics', metrics)
    print('************************************')
    t2 = datetime.now()
    model = model_selector(algo, best_parameters)
    model.fit(X_train_and_val, y_train_and_val)
    train_and_val_pred = model.predict(X_train_and_val)
    test_pred = model.predict(X_test)
    # if predictions are not binary
    if len(np.unique(test_pred)) > 2:
        # binarise them
        train_and_val_pred = [1 if x>0.5 else 0 for x in train_and_val_pred]
        test_pred = [1 if x>0.5 else 0 for x in test_pred]
    perf_df = pd.DataFrame(columns=['algo', 'score', 'metric', 'type'])
    row_i = 0
    perf_df.loc[row_i] = [algo, accuracy_score(y_test, test_pred), 'accuracy', 'test']
    row_i += 1
    perf_df.loc[row_i] = [algo, accuracy_score(y_train_and_val, train_and_val_pred), 'accuracy', 'train']
    row_i += 1
    perf_df.loc[row_i] = [algo, f1_score(y_test, test_pred), 'f1_score', 'test']
    row_i += 1
    perf_df.loc[row_i] = [algo, f1_score(y_train_and_val, train_and_val_pred), 'f1_score', 'train']
    row_i += 1
    perf_df.loc[row_i] = [algo, matthews_corrcoef(y_test, test_pred), 'mcc', 'test']
    row_i += 1
    perf_df.loc[row_i] = [algo, matthews_corrcoef(y_train_and_val, train_and_val_pred), 'mcc', 'train']
    row_i += 1
    perf_df.loc[row_i] = [algo, balanced_accuracy_score(y_test, test_pred), 'balancedaccuracy', 'test']
    row_i += 1
    perf_df.loc[row_i] = [algo, balanced_accuracy_score(y_train_and_val, train_and_val_pred), 'balancedaccuracy', 'train']
    try:
        train_roc_auc_score = roc_auc_score(y_train_and_val, train_and_val_pred)
        test_roc_auc_score = roc_auc_score(y_test, test_pred)
    except:
        train_roc_auc_score = 0
        test_roc_auc_score = 0
    row_i += 1
    perf_df.loc[row_i] = [algo, test_roc_auc_score, 'roc_auc_score', 'test']
    row_i += 1
    perf_df.loc[row_i] = [algo, train_roc_auc_score, 'roc_auc_score', 'train']
    row_i += 1
    perf_df.loc[row_i] = [algo, precision_score(y_test, test_pred), 'precision_score', 'test']
    row_i += 1
    perf_df.loc[row_i] = [algo, precision_score(y_train_and_val, train_and_val_pred), 'precision_score', 'train']
    row_i += 1
    perf_df.loc[row_i] = [algo, recall_score(y_test, test_pred), 'recall_score', 'test']
    row_i += 1
    perf_df.loc[row_i] = [algo, recall_score(y_train_and_val, train_and_val_pred), 'recall_score', 'train']
    row_i += 1
    perf_df.loc[row_i] = [algo, (t2 - t1).total_seconds(), 't2-t1', 'fit time']
    row_i += 1
    perf_df['split'] = [split_id]*perf_df.shape[0]
    perf_df['dataset'] = [dataset_name]*perf_df.shape[0]
    perf_df['bayesian_optimization_trials'] = [bayesian_optimization_trials]*perf_df.shape[0]
    n_nonzero_features, n_nodes, best_feature_id = compute_model_sparsity(model)
    perf_df['n_nonzero_features'] = [n_nonzero_features]*perf_df.shape[0]
    perf_df['n_nodes'] = [n_nodes]*perf_df.shape[0]
    perf_df['best_feature_id'] = [best_feature_id]*perf_df.shape[0]
    save_perf_path = os.path.join(results_dir, f'perf_df_metric_{metric}_{dataset_name}_{algo}')
    for spec in exp_specifics:
        perf_df[spec] = [exp_specifics[spec]]*perf_df.shape[0]
        save_perf_path = save_perf_path + f'_{spec}_{exp_specifics[spec]}'
    perf_df.to_csv(save_perf_path+f'_trials_{bayesian_optimization_trials}_{split_id}.csv', index=False)
    if experiment_df is not None:
        experiment_df['split'] = [split_id]*experiment_df.shape[0]
        exp_df_save_path = save_perf_path.replace('perf_df', 'experiment_logs')
        experiment_df.to_csv(exp_df_save_path+f'_trials_{bayesian_optimization_trials}_{split_id}.csv', index=False)
    if exp_snapshot_to_start is not None:
        os.remove(os.path.join(results_dir, exp_snapshot_to_start))
    snapshot_save_path = save_perf_path.replace('perf_df', 'exp_snapshot') + f'_trials_{bayesian_optimization_trials}_{split_id}.json'
    if optimethod == 'bayesianoptimization':
        ax_client.save_to_json_file(snapshot_save_path)
    model_filepath = snapshot_save_path.replace('exp_snapshot', 'model').replace('.json', '.pkl')
    with open(model_filepath, 'wb') as fo:
        pkl.dump(model, fo)

def run_univariate_exp(nsplits, dataset_name, result_dir):
    # being loading data
    data_path, dataset_filename = locate_dataset(dataset_name)
    with open(os.path.join(data_path, dataset_filename), 'rb') as fo:
        dataset = pkl.load(fo)
    X, y, features_names = dataset['X'], dataset['y'], dataset['features_names']
    res_df = pd.DataFrame(data={'feature': features_names})
    print('dataset_name', dataset_name)
    #data_df = pd.DataFrame(data=X, columns=features_names)
    #assert len(X) == len(y)
    #assert X.shape[1] == len(features_names)
    #print("X.shape", X.shape)
    #print("len(y)", len(y))
    #print(X)
    #print(X.dtype)
    #print(data_df.dtypes)
    for split_id in range(nsplits):
        print('split_id', split_id)
        X_train_and_val, X_test, y_train_and_val, y_test = generate_train_test_split(dataset_name, split_id=split_id)
        X_train_and_val_df = pd.DataFrame(data=X_train_and_val, columns=features_names)
        X_test_df = pd.DataFrame(data=X_test, columns=features_names)
        # end loading data
        # start univariate analysis
        perf_split_test_accuracy_list = []
        perf_split_test_f1_list = []
        perf_split_test_balanced_accuracy_list = []
        perf_split_test_roc_auc_score_list = []
        perf_split_test_precision_score_list = []
        perf_split_test_recall_score_list = []
        perf_split_test_mcc_score_list = []
        for feature_id in range(len(features_names)):
            if feature_id % (100*nsplits) == 0:
                print('    feature_id {} / {}'.format(feature_id, len(features_names)))
            model = DecisionTreeClassifier(max_depth=1, random_state=11)
            X_train_and_val_loc_df = X_train_and_val_df[[features_names[feature_id]]]
            X_test_loc_df = X_test_df[[features_names[feature_id]]]
            model.fit(X_train_and_val_loc_df, y_train_and_val)
            test_pred = model.predict(X_test_loc_df)
            perf_split_test_accuracy_list.append(accuracy_score(y_test, test_pred))
            perf_split_test_f1_list.append(f1_score(y_test, test_pred))
            perf_split_test_balanced_accuracy_list.append(balanced_accuracy_score(y_test, test_pred))
            try:
                perf_split_test_roc_auc_score_list.append(roc_auc_score(y_test, test_pred))
            except:
                perf_split_test_roc_auc_score_list.append(None)
            perf_split_test_precision_score_list.append(precision_score(y_test, test_pred))
            perf_split_test_recall_score_list.append(recall_score(y_test, test_pred))
            perf_split_test_mcc_score_list.append(matthews_corrcoef(y_test, test_pred))
        res_df[f'accuracy_split_{split_id}'] = perf_split_test_accuracy_list
        res_df[f'f1_split_{split_id}'] = perf_split_test_f1_list
        res_df[f'balancedaccuracy_split_{split_id}'] = perf_split_test_balanced_accuracy_list
        res_df[f'rocaucscore_split_{split_id}'] = perf_split_test_roc_auc_score_list
        res_df[f'precisionscore_split_{split_id}'] = perf_split_test_precision_score_list
        res_df[f'recallscore_split_{split_id}'] = perf_split_test_recall_score_list
        res_df[f'mccscore_split_{split_id}'] = perf_split_test_mcc_score_list
    # compute mean and std of each metric
    res_df['accuracy_avg'] = res_df[[f'accuracy_split_{split_id}' for split_id in range(nsplits)]].mean(axis=1)
    res_df['f1_avg'] = res_df[[f'f1_split_{split_id}' for split_id in range(nsplits)]].mean(axis=1)
    res_df['balancedaccuracy_avg'] = res_df[[f'balancedaccuracy_split_{split_id}' for split_id in range(nsplits)]].mean(axis=1)
    res_df['rocaucscore_avg'] = res_df[[f'rocaucscore_split_{split_id}' for split_id in range(nsplits)]].mean(axis=1)
    res_df['precisionscore_avg'] = res_df[[f'precisionscore_split_{split_id}' for split_id in range(nsplits)]].mean(axis=1)
    res_df['recallscore_avg'] = res_df[[f'recallscore_split_{split_id}' for split_id in range(nsplits)]].mean(axis=1)
    res_df['mccscore_avg'] = res_df[[f'mccscore_split_{split_id}' for split_id in range(nsplits)]].mean(axis=1)
    # save results df
    save_path = os.path.join(result_dir, f'univariate_df_{dataset_name}.csv')
    res_df.to_csv(save_path)
    return res_df