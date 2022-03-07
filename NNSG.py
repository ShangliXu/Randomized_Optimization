# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 23:22:52 2022

@author: passi
"""


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import mlrose_hiive
from mlrose_hiive import NNGSRunner
from sklearn.neural_network import MLPClassifier


import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit, GridSearchCV, learning_curve
from sklearn.metrics import balanced_accuracy_score


from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

random_seed = 42
test_data_size = 0.2
cv_data_size = 0.2
impute_strategy = 'mean'
n_jobs_val = -1 
cv_num = 5
max_iter_num = 200
score_method = 'balanced_accuracy' #balanced_accuracy_score
cv_splitter = ShuffleSplit(n_splits = cv_num, test_size = cv_data_size, random_state = random_seed)
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

estimator_names = ['Neural networks']
iterative_algorithms = ['Neural networks']
estimator_list = [MLPClassifier]
estimators = dict(zip(estimator_names, estimator_list))
    
hyperparameter_kepler = [{'hidden_layer_sizes': [(100,)], 'max_iter': [50],
    'alpha': [0.05], 'learning_rate': ['constant'],},]
hyperparameter_maternal = [{'hidden_layer_sizes': [(50,50,50)], 'max_iter': [500],
    'alpha': [0.0001], 'learning_rate': ['constant'],},]
dataset = ['kepler', 'health']
hyperparameters = dict(zip(dataset, [hyperparameter_kepler, hyperparameter_maternal]))

            
def data_preprocess(X, y):
    col_names = X.columns
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    imputer = SimpleImputer(missing_values = np.nan, strategy = impute_strategy)
    imputer = imputer.fit(X)
    X = pd.DataFrame(data = imputer.transform(X), index = y.index, columns = col_names)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size = test_data_size, random_state = random_seed)
    
    one_hot = OneHotEncoder()
    y_train = one_hot.fit_transform(y_train.to_numpy().reshape(-1, 1)).todense()
    y_test = one_hot.transform(y_test.to_numpy().reshape(-1, 1)).todense()

    return X_train, X_test, y_train, y_test


def get_kepler_data():
    kepler = pd.read_csv("Kepler Exoplanet Search Results.csv", index_col = 'kepid')
    col_to_drop = ['rowid', 'kepoi_name', 'koi_pdisposition', 'koi_tce_delivname']
    col_to_drop.extend(kepler.columns[kepler.isnull().sum()>kepler.shape[0]*0.1].to_list())
    kepler.drop(col_to_drop, axis = 1, inplace = True)
    
    le = LabelEncoder()
    kepler.koi_disposition = le.fit_transform(kepler.koi_disposition)
    y = kepler['koi_disposition']
    # print(y.unique(), le.inverse_transform(y.unique()))
    
    X = kepler.loc[:, kepler.columns != 'koi_disposition']
    return data_preprocess(X, y)


def get_maternal_data():
    maternal = pd.read_csv('Maternal Health Risk Data Set.csv')
    le = LabelEncoder()
    maternal.RiskLevel = le.fit_transform(maternal.RiskLevel)
    y = maternal.RiskLevel
    # print(y.unique(), le.inverse_transform(y.unique()))
    X = maternal.loc[:, maternal.columns != 'RiskLevel']
    return data_preprocess(X, y)


def fitness_and_time(run_stats_best_run_list, algo_names, title):    
    for run_stats_best_run in run_stats_best_run_list:
        run_stats_best_run = run_stats_best_run.drop_duplicates(subset=['Iteration'])
        plt.plot(run_stats_best_run['Iteration'], run_stats_best_run['Fitness'])
    plt.legend(algo_names)
    plt.title('Fitness vs Iteration ' + title)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.show()
    
    time_list = []
    for run_stats_best_run in run_stats_best_run_list:
        time_list.append(run_stats_best_run['Time'].iloc[-1])
    plt.bar(algo_names, height=time_list)
    plt.title('Time ' + title)
    plt.show()


X_train_scaled, X_test_scaled, y_train_hot, y_test_hot = get_maternal_data()


nn_kepler_kwargs = {'hidden_layer_sizes': [(100,)], 'iteration_list': 2 ** np.arange(7),}

rhc_kwargs = {'restarts': [25, 50, 5]}
sa_kwargs = {'schedule': [mlrose_hiive.ArithDecay(10.0), mlrose_hiive.ArithDecay(1.0), mlrose_hiive.ArithDecay(0.25)]}#{'temperature_list': [0.1, 0.5, 0.75, 1.0, 2.0, 5.0], 'decay_list': [mlrose_hiive.GeomDecay]}
ga_kwargs = {'pop_size': [100, 200, 1000], 'mutation_prob': [0.2, 0.5, 0.8]}
# mimic_kwargs = {'keep_percent_list': [0.25, 0.5, 0.75], 'pop_size': [100, 200, 1000], 'use_fast_mimic': [True]}
# algo_names = ['rhc', 'sa', 'ga', 'mimic']
        
grid_search_parameters = {
    "max_iters": [50],
    "learning_rate_init": [0.0001],
    "hidden_layers_sizes": [(100,)],
    "activation": [mlrose_hiive.neural.activation.relu],
    "is_classifier": [True]}
#     "mutation_prob": [0.1, 0.25, 0.5, 0.7],
#     "pop_size": [100, 200, 1000]    
# }
grid_search_parameters_rhc = grid_search_parameters.copy()
grid_search_parameters_rhc.update(rhc_kwargs)
grid_search_parameters_sa = grid_search_parameters.copy()
grid_search_parameters_sa.update(sa_kwargs)
grid_search_parameters_ga = grid_search_parameters.copy()
grid_search_parameters_ga.update(ga_kwargs)
# grid_search_parameters_mmc = grid_search_parameters.copy()
# grid_search_parameters_mmc.update(mimic_kwargs)


nngs = NNGSRunner(x_train=X_train_scaled,
                      y_train=y_train_hot,
                      x_test=X_test_scaled,
                      y_test=y_test_hot,
                      experiment_name='random_hill_climb',
                      algorithm=mlrose_hiive.algorithms.rhc.random_hill_climb,#ga.genetic_alg,#sa.simulated_annealing,
                      grid_search_parameters=grid_search_parameters_rhc,
                      bias=True,
                      early_stopping=False,
                      clip_max=1e+10,
                      max_attempts=500,
                      generate_curves=True,
                      seed=123456,
                      **nn_kepler_kwargs)

df_run_stats, df_run_curves, cv_results, best_est = nngs.run()
print('best_est.best_params_ ', best_est.best_params_)

df_run_curves_rhc  = df_run_curves
df_run_stats_rhc = df_run_stats
time_rhc = df_run_curves['Time'].iloc[-1]

nngs = NNGSRunner(x_train=X_train_scaled,
                      y_train=y_train_hot,
                      x_test=X_test_scaled,
                      y_test=y_test_hot,
                      experiment_name='simulated_annealing',
                      algorithm=mlrose_hiive.algorithms.sa.simulated_annealing,
                      grid_search_parameters=grid_search_parameters_sa,
                      bias=True,
                      early_stopping=False,
                      clip_max=1e+10,
                      max_attempts=500,
                      generate_curves=True,
                      seed=123456,
                      **nn_kepler_kwargs)

df_run_stats, df_run_curves, cv_results, best_est = nngs.run()
print('best_est.best_params_ ', best_est.best_params_)

df_run_curves_sa  = df_run_curves
df_run_stats_sa = df_run_stats
time_sa = df_run_curves['Time'].iloc[-1]


nngs = NNGSRunner(x_train=X_train_scaled,
                      y_train=y_train_hot,
                      x_test=X_test_scaled,
                      y_test=y_test_hot,
                      experiment_name='genetic_alg',
                      algorithm=mlrose_hiive.algorithms.ga.genetic_alg,#sa.simulated_annealing,
                      grid_search_parameters=grid_search_parameters_ga,
                      bias=True,
                      early_stopping=False,
                      clip_max=1e+10,
                      max_attempts=500,
                      generate_curves=True,
                      seed=123456,
                      **nn_kepler_kwargs)

df_run_stats, df_run_curves, cv_results, best_est = nngs.run()
print('best_est.best_params_ ', best_est.best_params_)

df_run_curves_ga  = df_run_curves
df_run_stats_ga = df_run_stats
time_ga = df_run_curves['Time'].iloc[-1]


# algo_names = 'ga'
# title = 'NN' + ' ' + algo_names
# fitness_and_time(df_run_curves, algo_names, title)

def fitness_and_time(run_stats_best_run_list, algo_names, title):    
    for run_stats_best_run in run_stats_best_run_list:
        run_stats_best_run = run_stats_best_run.drop_duplicates(subset=['Iteration'])
        plt.plot(run_stats_best_run['Iteration'], run_stats_best_run['Fitness'])
    plt.legend(algo_names)
    plt.title('Fitness vs Iteration ' + title)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.show()
    
    time_list = []
    for run_stats_best_run in run_stats_best_run_list:
        time_list.append(run_stats_best_run['Time'].iloc[-1])
    plt.bar(algo_names, height=time_list)
    plt.title('Time ' + title)
    plt.show()

run_stats_best_run_list = [df_run_curves_rhc, df_run_curves_sa, df_run_curves_ga]
fitness_and_time(run_stats_best_run_list, ['rhc', 'sa', 'ga'], 'NN')


