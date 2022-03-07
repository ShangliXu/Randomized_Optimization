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
from sklearn.metrics import accuracy_score


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

def best_run(algo_name, df_run_stats, df_run_curves, max_min):
    if max_min == 'min':
        best_fitness = df_run_curves['Fitness'].min()
    elif max_min == 'max':
        best_fitness = df_run_curves['Fitness'].max()
    
    # best_fitness = df_run_curves['Fitness'].min()
    best_runs = df_run_curves[df_run_curves['Fitness'] == best_fitness]
    # print('best_runs', best_runs)
    
    minimum_evaluations = best_runs['FEvals'].min()
    best_curve_run = best_runs[best_runs['FEvals'] == minimum_evaluations]

    if algo_name == 'rhc':
        best_init_restart = best_curve_run['Restarts'].iloc()[0]
        print(f'Best initial restart: {best_init_restart}')
        run_stats_best_run = df_run_curves[df_run_curves['Restarts'] == best_init_restart]
        
    elif algo_name == 'sa':
        # Which has the following identifying state information:
        best_init_temperature = best_curve_run['Temperature'].iloc()[0].init_temp
        print(f'Best initial temperature: {best_init_temperature}')
        # run_stats_best_run = df_run_stats[df_run_stats['schedule_init_temp'] == best_init_temperature]
        run_stats_best_run = df_run_curves[df_run_curves['Temperature'].apply(lambda x: x.init_temp==best_init_temperature)]

    elif algo_name == 'ga':
        best_init_population_size = best_curve_run['Population Size'].iloc()[0]
        best_init_mutation_rate = best_curve_run['Mutation Rate'].iloc()[0]
        print(f'Best initial population size: {best_init_population_size}')
        print(f'Best initial mutation size: {best_init_mutation_rate}')
        run_stats_best_run = df_run_curves[np.logical_and(df_run_curves['Population Size'] == best_init_population_size, \
                                          df_run_curves['Mutation Rate'] == best_init_mutation_rate)]
    elif algo_name == 'mimic':
        best_init_population_size = best_curve_run['Population Size'].iloc()[0]
        best_init_keep_percent = best_curve_run['Keep Percent'].iloc()[0]
        print(f'Best initial population size: {best_init_population_size}')
        print(f'Best initial keep percent: {best_init_keep_percent}')
        run_stats_best_run = df_run_curves[np.logical_and(df_run_curves['Population Size'] == best_init_population_size, \
                                          df_run_curves['Keep Percent'] == best_init_keep_percent)]
    
    return run_stats_best_run    


X_train_scaled, X_test_scaled, y_train_hot, y_test_hot = get_kepler_data()#get_maternal_data()

nn_kepler_kwargs = {'hidden_nodes': [100], 'max_iters': 50, 'learning_rate': 0.0001}
    # 'alpha': [0.05], 'learning_rate': ['constant'],}
# nn_health_kwargs = {'hidden_nodes': (50,50,50), 'max_iters': 500, 'learning_rate': 0.0001,}

rhc_kwargs = {'restarts': [5, 25, 500]}
sa_kwargs = {'schedule': [mlrose_hiive.ArithDecay(1.0), mlrose_hiive.ArithDecay(2.0), mlrose_hiive.ArithDecay(5.0)]}#{'temperature_list': [0.1, 0.5, 0.75, 1.0, 2.0, 5.0], 'decay_list': [mlrose_hiive.GeomDecay]}
ga_kwargs = {'pop_size': [100, 200, 1000], 'mutation_prob': [0.2, 0.5, 0.8]}

# Initialize neural network object and fit object - attempt 1
def algo(algo_name, **nn_kwargs):
    nn_model1 = mlrose_hiive.NeuralNetwork(activation ='relu', algorithm = algo_name, #'random_hill_climb', #'simulated_annealing', 'genetic_alg'
                                     bias = True, is_classifier = True, early_stopping = True, 
                                     clip_max = 5, max_attempts = 500, random_state = 3, 
                                     **nn_kwargs)
    
    nn_model1.fit(X_train_scaled, y_train_hot)
    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(X_train_scaled)
    
    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
    
    print(y_train_accuracy)
    
    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model1.predict(X_test_scaled)
    
    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
    
    print(y_test_accuracy)


# rhc_kwargs.update(nn_kepler_kwargs)
# algo('random_hill_climb', **nn_kepler_kwargs)
sa_kwargs.update(nn_kepler_kwargs)
algo('simulated_annealing', **nn_kepler_kwargs)
# ga_kwargs.update(nn_kepler_kwargs)
# algo('genetic_alg', **nn_kepler_kwargs)

# rhc_kwargs.update(nn_health_kwargs)
# algo('random_hill_climb', **rhc_kwargs)
# sa_kwargs.update(nn_health_kwargs)
# algo('simulated_annealing', **sa_kwargs)
# ga_kwargs.update(nn_health_kwargs)
# algo('genetic_alg', **ga_kwargs)

######## record printed accuracy and get backprop accuarcy fro assignment1
train_accuracy = [0.88, 0.46, 0.46, 0.62]
test_accuracy = [0.85, 0.47, 0.47, 0.57]

algo_names = ['backprop', 'rhc', 'sa', 'ga']
plt.bar(algo_names, height=train_accuracy)
plt.title('train_accuracy NN')
plt.ylabel('accuracy')
plt.show()
plt.bar(algo_names, height=test_accuracy)
plt.title('test_accuracy NN')
plt.ylabel('accuracy')
plt.show()