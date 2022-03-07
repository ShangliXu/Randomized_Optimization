# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 17:29:58 2022

@author: passi
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 15:06:58 2022

@author: passi
"""

import mlrose_hiive
import numpy as np
import logging
import networkx as nx
import matplotlib.pyplot as plt
import string


from ast import literal_eval
import chess

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from mlrose_hiive import QueensGenerator, MaxKColorGenerator, TSPGenerator
from mlrose_hiive import SARunner, GARunner, NNGSRunner, MIMICRunner, RHCRunner

def opt_prob_domains(prob_domains_name, problem_size):
    if prob_domains_name == 'Nqueens':
        problem = QueensGenerator().generate(seed=123456, size = problem_size)
    elif prob_domains_name == 'MaxKColor':
        problem = MaxKColorGenerator().generate(seed=123456, number_of_nodes = problem_size, max_connections_per_node=3, max_colors=3)
    elif prob_domains_name == 'TSP':
        problem = TSPGenerator().generate(seed=123456, number_of_cities = problem_size)
    return problem


def local_random_search_algo(algo_func, problem, experiment_name_, **kwargs):
    # create a runner class and solve the problem
    algo = algo_func(problem=problem,
                  experiment_name = experiment_name_,
                  output_directory=None, # note: specify an output directory to have results saved to disk
                  seed=123456,
                  iteration_list=2 ** np.arange(11),
                  max_attempts=500,
                  **kwargs)
    return algo.run()


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
        run_stats_best_run = df_run_stats[df_run_stats['Restarts'] == best_init_restart]
        run_curve_best_run = df_run_curves[df_run_curves['Restarts'] == best_init_restart]
        
    elif algo_name == 'sa':
        # Which has the following identifying state information:
        best_init_temperature = best_curve_run['Temperature'].iloc()[0].init_temp
        print(f'Best initial temperature: {best_init_temperature}')
        run_stats_best_run = df_run_stats[df_run_stats['schedule_init_temp'] == best_init_temperature]
        run_curve_best_run = df_run_curves[df_run_curves['Temperature'].apply(lambda x: x.init_temp==best_init_temperature)]

    elif algo_name == 'ga':
        best_init_population_size = best_curve_run['Population Size'].iloc()[0]
        best_init_mutation_rate = best_curve_run['Mutation Rate'].iloc()[0]
        print(f'Best initial population size: {best_init_population_size}')
        print(f'Best initial mutation size: {best_init_mutation_rate}')
        run_stats_best_run = df_run_stats[np.logical_and(df_run_stats['Population Size'] == best_init_population_size, \
                                          df_run_stats['Mutation Rate'] == best_init_mutation_rate)]
        run_curve_best_run = df_run_curves[np.logical_and(df_run_curves['Population Size'] == best_init_population_size, \
                                          df_run_curves['Mutation Rate'] == best_init_mutation_rate)]
            
    elif algo_name == 'mimic':
        best_init_population_size = best_curve_run['Population Size'].iloc()[0]
        best_init_keep_percent = best_curve_run['Keep Percent'].iloc()[0]
        print(f'Best initial population size: {best_init_population_size}')
        print(f'Best initial keep percent: {best_init_keep_percent}')
        run_stats_best_run = df_run_stats[np.logical_and(df_run_stats['Population Size'] == best_init_population_size, \
                                          df_run_stats['Keep Percent'] == best_init_keep_percent)]
        run_curve_best_run = df_run_curves[np.logical_and(df_run_curves['Population Size'] == best_init_population_size, \
                                          df_run_curves['Keep Percent'] == best_init_keep_percent)]
    
    return run_stats_best_run, run_curve_best_run


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

    fitness_list = []
    for run_stats_best_run in run_stats_best_run_list:
        fitness_list.append(run_stats_best_run['Fitness'].iloc[-1])
    # plt.bar(algo_names, height=fitness_list)
    # plt.title('Fitness')
    return fitness_list
    

def problem_size(problem_size_list, fitness_list, algo_names, title):
    fitness_list = np.array(fitness_list)
    for i in range(len(fitness_list.T)):
        plt.plot(problem_size_list, fitness_list[:,i])
    plt.legend(algo_names)
    plt.title('Problem size vs Fitness ' + title)
    plt.xlabel('Problem size')
    plt.ylabel('Fitness')
    plt.show()



prob_domains_name_list = ['TSP']#['Nqueens', 'MaxKColor', 'TSP']
problem_size_map = {'TSP':[10, 20, 50]}#[10, 20, 100]}#{'Nqueens':[8, 20], 'MaxKColor':[8, 20], 'TSP':[10, 20, 100]}#[8, 20, 50]
fitness_list_list = []
for prob_domains_name in prob_domains_name_list:
    fitness_list = []
    problem_size_list = problem_size_map[prob_domains_name]
    for problem_size_ in problem_size_list:
        problem = opt_prob_domains(prob_domains_name, problem_size = problem_size_)
        
        rhc_kwargs = {'restart_list': [25, 75, 100]}
        sa_kwargs = {'temperature_list': [0.1, 0.5, 0.75, 1.0, 2.0, 5.0], 'decay_list': [mlrose_hiive.GeomDecay]}
        ga_kwargs = {'population_sizes': [10, problem_size_], 'mutation_rates': [0.2, 0.5, 0.8]}
        mimic_kwargs = {'keep_percent_list': [0.25, 0.5, 0.75], 'population_sizes': [10, problem_size_], 'use_fast_mimic': [True]}
        algo_names = ['rhc', 'sa', 'ga', 'mimic']
        
        algo_func_list = [RHCRunner, SARunner, GARunner, MIMICRunner]
        kwargs_list = [rhc_kwargs, sa_kwargs, ga_kwargs, mimic_kwargs]
        
        run_stats_best_run_list, run_curve_best_run_list = [], []
        for algo_func, algo_kwargs, algo_name in zip(algo_func_list, kwargs_list, algo_names):
            print(prob_domains_name, problem_size_, algo_name)
            df_run_stats, df_run_curves = local_random_search_algo(algo_func, problem, prob_domains_name + '_' + algo_name, **algo_kwargs)
            run_stats_best_run, run_curve_best_run = best_run(algo_name, df_run_stats, df_run_curves, max_min = 'min')
            run_stats_best_run_list.append(run_stats_best_run)
            run_curve_best_run_list.append(run_curve_best_run)
        title = prob_domains_name + ' ' + str(problem_size_)
        fitness_list.append(fitness_and_time(run_curve_best_run_list, algo_names, title))
    problem_size(problem_size_list, fitness_list, algo_names, prob_domains_name)
    fitness_list_list.append(fitness_list)


# fitness_list_20 = fitness_list[0]
# [1362.9958890094397, 1121.1419444273401, 939.8363736930773, 1895.3599889159889]
# fitness_list_10 = fitness_list[0]
# [712.4692884672163, 712.4692884672163, 712.4692884672163, 1304.0119250307298]
    # fitness_list_50
    # [[2553.8499791367385,  2592.0340261371157,  1814.2950801194113,  5484.468329827498]]