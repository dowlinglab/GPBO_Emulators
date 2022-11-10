##https://towardsdatascience.com/model-validation-in-python-95e2f041f78c
##Load modules
import sys
import gpytorch
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from scipy.stats import qmc
from sklearn.model_selection import LeaveOneOut

from bo_functions_generic import train_GP_model, ExactGPModel, find_train_doc_path, clean_1D_arrays, set_ep, calc_GP_outputs, eval_GP

import matplotlib as mpl

###Load data
###Get constants

def LOO_Analysis(all_data, ep, Xexp, Yexp, true_model_coefficients, emulator, sparse_grid, obj, skip_param_types = 0, set_lengthscale = None, train_iter = 300,verbose = False):
    ep_init = ep
    loo = LeaveOneOut()
    loo.get_n_splits(all_data)
    #Loop over all test indecies & #Shuffle and split into training and testing data where 1 point is testing data
    for train_index, test_index in loo.split(all_data):
        data_train = all_data[train_index]
        data_test = all_data[test_index]
        
        #separate into y data and parameter data
        train_p = data_train[:,0:-1]
        test_p = data_test[:,0:-1]
        
        train_y = data_train[:,-1]
        test_y = data_test[:,-1]
         
        #Set likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_p, train_y, likelihood)
        #Train GP
        train_GP = train_GP_model(model, likelihood, train_p, train_y, train_iter, verbose=verbose)
        #Set exploration parameter (in thic case, 1)
        explore_bias = ep_init #Sets ep to the multiplicative scaler between 0.1 and 1
        #Evaluate GP
        # QUESTION: What do I actually want to evaluate? The test theta at all values of X1 and X2?
        #Create new functions to do the LOO GP analysis
        eval_components = eval_GP(test_p, train_p, explore_bias,Xexp, Yexp, true_model_coefficients, model, likelihood, verbose, emulator, sparse_grid, set_lengthscale, train_p, obj = obj, skip_param_types = skip_param_types)
        
        if verbose == True and emulator == False:
            ei,sse,var,stdev,best_error,z,ei_term_1,ei_term_2,CDF,PDF = eval_components
        elif emulator == True:
            ei,sse,var,stdev,best_error,gp_mean_all, gp_var_all = eval_components
        else:
            ei,sse,var,stdev,best_error = eval_components
            
        #Plot GP_mean test vs train for X1 and X2 vs Muller Potential
        #Fix these plotters to be what I want
        if emulator == True:
            LOO_Plots_3_Input(model, likelihood, Xexp, noise_std, emulator, set_lengthscale, t, obj, sep_fact, verbose = verbose, runs = runs, DateTime = DateTime, test_p = test_p, LOO = LOO, LSO = LSO, save_figure = save_fig)
        else:
            LOO_Plots_2_Input(model, likelihood, Xexp, noise_std, emulator, set_lengthscale, t, obj, sep_fact, verbose = verbose, runs = runs, DateTime = DateTime, test_p = test_p, LOO = LOO, LSO = LSO, save_figure = save_fig)
            
        #Calculate SSE
        #Make residual plots