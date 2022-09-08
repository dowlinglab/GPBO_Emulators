import numpy as np
import math
from scipy.stats import norm
import torch
import csv
import gpytorch
import scipy.optimize as optimize
import pandas as pd

from bo_functions import find_train_doc_path, train_GP_model, calc_y_exp, create_sse_data, create_y_data, find_train_doc_path, ExactGPModel, train_GP_model 
from bo_functions import calc_GP_outputs, test_train_split

import matplotlib.pyplot as plt
from bo_plotters import plot_org_train
from bo_plotters import plot_3GP_performance

import os
import Tasmanian

def LOO_test_train_split(all_data, Xexp, sep_fact=0.95, run = 0, shuffle_seed = None, LSO = False):
    """
    Splits y data into training and testing data
    
    Parameters
    ----------
        all_data: ndarray or tensor, The simulated parameter space and y data
        Xexp, ndarray, state points
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        run: int, # of run for LOO iterations. default is 1
        shuffle_seed, int, number of seed for shuffling training data. Default is None.
        LSO: bool, determines whether 1 set of training points or multiple training points are left out
    Returns:
        train_data: ndarray, The training data
        test_data: ndarray, The testing data
    
    """
    #Assert statements check that the types defined in the doctring are satisfied and sep_fact is between 0 and 1 
#     assert isinstance(sep_fact, (float, int))==True, "Separation factor must be a float or integer"
#     assert 0 <= sep_fact <= 1, "Separation factor must be between 0 and 1"
    
    #Shuffles Random Data
    #No shuffling for this test
    if shuffle_seed is not None:
        if run > 1:
            np.random.seed(run)
        else:
            np.random.seed(shuffle_seed)
    
    n = len(Xexp)
    
    if LSO:
        #Creates the index on which to split data
        train_enteries = int(len(all_data)*sep_fact)

        #Training and testing data are created and converted into tensors
        train_y =all_data[:train_enteries, -1] #1x(n*sep_fact)
        test_y = all_data[train_enteries:, -1] #1x(n-n*sep_fact)
        train_param = all_data[:train_enteries,:-1] #1x(n*sep_fact)
        test_param = all_data[train_enteries:,:-1] #1x(n-n*sep_fact)
    
    else:
        #Make every group of 5 points testing data
        test_ind = list(range(n*run,n*run+n))
    #     print(test_ind)

        #Training and testing data are created and converted into tensors
    #     print(all_data[:,-1])
        all_param = all_data[:,:-1]
        all_y = all_data[:,-1]
    #     print(all_y)
    #     print(all_param)

        train_y = np.delete(all_y,test_ind)
        train_param = np.delete(all_param,test_ind,axis = 0)
        test_y = all_data[test_ind[0]:test_ind[-1]+1, -1] #1x(n-n*sep_fact)
        test_param = all_data[test_ind[0]:test_ind[-1]+1,:-1] #1x(n-n*sep_fact)
    
    train_data = np.column_stack((train_param, train_y))
    test_data = np.column_stack((test_param, test_y))
    
    return torch.tensor(train_data),torch.tensor(test_data)
 
def LOO_eval_GP(train_y, model, likelihood, verbose, set_lengthscale):
    """
    Evaluates GP
    
    Parameters:
    -----------
        train_y: tensor or ndarray, The training y data
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        verbose: True/False: Determines whether z_term, ei_term_1, ei_term_2, CDF, and PDF terms are saved
        set_lengthscale: float/None: Determines whether Hyperparameter values will be set
    
    Returns:
    --------
 
    """
    #Find Number of training points    
    assert isinstance(train_y, np.ndarray) or torch.is_tensor(train_y) == True, "Train_sse must be ndarray or torch.tensor"
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    ##Set Hyperparameters to 1
    if isinstance(train_y, np.ndarray)==True:
        train_y = torch.tensor(train_y) #1xn
    
    #Set hyperparameters
    if set_lengthscale is not None:
        if verbose == True:
            print("Lengthscale Set To: " + set_lengthscale)
        outputscale = torch.tensor([1])
        lengthscale = torch.tensor([set_lengthscale])
        noise = torch.tensor([0.1])

        model.likelihood.noise = noise
        model.covar_module.base_kernel.lengthscale =lengthscale
        model.covar_module.outputscale = outputscale
    
    model.eval()
    #Puts likelihood in evaluation mode
    likelihood.eval()
    
    return

def set_eval_point():
    "Set the Eval point used in LOO_Analysis based off number of dimensions"
    
def LOO_Analysis(train_p,train_y, Theta_True, train_iter, Xexp, Yexp, noise_std, obj, run, sparse_grid, emulator, set_lengthscale, verbose = False,save_fig=False, runs = 1, DateTime=None, test_p = None, LOO = True, LSO = False):
    """
    Performs BO iterations
    
    Parameters:
    -----------
        train_p: tensor or ndarray, The training parameter space data
        train_y: tensor or ndarray, The training y data
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        Theta_True: ndarray, The array containing the true values of Theta1 and Theta2
        train_iter: int, number of training iterations to run. Default is 300
        Xexp: ndarray, The list of xs that will be used to generate y
        Yexp: ndarray, The experimental data for y (the true value)
        noise_std: float, int: The standard deviation of the noise
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        run: int, The iteration of the number of times new training points have been picked
        sparse_grid: True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        emulator: True/False, Determines if GP will model the function or the function error
        set_lengthscale: float or None, Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        verbose: True/False, Determines whether z_term, ei_term_1, ei_term_2, CDF, and PDF terms are saved, Default = False
        save_fig: True/False, Determines whether figures will be saved
        runs: int, # of total runs
        DateTime: None or bool, Determines whether the date and time will be used to save figures
        test_p: ndarray, test points for GP
        LOO: bool, Determines whether leave one out is a true LOO plot or has random values that are left out
        LSO: bool, determines whether 1 set of training points or multiple training points are left out
   
        
    Returns:
    --------
    
    """
    #Assert Statments
    assert all(isinstance(i, int) for i in [train_iter]), "BO_iters and train_iter must be integers"
    assert len(train_p) == len(train_y), "Training data must be the same length"
    assert len(Xexp) == len(Yexp), "Experimental data must have the same length"
    assert obj == "obj" or obj == "LN_obj", "Objective function choice, obj, MUST be sse or LN_sse"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    assert emulator==True or emulator==False, "Verbose must be True/False"
    
    #Find parameters
    m = Xexp[0].size #Dimensions of X
    n = len(Xexp) #Length of experimental data
    q = len(Theta_True) #Number of parameters to regress
    t = len(train_p) #Original length of training data

    #Ensures GP will take correct # of inputs
    if emulator == True:
        GP_inputs = q+m
        assert len(train_p.T) ==q+m, "train_p must have the same number of dimensions as the value of q+m"
    else:
        GP_inputs = q
        assert len(train_p.T) ==q, "train_p must have the same number of dimensions as the value of q"

    #Redefine likelihood and model based on new training data
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_p, train_y, likelihood)

    #Train GP
    train_GP = train_GP_model(model, likelihood, train_p, train_y, train_iter, verbose=False)

    #Evaluate GP
    eval_components = LOO_eval_GP(train_y, model, likelihood, verbose, set_lengthscale)

    #Create LOO Plots
    X_space = np.linspace(-2,2,50)
    y_sim = np.zeros(len(X_space))
    GP_mean = np.zeros(len(X_space))
    GP_stdev = np.zeros(len(X_space))
    test_y = np.zeros(len(Xexp))
    SSE_GP_Analy = 0

        
    for k in range(len(X_space)):
        if LOO == True and LSO == False:
            point = [test_p[0,0],test_p[0,1],X_space[k]]
            calc_eval_point = np.array([test_p[0,0],test_p[0,1]])
        else:
            point = [1, -1, X_space[k]]
            calc_eval_point = np.array([1,-1])
            
        eval_point = np.array([point])
        
        y_sim[k] = create_y_data(eval_point[0])
        GP_mean[k] = calc_GP_outputs(model, likelihood, eval_point[0:1])[3]
        GP_stdev[k] = calc_GP_outputs(model, likelihood, eval_point[0:1])[1]
        
        SSE_GP_Analy += (y_sim[k] - GP_mean[k])**2

    for k in range(len(Xexp)):
        if LOO == True and LSO == False:
            point = [test_p[0,0],test_p[0,1],Xexp[k]]
            calc_eval_point = np.array([test_p[0,0],test_p[0,1]])
        else:
            point = [1, -1, Xexp[k]]
            calc_eval_point = np.array([1,-1]) 
            
        eval_point = np.array([point])
#             test_y[k] = calc_GP_outputs(model, likelihood, eval_point[0:1])[3]
        test_y = calc_y_exp(calc_eval_point, Xexp, noise_std, noise_mean=0,random_seed=6)

    Theta = np.array([point[0:2]])[0]
    if verbose == True:
        print("Showing X/Y Plot for Theta = ",Theta , "with Xexp =", Xexp)
        print("SSE_Total is", SSE_GP_Analy)
         
        plot_3GP_performance(X_space, y_sim, GP_mean, GP_stdev, Theta, Xexp, test_p = test_p, test_y = test_y, verbose=verbose)

    return Theta, SSE_GP_Analy

def LSO_LOO_Analysis(theta_mesh,Theta_True,train_iter,explore_bias, Xexp, Yexp, noise_std, obj, sparse_grid, emulator,set_lengthscale, len_data, verbose = False, save_fig = False, shuffle_seed = None, DateTime=None, LOO = True, LSO = False):
    """
    Performs BO iterations with runs. A run contains of choosing different initial training data.
    
    Parameters:
    -----------
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        Theta_True: ndarray, The array containing the true values of Theta1 and Theta2
        train_iter: int, number of training iterations to run. Default is 300
        explore_bias: float,int,tensor,ndarray (1 value) The exploration bias parameter
        Xexp: ndarray, The list of xs that will be used to generate y
        Yexp: ndarray, The experimental data for y (the true value)
        noise_std: float, int: The standard deviation of the noise
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        sparse_grid: Determines whether a sparse grid or approximation is used for the GP emulator
        emulator: True/False, Determines if GP will model the function or the function error
        set_lengthscale: float or None, Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        len_data: int, original number of data points in the document
        verbose: True/False, Determines whether z_term, ei_term_1, ei_term_2, CDF, and PDF terms are saved, Default = False
        save_fig: True/False, Determines whether figures will be saved
        shuffle_seed, int, number of seed for shuffling training data. Default is None.  
        DateTime: None or bool, Determines whether the date and time will be used to save figures
        LOO: Bool, Determines whether leave one out is a true LOO plot or has random values that are left out
        LSO: bool, determines whether 1 set of training points or multiple training points are left out
        
    Returns:
    --------
        None
    
    """
    #Assert statements
    assert all(isinstance(i, int) for i in [train_iter]), "BO_iters, t, runs, and train_iter must be integers"
    assert len(Xexp) == len(Yexp), "Experimental data must have the same length"
    assert obj == "obj" or obj == "LN_obj", "Objective function choice, obj, MUST be sse or LN_sse"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    assert emulator==True or emulator==False, "Verbose must be True/False"
    
    
    #Find constants
    m = Xexp[0].size #Dimensions of X
    q = len(Theta_True) #Number of parameters to regress
    p = theta_mesh.shape[1] #Number of training points to evaluate in each dimension of q
    n = len(Xexp)
    BO_iters = 1
#     sep_facts = np.linspace(0.05,0.95,19)
    sep_facts = [0.95, 0.8, 0.7, 0.6, 0.5]
    ln_SSE_GP_Analy_List = []
    dim = m+q #dimensions in a CSV
    
    #Read data from a csv
    
    all_data_doc = find_train_doc_path(emulator, obj, len_data)
    all_data = np.array(pd.read_csv(all_data_doc, header=0,sep=","))   
    
    if LOO==True and LSO == False:
        runs = 20
    else: 
        runs = len(sep_facts)
    
    #Loop over # runs
    for i in range(runs): #Using number of runs to do the Leave one out analysis
        #Create training/testing data
        if LSO:
            sep_fact = sep_facts[i] 
        else:
            sep_fact = 0.8
        
        if verbose == True:
            print("Train/Test Separation Factor",sep_fact)
            
        if LOO:
            train_data, test_data = LOO_test_train_split(all_data, Xexp, sep_fact, i, shuffle_seed, LSO)    
        else:
            train_data, test_data = test_train_split(all_data, sep_fact, runs, shuffle_seed)
            
        if emulator:
            train_p = train_data[:,1:(q+m+1)]
            test_p = test_data[:,1:(q+m+1)]
            assert len(train_p.T) ==q+m, "train_p must have the same number of dimensions as the value of q+m"
            
        else:
            train_p = train_data[:,1:(q+1)]
            test_p = test_data[:,1:(q+1)]
            assert len(train_p.T) ==q, "train_p must have the same number of dimensions as the value of q"
            
        train_y = train_data[:,-1]
        assert len(train_p) == len(train_y), "Training data must be the same length"
                          
#         plot_org_train(theta_mesh,train_p,Theta_True)
        plot_org_train(theta_mesh,train_p, test_p, Theta_True, emulator, sparse_grid, obj, explore_bias, set_lengthscale, i, save_fig, BO_iters, runs, DateTime, verbose)

        #Run BO iteration
        Theta, SSE_for_sep_fact = LOO_Analysis(train_p, train_y, Theta_True, train_iter, Xexp, Yexp, noise_std, obj, i, sparse_grid, emulator, set_lengthscale, verbose, save_fig, runs, DateTime, test_p, LOO = LOO, LSO = LSO)
        ln_SSE_GP_Analy_List.append(np.log(SSE_for_sep_fact))
        
    if LSO==True:
        plt.figure()
        plt.plot(sep_facts, ln_SSE_GP_Analy_List, label = "ln(SSE)")
        plt.legend(loc = "best")
        plt.xlabel("Separation Factor")
        plt.ylabel("ln(SSE)")
        plt.title("Separation Factor Analysis at Theta = " +str(Theta))
        plt.grid(True)
        plt.show()
        
    return 