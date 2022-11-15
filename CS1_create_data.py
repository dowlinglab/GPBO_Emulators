import numpy as np
import math
from scipy.stats import norm
from scipy import integrate
import torch
import csv
import gpytorch
import scipy.optimize as optimize
from scipy.stats import qmc
import pandas as pd
import os
import Tasmanian
import itertools
from itertools import combinations_with_replacement
from itertools import combinations
from itertools import permutations

from bo_functions_generic import clean_1D_arrays

def calc_y_exp(Theta_True, x, noise_std, noise_mean=0,random_seed=6):
    """
    Creates y_data for the 2 input GP function
    
    Parameters
    ----------
        Theta_True: ndarray, The array containing the true values of Theta1 and Theta2
        x: ndarray, The list of xs that will be used to generate y
        noise_std: float, int: The standard deviation of the noise
        noise_mean: float, int: The mean of the noise
        random_seed: int: The random seed
        
    Returns:
        y_exp: ndarray, The expected values of y given x data
    """   
    
    #Asserts that test_T is a tensor with 2 columns
    assert isinstance(noise_std,(float,int)) == True, "The standard deviation of the noise must be an integer ot float."
    assert isinstance(noise_mean,(float,int)) == True, "The mean of the noise must be an integer ot float."
    assert len(Theta_True) ==2, "This function only has 2 unknowns, Theta_True can only contain 2 values."
    
    
    #Seed Random Noise (For Bug Testing)
    if random_seed != None:
        assert isinstance(random_seed,int) == True, "Seed number must be an integer or None"
        np.random.seed(random_seed)
        
    #Creates noise values with a certain stdev and mean from a normal distribution
    noise = np.random.normal(size=x.shape[1],loc = noise_mean, scale = noise_std) #1x n_x
    #     if isinstance(x, np.ndarray):
#         noise = np.random.normal(size=len(x),loc = noise_mean, scale = noise_std) #1x n_x
#     else:
#         noise = np.random.normal(size=1,loc = noise_mean, scale = noise_std) #1x n_x
    # True function is y=T1*x + T2*x^2 + x^3 with Gaussian noise
    y_exp =  Theta_True[0]*x + Theta_True[1]*x**2 +x**3 + noise #1x n_x #Put this as an input
  
    return y_exp

def create_sse_data_GP_val(q,train_T, x, y_exp, obj = "obj"): #Note - Broken
    """
    Creates y_data for the 2 input GP function
    
    Parameters
    ----------
        q: int, Number of parameters to be regressed
        train_T: ndarray, The array containing the training data for Theta1 and Theta2
        x: ndarray, The list of xs that will be used to generate y
        y_exp: ndarray, The experimental data for y (the true value)
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        
    Returns:
        sum_error_sq: ndarray, The SSE or ln(SSE) values that the GP will be trained on
    """   
    train_T = clean_1D_arrays(train_T)
    y_exp = clean_1D_arrays(y_exp)
    
    sum_error_sq = np.zeros(train_T.shape[0]) #1 x n_train^2

    #Iterates over evey combination of theta to find the SSE for each combination
    for i in range(train_T.shape[0]):
        theta_1 = train_T[i,0] #n_train^2x1 
        theta_2 = train_T[i,1] #n_train^2x1
        y_sim = theta_1*x + theta_2*x**2 +x**3 #n_train^2 x n_x
#         y_sim = clean_1D_arrays(y_sim)
        if obj == "obj":
            sum_error_sq[i] = sum((y_sim - y_exp)**2) #Scaler
        else:
            sum_error_sq[i] = np.log(sum((y_sim - y_exp)**2)) #Scaler
            
    sum_error_sq = torch.tensor(sum_error_sq)
    return sum_error_sq   

def create_sse_data(q,train_T, x, y_exp, obj = "obj"):
    """
    Creates y_data for the 2 input GP function
    
    Parameters
    ----------
        q: int, Number of parameters to be regressed
        train_T: ndarray, The array containing the training data for Theta1 and Theta2
        x: ndarray, The list of xs that will be used to generate y
        y_exp: ndarray, The experimental data for y (the true value)
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        
    Returns:
        sum_error_sq: ndarray, The SSE or ln(SSE) values that the GP will be trained on
    """   
    
    #Asserts that test_T is a tensor with 2 columns (May delete this)
    assert isinstance(q, int), "Number of inputs must be an integer"
#     print(train_T.T)    
    if torch.is_tensor(train_T)==True:
        assert len(train_T.permute(*torch.arange(train_T.ndim -1, -1, -1))) >=q, str("This is a "+str(q)+" input GP, train_T must have at least q columns of values.")
    else:
        assert len(train_T.T) >=q, str("This is a "+str(q)+" input GP, train_T must have at least q columns of values.")
    assert len(x) == len(y_exp), "Xexp and Yexp must be the same length"
    assert obj == "obj" or obj == "LN_obj", "Objective function choice, obj, MUST be sse or LN_sse"
    
    train_T = clean_1D_arrays(train_T)
    y_exp = clean_1D_arrays(y_exp)
    
#     try: 
    if train_T.shape[1] > 1: #For the case where more than 1 point is geing generated
        #Creates an array for train_sse that will be filled with the for loop
        sum_error_sq = torch.tensor(np.zeros(len(train_T))) #1 x n_train^2

        #Iterates over evey combination of theta to find the SSE for each combination
        for i in range(len(train_T)):
            theta_1 = train_T[i,0] #n_train^2x1 
            theta_2 = train_T[i,1] #n_train^2x1
            y_sim = theta_1*x + theta_2*x**2 +x**3 #n_train^2 x n_x
            y_sim = clean_1D_arrays(y_sim)
            if obj == "obj":
                sum_error_sq[i] = sum((y_sim - y_exp)**2) #Scaler
            else:
                sum_error_sq[i] = np.log(sum((y_sim - y_exp)**2)) #Scaler
#     except:
    else:
         #Creates a value for train_sse that will be filled with the for loop
        sum_error_sq = 0 #1 x n_train^2

        #Iterates over x to find the SSE for each combination
        theta_1 = train_T[0] #n_train^2x1 
        theta_2 = train_T[1] #n_train^2x1
        y_sim = theta_1*x + theta_2*x**2 +x**3 #n_train^2 x n_x
        if obj == "obj":
            sum_error_sq = sum((y_sim - y_exp)**2) #Scaler 
        else:
            sum_error_sq = np.log(sum((y_sim - y_exp)**2)) #Scaler 
    
    return sum_error_sq    

def create_y_data(param_space):
    """
    Creates y_data (training data) based on the function theta_1*x + theta_2*x**2 +x**3
    Parameters
    ----------
        param_space: (nx3) ndarray or tensor, parameter space over which the GP will be run
    Returns
    -------
        y_data: ndarray, The simulated y training data
    """
    #Assert statements check that the types defined in the doctring are satisfied
    assert len(param_space.T) >= 3, "Parameter space must have at least 3 parameters"
    
    #Converts parameters to numpy arrays if they are tensors
    if torch.is_tensor(param_space)==True:
        param_space = param_space.numpy()
        
    #Creates an array for train_data that will be filled with the for loop
    y_data = np.zeros(len(param_space)) #1 x n (row x col)
    
    try: #Used when multiple values of y are being calculated
        #Iterates over evey combination of theta to find the expected y value for each combination
        for i in range(len(param_space)):
            theta_1 = param_space[i,0] #nx1 
            theta_2 = param_space[i,1] #nx1
            x = param_space[i,2] #nx1 
            y_data[i] = theta_1*x + theta_2*x**2 +x**3 #Scaler
            #Returns all_y
    except:
        theta_1 = param_space[0] #nx1 
        theta_2 = param_space[1] #nx1
        x = param_space[2] #nx1 
        y_data = theta_1*x + theta_2*x**2 +x**3 #Scaler
    return y_data

def gen_y_Theta_GP(x_space, Theta, true_model_coefficients, skip_param_types = 0):
# def gen_y_Theta_GP(x_space, Theta):
    """
    Generates an array of Best Theta Value and X to create y data
    
    Parameters
    ----------
        x_space: ndarray, array of x value
        Theta: ndarray, Array of theta values
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
        x: ndarray, Array containing x data
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed
           
    Returns
    -------
        create_y_data_space: ndarray, array of parameters [Theta, x] to be used to generate y data
        
    """
    x_space = clean_1D_arrays(x_space)
    
    m = x_space.shape[1]
    q = Theta.shape[0]
    
    #Define dimensions and initialize parameter matricies
    dim = q+m
    lenX = len(x_space)
    create_y_data_space = np.zeros((lenX,dim))
    
    #Loop over # of x values
    for i in range(lenX):
        #Loop over number of theta values
        for j in range(q):
            #Fill matrix to include all Theta and x parameters
            create_y_data_space[i,j] = Theta[j]
#         print(create_y_data_space)
        create_y_data_space[i,q:] = x_space[i,:]
#     print(create_y_data_space)
    #Generate y data based on parameters
    y_GP_Opt_data = create_y_data(create_y_data_space)
#     y_GP_Opt_data = create_y_data(create_y_data_space, true_model_coefficients, x_space, skip_param_types = skip_param_types)
    return y_GP_Opt_data   

def eval_GP_emulator_BE(Xexp, Yexp, train_p, true_model_coefficients, obj = "obj", skip_param_types = 0):
    """ 
    Calculates the best error of the 3 input parameter GP
    Parameters
    ----------
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        train_p: ndarray (d, p x p), training data
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
        obj: str, LN_obj or obj, determines whether log or regular objective function is calculated
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed
    Returns
    -------
        best_error: float, the best error of the 3-Input GP model
    """
    #Asserts that inputs are correct
    assert len(Xexp)==len(Yexp), "Experimental data must have same length"
    
    n = len(Xexp)
    q = len(true_model_coefficients)
    
    t_train = len(train_p)
#     try:
#         t_train = len(train_p)
#     except:
#         print("train_p",train_p)
#         t_train = 1
#     print(true_model_coefficients)
#     print(true_model_coefficients.shape)
    #Will compare the rigorous solution and approximation later (multidimensional integral over each experiment using a sparse grid)
    SSE = np.zeros(t_train)
    for i in range(t_train):
        SSE[i] = create_sse_data(q,train_p[i], Xexp, Yexp, obj= obj) 
#         SSE[i] = create_sse_data(train_p[i], Xexp, Yexp, true_model_coefficients, obj = obj, skip_param_types = skip_param_types)

    #Define best_error as the minimum SSE or ln(SSE) value
    best_error = np.amin(SSE)
    
    return best_error

def make_next_point(train_p, train_y, theta_b, Xexp, Yexp, emulator, true_model_coefficients, obj, dim_param, skip_param_types=0, noise_std=None):
    """
    Augments the training data with the max(EI) point in parameter space at each iteration.
    
    Parameters:
    ----------
        train_p: tensor or ndarray, The training parameter space data
        train_y: tensor or ndarray, The training y data
        theta_b: ndarray, The point where the objective function is minimized in theta space
        Xexp: ndarray, The list of xs that will be used to generate y
        Yexp: ndarray, The experimental data for y (the true value)
        emulator: True/False, Determines if GP will model the function or the function error
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        dim_param: int, Number of parameters to be regressed
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed
        noise_std: float, int: The standard deviation of the noise
    
    Returns:
    --------
        train_p: tensor or ndarray, The training parameter space data with the augmented point
        train_y: tensor or ndarray, The training y data with the augmented point
    """
    n = Xexp.shape[0]
    #Make this a new function
    if emulator == False:   
        #Call the expensive function and evaluate at Theta_Best
#             print(theta_b.shape)
        sse_Best = create_sse_data(dim_param,theta_b, Xexp, Yexp, obj) #(1 x 1)
#             print(sse_Best)
        #Add Theta_Best to train_p and y_best to train_y
        train_p = np.concatenate((train_p, [theta_b]), axis=0) #(q x t)
#             print(train_y.shape, sse_Best)
        train_y = np.concatenate((train_y, sse_Best),axis=0) #(1 x t)
#             print(train_y.shape, sse_Best.shape)      

    else:
        #Loop over experimental data
#             print(Xexp)
        for k in range(n):
            Best_Point = theta_b
#                 print(theta_b, Theta_True)
            Best_Point = np.append(Best_Point, Xexp[k])
            #Create y-value/ experimental data ---- #Should use calc_y_exp correct? create_y_sim_exp       
            y_Best = calc_y_exp(theta_b, Xexp[k], noise_std)
            train_p = np.append(train_p, [Best_Point], axis=0) #(q x t)
            train_y = np.append(train_y, [y_Best]) #(1 x t)
#                 print(train_p.shape, train_y.shape)
    return train_p, train_y