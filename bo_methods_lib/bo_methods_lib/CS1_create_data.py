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

from .bo_functions_generic import clean_1D_arrays
from .normalize import normalize_p_true, normalize_x, normalize_p_set, normalize_p_data

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
    #Clean train_T to shape (len(train_T),1) and y_exp to (len(y_exp),1)
    train_T = clean_1D_arrays(train_T)
    y_exp = clean_1D_arrays(y_exp)
    
    #Initialize sse matrix
    sum_error_sq = np.zeros(train_T.shape[0]) #1 x n_train^2

    #Iterates over evey combination of theta to find the SSE for each combination
    for i in range(train_T.shape[0]):
        #Theta 1 and theta 2 represented by columns for this caste study
        theta_1 = train_T[i,0] #n_train^2x1 
        theta_2 = train_T[i,1] #n_train^2x1
        #Calc y_sim
        y_sim = theta_1*x + theta_2*x**2 +x**3 #n_train^2 x n_x
#         y_sim = clean_1D_arrays(y_sim)
        if obj == "obj":
            sum_error_sq[i] = sum((y_sim - y_exp)**2) #Scaler
        else:
            sum_error_sq[i] = np.log(sum((y_sim - y_exp)**2)) #Scaler
            
#     sum_error_sq = torch.tensor(sum_error_sq)
    sum_error_sq = torch.from_numpy(sum_error_sq)
    
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
    
    #Clean train_T to shape (1, len(train_T)) and y_exp to (len(y_exp),1)
    train_T = clean_1D_arrays(train_T, param_clean = True)
#     print(train_T.shape)
    y_exp = clean_1D_arrays(y_exp)    

    #Creates an array for train_sse that will be filled with the for loop
    sum_error_sq = np.zeros((train_T.shape[0]))

    #Iterates over evey combination of theta to find the SSE for each combination
    #For each point in train_T
    for i in range(len(train_T)):
        #Theta 1 and theta 2 represented by columns for this case study
        theta_1 = train_T[i,0] #n_train^2x1 
        theta_2 = train_T[i,1] #n_train^2x1
        #Calc y_sim
        y_sim = theta_1*x + theta_2*x**2 +x**3 #n_train^2 x n_x
        #Clean y_sim and calculate sse or log(sse)
        y_sim = clean_1D_arrays(y_sim)
#         print(type(y_sim))
        if obj == "obj":
            sum_error_sq[i] = sum((y_sim - y_exp)**2) #Scaler
        else:
            sum_error_sq[i] = np.log(sum((y_sim - y_exp)**2)) #Scaler
    sum_error_sq = torch.from_numpy(sum_error_sq)
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
    
    #clean 1D_arrays to shape (1, len(param_space))
    param_space = clean_1D_arrays(param_space, param_clean = True)
    
    #Creates an array for train_data that will be filled with the for loop
    y_data = np.zeros(param_space.shape[0]) #1 x n (row x col)
    
    #Used when multiple values of y are being calculated
    #Iterates over evey combination of theta to find the expected y value for each combination
    for i in range(len(param_space)):
        #Theta1, theta2, and xexp are defined as coulms of param_space
        theta_1 = param_space[i,0] #nx1 
        theta_2 = param_space[i,1] #nx1
        x = param_space[i,2] #nx1 
        #Calculate y_data
        y_data[i] = theta_1*x + theta_2*x**2 +x**3 #Scaler
        #Returns all_y
    
    #Flatten y_data
    y_data = y_data.flatten()
    return y_data

def gen_y_Theta_GP(x_space, Theta, true_model_coefficients, skip_param_types = 0, norm_scalers = None, emulator = False):
# def gen_y_Theta_GP(x_space, Theta):
    """
    Generates an array of Best Theta Value and X to create y data
    
    Parameters
    ----------
        x_space: ndarray, array of x value
        Theta: ndarray, Array of theta values
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed. 
        norm_scalers: None or list of MinMaxScaler(), if data is being normalized, the scalers used to normalize the data. Default None
        emulator: bool, determines whether emulator approach is used
           
    Returns
    -------
        create_y_data_space: ndarray, array of parameters [Theta, x] to be used to generate y data
        
    """
    #clean x_space to shape (len(Xexp),1)
    x_space = clean_1D_arrays(x_space)
    m = x_space.shape[1]
    q = Theta.shape[0]
    
    #Unscale Data
    if str(norm_scalers) != "None":
        norm = False
        scaler_x, scaler_theta, scaler_C_before, scaler_C_after = norm_scalers
        Theta_unscl = normalize_p_set(clean_1D_arrays(Theta, param_clean = True), scaler_theta, norm)[0]
        x_space_unscl = normalize_x(x_space, None, norm, scaler_x)[0]
        Theta_use = Theta_unscl
        x_space_use = x_space_unscl
    else:
        Theta_use = Theta
        x_space_use = x_space
        
    #Define dimensions of GP (dim) and the number of experimental data (lenX), and initialize parameter matricies
    dim = q+m
    lenX = x_space.shape[0]
    create_y_data_space = np.zeros((lenX,dim))
    
    #Loop over # of x values
    for i in range(lenX):
        #Loop over number of dimensions
        for j in range(q):
            #Fill matrix to include all Theta and x parameters
            create_y_data_space[i,j] = Theta_use[j]
        create_y_data_space[i,q:] = x_space_use[i,:]
    #Generate y data based on parameters
    y_GP_Opt_data = create_y_data(create_y_data_space)
    return y_GP_Opt_data   

def eval_GP_emulator_BE(Xexp, Yexp, train_p, true_model_coefficients, emulator=True, obj = "obj", skip_param_types = 0, norm_scalers=None):
    """ 
    Calculates the best error of the emulator approach
    Parameters
    ----------
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        train_p: ndarray (d, p x p), training data
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
        obj: str, LN_obj or obj, determines whether log or regular objective function is calculated
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed
        norm_scalers: None or list of MinMaxScaler(), if data is being normalized, the scalers used to normalize the data. Default None
    Returns
    -------
        best_error: float, the best error of the 3-Input GP model
    """
    #Asserts that inputs are correct
    assert len(Xexp)==len(Yexp), "Experimental data must have same length"
    
    #Infer number of experimental data (n), number of parameters to regress (q), and length of training data (t)
    n,m = clean_1D_arrays(Xexp).shape
    q = len(true_model_coefficients) #In this case len(true_model_coefficients) == len(true_p)
    t_train = len(train_p)
    
    #Initialize SSE matrix
    SSE = np.zeros(t_train)
    
    #Unscale Data for data generation
    if str(norm_scalers) != "None":
#         print("norming...")
        norm = False
        train_p_unscl = train_p.clone()
        scaler_x, scaler_theta, scaler_C_before, scaler_C_after = norm_scalers
#         train_p_unscl = normalize_p_set(train_p, scaler_theta, norm)
        train_p_unscl[:,0:-m] = normalize_p_data(train_p[:,0:-m], m, emulator, norm, scaler_theta) 
        train_p_unscl[:,-m:] = normalize_x(Xexp, train_p[:,-m:], norm, scaler_x)[0]
        Xexp_unscl = normalize_x(Xexp, None, norm, scaler_x)[0]
        train_p_use = train_p_unscl
        Xexp_use = Xexp_unscl
    else:
        train_p_use = train_p
        Xexp_use = Xexp
      
    #Loop over each training point
    for i in range(t_train):
        #Caclulate SSE and save to list
        SSE[i] = create_sse_data(q,train_p_use[i], Xexp_use, Yexp, obj= obj) 

    #Define best_error as the minimum SSE or ln(SSE) value
    best_error = np.amin(SSE)
#     print(best_error)
    return best_error

def make_next_point(train_p, train_y, theta_b, Xexp, Yexp, emulator, true_model_coefficients, obj, dim_param, skip_param_types=0, noise_std=None, norm_scalers = None):
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
        norm_scalers: None or list of MinMaxScaler(), if data is being normalized, the scalers used to normalize the data. Default None
    
    Returns:
    --------
        train_p: tensor or ndarray, The training parameter space data with the augmented point
        train_y: tensor or ndarray, The training y data with the augmented point
    """
    
    #Infer dimensions (m) and length (n) of experimental data
    n,m = clean_1D_arrays(Xexp).shape
    #Unscale for Data Generation
    if str(norm_scalers) != "None":
        norm = False
        scaler_x, scaler_theta, scaler_C_before, scaler_C_after = norm_scalers
        theta_b_unscl = normalize_p_true(theta_b, scaler_theta, norm)
        Xexp_unscl = normalize_x(Xexp, None, norm, scaler_x)[0]
        theta_b_use = theta_b_unscl
        Xexp_use = Xexp_unscl
    else:
        theta_b_use = theta_b
        Xexp_use = Xexp  
    
    if emulator == False:   
        #Call the expensive function and evaluate at Theta_Best
        sse_Best = create_sse_data(dim_param,theta_b_use, Xexp_use, Yexp, obj) #(1 x 1)
#             print(sse_Best)
        #Add Theta_Best to train_p and y_best to train_y
        train_p = np.concatenate((train_p, [theta_b]), axis=0) #(q x t)
#             print(train_y.shape, sse_Best)
        train_y = np.concatenate((train_y, sse_Best),axis=0) #(1 x t)
#             print(train_y.shape, sse_Best.shape)      

    else:
        #Loop over experimental data
        for k in range(n):
            #Append state point to best value
            Best_Point = theta_b
            Best_Point = np.append(Best_Point, Xexp[k])
            #Create y-value/ experimental data ---- #Should use calc_y_exp correct? create_y_sim_exp   
            y_Best = calc_y_exp(theta_b_use, clean_1D_arrays(Xexp_use[k]), noise_std)
            train_p = np.append(train_p, [Best_Point], axis=0) #(q x t)
            train_y = np.append(train_y, [y_Best]) #(1 x t)

    return train_p, train_y