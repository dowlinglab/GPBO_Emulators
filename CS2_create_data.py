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
from normalize import normalize_p_true, normalize_x, normalize_p_set, normalize_p_data, normalize_constants

def calc_muller(x, model_coefficients, noise = 0):
    """
    Caclulates the Muller Potential
    
    Parameters
    ----------
        x: ndarray, Values of X
        model_coefficients: ndarray, The array containing the values of Muller constants
        noise: ndarray, Any noise associated with the model calculation
    
    Returns:
    --------
        y_mul: float, value of Muller potential
    """
    x = clean_1D_arrays(x)
    if len(x.shape) > 1:
        X1, X2 = x
    else:
        X1 = x[0], X2 = x[1]
    A, a, b, c, x0, y0 = model_coefficients
    Term1 = a*(X1 - x0)**2
    Term2 = b*(X1 - x0)*(X2 - y0)
    Term3 = c*(X2 - y0)**2
    y_mul = np.sum(A*np.exp(Term1 + Term2 + Term3) ) + noise
    return y_mul

def calc_y_exp(true_model_coefficients, x, noise_std, noise_mean=0,random_seed=9):
    """
    Creates y_data (Muller Potential) for the 2 input GP function
    
    Parameters
    ----------
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
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
    
    x = clean_1D_arrays(x)
    len_x = x.shape[0]
    
#     print(len_x)
    #Seed Random Noise (For Bug Testing)
    if random_seed != None:
        assert isinstance(random_seed,int) == True, "Seed number must be an integer or None"
        np.random.seed(random_seed)
        
    #Creates noise values with a certain stdev and mean from a normal distribution
    noise = np.random.normal(size= 1 ,loc = noise_mean, scale = noise_std) #1x n_x
    
    # True function is Muller Potential
    
    y_exp = np.zeros(len_x)
    
    for i in range(len_x):
#         print(true_model_coefficients.shape)
        y_exp[i] = calc_muller(x[i], true_model_coefficients, noise)
  
    return y_exp

def create_sse_data(param_space, x, y_exp, true_model_coefficients, obj = "obj", skip_param_types = 0):
    """
    Creates y_data for the 2 input GP function
    
    Parameters
    ----------
        param_space: ndarray, The array containing the data for Theta1 and Theta2
        x: ndarray, The list of xs that will be used to generate y
        y_exp: ndarray, The experimental data for y (the true value)
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed
        
    Returns:
        sum_error_sq: ndarray, The SSE or ln(SSE) values that the GP will be trained on
    """  
#     print(x)
#     print(skip_param_types)
    if isinstance(param_space, pd.DataFrame):
        param_space = param_space.to_numpy()

    #Will need assert statement
    assert obj == "obj" or obj == "LN_obj", "Objective function choice, obj, MUST be sse or LN_sse"
    
    x = clean_1D_arrays(x)
    param_space = clean_1D_arrays(param_space, param_clean = True)
    len_x, dim_x = x.shape[0], x.shape[1]
    
#     len_data, dim_data = param_space.shape[1], param_space.shape[0] 
#     print(len_data, dim_data)
#     print(param_space.shape)
    len_data, dim_data = param_space.shape[0], param_space.shape[1]
#     print(len_data, dim_data)
    dim_param = dim_data
#     print(true_model_coefficients)
    try:
        num_constant_type, len_constants = true_model_coefficients.shape[0], true_model_coefficients.shape[1] # 6,4
    except:
        true_model_coefficients = clean_1D_arrays(true_model_coefficients)
        num_constant_type, len_constants = true_model_coefficients.shape[0], true_model_coefficients.shape[1]
#         print(true_model_coefficients, true_model_coefficients.shape)
    num_param_type_guess = int(dim_param/len_constants)
        
    #For the case where more than 1 point is geing generated
    #Creates an array for train_sse that will be filled with the for loop
    sum_error_sq = torch.tensor(np.zeros(len_data)) #1 x n_train^2 
    model_coefficients = true_model_coefficients.copy()
    
    #Iterates over evey combination of theta to find the SSE for each combination
    for i in range(len_data):
        #Set dig out values of a from train_p
        #Set constants to change the a row to the index of the first loop
        for j in range(num_param_type_guess):
            j_model = skip_param_types + j
            model_coefficients[j_model] = param_space[i][len_constants*j: len_constants*(j+1)]
        
        y_sim = np.zeros(len_x)
        #Loop over state points (5)
        for k in range(len_x):
#             print(x[k])
            y_sim[k] = calc_muller(x[k], model_coefficients)
                
        if obj == "obj":
            sum_error_sq[i] = sum((y_sim - y_exp)**2) #Scaler
#                 print(sum_error_sq[i])
        else:
            sum_error_sq[i] = np.log(sum((y_sim - y_exp)**2)) #Scaler
    
    return sum_error_sq


def create_y_data(param_space, true_model_coefficients, x, skip_param_types = 0, noise_std=None, noise_mean=0,random_seed=9):
    """
    Creates y_data (training data) based on the function theta_1*x + theta_2*x**2 +x**3
    Parameters
    ----------
        param_space: (nx3) ndarray or tensor, parameter space over which the GP will be run
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
        x: ndarray, Array containing x data
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed
        noise_std: float, int: The standard deviation of the noise
        noise_mean: float, int: The mean of the noise
        random_seed: int: The random seed
    Returns
    -------
        y_sim: ndarray, The simulated y training data
    """
    #Assert statements check that the types defined in the doctring are satisfied
    
    #Converts parameters to numpy arrays if they are tensors
    if torch.is_tensor(param_space)==True:
        param_space = param_space.numpy()
        
    if isinstance(param_space, pd.DataFrame):
        param_space = param_space.to_numpy()
    
    if random_seed != None:
        assert isinstance(random_seed,int) == True, "Seed number must be an integer or None"
        np.random.seed(random_seed)
        
    #Creates noise values with a certain stdev and mean from a normal distribution
    if noise_std != None:
        noise = np.random.normal(size= 1 ,loc = noise_mean, scale = noise_std) #1x n_x
    else:
        noise = np.random.normal(size= 1 ,loc = noise_mean, scale = 0) #1x n_x
    
    param_space = clean_1D_arrays(param_space, param_clean = True) 
    x = clean_1D_arrays(x) 
    len_data, dim_data = param_space.shape[0], param_space.shape[1] #300, 10
#     print(len_data, dim_data)
    dim_x = x.shape[1] # 2
    dim_param = dim_data - dim_x
#     print(len_data, dim_data, dim_x, dim_param)
    
    num_constant_type, len_constants = true_model_coefficients.shape[0], true_model_coefficients.shape[1] # 6,4
    num_param_type_guess = int(dim_param/len_constants)
#     print(num_param_type_guess)
        
    #For the case where more than 1 point is geing generated
    #Creates an array for train_sse that will be filled with the for loop
    #Initialize y_sim
    y_sim = np.zeros(len_data) #1 x n_train^2
    model_coefficients = true_model_coefficients.copy()

    #Iterates over evey data point to find the y for each combination
    for i in range(len_data):
        #Set dig out values of a from train_p
        #Set constants to change the a row to the index of the first loop

        #loop over number of param types (A, a, b c, x0, y0)
        for j in range(num_param_type_guess):
            j_model = skip_param_types + j
            model_coefficients[j_model] = param_space[i][len_constants*j: len_constants*(j+1)]
#         print(model_coefficients)
#         print(model_coefficients)
        A, a, b, c, x0, y0 = model_coefficients         
        #Calculate y_sim
        x = param_space[i][dim_param:dim_data]
#         print(x,x.shape)
        y_sim[i] = calc_muller(x, model_coefficients, noise)
   
    return y_sim

def gen_y_Theta_GP(x_space, Theta, true_model_coefficients, skip_param_types = 0, norm_scalers = None):
# def gen_y_Theta_GP(x_space, Theta):
    """
    Generates an array of Best Theta Value and X to create y data
    
    Parameters
    ----------
        x_space: ndarray, array of x value
        Theta: ndarray, Array of parameter values
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
        x: ndarray, Array containing x data
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed
        norm_scalers: None or list of MinMaxScaler(), if data is being normalized, the scalers used to normalize the data. Default None
           
    Returns
    -------
        create_y_data_space: ndarray, array of parameters [Theta, x] to be used to generate y data
        
    """
    x_space = clean_1D_arrays(x_space)
    
    m = x_space.shape[1]
    q = Theta.shape[0]
    
    #Unscale Data
    if norm_scalers is not None:
        norm = False
        m = x_space.shape[0]
        scaler_x, scaler_theta, scaler_C_before, scaler_C_after = norm_scalers
        true_model_coefficients_unscl = normalize_constants(Constants, p_true, scaler_theta, skip_params, CS, norm, scaler_C_before, scaler_C_after)[0]
        Theta_unscl = normalize_p_set(clean_1D_arrays(Theta, param_clean = True), scaler_theta, norm)[0]
        x_space_unscl = normalize_x(x_space, m, x_space, emulator, norm, scaler_x)[0]
        Theta_use = Theta_unscl
        x_space_use = x_space_unscl
    else:
        Theta_use = Theta
        x_space_use = x_space
        true_model_coefficients_use = true_model_coefficients_unscl
    
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
            create_y_data_space[i,j] = Theta_use[j]
#         print(create_y_data_space)
        create_y_data_space[i,q:] = x_space_use[i,:]
#     print(create_y_data_space)
    #Generate y data based on parameters
#     y_GP_Opt_data = create_y_data(create_y_data_space)
    y_GP_Opt_data = create_y_data(create_y_data_space, true_model_coefficients_use, x_space_use, skip_param_types = skip_param_types)
    return y_GP_Opt_data   

def eval_GP_emulator_BE(Xexp, Yexp, train_p, true_model_coefficients, emulator = True, obj = "obj", skip_param_types = 0, norm_scalers = None):
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
        norm_scalers: None or list of MinMaxScaler(), if data is being normalized, the scalers used to normalize the data. Default None
    Returns
    -------
        best_error: float, the best error of the 3-Input GP model
    """
    #Asserts that inputs are correct
    assert len(Xexp)==len(Yexp), "Experimental data must have same length"
    
    Xexp_shape = clean_1D_arrays(Xexp)
    n,m = np.shape(Xexp)
    q = train_p.shape[1] - m

    t_train = len(train_p)
    true_p_shape = np.zeros(q)

    #Unscale Data for data generation
    if norm_scalers is not None:
#         print(norm_scalers)
        norm = False
        CS = 2.2
        
        scaler_x, scaler_theta, scaler_C_before, scaler_C_after = norm_scalers
        true_model_coefficients_unscl = normalize_constants(true_model_coefficients, true_p_shape, scaler_theta, skip_param_types, CS, norm, scaler_C_before, scaler_C_after)[0]
        train_p_unscl = normalize_p_data(train_p, scaler_theta, norm)[0]
        Xexp_unscl = normalize_x(Xexp, m, Xexp, emulator, norm, scaler_x)[0]
        train_p_use = train_p_unscl
        Xexp_use = Xexp_unscl
        true_model_coefficients_use = true_model_coefficients_unscl
    else:
        train_p_use = train_p
        Xexp_use = Xexp
        true_model_coefficients_use = true_model_coefficients
    
#     print(Xexp_use)
    SSE = np.zeros(t_train)
    for i in range(t_train):
        SSE[i] = create_sse_data(train_p_use[i], Xexp_use, Yexp, true_model_coefficients_use, obj = obj, skip_param_types = skip_param_types)

    #Define best_error as the minimum SSE or ln(SSE) value
    best_error = np.amin(SSE)
    
    return best_error


def make_next_point(train_p, train_y, theta_b, Xexp, Yexp, emulator, true_model_coefficients, obj, dim_param, skip_param_types = 0, noise_std = None, norm_scalers = None):
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
    #Note train_p is already unscaled
    #Unscale for Data Generation
    q = train_p.shape[1] - Xexp.shape[1]
    true_p_shape = np.zeros(q)
    
    if norm_scalers is not None:
#         print(norm_scalers)
        norm = False
        CS = 2.2
        m = Xexp.shape[0]
        scaler_x, scaler_theta, scaler_C_before, scaler_C_after = norm_scalers
        theta_b_unscl = normalize_p_true(theta_b, scaler_theta, norm)
        Xexp_unscl = normalize_x(Xexp, m, Xexp, emulator, norm, scaler_x)[0]
        true_model_coefficients_unscl = normalize_constants(Constants, true_p_shape, scaler_theta, skip_param_types, CS, norm, scaler_C_before, scaler_C_after)[0]
        theta_b_use = theta_b_unscl
        Xexp_use = Xexp_unscl
        true_model_coefficients_use = true_model_coefficients_unscl
    else:
        theta_b_use = theta_b
        Xexp_use = Xexp
        true_model_coefficients_use = true_model_coefficients
        
    n = Xexp.shape[0]
    if emulator == False:   
        #Call the expensive function and evaluate at Theta_Best
#             print(theta_b.shape)
        sse_Best = create_sse_data(theta_b_use, Xexp_use, Yexp, true_model_coefficients_use, obj, skip_param_types)
#             print(sse_Best)
        #Add Theta_Best to train_p and y_best to train_y
        train_p = np.concatenate((train_p, [theta_b_use]), axis=0) #(q x t)
#             print(train_y.shape, sse_Best)
        train_y = np.concatenate((train_y, sse_Best),axis=0) #(1 x t)
#             print(train_y.shape, sse_Best.shape)      

    else:
        #Loop over experimental data
#             print(Xexp)
        for k in range(n):
            Best_Point = theta_b
            Best_Point = np.append(Best_Point, Xexp_use[k])
            #Create y-value/ experimental data ---- #Should use calc_y_exp correct? create_y_sim_exp
#                 y_Best = calc_y_exp(theta_b, Xexp[k].reshape((1,-1)), noise_std, noise_mean=0,random_seed=6)
            #Adding the noise creates experimental data at theta_b using create_y_data
            y_Best = create_y_data(Best_Point, true_model_coefficients_use, Xexp_use[k].reshape((1,-1)), skip_param_types, noise_std)       
            train_p = np.append(train_p, [Best_Point], axis=0) #(q x t)
            train_y = np.append(train_y, [y_Best]) #(1 x t)
#                 print(train_p.shape, train_y.shape)
    return train_p, train_y