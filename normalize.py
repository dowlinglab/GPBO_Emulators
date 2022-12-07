import random
import numpy as np
import torch
from bo_functions_generic import clean_1D_arrays, norm_unnorm

def normalize_general(train_p, test_p, Xexp, theta_set, Theta_True, true_model_coefficients, emulator, skip_param_types, case_study):
    """
    normalizes many constants at once
    
    Parameters:
    -----------
        train_p: tensor or ndarray, The training parameter space data
        test_p: tensor, or ndarray, The testing parameter space data
        Xexp: ndarray, The list of xs that will be used to generate y
        theta_set: ndarray (len_set x dim_param), array of Theta values
        Theta_True: ndarray, The array containing the true values of theta values (must be 1D)
        true_model_coefficients: ndarray, The array containing the true values of the constants (may be same as theta_true)
        emulator: bool, Determines if GP will model the function or the function error
        skip_param_types: int, The offset of which parameter types (A - y0) that are being guessed. Default 0
        case_study: float, the number of the case study to be evaluated. Default is 1, other option is 2.2
        
    Returns:
    --------
         norm_vals: list of ndarray/tensors, The normalized values of train_p, test_p, Xexp, theta_set, Theta_True, and true_model_coefficients
         norm_scalers: list of MinMaxScaler(), The scalers associated with x, theta, and the constants
       
    """
    #Define norm as true: This function only normalizes data
    norm = True
    
    #Define dimensionality of X
    m = Xexp.shape[1]
#     print(type(test_p), type(train_p))

    #Clone training and testing data
    if torch.is_tensor(train_p) == True or torch.is_tensor(test_p):
        train_p_scl = train_p.detach().clone()
        test_p_scl = test_p.detach().clone()
    else:
        train_p_scl = train_p.copy()
        test_p_scl = test_p.copy()
    
    if emulator == True:
    #Overwrite x_data in train_p w/ normalized values
#         print(type(test_p[:,-m:]), type(train_p[:,-m:]))
        train_p_scl[:,-m:], scaler_x = normalize_x(train_p[:,-m:], m, Xexp, emulator, norm)
        #Normalize Xexp data
        Xexp_scl = normalize_x(Xexp, m, Xexp, emulator, norm, scaler_x)[0]
        #Normalize train_p data
        train_p_scl[:,0:-m], scaler_theta = normalize_p_data(train_p, m, emulator, norm)
        #normalize testing data if there is any
        if test_p.shape[0] >= 1:
            test_p_scl[:,-m:] = normalize_x(test_p[:,-m:], m, Xexp, emulator, norm, scaler_x)[0]
            test_p_scl[:,0:-m] = normalize_p_data(test_p, m, emulator, norm, scaler_theta)[0]
    else:
        #Normalize x data with Xexp if there is no X training data
        Xexp_scl, scaler_x = normalize_x(Xexp, m, Xexp, emulator, norm)
        #Normalize train_p data
        train_p_scl, scaler_theta = normalize_p_data(train_p, m, emulator, norm)
        #normalize testing data if there is any
        if test_p.shape[0] >= 1:
            test_p_scl = normalize_p_data(test_p, m, emulator, norm)

    #Overwrite theta values with normalized values in theta_set, p_true, and constants
    theta_set_scl = normalize_p_set(theta_set, scaler_theta, norm)
    Theta_True_scl = normalize_p_true(Theta_True, scaler_theta, norm)
#     print(true_model_coefficients, Theta_True)
    true_model_coefficients_scl, scaler_C_before, scaler_C_after  = normalize_constants(true_model_coefficients, Theta_True,
                                                                                    scaler_theta, skip_param_types,
                                                                                    case_study, norm)
    #Define list of normalized values and scalers used to normalize the values and return them
    norm_vals_and_scalers = [train_p_scl, test_p_scl, Xexp_scl, theta_set_scl, Theta_True_scl, true_model_coefficients_scl, scaler_x, scaler_theta, scaler_C_before, scaler_C_after]
    norm_vals = norm_vals_and_scalers[:6]
    norm_scalers = norm_vals_and_scalers[6:]
    
    #norm_vals, norm_scalers = np.split(norm_vals_and_scalers, [6])
    
    return  norm_vals, norm_scalers

def normalize_x(x_vals, m, Xexp, emulator, norm = True, scaler = None):
    """
    Normalizes or unnormalizes x data from training/testing data and experimental data
    
    Parameters
    ----------
        x_vals: ndarray, x_values to normalize from training/testing data or Xexp when standard approaches are used
        m: int, dimensionality of x data
        Xexp: ndarraay, experimental x values
        emulator: bool, whether GP is emulating fxn or error
        norm: bool, whether the value will be normalized to 0 and 1 (True) or from 0 and 1 (False). Default True
        scaler: None or MinMaxScaler(), used to un-normalize data or normalize data based on another sets normalization
        
    Returns
    -------
        x_scl: ndarray, rescaled values of x
        scaler_x: MinMaxScaler(), scaler used to obtain these values
    """
    #Change 1D array to 2S with shape (len(X),1)
    x_vals = clean_1D_arrays(x_vals)
    if emulator == True:
        #If using emulator approach, scale x data using x training data
        x_scl, scaler_x = norm_unnorm(x_vals, norm, scaler)
#         x_scl, scaler_x = norm_unnorm(x_vals[:,-m:], norm, scaler)
    else:
        #If not using emulator approach, scale x data using experimental x data
        x_scl, scaler_x = norm_unnorm(Xexp, norm, scaler)
    return x_scl, scaler_x

def normalize_p_data(param_vals_data, m, emulator, norm = True, scaler = None):
    """
    Normalizes or unnormalizes parameter data from training/testing data
    
    Parameters
    ----------
        param_vals_data: ndarray, parameter values to normalize from training/testing data
        m: int, dimensionality of x data
        emulator: bool, whether GP is emulating fxn or error
        norm: bool, whether the value will be normalized to 0 and 1 (True) or from 0 and 1 (False). Default True
        scaler: None or MinMaxScaler(), used to un-normalize data or normalize data based on another sets normalization
        
    Returns
    -------
        param_data_scl: ndarray, rescaled values of x
        scaler_theta: MinMaxScaler(), scaler used to obtain these values
    """
    if emulator == True:
        if norm == True:
            #If using emulator approach and normalizing data, overwrite normal parameter values with scaled values
            param_data_scl, scaler_theta = norm_unnorm(param_vals_data[:,0:-m], norm, scaler)
        else:
            #If using emulator approach and un-normalizing data, overwrite scaled values with normal values
            param_data_scl, scaler_theta = norm_unnorm(param_vals_data, norm, scaler)
    else:
         #If using standard approach overwrite scaled/normal values with normal/scaled values
        param_data_scl, scaler_theta = norm_unnorm(param_vals_data, norm, scaler)
    return param_data_scl, scaler_theta

def normalize_p_set(p_set, scaler_theta, norm = True):
    """
    Normalizes or unnormalizes parameter data for theta_set
    
    Parameters
    ----------
        param_vals_data: ndarray, parameter values to normalize from training/testing data
        scaler_theta: None or MinMaxScaler(), used to (un)normalize data based on another sets normalization
        norm: bool, whether the value will be normalized to 0 and 1 (True) or from 0 and 1 (False). Default True
           
    Returns
    -------
        p_set_scl: ndarray, rescaled values of x
    """
    #Normalize or unnormalize set data
    p_set_scl, scaler_theta = norm_unnorm(p_set, norm, scaler = scaler_theta)
    return p_set_scl

def normalize_p_true(p_true, scaler_theta, norm= True):
    """
    Normalizes or unnormalizes parameter data for theta_true
    
    Parameters
    ----------
        p_true: ndarray, True parameter values
        scaler_theta: None or MinMaxScaler(), used to (un)normalize data based on another sets normalization
        norm: bool, whether the value will be normalized to 0 and 1 (True) or from 0 and 1 (False). Default True
           
    Returns
    -------
        theta_true_scl.flatten(): ndarray, rescaled values of theta_true flattened to correct dimensions
    """    
    #Normalize or un normalize a 2D shape (1, len(theta_true) array and flatten
    theta_true_scl, scaler_theta = norm_unnorm(clean_1D_arrays(p_true, param_clean = True), 
                                               norm, scaler_theta)
    return theta_true_scl.flatten()

def normalize_constants(Constants, p_true, scaler_theta, skip_params, CS, norm = True, scaler_C_before = None, 
                        scaler_C_after = None):
    """
    Normalizes or unnormalizes data for constants
    
    Parameters
    ----------
        Constants: ndarray, ndarray, True values of Muller potential constants OR p_true (CS1 only)
        p_true: ndarray, True parameter values
        scaler_theta: None or MinMaxScaler(), used to (un)normalize data based on another sets normalization
        skip_params: int, number of sets of parameters in Constants to skip before reaching the first iterable parameter row
        CS: float, case study label 
        norm: bool, whether the value will be normalized to 0 and 1 (True) or from 0 and 1 (False). Default True
        scaler_C_before: None or MinMaxScaler(), used to un-normalize normalized constansts before theta constants
        scaler_C_after: None or MinMaxScaler(), used to un-normalize normalized constansts after theta constants
        
    Returns
    -------
        x_scl: ndarray, rescaled values of x
        scaler_x: MinMaxScaler(), scaler used to obtain these values
    """   
    #Determine number of parameter types and the length of each: Ex, Muller Case study has 6, and 2D case study has 1
    num_param_types = clean_1D_arrays(Constants).shape[1] #4 - Represents how many indecies are in each param type A, a, b, c, x0, y0
    len_param_type = int(len(p_true)/num_param_types)
    
    #For the case study, the constants are identical to theta_True and we scale them as such
    if CS == 1:
        Constants_scl = normalize_p_true(p_true, scaler_theta, norm)
    #For the Muller potential case studies, we need to normalize constants in chunks
    else:
        #Define constants before, after, and representing theta_true and normalize them by splitting the constants array is 3 parts
#         print(Constants)
        Constants_before, Constants_theta, Constants_after = np.split(Constants, [skip_params,len_param_type+1])
#         print(Constants_before.shape, Constants_theta.shape, Constants_after.shape)
        Constants_before_scl, scaler_C_before = norm_unnorm(Constants_before.T, norm, scaler_C_before)
        
        Constants_theta_scl, scaler_theta =  norm_unnorm(clean_1D_arrays(Constants_theta.flatten(), param_clean = True),
                                                 norm, scaler_theta)
        Constants_theta_scl = Constants_theta_scl.reshape((len_param_type,num_param_types)) #(2,4) for 2.2
        
        Constants_after_scl, scaler_C_after = norm_unnorm(Constants_after.T, norm, scaler_C_after)
        
        #Restack scaled constants and return their value and necessary scalers
        Constants_scl = np.vstack((Constants_before_scl.T, Constants_theta_scl, Constants_after_scl.T))
    return Constants_scl, scaler_C_before, scaler_C_after      