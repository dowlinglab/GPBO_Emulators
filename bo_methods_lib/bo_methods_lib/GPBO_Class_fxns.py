import numpy as np
from scipy.stats import qmc
import pandas as pd
import bo_methods_lib
from .GPBO_Classes_New import Simulator

#Add your function here. SHould take theta_ref and x values
def calc_cs1_polynomial(true_model_coefficients, x):
    """
    Calculates the value of y for case study 1
    
    Parameters
    ----------
    true_model_coefficients: ndarray, The array containing the true values of Theta1 and Theta2
    x: ndarray, The list of xs that will be used to generate y
    
    Returns
    --------
    y_poly: ndarray, The noiseless values of y given theta_true and x
    """
    
    y_poly =  true_model_coefficients[0]*x + true_model_coefficients[1]*x**2 +x**3
    
    return y_poly

def calc_cs3_polynomial(true_model_coefficients, x):
    """
    Calculates the value of y for case study 1
    
    Parameters
    ----------
    true_model_coefficients: ndarray, The array containing the true values of Theta1 and Theta2
    x: ndarray, The list of xs that will be used to generate y
    
    Returns
    --------
    y_poly: ndarray, The noiseless values of y given theta_true and x
    """
    assert len(true_model_coefficients) == 5, "5 Coefficients"
    t1, t2, t3, t4, t5 = true_model_coefficients
    
    #If array is not 2D, give it shape (len(array), 1)
    if not len(x.shape) > 1:
        x = x.reshape(-1,1)

    assert x.shape[0] == 2, "Polynomial x_data must be 2 dimensional"
    x1, x2 = x #Split x into 2 parts by splitting the rows

    y_model =  t1*x1 + t2*x2 + t3*x1*x2 + t4*x1**2 + t5*x2**2
       
    return y_model

def calc_cs4_isotherm(true_model_coefficients, x):
    """
    Calculates the value of y for case study 1
    
    Parameters
    ----------
    true_model_coefficients: ndarray, The array containing the true values of Theta1 and Theta2
    x: ndarray, The list of xs that will be used to generate y
    
    Returns
    --------
    y_poly: ndarray, The noiseless values of y given theta_true and x
    """
    assert len(true_model_coefficients) == 4, "true_model_coefficients must be length 4"
    t1, t2, t3, t4 = true_model_coefficients
    
    #If array is not 2D, give it shape (len(array), 1)
    if not len(x.shape) > 1:
        x = x.reshape(-1,1)

    assert x.shape[0] == 2, "Isotherm x_data must be 2 dimensional"
    x1, x2 = x #Split x into 2 parts by splitting the rows

    y_model =  (t1*t2*x1)/(1+t2*x1) + (t3*t4*x2)/(1+t4*x2)
       
    return y_model

def calc_muller(model_coefficients, x):
    """
    Caclulates the Muller Potential
    
    Parameters
    ----------
        model_coefficients: ndarray, The array containing the values of Muller constants
        x: ndarray, Values of X
        noise: ndarray, Any noise associated with the model calculation
    
    Returns:
    --------
        y_mul: float, value of Muller potential
    """
    #Reshape x to matrix form
    #If array is not 2D, give it shape (len(array), 1)
    if not len(x.shape) > 1:
        x = x.reshape(-1,1)
        
    assert x.shape[0] == 2, "Muller Potential x_data must be 2 dimensional"
    X1, X2 = x #Split x into 2 parts by splitting the rows
    
    #Separate all model parameters into their appropriate pieces
    model_coefficients_reshape = model_coefficients.reshape(6, 4)
        
    #Calculate Muller Potential
    A, a, b, c, x0, y0 = model_coefficients_reshape
    term1 = a*(X1 - x0)**2
    term2 = b*(X1 - x0)*(X2 - y0)
    term3 = c*(X2 - y0)**2
    y_mul = np.sum(A*np.exp(term1 + term2 + term3) )
    
    return y_mul

#Define Simulator Class Helper
def simulator_helper_test_fxns(cs_name, indecies_to_consider, noise_mean, noise_std, normalize, seed):
    """
    Sets the model for calculating y based off of the case study identifier.

    Parameters
    ----------
    cs_name: Class, The name/enumerator associated with the case study being evaluated

    Returns
    -------
    calc_y_fxn: function, the function used for calculation is case study cs_name.name
    """
    #Note: Add your function name from GPBO_Class_fxns.py here
    #CS1
    if cs_name.value == 1:      
        theta_names = ['theta_1', 'theta_2']
        bounds_x_l = [-2]
        bounds_x_u = [2]
        bounds_theta_l = [-2, -2]
        bounds_theta_u = [ 2,  2]
        theta_ref = np.array([1.0, -1.0])     
        calc_y_fxn = calc_cs1_polynomial
        
    #CS2_4 to CS2_24
    elif 2 <= cs_name.value <= 7:                          
        theta_names = ['A_1', 'A_2', 'A_3', 'A_4', 'a_1', 'a_2', 'a_3', 'a_4', 'b_1', 'b_2', 'b_3', 'b_4', 'c_1', 
                       'c_2', 'c_3', 'c_4', 'x0_1', 'x0_2', 'x0_3', 'x0_4', 'y0_1', 'y0_2', 'y0_3', 'y0_4']
        bounds_x_l = [-1.5, -0.5]
        bounds_x_u = [1, 2]
        bounds_theta_l = [-300,-200,-250, 5,-2,-2,-10, -2, -2,-2,5,-2,-20,-20, -10,-1 ,-2,-2,-2, -2,-2,-2,0,-2]
        bounds_theta_u = [-100,  0, -150, 20,2, 2, 0,  2,  2,  2, 15,2, 0,0   , 0,  2, 2,  2, 2, 2 ,2 , 2, 2,2]
        theta_ref = np.array([-200,-100,-170,15,-1,-1,-6.5,0.7,0,0,11,0.6,-10,-10,-6.5,0.7,1,0,-0.5,-1,0,0.5,1.5,1])      
        calc_y_fxn = calc_muller
       
    #5 parameter Polynomial (CS3)
    elif cs_name.value == 8:
        theta_names = ['theta_1', 'theta_2', 'theta_3', 'theta_4', 'theta_5']
        bounds_x_l = [-5, -5]
        bounds_x_u = [ 5,  5]
        bounds_theta_l = [-300,-5.0,-5.0, -20, -20]
        bounds_theta_u = [   0, 5.0, 5.0,  20,  20]
        theta_ref = np.array([-100, 1.0, -0.1, 10, -10])      
        calc_y_fxn = calc_cs3_polynomial
    
    #4 parameter Isotherm (CS4)
    elif cs_name.value == 9:
        theta_names = ['theta_1', 'theta_2', 'theta_3', 'theta_4']
        bounds_x_l = [-5, -5]
        bounds_x_u = [ 5,  5]
        bounds_theta_l = [1e-5, 1e-5, 1e-5, 1e-5]
        bounds_theta_u = [50  , 50  , 50  ,  50]
        theta_ref =  np.array([25, 30, 15, 20])
        calc_y_fxn = calc_cs4_isotherm
        
    else:
        print(cs_name.value)
        raise ValueError("self.CaseStudyParameters.cs_name.value must exist!")

    return Simulator(indecies_to_consider, 
                     theta_ref,
                     theta_names,
                     bounds_theta_l, 
                     bounds_x_l, 
                     bounds_theta_u, 
                     bounds_x_u, 
                     noise_mean,
                     noise_std,
                     normalize,
                     seed,
                     calc_y_fxn)


def set_param_str(cs_name_val):
    """
    Sets parameter value string
    
    Parameters
    ----------
    cs_name_val: int, the string of the case study name
    """
    assert 9 >= cs_name_val >= 1 and isinstance(cs_name_val, int), "cs_name_val must be an integer between 1 and 9 inclusive"
    
    if cs_name_val == 1:
        param_name_str = "t1t2"
    elif cs_name_val == 2:
        param_name_str = "a"
    elif cs_name_val == 3:
        param_name_str = "x0y0"
    elif cs_name_val == 4:
        param_name_str = "abc"
    elif cs_name_val == 5:
        param_name_str = "abcx0"
    elif cs_name_val == 6:
        param_name_str = "abcx0y0"
    elif cs_name_val == 7:
        param_name_str = "Aabcx0y0"
    elif cs_name_val == 8:
        param_name_str = "t1t2t3t4t5"
    elif cs_name_val == 9:
        param_name_str = "t1t2t3t4"
        
    return param_name_str
        

def set_idcs_to_consider(cs_name_val, param_name_str):
    """
    Sets indecies to consider based on problem name
    
    Parameters
    ----------
    cs_name_val: int, the string of the case study name
    param_name_str: str, string of parameter names to include. t1 and t2 for CS1 and A,a,b,cx0,and y0 for CS2 Ex: 't1t2' or 'Aabcx0y0'.
    
    Returns
    -------
    indecies_to_consider, list. List of indecies to consider
    """
    assert 9 >= cs_name_val >= 1 and isinstance(cs_name_val, int), "cs_name_val must be an integer between 1 and 9 inclusive"
    assert isinstance(param_name_str, str), "param_list must be str"
    
    if 7 >= cs_name_val > 1:
        indecies_to_consider = []
        all_param_idx = [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19, 20,21,22,23]

        if "A" in param_name_str:
            indecies_to_consider += all_param_idx[0:4]
        if "a" in param_name_str:
            indecies_to_consider += all_param_idx[4:8]
        if "b" in param_name_str:
            indecies_to_consider += all_param_idx[8:12]                
        if "c" in param_name_str:
            indecies_to_consider += all_param_idx[12:16]
        if "x0" in param_name_str:
            indecies_to_consider += all_param_idx[16:20]
        if "y0" in param_name_str:
            indecies_to_consider += all_param_idx[20:]
        
            
    elif 9 >= cs_name_val >= 8 or cs_name_val == 1:
        indecies_to_consider = []
        all_param_idx = [0,1,2,3,4]
        if "t1" in param_name_str:
            indecies_to_consider += [all_param_idx[0]]
        if "t2" in param_name_str:
            indecies_to_consider += [all_param_idx[1]]
        if "t3" in param_name_str:
            indecies_to_consider += [all_param_idx[2]]
        if "t4" in param_name_str:
            indecies_to_consider += [all_param_idx[3]]
        if "t5" in param_name_str:
            indecies_to_consider += [all_param_idx[4]]
            
    else:
        raise Warning("Try again")
    
    return indecies_to_consider
