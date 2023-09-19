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
                       'c_2', 'c_3', 'c_4', 'x0_1', 'x0_2', 'x0_3', 'x0_4', 'x1_1', 'x1_2', 'x1_3', 'x1_4']
        bounds_x_l = [-1.5, -0.5]
        bounds_x_u = [1, 2]
        bounds_theta_l = [-300,-200,-250, 5,-2,-2,-10, -2, -2,-2,5,-2,-20,-20, -10,-1 ,-2,-2,-2, -2,-2,-2,0,-2]
        bounds_theta_u = [-100,  0, -150, 20,2, 2, 0,  2,  2,  2, 15,2, 0,0   , 0,  2, 2,  2, 2, 2 ,2 , 2, 2,2]
        theta_ref = np.array([-200,-100,-170,15,-1,-1,-6.5,0.7,0,0,11,0.6,-10,-10,-6.5,0.7,1,0,-0.5,-1,0,0.5,1.5,1])      
        calc_y_fxn = calc_muller
        
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

def set_idcs_to_consider(cs_name_val):
    """
    Sets indecies to consider based on problem name
    
    Parameters
    ----------
    cs_name_val: int, the string of the case study name
    
    Returns
    -------
    indecies_to_consider
    """
    cs_val_idx = cs_name_val - 1
    inc_to_consider_lo = [0, 0, 0, 0, 0, 0, 0]
    inc_to_consider_hi = [2, 4, 8, 12, 16, 20, 24]
        
    idc_lo = inc_to_consider_lo[cs_val_idx]
    idc_hi = inc_to_consider_hi[cs_val_idx]
    
    return idc_lo, idc_hi
