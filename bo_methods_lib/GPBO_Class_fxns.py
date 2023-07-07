import numpy as np
from scipy.stats import qmc
import pandas as pd

    """
    Turns arrays that are shape (n,) into (n, 1) arrays
    
    Parameters:
        array: ndarray, n dimensions
    Returns:
        array: ndarray,  if n > 1, return original array. Otherwise, return 2D array with shape (-1,n)
    """
    #If array is not 2D, give it shape (len(array), 1)
    if not len(array.shape) > 1:
        array = array.reshape(-1,1)
    return array

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
    x = vector_to_1D_array(x) 
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