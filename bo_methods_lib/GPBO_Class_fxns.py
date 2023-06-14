import numpy as np
from scipy.stats import qmc
import pandas as pd

def lhs_design(num_points, dimensions, seed = None, bounds = None):
    """
    Design LHS Samples
    
    Parameters
    ----------
        num_points: int, number of points in LHS, should be greater than # of dimensions
        dimensions: int, number of parameters to be regressed
        seed (optional): int, seed of random generation
        bounds (optional): ndarray, array containing upper and lower bounds of elements in LHS sample. Defaults of 0 and 1
    
    Returns
    -------
        LHS: ndarray, Array of LHS sampling points
    """
    sampler = qmc.LatinHypercube(d=dimensions, seed = seed)
    LHS = sampler.random(n=num_points)
    
    if bounds is not None:
        LHS = qmc.scale(LHS, bounds[0], bounds[1])

    return LHS

def vector_to_1D_array(array):
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

#How to pass the function given that inputs may be different?
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

def calc_y_exp(calc_y_fxn, true_model_coefficients, x, noise_std, noise_mean=0,random_seed=6):
    """
    Creates y_data for any case study
    
    Parameters
    ----------
        calc_y_fxn: function, the function that calculates the experimental value of y
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
    
    x = vector_to_1D_array(x)
    len_x = x.shape[0]
    
    #Seed Random Noise (For Bug Testing)
    if random_seed != None:
        assert isinstance(random_seed,int) == True, "Seed number must be an integer or None"
        np.random.seed(random_seed)
        
    #Creates noise values with a certain stdev and mean from a normal distribution
    noise = np.random.normal(size=x.shape[0],loc = noise_mean, scale = noise_std) #1x n_x
    
    #Define an array to store y values in
    y_exp = np.zeros(len_x)
    
    #Loop over x values and calculate y
    for i in range(len_x):
        y_exp[i] = calc_y_fxn(true_model_coefficients, x[i])
    
    #Add noise and flatten array
    y_exp = y_exp + noise
    y_exp = y_exp.flatten()
  
    return y_exp

def calc_y_sim(calc_y_fxn, sim_data, exp_data, true_model_coefficients, indecies_to_consider):
    """
    Creates y_data (training data) based on the function theta_1*x + theta_2*x**2 +x**3
    Parameters
    ----------
        calc_y_fxn: function, The function used to evaluate the simulation
        sim_data: Class, Class containing at least the theta_vals for simulation
        exp_data: Class, Class containing at least the x_data and y_data for the experimental data
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
        indecies_to_consider: list of int, The indecies corresponding to which parameters are being guessed
        
    Returns
    -------
        y_data: ndarray, The simulated y training data
    """
    #Define an array to store y values in
    y_sim = []
    
    #Loop over x values and calculate y
    for i in range(sim_data.theta_vals.shape[0]):
        #Create model coefficient from true space substituting in the values of param_space at the correct indecies
        model_coefficients = true_model_coefficients
        model_coefficients[indecies_to_consider] = sim_data.theta_vals[i]
        for j in range(exp_data.x_vals.shape[0]):
            #Create model coefficients
            y_sim.append(calc_y_fxn(model_coefficients, exp_data.x_vals[j]))
    
    #Convert list to array and flatten array
    y_sim = np.array(y_sim).flatten()
    
    return y_sim

def calc_sse(calc_y_fxn, sim_data, exp_data, true_model_coefficients, indecies_to_consider, obj = "obj"):
    """
    Creates y_data for the 2 input GP function
    
    Parameters
    ----------
        calc_y_fxn: function, The function used to evaluate the simulation
        sim_data: Class, Class containing at least the theta_vals for simulation
        exp_data: Class, Class containing at least the x_data and y_data for the experimental data
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
        indecies_to_consider: list of int, The indecies corresponding to which parameters are being guessed
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        
    Returns:
        sum_error_sq: ndarray, The SSE or ln(SSE) values that the GP will be trained on
    """   
    #Assert statement
    assert obj == "obj" or obj == "LN_obj", "Objective function choice, obj, MUST be sse or LN_sse"
    
    #Calculate noise. Do we actually need noise here? The SSE is never a measured value
#     noise = np.random.normal(size= 1 ,loc = noise_mean, scale = noise_std) #1x n_x
    
    #How could I use calc_y_sim here rather than writing the same lines of code?
    sum_error_sq = np.zeros(sim_data.theta_vals.shape[0])
    #Iterates over evey combination of theta to find the SSE for each combination
    for i in range(sim_data.theta_vals.shape[0]):
        #Create model coefficient from true space substituting in the values of param_space at the correct indecies
        model_coefficients = true_model_coefficients
        model_coefficients[indecies_to_consider] = sim_data.theta_vals[i]
        y_sim = np.zeros(exp_data.x_vals.shape[0])
        for j in range(exp_data.x_vals.shape[0]):
            #Create model coefficients
            y_sim[j] = calc_y_fxn(model_coefficients, exp_data.x_vals[j])
    
        sum_error_sq[i] = sum((y_sim - exp_data.y_vals)**2) #Scaler
        
    if obj == "LN_obj":
        sum_error_sq = np.log(sum_error_sq) #Scaler
        
    return sum_error_sq 