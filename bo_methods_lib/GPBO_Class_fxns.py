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
        LHS = qmc.scale(LHS, bounds[0], bounds[1]) #Again, using this because I like that bounds can be different shapes

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

def calc_y_exp(CaseStudyParameters, Simulator, exp_data):
    """
    Creates y_data for any case study
    
    Parameters
    ----------
        CaseStudyParameters: class, class containing at least the theta_true, x_data, noise_mean, noise_std, and seed
        Simulator: Class, class containing at least calc_y_fxn
        exp_data: instance of a class. Contains at least the experimental x data
        
    Returns:
        y_exp: ndarray, The expected values of y given x data
    """   
    noise_std = CaseStudyParameters.noise_std
    noise_mean = CaseStudyParameters.noise_mean
    x = exp_data.x_vals
    random_seed = CaseStudyParameters.seed
    true_model_coefficients = Simulator.theta_ref
    calc_y_fxn = Simulator.calc_y_fxn
    len_x = exp_data.get_num_x_vals()
    
    #Asserts that test_T is a tensor with 2 columns
    assert isinstance(noise_std,(float,int)) == True, "The standard deviation of the noise must be an integer ot float."
    assert isinstance(noise_mean,(float,int)) == True, "The mean of the noise must be an integer ot float."
    
    x = vector_to_1D_array(x)
    
    #Seed Random Noise (For Bug Testing)
    if random_seed != None:
        assert isinstance(random_seed,int) == True, "Seed number must be an integer or None"
        np.random.seed(random_seed)
        
    #Creates noise values with a certain stdev and mean from a normal distribution
    noise = np.random.normal(size=len_x, loc = noise_mean, scale = noise_std) #1x n_x
    
    #Define an array to store y values in
    y_exp = np.zeros(len_x)
    
    #Loop over x values and calculate y
    for i in range(len_x):
        y_exp[i] = calc_y_fxn(true_model_coefficients, x[i])
    
    #Add noise and flatten array
    y_exp = y_exp + noise
    y_exp = y_exp.flatten()
  
    return y_exp

def calc_y_sim(CaseStudyParameters, Simulator, sim_data, exp_data):
    """
    Creates y_data (training data) based on the function theta_1*x + theta_2*x**2 +x**3
    Parameters
    ----------
        CaseStudyParameters: class, class containing at least the theta_true, x_data, noise_mean, noise_std, and seed
        Simulator: Class, class containing at least calc_y_fxn
        sim_data: Class, Class containing at least the theta_vals for simulation
        exp_data: Class, Class containing at least the x_data and y_data for the experimental data
        
    Returns
    -------
        y_data: ndarray, The simulated y training data
    """
    #Define an array to store y values in
    y_sim = []
    len_theta = sim_data.get_num_theta() #Have to do it this way to be able to generalize between all the theta values and just 1 value
    len_x = sim_data.get_num_x_vals()
    calc_y_fxn = Simulator.calc_y_fxn
    true_model_coefficients = Simulator.theta_ref
    indecies_to_consider = Simulator.indecies_to_consider
    #Loop over all theta values
    for i in range(len_theta):
        #Create model coefficient from true space substituting in the values of param_space at the correct indecies
        model_coefficients = true_model_coefficients
        model_coefficients[indecies_to_consider] = sim_data.theta_vals[i]
        #Loop over x values and calculate y
        for j in range(len_x):
            #Create model coefficients
            y_sim.append(calc_y_fxn(model_coefficients, exp_data.x_vals[j])) 
    
    #Convert list to array and flatten array
    y_sim = np.array(y_sim).flatten()
    
    return y_sim

def calc_sse(CaseStudyParameters, Simulator, Method, sim_data, exp_data):
    """
    Creates y_data for the 2 input GP function
    
    Parameters
    ----------
        CaseStudyParameters: class, class containing at least the theta_true, x_data, noise_mean, noise_std, and seed
        Simulator: Class, class containing at least calc_y_fxn
        method: class, fully defined methods class which determines which method will be used
        sim_data: Class, Class containing at least the theta_vals for simulation
        exp_data: Class, Class containing at least the x_data and y_data for the experimental data     
        
    Returns:
        sum_error_sq: ndarray, The SSE or ln(SSE) values that the GP will be trained on
    """   
    #Assert statement
    #How would I fix this assert statement to assert the correct typr?
#     assert obj == "obj" or obj == "LN_obj", "Objective function choice, obj, MUST be sse or LN_sse"
    
    #Calculate noise. Do we actually need noise here? The SSE is never a measured value
#     noise = np.random.normal(size= 1 ,loc = noise_mean, scale = noise_std) #1x n_x
    len_theta = sim_data.get_num_theta() #Have to do it this way to be able to generalize between all the theta values and just 1 value
    len_x = sim_data.get_num_x_vals()
    calc_y_fxn = Simulator.calc_y_fxn
    true_model_coefficients = Simulator.theta_ref
    indecies_to_consider = Simulator.indecies_to_consider
    obj = Method.obj
    
    #How could I use calc_y_sim here rather than writing the same lines of code?
    sum_error_sq = np.zeros(len_theta)
    #Iterates over evey combination of theta to find the SSE for each combination
    for i in range(len_theta):
        #Create model coefficient from true space substituting in the values of param_space at the correct indecies
        model_coefficients = true_model_coefficients
        model_coefficients[indecies_to_consider] = sim_data.theta_vals[i]
        y_sim = np.zeros(len_x)
        for j in range(len_x):
            #Create model coefficients
            y_sim[j] = calc_y_fxn(model_coefficients, exp_data.x_vals[j])
    
        sum_error_sq[i] = sum((y_sim - exp_data.y_vals)**2) #Scaler
        
    if obj.value == 2:
        sum_error_sq = np.log(sum_error_sq) #Scaler
        
    return sum_error_sq 