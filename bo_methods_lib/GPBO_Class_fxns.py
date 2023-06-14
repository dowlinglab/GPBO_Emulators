import numpy as np
from scipy.stats import qmc
import pandas as pd

def LHS_Design(num_points, dimensions, seed = None, bounds = None):
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

def clean_1D_arrays(array):
    """
    Turns arrays that are shape (n,) into (n, 1) arrays
    
    Parameters:
        array: ndarray, 1D array
    Returns:
        array: ndarray, 2D array with shape (n,1)
    """
    #If array is not 2D, give it shape (len(array), 1)
    if not len(array.shape) > 1:
        array == array.reshape(-1,1)
    return array

def cs1_calc_y_exp(Theta_True, x, noise_std, noise_mean=0,random_seed=6):
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
    noise = np.random.normal(size=x.shape[0],loc = noise_mean, scale = noise_std) #1x n_x
    #     if isinstance(x, np.ndarray):
#         noise = np.random.normal(size=len(x),loc = noise_mean, scale = noise_std) #1x n_x
#     else:
#         noise = np.random.normal(size=1,loc = noise_mean, scale = noise_std) #1x n_x
    # True function is y=T1*x + T2*x^2 + x^3 with Gaussian noise
    y_exp =  Theta_True[0]*x + Theta_True[1]*x**2 +x**3 + noise #1x n_x #Put this as an input
  
    return y_exp


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
    #Reshape x to matrix form
    x = clean_1D_arrays(x)
    model_coefficients = clean_1D_arrays(model_coefficients)
    X1, X2 = x
    
    #Separate all model parameters into their appropriate pieces
    sublists = []
    chunk_size = model_coefficients.shape[0] // 6 # Calculate the size of each sublist, 6 different parameters

    for i in range(6):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size
        sublist = model_coefficients[start_index:end_index]
        sublists.append(sublist)
        
    #Calculate Muller Potential
    A, a, b, c, x0, y0 = sublists
    Term1 = a*(X1 - x0)**2
    Term2 = b*(X1 - x0)*(X2 - y0)
    Term3 = c*(X2 - y0)**2
    y_mul = np.sum(A*np.exp(Term1 + Term2 + Term3) ) + noise
    
    return y_mul

def cs2_calc_y_exp(true_model_coefficients, x, noise_std, noise_mean=0,random_seed=9):
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

def cs1_calc_sse(Sim_data, Exp_data, obj = "obj"):
    """
    Creates y_data for the 2 input GP function
    
    Parameters
    ----------
        Sim_data: Class, Class containing at least the theta_vals for simulation
        Exp_data: Class, Class containing at least the x_data and y_data for the experimental data
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        
    Returns:
        sum_error_sq: ndarray, The SSE or ln(SSE) values that the GP will be trained on
    """   
    assert obj == "obj" or obj == "LN_obj", "Objective function choice, obj, MUST be sse or LN_sse" 

    #Creates an array for train_sse that will be filled with the for loop
    sum_error_sq = np.zeros((Sim_data.theta_vals.shape[0]))

    #Iterates over evey combination of theta to find the SSE for each combination
    #For each point in train_T
    for i in range(Sim_data.theta_vals.shape[0]):
        #Theta 1 and theta 2 represented by columns for this case study
        theta_1 = Sim_data.theta_vals[i,0] #n_train^2x1 
        theta_2 = Sim_data.theta_vals[i,1] #n_train^2x1
        #Calc y_sim
        y_sim = theta_1*Exp_data.x_vals + theta_2*Exp_data.x_vals**2 +Exp_data.x_vals**3 #n_train^2 x n_x
        #Clean y_sim and calculate sse or log(sse)
        y_sim = y_sim.flatten()
        
        if obj == "obj":
            sum_error_sq[i] = sum((y_sim - Exp_data.y_vals)**2) #Scaler
        else:
            sum_error_sq[i] = np.log(sum((y_sim - Exp_data.y_vals)**2)) #Scaler
    
    return sum_error_sq 

def cs1_calc_y_sim(Sim_data, Exp_data):
    """
    Creates y_data (training data) based on the function theta_1*x + theta_2*x**2 +x**3
    Parameters
    ----------
        Sim_data: Class, Class containing at least the theta_vals for simulation
        Exp_data: Class, Class containing at least the x_data and y_data for the experimental data
    Returns
    -------
        y_data: ndarray, The simulated y training data
    """       
    #Creates an array for train_data that will be filled with the for loop
    y_data = [] #1 x n (row x col)
    
    #Used when multiple values of y are being calculated
    #Iterates over evey combination of theta to find the expected y value for each combination
    for i in range(Sim_data.theta_vals.shape[0]):
        #Theta1 and theta2 are defined as coulms of param_space
        theta_1, theta_2 = Sim_data.theta_vals[i] #1 x n
        #Calculate y_data and append to list
        y_vals = theta_1*Exp_data.x_vals + theta_2*Exp_data.x_vals**2 +Exp_data.x_vals**3
        y_data.append(y_vals) #Scaler
        
    #Flatten y_data
    y_data = np.array(y_data).flatten()
    
    return y_data

def cs2_calc_sse(Sim_data, Exp_data, true_model_coefficients, obj, indecies_to_consider, 
                 noise_mean = 0, noise_std = 0, seed = None):
    """
    Creates y_data for the 2 input GP function
    
    Parameters
    ----------
        Sim_data: Class, Class containing at least the theta_vals for simulation
        Exp_data: Class, Class containing at least the x_data and y_data for the experimental data
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        indecies_to_consider: list of int, The indecies corresponding to which parameters are being guessed
        noise_mean: float, int: The mean of the noise
        noise_std: float, int: The standard deviation of the noise
        seed: int: The random seed
        
    Returns:
        sum_error_sq: ndarray, The SSE or ln(SSE) values that the GP will be trained on
    """  
    #Will need assert statement
    assert obj == "obj" or obj == "LN_obj", "Objective function choice, obj, MUST be sse or LN_sse"
    
    noise = np.random.normal(size= 1 ,loc = noise_mean, scale = noise_std) #1x n_x
    
    sum_error_sq = np.zeros(Sim_data.theta_vals.shape[0])
    #Iterates over evey combination of theta to find the SSE for each combination
    for i in range(Sim_data.theta_vals.shape[0]):
        model_coefficients = true_model_coefficients
        model_coefficients[indecies_to_consider] = Sim_data.theta_vals[i]
        y_sim = np.zeros(Exp_data.x_vals.shape[0])
        #Loop over state points
        for j in range(Exp_data.x_vals.shape[0]):
            noise = np.random.normal(size= 1 ,loc = 0, scale = 0) #1x n_x
            y_sim[j] = calc_muller(Exp_data.x_vals[j], model_coefficients, noise)
                
        if obj == "obj":
            sum_error_sq[i] = sum((y_sim - Exp_data.y_vals)**2) #Scaler
        else:
            sum_error_sq[i] = np.log(sum((y_sim - Exp_data.y_vals)**2)) #Scaler
    
    return sum_error_sq


def cs2_calc_y_sim(Sim_data, true_model_coefficients, indecies_to_consider, noise_mean = 0, noise_std = 0, seed = None):
    """
    Creates y_data (training data) based on the function theta_1*x + theta_2*x**2 +x**3
    Parameters
    ----------
        Sim_data: Class, Class containing at least the x_data and theta_vals for simulation
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
        indecies_to_consider: list of int, The indecies corresponding to which parameters are being guessed
        noise_mean: float, int: The mean of the noise
        noise_std: float, int: The standard deviation of the noise
        seed: int: The random seed
    Returns
    -------
        y_sim: ndarray, The simulated y training data
    """
    #Assert statements check that the types defined in the doctring are satisfied
    if seed != None:
        assert isinstance(seed,int) == True, "Seed number must be an integer or None"
        np.random.seed(seed)
        
    #Creates noise values with a certain stdev and mean from a normal distribution
    noise = np.random.normal(size= 1 ,loc = noise_mean, scale = noise_std) #1x n_x
    
    #Generate empy list to store y_sim values
    y_sim = []
    #Loop over theta values
    for i in range(Sim_data.theta_vals.shape[0]):
        #Loop over x values
        for j in range(Sim_data.x_vals.shape[0]):
            #Create model coefficients
            #Create model coefficient from true space substituting in the values of param_space at the correct indecies
            model_coefficients = true_model_coefficients
            model_coefficients[indecies_to_consider] = Sim_data.theta_vals[i]
            #Calculate the value of y_sim
            y_sim.append(calc_muller(Sim_data.x_vals[j], model_coefficients, noise))
            
    #Change list to np array
    y_sim = np.array(y_sim).flatten()
   
    return y_sim