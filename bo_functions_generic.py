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

from CS2_bo_plotters import plot_org_train

def clean_1D_arrays(array, param_clean = False):
    """
    Turns arrays that are shape (n,) into (n,1) arrays
    
    Parameters:
        array: ndarray, 1D array
    Returns:
        array: ndarray, 2D array with shape (n,1)
    """
    if not len(array.shape) > 1:
        array = array.reshape(-1,1)
        if param_clean == True:
            array = array.reshape(1,-1)
    return array

def gen_theta_set(LHS = True, n_points = 10, dimensions = 2, bounds = None):
    """
    Generates theta_set from either a meshgrid search or and LHS
    
    Parameters:
    -----------
        LHS: bool, Determines whether a meshgrid or LHS sample is generated, default True
        n_points:, int, number of meshgrid points/ parameter or the square root of the number of LHS samples
        dimensions: int, Number of parameters to regress
        bounds: None or ndarray, contains bounds for LHS generation if necessary
    """
    if LHS == False:
        if bounds is not None:
            Theta = np.linspace(bounds[0,0],bounds[1,0],n_points)
        else:
            Theta = np.linspace(0,1,n_points)
        df = pd.DataFrame(list(itertools.product(Theta, repeat=dimensions)))
        df2 = df.drop_duplicates()
        theta_set = df2.to_numpy()
    else:
        theta_set = LHS_Design(n_points**2, dimensions, seed = 9, bounds = bounds)
    return theta_set

def LHS_Design(num_points, dimensions, seed = None, bounds = None):
    """
    Design LHS Samples
    
    Parameters
    ----------
        num_points: int, number of points in LHS, should be greater than # of dimensions
        dimensions: int, number of parameters to be regressed
        bounds (optional): ndarray, array containing upper and lower bounds of elements in LHS sample. Defaults of 0 and 1
        seed (optional): int, seed of random generation
    Returns
    -------
        LHS: ndarray, Array of LHS sampling points
    """
    sampler = qmc.LatinHypercube(d=dimensions, seed = seed)
    LHS = sampler.random(n=num_points)
    
    if bounds is not None:
        LHS = qmc.scale(LHS, bounds[0], bounds[1])

    return LHS

# def calc_y_exp(Theta_True, x, noise_std, noise_mean=0,random_seed=6):
#     """
#     Creates y_data for the 2 input GP function
    
#     Parameters
#     ----------
#         Theta_True: ndarray, The array containing the true values of Theta1 and Theta2
#         x: ndarray, The list of xs that will be used to generate y
#         noise_std: float, int: The standard deviation of the noise
#         noise_mean: float, int: The mean of the noise
#         random_seed: int: The random seed
        
#     Returns:
#         y_exp: ndarray, The expected values of y given x data
#     """   
    
#     #Asserts that test_T is a tensor with 2 columns
#     assert isinstance(noise_std,(float,int)) == True, "The standard deviation of the noise must be an integer ot float."
#     assert isinstance(noise_mean,(float,int)) == True, "The mean of the noise must be an integer ot float."
#     assert len(Theta_True) ==2, "This function only has 2 unknowns, Theta_True can only contain 2 values."
    
    
#     #Seed Random Noise (For Bug Testing)
#     if random_seed != None:
#         assert isinstance(random_seed,int) == True, "Seed number must be an integer or None"
#         np.random.seed(random_seed)
        
#     #Creates noise values with a certain stdev and mean from a normal distribution
#     if isinstance(x, np.ndarray):
#         noise = np.random.normal(size=len(x),loc = noise_mean, scale = noise_std) #1x n_x
#     else:
#         noise = np.random.normal(size=1,loc = noise_mean, scale = noise_std) #1x n_x
#     # True function is y=T1*x + T2*x^2 + x^3 with Gaussian noise
#     y_exp =  Theta_True[0]*x + Theta_True[1]*x**2 +x**3 + noise #1x n_x #Put this as an input
  
#     return y_exp

# def create_sse_data(q,train_T, x, y_exp, obj = "obj"):
#     """
#     Creates y_data for the 2 input GP function
    
#     Parameters
#     ----------
#         q: int, Number of parameters to be regressed
#         train_T: ndarray, The array containing the training data for Theta1 and Theta2
#         x: ndarray, The list of xs that will be used to generate y
#         y_exp: ndarray, The experimental data for y (the true value)
#         obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        
#     Returns:
#         sum_error_sq: ndarray, The SSE or ln(SSE) values that the GP will be trained on
#     """   
    
#     #Asserts that test_T is a tensor with 2 columns (May delete this)
#     assert isinstance(q, int), "Number of inputs must be an integer"
# #     print(train_T.T)    
#     if torch.is_tensor(train_T)==True:
#         assert len(train_T.permute(*torch.arange(train_T.ndim -1, -1, -1))) >=q, str("This is a "+str(q)+" input GP, train_T must have at least q columns of values.")
#     else:
#         assert len(train_T.T) >=q, str("This is a "+str(q)+" input GP, train_T must have at least q columns of values.")
#     assert len(x) == len(y_exp), "Xexp and Yexp must be the same length"
#     assert obj == "obj" or obj == "LN_obj", "Objective function choice, obj, MUST be sse or LN_sse"
    
#     train_T = clean_1D_arrays(train_T)
#     y_exp = clean_1D_arrays(y_exp)
# #     try: 
#     if train_T.shape[1] > 1: #For the case where more than 1 point is geing generated
#         #Creates an array for train_sse that will be filled with the for loop
#         sum_error_sq = torch.tensor(np.zeros(len(train_T))) #1 x n_train^2

#         #Iterates over evey combination of theta to find the SSE for each combination
#         for i in range(len(train_T)):
#             theta_1 = train_T[i,0] #n_train^2x1 
#             theta_2 = train_T[i,1] #n_train^2x1
#             y_sim = theta_1*x + theta_2*x**2 +x**3 #n_train^2 x n_x
#             if obj == "obj":
#                 sum_error_sq[i] = sum((y_sim - y_exp)**2) #Scaler
#             else:
#                 sum_error_sq[i] = np.log(sum((y_sim - y_exp)**2)) #Scaler
# #     except:
#     else:
#          #Creates a value for train_sse that will be filled with the for loop
#         sum_error_sq = 0 #1 x n_train^2

#         #Iterates over x to find the SSE for each combination
#         theta_1 = train_T[0] #n_train^2x1 
#         theta_2 = train_T[1] #n_train^2x1
#         y_sim = theta_1*x + theta_2*x**2 +x**3 #n_train^2 x n_x
#         if obj == "obj":
#             sum_error_sq = sum((y_sim - y_exp)**2) #Scaler 
#         else:
#             sum_error_sq = np.log(sum((y_sim - y_exp)**2)) #Scaler 
    
#     return sum_error_sq    

# def create_y_data(param_space):
#     """
#     Creates y_data (training data) based on the function theta_1*x + theta_2*x**2 +x**3
#     Parameters
#     ----------
#         param_space: (nx3) ndarray or tensor, parameter space over which the GP will be run
#     Returns
#     -------
#         y_data: ndarray, The simulated y training data
#     """
#     #Assert statements check that the types defined in the doctring are satisfied
#     assert len(param_space.T) >= 3, "Parameter space must have at least 3 parameters"
    
#     #Converts parameters to numpy arrays if they are tensors
#     if torch.is_tensor(param_space)==True:
#         param_space = param_space.numpy()
        
#     #Creates an array for train_data that will be filled with the for loop
#     y_data = np.zeros(len(param_space)) #1 x n (row x col)
    
#     try: #Used when multiple values of y are being calculated
#         #Iterates over evey combination of theta to find the expected y value for each combination
#         for i in range(len(param_space)):
#             theta_1 = param_space[i,0] #nx1 
#             theta_2 = param_space[i,1] #nx1
#             x = param_space[i,2] #nx1 
#             y_data[i] = theta_1*x + theta_2*x**2 +x**3 #Scaler
#             #Returns all_y
#     except:
#         theta_1 = param_space[0] #nx1 
#         theta_2 = param_space[1] #nx1
#         x = param_space[2] #nx1 
#         y_data = theta_1*x + theta_2*x**2 +x**3 #Scaler
#     return y_data

def create_y_sim_exp(true_model_coefficients, x, param_space = None, skip_param_types = 0, noise_std=None, noise_mean=0,random_seed=9):
    """
    Creates y_data (training data) based on the function theta_1*x + theta_2*x**2 +x**3
    Parameters
    ----------
        param_space: (nx3) ndarray or tensor, parameter space over which the GP will be run
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
        x: ndarray, Array containing x data
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed
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
          
    x = clean_1D_arrays(x) 
    len_x, dim_x = x.shape[0], x.shape[1] # 2
    
    num_constant_type, len_constants = true_model_coefficients.shape[0], true_model_coefficients.shape[1] # 6,4
    
    if random_seed != None:
        assert isinstance(random_seed,int) == True, "Seed number must be an integer or None"
        np.random.seed(random_seed)
        
    if noise_std != None:
        noise = np.random.normal(size= 1 ,loc = noise_mean, scale = noise_std) #1x n_x
    else:
        noise = 0
       
    #For the case where more than 1 point is geing generated
    #Creates an array for train_sse that will be filled with the for loop
    #Initialize y_sim 
        
    if param_space is not None:       
        param_space = clean_1D_arrays(param_space) 
        len_data, dim_data = param_space.shape[0], param_space.shape[1] #300, 10
        dim_param = dim_data - dim_x
        num_param_type_guess = int(dim_param/len_constants)
    
        model_coefficients = true_model_coefficients.copy()
        y_create = np.zeros(len_data) #1 x n_train^2
    
        #Iterates over evey data point to find the y for each combination
        for i in range(len_data):
            #Set dig out values of a from train_p
            #Set constants to change the a row to the index of the first loop

            #loop over number of param types (A, a, b c, x0, y0)
            for j in range(num_param_type_guess):
                j_model = skip_param_types + j
                model_coefficients[j_model] = param_space[i][len_constants*j: len_constants*(j+1)]
    #         print(model_coefficients)
            A, a, b, c, x0, y0 = model_coefficients         
            #Calculate y_sim
            x = param_space[i][dim_param:dim_data]
            y_create[i] = calc_muller(x, model_coefficients, noise)
    
    else:      
        y_create = np.zeros(len_x) #1 x n_train^2
            
        for i in range(len_x):
            #Creates noise values with a certain stdev and mean from a normal distribution
            y_create[i] = calc_muller(x[i], true_model_coefficients, noise)
   
    return y_create

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
        y_exp: float, value of Muller potential
    """
    X1, X2 = x
    A, a, b, c, x0, y0 = model_coefficients
    Term1 = a*(X1 - x0)**2
    Term2 = b*(X1 - x0)*(X2 - y0)
    Term3 = c*(X2 - y0)**2
    y_mul = np.sum(A*np.exp(Term1 + Term2 + Term3) ) + noise
    return y_mul

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
        print(true_model_coefficients)
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
            y_sim[k] = calc_muller(x[k], model_coefficients)
                
        if obj == "obj":
            sum_error_sq[i] = sum((y_sim - y_exp)**2) #Scaler
#                 print(sum_error_sq[i])
        else:
            sum_error_sq[i] = np.log(sum((y_sim - y_exp)**2)) #Scaler
    
    return sum_error_sq


def create_y_data(param_space, true_model_coefficients, x, skip_param_types = 0):
    """
    Creates y_data (training data) based on the function theta_1*x + theta_2*x**2 +x**3
    Parameters
    ----------
        param_space: (nx3) ndarray or tensor, parameter space over which the GP will be run
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
        x: ndarray, Array containing x data
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed
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
    
    param_space = clean_1D_arrays(param_space) 
    x = clean_1D_arrays(x) 
    len_data, dim_data = param_space.shape[0], param_space.shape[1] #300, 10
#     print(len_data, dim_data)
    dim_x = x.shape[1] # 2
    dim_param = dim_data - dim_x
    
    num_constant_type, len_constants = true_model_coefficients.shape[0], true_model_coefficients.shape[1] # 6,4
    num_param_type_guess = int(dim_param/len_constants)
        
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
        A, a, b, c, x0, y0 = model_coefficients         
        #Calculate y_sim
        x = param_space[i][dim_param:dim_data]
        y_sim[i] = calc_muller(x, model_coefficients)
   
    return y_sim

def train_test_plot_preparation(param_dim, exp_data_dim, theta_set, train_p, test_p, p_True, Xexp, emulator, sparse_grid, obj, ep0, len_scl, run, save_fig, tot_iters, tot_runs, DateTime, verbose, sep_fact = 1):  
    """
    Puts training data into a for loop to print all possible 3D angles of the training data
    
    Parameters:
    -----------
        test_set: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid() or LHS samples
        train_p: tensor or ndarray, The training parameter space data
        test_p: tensor or ndarray, The training parameter space data
        p_true: ndarray, A 2x1 containing the true input parameters
        emulator: True/False, Determines if GP will model the function or the function error
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        ep0: float, float,int,tensor,ndarray (1 value) The initial exploration bias parameter
        len_scl: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        run, int or None, The iteration of the number of times new training points have been picked
        save_fig: True/False, Determines whether figures will be saved
        tot_iters: int or None, Total number of BO Iters
        tot_runs, int or None, The total number of times new training points have been picked
        verbose: True/False, Determines whether z_term, ei_term_1, ei_term_2, CDF, and PDF terms are saved, Default = False
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        sep_fact: float, Between 0 and 1. Determines fraction of all data that will be used to train the GP. Default is 1.
    Returns:
    --------
        Prints or saves plots of starting training data for any set of dimensions
        
    """
    dim_param_list = np.linspace(0,param_dim-1,param_dim) #Note - Need to figure this out when plotting w/ multidimensional x
    mesh_combos = np.array(list(combinations(dim_param_list, 2)), dtype = int)

    for i in range(len(mesh_combos)):
        indecies = mesh_combos[i]
        #Concatenate test data and train data from indecie combination
        if len(test_p) > 0:
            test_data_piece = torch.cat((torch.reshape(test_p[:,indecies[0]],(-1,1)),torch.reshape(test_p[:,indecies[1]],(-1,1))),axis= 1)
        else:
            test_data_piece = test_p
#         print(train_p[:,indecies[0]].shape)
        train_data_piece = torch.cat((torch.reshape(train_p[:,indecies[0]],(-1,1)),torch.reshape(train_p[:,indecies[1]],(-1,1))),axis = 1)
#         print(train_data_piece)

        theta_set_piece = np.array((theta_set[:,indecies[0]],theta_set[:,indecies[1]]))

        if emulator == True:
            #Loop over each X dimension
            for i in range(exp_data_dim):
#                 print(indecies)
#                 print(param_dim)
                #Concatenate array corresponding to x values (looped) to training data to plot
                train_data_piece = torch.cat( (train_data_piece, torch.reshape(train_p[:,indecies[param_dim-1]+(i+1)],(-1,1))), axis = 1 )
#                 print(train_data_piece.shape)
                if len(test_p) > 0:
                    test_data_piece = torch.cat( (test_data_piece, torch.reshape(test_p[:,indecies[param_dim-1]+(i+1)],(-1,1))), axis = 1 )
                else:
                    test_data_piece = test_p
                plot_org_train(theta_set_piece,train_data_piece, test_data_piece, p_True, Xexp, emulator, sparse_grid, obj, ep0, len_scl, run, save_fig, tot_iters, tot_runs, DateTime, verbose, sep_fact = sep_fact)
        else:
            plot_org_train(theta_set_piece,train_data_piece, test_data_piece, p_True, Xexp, emulator, sparse_grid, obj, ep0, len_scl, run, save_fig, tot_iters, tot_runs, DateTime, verbose, sep_fact = sep_fact)
    return 

#This will need to change eventually
def set_ep(emulator, obj, sparse):
    '''
    Sets the ep of the method based on results of the ep sensitivity analysis for all 5 methods
    
    Parameters:
    -----------
        emulator: True/False, Determines if GP will model the function or the function error
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        sparse_grid: Determines whether a sparse grid or approximation is used for the GP emulator
    Returns:
    --------
        ep: float, the optimal ep for the method based on previous analysis
    '''
    
    if emulator == False:
        if obj == "obj":
            ep =0.3
        else:
            ep = 0.5
    
    if emulator == True:
        if sparse == True:
            ep = 1
        else:
            if obj == "obj":
                ep = 0.8
            else:
                ep = 1
    return ep

# def LHS_Design(csv_file):
#     """
#     Creates LHS Design based on a CSV
#     Parameters
#     ----------
#         csv_file: str, the name of the file containing the LHS design from Matlab. Values should not be scaled between 0 and 1.
#     Returns
#     -------
#         param_space: ndarray , the parameter space that will be used with the GP
#     """
#     #Asserts that the csv filename is a string
#     assert isinstance(csv_file, str)==True, "csv_file must be a sting containing the name of the file"
    
#     reader = csv.reader(open(csv_file), delimiter=",") #Reads CSV containing nx3 LHS design
#     lhs_design = list(reader) #Creates list from CSV
#     param_space = np.array(lhs_design).astype("float") #Turns LHS design into a useable python array (nx3)
#     return param_space

##LEFT OFF DEBUGGING THIS FUNCTION
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
#     y_GP_Opt_data = create_y_data(create_y_data_space)
    y_GP_Opt_data = create_y_data(create_y_data_space, true_model_coefficients, x_space, skip_param_types = skip_param_types)
    return y_GP_Opt_data   

# def gen_y_Theta_GP(x_space, Theta):
#     """
#     Generates an array of Best Theta Value and X to create y data
    
#     Parameters
#     ----------
#         x_space: ndarray, array of x value
#         Theta: ndarray, Array of theta values
           
#     Returns
#     -------
#         create_y_data_space: ndarray, array of parameters [Theta, x] to be used to generate y data
        
#     """
#     x_space = clean_1D_arrays(x_space)
#     Theta = clean_1D_arrays(Theta)
#     m = x_space.shape[1]
#     q = Theta.shape[1]
    
#     #Define dimensions and initialize parameter matricies
#     dim = q+m
#     lenX = len(x_space)
#     create_y_data_space = np.zeros((lenX,dim))
    
#     #Loop over # of x values
#     for i in range(lenX):
#         #Loop over number of theta values
#         for j in range(q):
#             #Fill matrix to include all Theta and x parameters
#             create_y_data_space[i,j] = Theta[j]
#         create_y_data_space[i,q] = x_space[i]
#     #Generate y data based on parameters
#     y_GP_Opt_data = create_y_data(create_y_data_space)
#     y_GP_Opt_data = create_y_data(create_y_data_space)
#     return y_GP_Opt_data      

def test_train_split(all_data, sep_fact=0.8, runs = 0, shuffle_seed = None):
    """
    Splits y data into training and testing data
    
    Parameters
    ----------
        all_data: ndarray or tensor, The simulated parameter space and y data
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
        runs: int, # of runs for bo iterations. default is 0
        shuffle_seed, int, number of seed for shuffling training data. Default is None.
    Returns:
        train_data: ndarray, The training data
        test_data: ndarray, The testing data
    
    """
    #Assert statements check that the types defined in the doctring are satisfied and sep_fact is between 0 and 1 
    assert isinstance(sep_fact, (float, int))==True or torch.is_tensor(sep_fact)==True, "Separation factor must be a float, int, or tensor"
    assert 0 <= sep_fact <= 1, "Separation factor must be between 0 and 1"
    
    #Shuffles Random Data
#     if sep_fact 
    if shuffle_seed is not None:
        if runs > 1:
             for i in range(runs):
                np.random.seed(i)
        else:
            np.random.seed(shuffle_seed)
    
    np.random.shuffle(all_data) 
        
    #Creates the index on which to split data
    train_enteries = int(len(all_data)*sep_fact)
    
    #Training and testing data are created and converted into tensors
    train_y =all_data[:train_enteries, -1] #1x(n*sep_fact)
    test_y = all_data[train_enteries:, -1] #1x(n-n*sep_fact)
    train_param = all_data[:train_enteries,:-1] #1x(n*sep_fact)
    test_param = all_data[train_enteries:,:-1] #1x(n-n*sep_fact)
    
    train_data = np.column_stack((train_param, train_y))
    test_data = np.column_stack((test_param, test_y))
    return torch.tensor(train_data),torch.tensor(test_data)

def find_train_doc_path(emulator, obj, d, t):
    """
    Finds the document that contains the correct training data based on the GP objective function, number of dimensions, and number of training inputs
    
    Parameters
    ----------
    obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
    emulator: True/False, Determines if GP will model the function or the function error
    d: The number of dimensions of the problem (number of parameters to be regressed)
    t: int, The number of total data points
    
    Returns
    -------
    all_data_doc: csv name as a string, contains all training data for GP
    
    """
    if emulator == False:
        if obj == "obj":
            all_data_doc = "Input_CSVs/Train_Data/d="+str(d)+"/all_st_data/t="+str(t)+".csv"   
        else:
            all_data_doc = "Input_CSVs/Train_Data/d="+str(d)+"/all_st_ln_obj_data/t="+str(t)+".csv" 
    else:    
        all_data_doc = "Input_CSVs/Train_Data/d="+str(d)+"/all_emul_data/t="+str(t)+".csv" 
            
    return all_data_doc

class ExactGPModel(gpytorch.models.ExactGP): #Exact GP does not add noise
    """
    The base class for any Gaussian process latent function to be used in conjunction
    with exact inference.

    Parameters
    ----------
    torch.Tensor train_inputs: (size n x d) The training features :math:`\mathbf X`.
    
    torch.Tensor train_targets: (size n) The training targets :math:`\mathbf y`.
    
    ~gpytorch.likelihoods.GaussianLikelihood likelihood: The Gaussian likelihood that defines
        the observational distribution. Since we're using exact inference, the likelihood must be Gaussian.
    
    Methods
    -------
    The :meth:`__init__` function takes training data and a likelihood and computes the objects of mean and covariance 
    for the forward method

    The :meth:`forward` function should describe how to compute the prior latent distribution
    on a given input. Typically, this will involve a mean and kernel function.
    The result must be a :obj:`~gpytorch.distributions.MultivariateNormal`.
    
    Returns
    -------
    Calling this model will return the posterior of the latent Gaussian process when conditioned
    on the training data. The output will be a :obj:`~gpytorch.distributions.MultivariateNormal`.
    """

    def __init__(self, train_param, train_data, likelihood):
        """
        Initializes the model
        
        Parameters
        ----------
        self : A class,The model itself. In this case, gpytorch.models.ExactGP
        train_param : tensor or ndarray, The inputs of the training data
        train_data : tensor or ndarray, the output of the training data
        likelihood : bound method, the lieklihood of the model. In this case, it must be Gaussian
        
        """
        #Asserts that likeliehood is Gaussian and will work with the exact gp model
        assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"  
        
        #Converts training data and parameters to tensors if they are numpy arrays
        if isinstance(train_param, np.ndarray)==True:
            param_space = torch.tensor(train_param) #1xn
        if isinstance(train_data, np.ndarray)==True:
            train_data = torch.tensor(train_data) #1xn
 
        #Initializes the GP model with train_param, train_data, and the likelihood
        ##Calls the __init__ method of parent class
        super(ExactGPModel, self).__init__(train_param, train_data, likelihood)
        #Defines a constant prior mean on the GP. Used in the forward method
        self.mean_module = gpytorch.means.ConstantMean()
        #Defines prior covariance matrix of GP to a scaled RFB Kernel. Used in the forward method
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) 

    def forward(self, x):
        """
        A forward method that takes in some (n×d) data, x, and returns a MultivariateNormal with the prior mean and 
        covariance evaluated at x. In other words, we return the vector μ(x) and the n×n matrix Kxx representing the 
        prior mean and covariance matrix of the GP.
        
        Parameters
        ----------
        self : A class,The model itself. In this case, gpytorch.models.ExactGP
        x : tensor, first input when class is called
        
        Returns:
        Vector μ(x)
        
        """
        #Defines the mean of the GP based off of x
        mean_x = self.mean_module(x) #1xn_train
        #Defines the covariance matrix based off of x
        covar_x = self.covar_module(x) #n_train x n_train covariance matrix
        #Constructs a multivariate normal random variable, based on mean and covariance. 
            #Can be multivariate, or a batch of multivariate normals
            #Returns multivariate normal distibution gives the mean and covariance of the GP        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) #Multivariate dist based on 1xn_train^2 tensor


def train_GP_model(model, likelihood, train_param, train_data, iterations=500, verbose=False):
    """
    Trains the GP model and finds hyperparameters with the Adam optimizer with an lr =0.1
    
    Parameters
    ----------
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be Gaussian
        train_param: tensor or ndarray, The training parameter space data
        train_data: tensor or ndarray, The training y data
        iterations: float or int, number of training iterations to run. Default is 300
        verbose: Set verbose to "True" to view the associated loss and hyperparameters for each training iteration. False by default
    
    Returns
    -------
        noise_list: ndarray, List containing value of noise hyperparameter at every iteration
        lengthscale_list: ndarray, List containing value of lengthscale hyperparameter at every iteration
        outputscale_list: ndarray, List containing value of outputscale hyperparameter at every iteration
    """
    #Assert statements check that inputs are the correct types and lengths
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert isinstance(iterations, int)==True, "Number of training iterations must be an integer" 
    assert len(train_param) == len(train_data), "training data must be the same length as each other"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    
    #Converts training data and parameters to tensors if they are a numpy arrays
    if isinstance(train_param, np.ndarray)==True:
        train_param = torch.tensor(train_param) #1xn
    if isinstance(train_data, np.ndarray)==True:
        train_data = torch.tensor(train_data) #1xn

    #Find optimal model hyperparameters
    training_iter = iterations

    #Puts the model in training mode
    model.train()

    #Puts the likelihood in training mode
    likelihood.train()

    # Use the adam optimizer
        #algorithm for first-order gradient-based optimization of stochastic objective functions
        # The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients. 
        #The hyper-parameters have intuitive interpretations and typically require little tuning.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  #Needs GaussianLikelihood parameters, and a learning rate
        #lr default is 0.001

    # Calculate"Loss" for GPs

    #The marginal log likelihood (the evidence: quantifies joint probability of the data under the prior)
    #returns an exact MLL for an exact Gaussian process with Gaussian likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) #Takes a Gaussian likelihood and a model, a bound Method
    #iterates a give number of times
    noise_list = np.zeros(training_iter)
    lengthscale_list = np.zeros(training_iter)
    outputscale_list = np.zeros(training_iter)
    for i in range(training_iter): #0-299
        # Zero gradients from previous iteration - Prevents past gradients from influencing the next iteration
        optimizer.zero_grad() 
        # Output from model
        output = model(train_param) # A multivariate norm of a 1 x n_train^2 tensor
        # Calc loss and backprop gradients
        #Minimizing -logMLL lets us fit hyperparameters
        loss = -mll(output, train_data) #A number (tensor)
        #computes dloss/dx for every parameter x which has requires_grad=True. 
        #These are accumulated into x.grad for every parameter x
        loss.backward()
        noise_list[i] = model.likelihood.noise.item()
        lengthscale_list[i] =  model.covar_module.base_kernel.lengthscale.item()
        outputscale_list[i] = model.covar_module.outputscale.item()
        if verbose == True:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f   output scale: %.3f '% (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                 model.likelihood.noise.item(), model.covar_module.outputscale.item()
            ))
        #optimizer.step updates the value of x using the gradient x.grad. For example, the SGD optimizer performs:
        #x += -lr * x.grad
        optimizer.step()
    return noise_list,lengthscale_list,outputscale_list

def calc_GP_outputs(model,likelihood,test_param):
    #Checked for Correctness 5/19/22
    """
    Calculates the GP model's approximation of y, and its mean, variance and standard devaition.
    
    Parameters
    ----------
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        test_param: tensor or ndarray (1x1), The testing parameter space data
    
    Returns
    -------
        model_mean: tensor, The GP model's mean
        model_variance: tensor, The GP model's variance
        model_stdev: tensor, The GP model's standard deviation
        model_prediction: tensor, The GP model's approximation of y
    """
    #Assert statements check that inputs are correct types and lengths
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    #https://www.geeksforgeeks.org/type-isinstance-python/

    #Converts test parameters to tensors if they are a numpy arrays
    if isinstance(test_param, np.ndarray)==True:
        test_param = torch.tensor(test_param) #1xn
        
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
    #torch.no_grad() 
        #Disabling gradient calculation is useful for inference, 
        #when you are sure that you will not call Tensor.backward(). It will reduce memory consumption
        #Note: Can't use np operations on tensors where requires_grad = True
    #gpytorch.settings.fast_pred_var() 
        #Use this for improved performance when computing predictive variances. 
        #Good up to 10,000 data points
    #Predicts data points for model (sse) by sending the model through the likelihood
        observed_pred = likelihood(model(test_param)) #1 x n_test

    #Calculates model mean  
    model_mean = observed_pred.mean #1 x n_test
    #Calculates the variance of each data point
    model_variance = observed_pred.variance #1 x n_test
    #Calculates the standard deviation of each data point
    model_stdev = np.sqrt(observed_pred.variance.detach().numpy()) #THIS LINE WAS CHANGED TO HAVE .detach
    model_prediction = observed_pred.loc #1 x n_test
    return model_mean, model_variance, model_stdev, model_prediction    

def explore_parameter(Bo_iter, ep, mean_of_var, best_error, ep_o = 1, ep_inc = 1.5, ep_f = 0.01, ep_method = None, improvement = False):
    """
    Creates a value for the exploration parameter
    
    Parameters
    ----------
        Bo_iter: int, The value of the current BO iteration
        mean_of_var: float, The value of the average of all posterior variances
        best_error: float, The best error of the GP
        OPTIONAL:
        ep_o: float, The initial exploration parameter value: Default is 1
        e_inc: float, the increment for the Boyle's method for calculating exploration parameter: Default is 1.5
        ep_f: float, The final exploration parameter value: Default is 0.01
        ep_method: float, determines if Boyle, Jasrasaria, or exponential method will be used: Defaults to exponential method
        improvement: Bool, Determines whether last objective was an improvement
    Returns
    --------
        ep: The exploration parameter for the iteration
    """
    if Bo_iter == 0:
        ep = ep_o
        
    elif ep_method == "Boyle": #Works
        if improvement == True:
            ep = ep*ep_inc
        else:
            ep = ep/ep_inc
    
    elif ep_method == "Jasrasaria": #Works, but is bad because of high variances
        ep = mean_of_var/best_error
        
    elif ep_method == "Constant":
        ep = ep_o
    
    else:
        if Bo_iter < 30 and ep_o > 0: #Works
            alpha = -np.log(ep_f/ep_o)/30
            ep = ep_o*np.exp(-alpha*Bo_iter)
        else: 
            ep = 0.01
    
    return ep

#Approximation
def ei_approx_ln_term(epsilon, error_best, pred_mean, pred_stdev, y_target, ep): 
    """ 
    Calculates the integrand of expected improvement of the 3 input parameter GP using the log version
    Parameters
    ----------
        epsilon: The random variable. This is the variable that is integrated w.r.t
        error_best: float, the best predicted error encountered
        pred_mean: ndarray, model mean
        pred_stdev: ndarray, model stdev
        y_target: ndarray, the expected value of the function from data or other source
        ep: float, the numerical bias towards exploration, zero is the default
    
    Returns
    -------
        ei: ndarray, the expected improvement for one term of the GP model
    """
#     EI = ( (error_best - ep) - np.log( (y_target - pred_mean - pred_stdev*epsilon)**2 ) )*norm.pdf(epsilon)

    ei_term_2_integral = np.log( abs((y_target - pred_mean - pred_stdev*epsilon)) )*norm.pdf(epsilon)
#     ei_term_2_integral = np.log( (y_target - pred_mean - pred_stdev*epsilon)**2 )*norm.pdf(epsilon)
    return ei_term_2_integral
    
def calc_ei_emulator(error_best,pred_mean,pred_var,y_target, explore_bias=0.0, obj = "obj"): #Will need obj toggle soon
    """ 
    Calculates the expected improvement of the 3 input parameter GP
    Parameters
    ----------
        error_best: float, the best predicted error encountered
        pred_mean: ndarray, model mean
        pred_var: ndarray, model variance
        y_target: ndarray, the expected value of the function from data or other source
        explore_bias: float, the numerical bias towards exploration, zero is the default
        obj: str, LN_obj or obj, determines whether log or regular EI function is calculated
    
    Returns
    -------
        ei: ndarray, the expected improvement for one term of the GP model
    """
    #Asserts that f_pred is a float, and y_target is an ndarray
    if isinstance(explore_bias, float)!=True:
        explore_bias = explore_bias.numpy()
#     print(explore_bias)
#     explore_bias = float(explore_bias)
    assert isinstance(error_best, (float,int))==True, "error_best must be a float or integer"
    
    #Coverts any tensors given as inputs to ndarrays         
    #Checks for equal lengths
    
    if not isinstance(y_target, float)==True:
        y_target = float(y_target)
    assert isinstance(pred_mean, float)==True, "y_target, pred_mean, and pred_var must be floats"
    assert isinstance(pred_var, float)==True, "y_target, pred_mean, and pred_var must be floats"
    
    #Defines standard devaition
    pred_stdev = np.sqrt(pred_var) #1xn
    
    #If variance is zero this is important
    if obj == "obj":
        with np.errstate(divide = 'warn'):
            #Creates upper and lower bounds and described by Alex Dowling's Derivation
#             bound_a = ((y_target - pred_mean) +np.sqrt(error_best - explore_bias))/pred_stdev #1xn
#             bound_b = ((y_target - pred_mean) -np.sqrt(error_best - explore_bias))/pred_stdev #1xn
            bound_a = ((y_target - pred_mean) +np.sqrt(error_best*explore_bias))/pred_stdev #1xn
            bound_b = ((y_target - pred_mean) -np.sqrt(error_best*explore_bias))/pred_stdev #1xn
            bound_lower = np.min([bound_a,bound_b])
            bound_upper = np.max([bound_a,bound_b])        

            #Creates EI terms in terms of Alex Dowling's Derivation
            ei_term1_comp1 = norm.cdf(bound_upper) - norm.cdf(bound_lower) #1xn
#             ei_term1_comp2 = (error_best - explore_bias) - (y_target - pred_mean)**2 #1xn
            ei_term1_comp2 = (error_best*explore_bias) - (y_target - pred_mean)**2 #1xn

            ei_term2_comp1 = 2*(y_target - pred_mean)*pred_stdev #1xn
            ei_eta_upper = -np.exp(-bound_upper**2/2)/np.sqrt(2*np.pi)
            ei_eta_lower = -np.exp(-bound_lower**2/2)/np.sqrt(2*np.pi)
            ei_term2_comp2 = (ei_eta_upper-ei_eta_lower)

            ei_term3_comp1 = bound_upper*ei_eta_upper #1xn
            ei_term3_comp2 = bound_lower*ei_eta_lower #1xn

            ei_term3_comp3 = (1/2)*math.erf(bound_upper/np.sqrt(2)) #1xn
            ei_term3_comp4 = (1/2)*math.erf(bound_lower/np.sqrt(2)) #1xn     

            ei_term3_psi_upper = ei_term3_comp1 + ei_term3_comp3 #1xn
            ei_term3_psi_lower = ei_term3_comp2 + ei_term3_comp4 #1xn
            ei_term1 = ei_term1_comp1*ei_term1_comp2 #1xn

            ei_term2 = ei_term2_comp1*ei_term2_comp2 #1xn
            ei_term3 = -pred_var*(ei_term3_psi_upper-ei_term3_psi_lower) #1xn
            EI = ei_term1 + ei_term2 + ei_term3 #1xn
    else:
#         print("It's working")
        with np.errstate(divide = 'warn'):
            #Creates upper and lower bounds and described by Alex Dowling's Derivation
#             bound_a = ((y_target - pred_mean) +np.sqrt(np.exp(error_best - explore_bias)))/pred_stdev #1xn
#             bound_b = ((y_target - pred_mean) -np.sqrt(np.exp(error_best - explore_bias)))/pred_stdev #1xn
            bound_a = ((y_target - pred_mean) +np.sqrt(np.exp(error_best*explore_bias)))/pred_stdev #1xn
            bound_b = ((y_target - pred_mean) -np.sqrt(np.exp(error_best*explore_bias)))/pred_stdev #1xn
            bound_lower = np.min([bound_a,bound_b])
            bound_upper = np.max([bound_a,bound_b])
            
            args = (error_best, pred_mean, pred_stdev, y_target, explore_bias)
#             print(bound_lower,bound_upper)
#             print(error_best, pred_mean, pred_stdev, y_target, explore_bias)
            #This first way is very slow
#             ei, abs_err = integrate.quad(ei_approx_ln_term, bound_lower, bound_upper, args = args) 
            #This 2nd way throws the error -> too many values to unpack (expected 3) even though 3 values are being unpacked unless you do it like this and not, EI, abs_err, infordict =
            ei_term_1 = (error_best*explore_bias)*( norm.cdf(bound_upper)-norm.cdf(bound_lower) )
            ei_term_2_out = integrate.quad(ei_approx_ln_term, bound_lower, bound_upper, args = args, full_output = 1)
            ei_term_2 = (-2)*ei_term_2_out[0] 
#             ei_term_2 = (-1)*ei_term_2_out[0] 
            term_2_abs_err = ei_term_2_out[1]
            EI = ei_term_1 + ei_term_2
#             print(EI)
   
    ei = EI         
    return ei

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
#         SSE[i] = create_sse_data(q,train_p[i], Xexp, Yexp, obj= obj) 
        SSE[i] = create_sse_data(train_p[i], Xexp, Yexp, true_model_coefficients, obj = obj, skip_param_types = skip_param_types)

    #Define best_error as the minimum SSE or ln(SSE) value
    best_error = np.amin(SSE)
    
    return best_error 

def get_sparse_grids(dim,output=0,depth=3, rule="gauss-hermite", verbose = False, alpha = 0):
    '''
    This function shows the sparse grids generated with different rules
    Parameters:
    -----------
        dim: int, sparse grids dimension. Default is zero
        output: int, output level for function that would be interpolated
        depth: int, depth level. Controls density of abscissa points
        rule: str, quadrature rule. Default is 'gauss-legendre'
        verbose: bool, determines Whether or not plot of sparse grid is shown. False by default
        alpha: int, specifies the $\alpha$ parameter for the integration weight $\rho(x)$, ignored when rule doesn't have this parameter
    
    Returns:
    --------
        points_p: ndarray, The sparse grid points
        weights_p: ndarray, The Gauss-Legendre Quadrature Rule Weights    
    
    Other:
    ------
        A figure shows 2D sparse grids (if verbose = True)
    '''
    grid_p = Tasmanian.SparseGrid()
    grid_p.makeGlobalGrid(dim,output,depth,"level",rule)
    points_p = grid_p.getPoints()
    weights_p = grid_p.getQuadratureWeights()
    if verbose == True:
        for i in range(len(points_p)):
            plt.scatter(points_p[i,0], points_p[i,1])
            plt.title('Sparse Grid of'+rule)
        plt.show()
    return points_p, weights_p

def eval_GP_sparse_grid(Xexp, Yexp, GP_mean, GP_stdev, best_error, ep, verbose = False):
    """Evaluate GP using the spare grid instead of an approximation.
    
    Parameters
    ----------
        Xexp: ndarray, experimental x value
        Yexp: ndarray, experimental y values
        GP_mean: ndarray, Array of GP mean values at each experimental data point
        GP_stdev: ndarray, Array of GP standard deviation values at each experimental data point
        best_error: float, the best error of the 3-Input GP model
        ep: float, the exploration parameter
        verbose: bool, determines whether plot of sparse grid points is printed
    
    Returns
    ----------
        EI_Temp: float: The expected improvement of a given point 
    """
    #Back out important parameters from inputs
    n = len(Yexp) #Number of experimental data points
    
    #Obtain Sparse Grid points and weights
    points_p, weights_p = get_sparse_grids(n,output=0,depth=3, rule='gauss-hermite', verbose = False)
    
    #Initialize EI
    EI_Temp = 0
    #Loop over sparse grid weights and nodes
    for i in range(len(points_p)):
        #Initialize SSE
        SSE_Temp = 0
        #Loop over experimental data points
        for j in range(n):
            SSE_Temp += (Yexp[j] - GP_mean[j] - GP_stdev[j]*points_p[i,j])**2
#             SSE_Temp += (Yexp[j] - GP_mean[j] - ep - GP_stdev[j]*points_p[i,j])**2 #If there is an ep, need to add
        #Apply max operator  
#         EI_Temp += weights_p[i]*(-np.min(SSE_Temp - best_error,0)) #Leades to negative EIs
#         EI_Temp += weights_p[i]*(-np.min([SSE_Temp - (best_error-ep),0])) #Leads to zero EIs: #Min values is never negative, so EI is always 0
        EI_Temp += weights_p[i]*(-np.min([SSE_Temp - (best_error*ep),0]))
    return EI_Temp

def calc_ei_basic(f_best,pred_mean,pred_var, explore_bias=0.0, verbose=False):
    """ 
    Calculates the expected improvement of the 2 input parameter GP
    Parameters
    ----------
        f_best: float, the best predicted sse encountered
        pred_mean: tensor, model mean
        pred_var, tensor, model variance
        explore_bias: float, the numerical bias towards exploration, zero is the default
        verbose: True/False: Determines whether z and ei terms are printed
    
    Returns
    -------
        ei: float, The expected improvement of a given point
        If verbose == True
            z: float, The z value of the point
            ei_term_1: float, The first term of the ei equation
            ei_term_2: float, The second term of the ei equation
            norm.cdf(z): float, The CDF of the point
            norm.pdf(z): float, The PDF of the point
    """
        #Checks for equal lengths
#     assert isinstance(f_best, (np.float64,int))==True or torch.is_tensor(f_best)==True, "f_best must be a float or int"
#     assert isinstance(pred_mean, (np.float64,int))==True, "pred_mean and pred_var must be the same length"
#     assert isinstance(pred_var, (np.float64,int))==True, "pred_mean and pred_var must be the same length"
    
    #Converts tensors to np arrays and defines standard deviation
    if torch.is_tensor(pred_mean)==True:
        pred_mean = pred_mean.numpy() #1xn
    if torch.is_tensor(explore_bias)==True:
        explore_bias = explore_bias.numpy() #1xn
    if torch.is_tensor(pred_var)==True:
        pred_var = pred_var.detach().numpy() #1xn
    pred_stdev = np.sqrt(pred_var) #1xn_test
#     print("stdev",pred_stdev)
    #Checks that all standard deviations are positive
    if pred_stdev > 0:
        #Calculates z-score based on Ke's formula
#         z = (pred_mean - f_best - explore_bias)/pred_stdev #scaler
        z = (f_best*explore_bias - pred_mean)/pred_stdev #scaler
#         z = (pred_mean - f_best*explore_bias)/pred_stdev #scaler
        
        #Calculates ei based on Ke's formula
        #Explotation term
#         ei_term_1 = (pred_mean - f_best - explore_bias)*norm.cdf(z) #scaler
        ei_term_1 = (f_best*explore_bias - pred_mean)*norm.cdf(z) #scaler
#         ei_term_1 = (pred_mean - f_best*explore_bias)*norm.cdf(z) #scaler
        #Exploration Term
        ei_term_2 = pred_stdev*norm.pdf(z) #scaler
        ei = ei_term_1 +ei_term_2 #scaler
#         if verbose == True:
#             print("z",z)
#             print("Exploitation Term",ei_term_1)
#             print("CDF", norm.cdf(z))
#             print("Exploration Term",ei_term_2)
#             print("PDF", norm.pdf(z))
#             print("EI",ei,"\n")
    else:
        #Sets ei to zero if standard deviation is zero
        ei = 0
    if verbose == True:
        return ei,z, ei_term_1,ei_term_2,norm.cdf(z),norm.pdf(z)
    else:
        return ei