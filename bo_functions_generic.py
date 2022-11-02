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
        param_clean: bool, determines whether parameter values or Xexp values are being cleaned
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

def train_test_plot_preparation(param_dim, exp_data_dim, theta_set, train_p, test_p, p_True, Xexp, emulator, sparse_grid, obj, ep0, len_scl, run, save_fig, tot_iters, tot_runs, DateTime, verbose, param_dict, sep_fact = 1):  
    """
    Puts training data into a for loop to print all possible 3D angles of the training data
    
    Parameters:
    -----------
        
        theta_set: ndarray, (n x dim_param) arrays containing all values of the input parameters. Created with np.meshgrid() or LHS samples
        train_p: tensor or ndarray, The training parameter space data
        test_p: tensor or ndarray, The training parameter space data
        p_true: ndarray, A 2x1 containing the true input parameters
        Xexp: ndarray, experimental x values
        emulator: True/False, Determines if GP will model the function or the function error
        sparse_grid: True/False, True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        ep0: float, float,int,tensor,ndarray (1 value) The initial exploration bias parameter
        len_scl: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        run, int or None, The iteration of the number of times new training points have been picked
        save_fig: True/False, Determines whether figures will be saved
        tot_iters: int or None, Total number of BO Iters
        tot_runs, int or None, The total number of times new training points have been picked
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        verbose: True/False, Determines whether z_term, ei_term_1, ei_term_2, CDF, and PDF terms are saved, Default = False
        param_dict: dictionary, dictionary of names of each parameter that will be plotted named by indecie w.r.t Theta_True
        sep_fact: float, Between 0 and 1. Determines fraction of all data that will be used to train the GP. Default is 1.
    Returns:
    --------
        Prints or saves plots of starting training data for any set of dimensions
        
    """
    dim_param_list = np.linspace(0,param_dim-1,param_dim) #Note - Need to figure this out when plotting w/ multidimensional x
    mesh_combos = np.array(list(combinations(dim_param_list, 2)), dtype = int)  
    
    test_p = clean_1D_arrays(test_p, True)
    train_p = clean_1D_arrays(train_p, True)

    for i in range(len(mesh_combos)):
        indecies = mesh_combos[i]
        #Find the names of the parameter space associated with each index
        param_names_list = [param_dict[indecies[0]], param_dict[indecies[1]]]
        #Concatenate test data and train data from indecie combination
#         if len(test_p) > 0:
        test_data_piece = torch.cat((torch.reshape(test_p[:,indecies[0]],(-1,1)),torch.reshape(test_p[:,indecies[1]],(-1,1))),axis= 1)
#         else:
#             test_data_piece = test_p
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
#                 if len(test_p) > 0:
                test_data_piece = torch.cat( (test_data_piece, torch.reshape(test_p[:,indecies[param_dim-1]+(i+1)],(-1,1))), axis = 1 )
#                 else:
#                     test_data_piece = test_p
                plot_org_train(theta_set_piece,train_data_piece, test_data_piece, p_True, Xexp, emulator, sparse_grid, obj, ep0, len_scl, run, save_fig, param_names_list, tot_iters, tot_runs, DateTime, verbose, sep_fact = sep_fact)
        else:
#             print(test_data_piece)
            plot_org_train(theta_set_piece,train_data_piece, test_data_piece, p_True, Xexp, emulator, sparse_grid, obj, ep0, len_scl, run, save_fig, param_names_list, tot_iters, tot_runs, DateTime, verbose, sep_fact = sep_fact)
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