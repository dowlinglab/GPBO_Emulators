import numpy as np
import math
from scipy.stats import norm
from scipy import integrate
import torch
import csv
import gpytorch
import scipy.optimize as optimize
import pandas as pd
import os
import Tasmanian

from CS2_bo_plotters import value_plotter
from CS2_bo_plotters import plot_org_train
from CS2_bo_plotters import plot_xy
from CS2_bo_plotters import plot_Theta
from CS2_bo_plotters import plot_obj
from CS2_bo_plotters import plot_obj_abs_min
from CS2_bo_plotters import plot_3GP_performance
from CS2_bo_plotters import plot_sep_fact_min

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

def calc_y_exp(Constants_True, x, noise_std, noise_mean=0,random_seed=9):
    """
    Creates y_data (Muller Potential) for the 2 input GP function
    
    Parameters
    ----------
        Constants_True: ndarray, The array containing the true values of Muller constants
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
    
    if len(x.shape) > 1:
        len_x = x.shape[0]
    else:
        len_x = 1
    
#     print(len_x)
    #Seed Random Noise (For Bug Testing)
    if random_seed != None:
        assert isinstance(random_seed,int) == True, "Seed number must be an integer or None"
        np.random.seed(random_seed)
        
    #Creates noise values with a certain stdev and mean from a normal distribution
    noise = np.random.normal(size= 1 ,loc = noise_mean, scale = noise_std) #1x n_x
    
    # True function is Muller Potential
    A, a, b, c, x0, y0 = Constants_True
    y_exp = np.zeros(len_x)
    
    for i in range(len_x):
        X1, X2 = x[i]
        Term1 = a*(X1 - x0)**2
        Term2 = b*(X1 - x0)*(X2 - y0)
        Term3 = c*(X2 - y0)**2
        y_exp[i] = np.sum(A*np.exp(Term1 + Term2 + Term3) ) + noise
  
    return y_exp

def create_sse_data(param_space, x, y_exp, Constants, obj = "obj"):
    """
    Creates y_data for the 2 input GP function
    
    Parameters
    ----------
        param_space: ndarray, The array containing the data for Theta1 and Theta2
        x: ndarray, The list of xs that will be used to generate y
        y_exp: ndarray, The experimental data for y (the true value)
        Constants: ndarray, The array containing the true values of Muller constants
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        
    Returns:
        sum_error_sq: ndarray, The SSE or ln(SSE) values that the GP will be trained on
    """   
    if isinstance(param_space, pd.DataFrame):
        param_space = param_space.to_numpy()

    #Will need assert statement
    assert obj == "obj" or obj == "LN_obj", "Objective function choice, obj, MUST be sse or LN_sse"
    
    if len(x.shape) > 1:
        len_x = x.shape[0]
    else:
        len_x = 1
        len_a = 1
    
    len_data = len(param_space)
        
    #For the case where more than 1 point is geing generated
    #Creates an array for train_sse that will be filled with the for loop
    sum_error_sq = torch.tensor(np.zeros(len_data)) #1 x n_train^2

    #Iterates over evey combination of theta to find the SSE for each combination
    for i in range(len_data):
        #Set dig out values of a from train_p
        #Set constants to change the a row to the index of the first loop
        A, a, b, c, x0, y0 = Constants
        len_a = a.shape
        a = param_space[i]
#         print(a)
        y_sim = np.zeros(len_x)
        #Loop over state points (5)
        for j in range(len_x):
            #Calculate y_sim
            X1, X2 = x[j]
            Term1 = a*(X1 - x0)**2
            Term2 = b*(X1 - x0)*(X2 - y0)
            Term3 = c*(X2 - y0)**2
            y_sim[j] = np.sum(A*np.exp(Term1 + Term2 + Term3) )
#             print(y_sim)
#             print(y_exp)

        if obj == "obj":
            sum_error_sq[i] = sum((y_sim - y_exp)**2) #Scaler
#                 print(sum_error_sq[i])
        else:
            sum_error_sq[i] = np.log(sum((y_sim - y_exp)**2)) #Scaler
    
    return sum_error_sq

def create_y_data(param_space, Constants):
    """
    Creates y_data (training data) based on the function theta_1*x + theta_2*x**2 +x**3
    Parameters
    ----------
        param_space: (nx3) ndarray or tensor, parameter space over which the GP will be run
        Constants: ndarray, The array containing the true values of Muller constants
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
    
    try:
        len_x = x.shape[1]
    except:
        len_x = 1
    
    len_data = len(param_space)
    num_params = len(param_space.T)
        
    #For the case where more than 1 point is geing generated
    #Creates an array for train_sse that will be filled with the for loop
    #Initialize y_sim
    y_sim = np.zeros(len_data) #1 x n_train^2

    #Iterates over evey data point to find the y for each combination
    for i in range(len_data):
        #Set dig out values of a from train_p
        #Set constants to change the a row to the index of the first loop
        A, a, b, c, x0, y0 = Constants
        len_a = a.shape[0]
        a = param_space[i][0:(len_a)]
#         print(a)

        #Calculate y_sim
        X1, X2 = param_space[i][(len_a):num_params]
        Term1 = a*(X1 - x0)**2
        Term2 = b*(X1 - x0)*(X2 - y0)
        Term3 = c*(X2 - y0)**2
        y_sim[i] = np.sum(A*np.exp(Term1 + Term2 + Term3) )
   
    return y_sim


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


def gen_y_Theta_GP(x_space, Theta):
    """
    Generates an array of Best Theta Value and X to create y data
    
    Parameters
    ----------
        x_space: ndarray, array of x value
        Theta: ndarray, Array of theta values
           
    Returns
    -------
        create_y_data_space: ndarray, array of parameters [Theta, x] to be used to generate y data
        
    """
    m = x_space[0].size
    q = len(Theta)
    
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
        create_y_data_space[i,q] = x_space[i]
    #Generate y data based on parameters
    y_GP_Opt_data = create_y_data(create_y_data_space)
    return y_GP_Opt_data      

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

def find_train_doc_path(emulator, obj, t):
    """
    Finds the document that contains the correct training data based on the GP objective function and number of training inputs
    
    Parameters
    ----------
    obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
    emulator: True/False, Determines if GP will model the function or the function error
    t: int, The number of total data points
    
    Returns
    -------
    all_data_doc: csv name as a string, contains all training data for GP
    
    """
    if emulator == False:
        if obj == "obj":
            all_data_doc = "Input_CSVs/Train_Data/all_2_data/t="+str(t)+".csv"   
        else:
            all_data_doc = "Input_CSVs/Train_Data/all_2_ln_obj_data/t="+str(t)+".csv" 
    else:    
        all_data_doc = "Input_CSVs/Train_Data/all_3_data/t="+str(t)+".csv" 
            
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
    assert isinstance(y_target, float)==True, "y_target, pred_mean, and pred_var must be floats"
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

def eval_GP_emulator_BE(Xexp,Yexp, train_p, q=2, obj = "obj"):
    """ 
    Calculates the best error of the 3 input parameter GP
    Parameters
    ----------
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        train_p: ndarray (d, p x p), training data
        q: Number of theta parameter to optimize - Will need to ensure this argumnet is always accurate
        obj: str, LN_obj or obj, determines whether log or regular objective function is calculated
    Returns
    -------
        best_error: float, the best error of the 3-Input GP model
    """
    #Asserts that inputs are correct
    assert len(Xexp)==len(Yexp), "Experimental data must have same length"
    
    n = len(Xexp)
    t_train = len(train_p)
    
    #Will compare the rigorous solution and approximation later (multidimensional integral over each experiment using a sparse grid)
    SSE = np.zeros(t_train)
    for i in range(t_train):
        SSE[i] = create_sse_data(q,train_p[i], Xexp, Yexp, obj= obj) 
        #SSE[i] = create_sse_data(train_p[i], Xexp, Yexp, Constants, obj = obj)

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


def eval_GP_emulator_tot(Xexp, Yexp, theta_mesh, model, likelihood, sparse_grid, explore_bias = 0.0, verbose = False, train_p = None, obj = "obj"):
    """ 
    Calculates the expected improvement of the 3 input parameter GP
    Parameters
    ----------
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        sparse_grid: True/False: Determines whether an assumption or sparse grid method is used
        explore_bias: float, the numerical bias towards exploration, zero is the default
        verbose: bool, Determines whether output is verbose
        train_p: tensor or ndarray, The training parameter space data
        obj: str, LN_obj or obj, determines whether log or regular objective function is calculated
    
    Returns
    -------
        EI: ndarray, the expected improvement of the GP model
        SSE: ndarray, The SSE of the model 
        SSE_var_GP: ndarray, The varaince of the SSE pf the GP model
        SSE_stdev_GP: ndarray, The satndard deviation of the SSE of the GP model
        best_error: ndarray, The best_error of the GP model
    """
    #Asserts that inputs are correct
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert len(Xexp)==len(Yexp), "Experimental data must have same length"
    n = len(Xexp)
    p = theta_mesh.shape[1]
    
    #Will compare the rigorous solution and approximation later (multidimensional integral over each experiment using a sparse grid)
    #Create theta1 and theta2 mesh grids
    
    #The next 3 lines won't be needed
    theta1_mesh = theta_mesh[0]
    theta2_mesh = theta_mesh[1]
    assert len(theta2_mesh)==len(theta1_mesh), "theta_mesh must be dim, pxp arrays"
    
    #Create an array in which to store expected improvement values
    
    #These will need to be 8 dimensional and will be redefined - See below
    EI = np.zeros((p,p)) #(p1 x p2) 
    SSE_var_GP = np.zeros((p,p))
    SSE_stdev_GP = np.zeros((p,p))
    SSE = np.zeros((p,p))
    
    ##Calculate Best Error
    # Loop over theta 1
    best_error = eval_GP_emulator_BE(Xexp,Yexp, train_p, q=2, obj = obj) #Need to take q as an argument now
    
    #Create all iteration permulations - Takes a very long time for 8 dimensions
    #Theta = np.linspace(-2,2,10) (insert this instead of a a theta mesh (everything will be scaled from 0-1 in Muller problem)
    #df = pd.DataFrame(list(itertools.product(Theta, repeat=8)))
    #df2 = df.drop_duplicates()
    #theta_list = df2.to_numpy()
    
    #Alternatively can we just do an LHS here too? IF SO:
    #Set bounds and seed
    #theta_list = LHS_Design(num_points, dimensions, seed = 9, bounds = bounds)
    #dim_list = np.linspace(0,q-1,q)
    #mesh_combos = np.array(list(combinations_with_replacement(a, 2)))
    #For LHS version initialize as shape (len(mesh_combos), p1, p2) and use same method as what is here already.
    
    #Save as an array of length(theta_list) and reshape at the end
    #EI = np.zeros(len(theta_list)))
    #SSE_var_GP = np.zeros(len(theta_list)))
    #SSE_stdev_GP = np.zeros(len(theta_list)))
    #SSE = np.zeros(len(theta_list)))

    #Commented code for next few lines is for LHS method only
    #For LHS, Loop over number of theta combinations
    #for i in range(len(mesh_combos)):
    #Create meshgrid
        #theta1_mesh, theta2_mesh = np.meshgrid(LHS_reshape[int(l[i,0])],LHS_reshape[int(l[i,1])])
    #
    # Loop over theta 1
    for i in range(p): #Loop over number of combinations instead (for itertools version)
    #For i in range(theta_list.shape[0]):    
        #Loop over theta2
        for j in range(p):
            #Loop over Xexp
            #Create lists in which to store GP mean and variances
            GP_mean = np.zeros(n)
            GP_var = np.zeros(n)
            
            ##Calculate EI
            for k in range(n):
                #Caclulate EI for each value n given the best error
                point = [theta1_mesh[i,j],theta2_mesh[i,j],Xexp[k]] #point = [theta_list[i],Xexp[k]]
                eval_point = np.array([point])
                GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
                model_mean = GP_Outputs[3].numpy()[0] #1xn
                model_variance= GP_Outputs[1].detach().numpy()[0] #1xn
                GP_mean[k] = model_mean
                GP_var[k] = model_variance               
                              
                #Compute SSE and SSE variance for that point
                SSE[i,j] += (model_mean - Yexp[k])**2
                #Replace with SSE[i] = ....
                
                error_point = (model_mean - Yexp[k]) #This SSE_variance CAN be negative
                SSE_var_GP[i,j] += 2*error_point*model_variance #Error Propogation approach
                #Replace with SSE_var_GP[i] = ....
                
                if SSE_var_GP[i,j] > 0:
                    SSE_stdev_GP[i,j] = np.sqrt(SSE_var_GP[i,j])
                    #Replace with SSE_stdev_GP[i] = ....
                else:
                    SSE_stdev_GP[i,j] = np.sqrt(np.abs(SSE_var_GP[i,j]))
                    #Replace with SSE_stdev_GP[i] = ....
                    
                if sparse_grid == False:
                    #Compute EI w/ approximation
                    EI_temp = calc_ei_emulator(best_error, model_mean, model_variance, Yexp[k], explore_bias, obj)
#                     print(EI_temp)
                    EI[i,j] += EI_temp
                    #Replace with EI[i] = ....
                
            GP_stdev = np.sqrt(GP_var)
            
            #Get testing values for integration
#             if i == j == 0:
#                 print("Model mean", GP_mean)
#                 print("Model stdev", GP_stdev)
#                 print("EP", explore_bias)
#                 print("best error", best_error)
#                 print("y_target", Yexp)
                
            if sparse_grid == True:
                #Compute EI using eparse grid (Note theta_mesh not actually needed here)
                EI[i,j] = eval_GP_sparse_grid(Xexp, Yexp, GP_mean, GP_stdev, best_error, explore_bias, verbose)
                #Replace with EI[i] = ....
        
#     SSE_stdev_GP = np.sqrt(SSE_var_GP)
#     if verbose == True:
#         print(EI)

    #Reshape to correct dimensions
    #For itertools combos method
    #EI.reshape((20, 20,2)).T #(final shape should be (q,20,20)
    #For LHS method, final shape for lists will be (num_combos, 20,20)
    #SSE.reshape(...
    #SSE_var_GP.reshape(...
    #SSE_stdev_GP.reshape(...
    return EI, SSE, SSE_var_GP, SSE_stdev_GP, best_error

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

def eval_GP_basic_tot(theta_mesh, train_sse, model, likelihood, explore_bias=0.0, verbose = False):
    """ 
    Calculates the expected improvement of the 2 input parameter GP
    Parameters
    ----------
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        train_sse: ndarray (1 x t), Training data for sse
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        explore_bias: float, the numerical bias towards exploration, zero is the default
        verbose: True/False: Determines whether z and ei terms are printed
    
    Returns
    -------
        ei: ndarray, the expected improvement of the GP model
        sse: ndarray, the sse/ln(sse) of the GP model
        var: ndarray, the variance of the GP model
        stdev: ndarray, the standard deviation of the GP model
        f_best: ndarray, the best value so far
    """
        #Asserts that inputs are correct
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(train_sse, np.ndarray) or torch.is_tensor(train_sse) == True, "Train_sse must be ndarray or torch.tensor"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    
    #Calculate and save best error
    #Negative sign because -max(-train_sse) = min(train_sse)
    best_error = -max(-train_sse).numpy() 
#     best_error = max(-train_sse).numpy()

    p = theta_mesh.shape[1]
    
    #These will be redone
    #Initalize matricies to save GP outputs and calculations using GP outputs
    ei = np.zeros((p,p))
    sse = np.zeros((p,p))
    var = np.zeros((p,p))
    stdev = np.zeros((p,p))

    if verbose == True:
        z_term = np.zeros((p,p))
        ei_term_1 = np.zeros((p,p))
        ei_term_2 = np.zeros((p,p))
        CDF = np.zeros((p,p))
        PDF = np.zeros((p,p))
        
        
    #Create all iteration permulations - Takes a very long time for 8 dimensions
    #Theta = np.linspace(-2,2,10) (insert this instead of a a theta mesh (everything will be scaled from 0-1 in Muller problem)
    #df = pd.DataFrame(list(itertools.product(Theta, repeat=8)))
    #df2 = df.drop_duplicates()
    #theta_list = df2.to_numpy()
    
    #Alternatively can we just do an LHS here too? IF SO:
    #Set bounds and seed
    #theta_list = LHS_Design(num_points, dimensions, seed = 9, bounds = bounds)
    #dim_list = np.linspace(0,q-1,q)
    #mesh_combos = np.array(list(combinations_with_replacement(a, 2)))
    #For LHS version initialize as shape (len(mesh_combos), p1, p2) and use same method as what is here already.
    
    #Commented code for next few lines is for LHS method only
    #For LHS, Loop over number of theta combinations
    #for i in range(len(mesh_combos)):
    #Create meshgrid
        #theta1_mesh, theta2_mesh = np.meshgrid(LHS_reshape[int(l[i,0])],LHS_reshape[int(l[i,1])])
    
    #Save as an array of length(theta_list) and reshape at the end
    #ei = np.zeros(len(theta_list)))
    #sse = np.zeros(len(theta_list)))
    #var = np.zeros(len(theta_list)))
    #stdev = np.zeros(len(theta_list)))
#     if verbose == True:
#         z_term = np.zeros(len(theta_list))
#         ei_term_1 = np.zeros(len(theta_list))
#         ei_term_2 = np.zeros(len(theta_list))
#         CDF = np.zeros(len(theta_list))
#         PDF = np.zeros(len(theta_list))
    
    #Separate Theta_mesh (We will delete these)
    theta1_mesh = theta_mesh[0]
    theta2_mesh = theta_mesh[1]
    
    for i in range(p):
    #For i in range(theta_list.shape[0]) (Loop over just combos instead)
        #Loop over Theta_2
        for j in range(p):
            #Choose and evaluate point
            point = [theta1_mesh[i,j],theta2_mesh[i,j]]
            #point = [theta_list[i]]
            eval_point = np.array([point])
            GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
            model_sse = GP_Outputs[3].numpy()[0] #1xn
            model_variance= GP_Outputs[1].detach().numpy()[0] #1xn
#             if verbose == True:
#                 print("Point",eval_point)
#                 print("Model Mean",model_sse)
#                 print("Model Var", model_variance)
            #Save GP outputs
            sse[i,j] = model_sse
            #sse[i] = ...
            var[i,j] = model_variance
            #var[i] = ...
            stdev[i,j] = np.sqrt(model_variance)  
            #stdev[i] = ...

            #Negative sign because -max(-train_sse) = min(train_sse)
            #Print and save certain values based on verboseness
            if verbose == True:
                out1, out2, out3, out4, out5, out6 = calc_ei_basic(best_error,model_sse,model_variance,explore_bias,verbose)
#                 out1, out2, out3, out4, out5, out6 = calc_ei_basic(best_error,-model_sse,model_variance,explore_bias,verbose)
                ei[i,j] = out1
                #ei[i] = ...
                z_term[i,j] = out2
                #z_term[i] = ...
                ei_term_1[i,j] = out3
                #ei_term_1[i] = ...
                ei_term_2[i,j] = out4
                #ei_term_2[i] = ...
                CDF[i,j] = out5
                #CDF[i] = ...
                PDF[i,j] = out6
                #PDF[i] = ...
            else:
                ei[i,j] = calc_ei_basic(best_error,model_sse,model_variance,explore_bias,verbose)
                #ei[i] = ...
#                 ei[i,j] = calc_ei_basic(best_error,-model_sse,model_variance,explore_bias,verbose)
    #Reshape Variables
    #ei.reshape((20, 20,2)).T (and others)
    
    if verbose == True:
        return ei, sse, var, stdev, best_error, z_term, ei_term_1, ei_term_2, CDF, PDF
    else:
        return ei, sse, var, stdev, best_error #Prints just the value
#         return ei, sse, var, stdev, f_best

def find_opt_and_best_arg(theta_mesh, sse, ei, train_p): #Not quite sure how to fix setting of points yet
    """
    Finds the Theta value where min(sse) or min(-ei) is true using argmax and argmin
    
    Parameters:
    -----------
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        sse: ndarray (d, p x p), meshgrid of sse values for all points in theta_mesh
        ei: ndarray (d, p x p), meshgrid of ei values for all points in theta_mesh
        train_p: tensor or ndarray, The training parameter space data
    
    Returns:
    --------
        Theta_Opt_GP: ndarray, The point where the objective function is minimized in theta_mesh
        Theta_Best: ndarray, The point where the ei is maximized in theta_mesh
    """
    theta1_mesh = theta_mesh[0]
    theta2_mesh = theta_mesh[1]
    
    #Point that the GP thinks is best has the lowest SSE
    #Find point in sse matrix where sse is lowest (argmin(SSE))
    argmin = np.array(np.where(np.isclose(sse, np.amin(sse),atol=np.amin(sse)*1e-6)==True))
    
    #ensures that only one point is used if multiple points yield a minimum
    
    if len(argmin[0]) > 1:
#         rand_ind = np.random.randint(np.max(argmin[0])) #Chooses a random point with the minimum value
#         argmin = argmin[:,rand_ind]
        argmin = np.array([[argmin[0,1]],[argmin[1,1]]]) #Replace with above
    
    #Find theta value corresponding to argmin(SSE)
    
    #Initialize Theta_Opt_GP
    #Theta_Opt_GP = np.zeros(sse.shape[0]) #2,4,8,12,etc...
    #Loop over number of thetas
    #for i in range(sse.shape[0]
        #Theta_Opt_GP[i] = theta_mesh[i,argmin]
        #Will this syntax actually work?
        #How to actually back out thetas? Should I still be using a mesh at all?
    Theta_1_Opt = float(theta1_mesh[argmin[0],argmin[1]])
    Theta_2_Opt = float(theta2_mesh[argmin[0],argmin[1]])
    Theta_Opt_GP = np.array((Theta_1_Opt,Theta_2_Opt))
    
    #calculates best theta value
    #Find point in ei matrix where ei is highest (argmax(EI))
    argmax = np.array(np.where(np.isclose(ei, np.amax(ei),atol=np.amax(ei)*1e-6)==True))

    #ensures that only one point is used if multiple points yield a maximum
    if len(argmax[0]) > 1:
        argmax = argmax_multiple(argmax, train_p, theta_mesh)
            
    #Find theta value corresponding to argmax(EI)
    #Initialize Theta_Best
    #Theta_Best = np.zeros(sse.shape[0]) #2,4,8,12,etc...
    #Loop over number of thetas
    #for i in range(sse.shape[0]
        #Theta_Best[i] = theta_mesh[i,argmax]
        #Will this syntax actually work?
        #How to actually back out thetas? Should I still be using a mesh at all?
    Theta_1_Best = float(theta1_mesh[argmax[0],argmax[1]])
    Theta_2_Best = float(theta2_mesh[argmax[0],argmax[1]])
    Theta_Best = np.array((Theta_1_Best,Theta_2_Best))
    
    return Theta_Best, Theta_Opt_GP

def argmax_multiple(argmax, train_p, theta_mesh): #not sure how to fix setting of points here either
    """
    Finds the best ei point argument when more than one point has the maximum ei
    
    Parameters:
    -----------
        argmax: ndarray, The indecies of all parameters that have the maximum ei
        train_p: tensor or ndarray, The training parameter space data
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        
    Returns:
    --------
        argmax_best: ndarray, The indecies of the parameters that have the maximum ei that is furthest from the rest of the training points
    """
    #Initialize max distance and theta arrays
    max_distance_sq = 0
    q = len(argmax)
    
    #Initialize 
    theta1_mesh = theta_mesh[0]
    theta2_mesh = theta_mesh[1]
    #argmax_best = np.zeros(len(argmin[:,0]))
    argmax_best = np.array([[0],[0]])
    #Only use this algorithm when >1 points have the max ei
    #Create avg x y pt for training data
    #train_T12_avg = np.average(train_p, axis =0)
    train_T12_avg = np.array([np.average(train_p[:,0]), np.average(train_p[:,1])])
#     assert len(argmax[0]) == len(argmax[1]), "Ensure argmax arrays are the same length"

    #Check each point in argmax with all training points and find max distance
    #Loop over all coord points
    for i in range(len(argmax[0])):
        #Create the corresponding argmax point that maps to theta 1 and theta 2 values
        # point= np.array(argmin[:,i])
        point = np.array([[argmax[0,i]],[argmax[1,i]]])

        #Find theta value corresponding to argmax(EI)
        
        #Initialize Theta_Arr
        #Theta_Arr = np.zeros(len(point))
        #Loop over # of thetas and add best thetas to the list
        #for i in range(len(point)):
            #Theta_Arr[i] = float(theta_mesh[point])
        Theta_1 = float(theta1_mesh[point[0],point[1]])
        Theta_2 = float(theta2_mesh[point[0],point[1]])
        Theta_Arr = np.array((Theta_1,Theta_2))

        #Calculate Distance
        #distance_sq = np.sum((train_T12_avg - Theta_Arr)**2)
        distance_sq = (train_T12_avg[0] - Theta_Arr[0])**2 + (train_T12_avg[1] - Theta_Arr[1])**2

        #Set distance to max distance if it is applicable. At the end of the loop, argmax will be the point with the greatest distance.
        if distance_sq > max_distance_sq:
            max_distance_sq = distance_sq
            argmax_best = point
            
    return argmax_best
             
##FOR USE WITH SCIPY##################################################################
def eval_GP_scipy(theta_guess, train_sse, train_p, Xexp,Yexp, theta_mesh, model, likelihood, emulator, sparse_grid, explore_bias=0.0, ei_sse_choice = "ei", verbose = False, obj = "obj"):
    """ 
    Calculates either -ei or sse (a function to be minimized). To be used in calculating best and optimal parameter sets.
    Parameters
    ----------
        theta_guess: ndarray (1xp), The theta value that will be guessed to optimize 
        train_sse: ndarray (1 x t), Training data for sse
        train_p: tensor or ndarray, The training parameter space data
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        emulator: True/False: Determines whether the GP is a property emulator of error emulator
        sparse_grid: True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        explore_bias: float, Exploration parameter used for calculating 2-Input GP expected improvement
        ei_sse_choice: "neg_ei" or "sse" - Choose which one to optimize
        verbose: True/False - Determines verboseness of output
        obj: str, LN_obj or obj, determines whether log or regular objective function is calculated
    
    Returns
    -------
        -ei: ndarray, the negative expected improvement of the GP model
        OR
        sse: ndarray, the sse/ln(sse) of the GP model
        
    """
        #Asserts that inputs are correct
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(train_sse, np.ndarray) or torch.is_tensor(train_sse) == True, "Train_sse must be ndarray or torch.tensor"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert ei_sse_choice == "neg_ei" or ei_sse_choice == "sse", "ei_sse_choice must be string 'ei' or 'sse'"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    
    #Separates meshgrid
    q = len(theta_guess)
    p = theta_mesh.shape[1] #Infer from something else
    n = len(Xexp)
    
    theta1_guess = theta_guess[0]
    theta2_guess = theta_guess[1]

    #Evaluate a point with the GP and save values for GP mean and var
    if emulator == False:
        point = [theta1_guess,theta2_guess]
        #point = [theta_guess]
        eval_point = np.array([point])
        GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
        model_sse = GP_Outputs[3].numpy()[0] #1xn 
        model_variance= GP_Outputs[1].detach().numpy()[0] #1xn

        #Calculate best error and sse
        #Does the objective function change this? No - As long as they're both logs this will work
        best_error = -max(-train_sse) #Negative sign because -max(-train_sse) = min(train_sse)
#         best_error = max(-train_sse) #Negative sign because -max(-train_sse) = min(train_sse)
        sse = model_sse
            #Calculate ei. If statement depends whether ei is the only thing returned by calc_ei_basic function
        if verbose == True:
            ei = calc_ei_basic(best_error,model_sse,model_variance,explore_bias,verbose)[0]
#             ei = calc_ei_basic(best_error,-model_sse,model_variance,explore_bias,verbose)[0]
        else:
            ei = calc_ei_basic(best_error,model_sse,model_variance,explore_bias,verbose)
#             ei = calc_ei_basic(best_error,-model_sse,model_variance,explore_bias,verbose)
    
    else:
        ei = 0
        sse = 0
        best_error = eval_GP_emulator_BE(Xexp,Yexp, train_p, q)
        GP_mean = np.zeros(n)
        GP_stdev = np.zeros(n)
        for k in range(n):
            #Caclulate EI for each value n given the best error
            point = [theta1_guess,theta2_guess,Xexp[k]]
            eval_point = np.array([point])
            GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
            model_mean = GP_Outputs[3].numpy()[0] #1xn
            model_variance= GP_Outputs[1].detach().numpy()[0] #1xn
            
            GP_mean[k] = model_mean
            GP_stdev[k] = np.sqrt(model_variance) 
            sse += (model_mean - Yexp[k])**2

            if sparse_grid == False:
                #Compute EI w/ approximation
                ei += calc_ei_emulator(best_error, model_mean, model_variance, Yexp[k], explore_bias, obj)
           
        if sparse_grid == True:
            #Compute EI using sparse grid #Note theta_mesh not actually needed here
            ei = eval_GP_sparse_grid(Xexp, Yexp, GP_mean, GP_stdev, best_error, explore_bias)
                
            
    #Return either -ei or sse as a minimize objective function
    if ei_sse_choice == "neg_ei":
#         print("EI chosen")
        return -ei #Because we want to maximize EI and scipy.optimize is a minimizer by default
    else:
#         print("sse chosen")
        return sse #We want to minimize sse or ln(sse)

def find_opt_best_scipy(Xexp, Yexp, theta_mesh, train_y,train_p, theta0_b,theta0_o,sse,ei,model,likelihood,explore_bias,emulator,sparse_grid,obj):
    """
    Finds the Theta value where min(sse) or min(-ei) is true using scipy.minimize and the L-BFGS-B method
    
    Parameters:
    -----------
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        train_y: tensor or ndarray, The training y data
        train_p: tensor or ndarray, The training parameter space data
        theta0_b: Initial guess of the Theta value where ei is maximized
        theta0_o: Initial guess of the Theta value where sse is minimized
        sse: ndarray (d, p x p), meshgrid of sse values for all points in theta_mesh
        ei: ndarray (d, p x p), meshgrid of ei values for all points in theta_mesh
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        explore_bias: float,int,tensor,ndarray (1 value) The exploration bias parameter
        emulator: True/False: Determines whether the GP is a property emulator of error emulator
        sparse_grid: True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        obj: ob or LN_obj: Determines which objective function is used for the 2 input GP
    
    Returns:
    --------
        theta_b: ndarray, The point where the objective function is minimized in theta_mesh
        theta_o: ndarray, The point where the ei is maximized in theta_mesh
    """
    #Assert statements to ensure no bugs
    assert isinstance(train_y, np.ndarray) or torch.is_tensor(train_y) == True, "Train_sse must be ndarray or torch.tensor"
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert len(theta0_b) == len(theta0_o), "Initial guesses must be the same length."
    
    #Set theta meshes and bounds
    theta1_mesh = theta_mesh[0]
    theta2_mesh = theta_mesh[1]
    
    # bound = np.array([0,1])
    # bounds = (np.repeat(x,8))
    # bnds = bounds.reshape(-1,8).T
    bnds = [[np.amin(theta1_mesh), np.amax(theta1_mesh)], [np.amin(theta2_mesh), np.amax(theta2_mesh)]]
    
    #Use L-BFGS Method with scipy.minimize to find theta_opt and theta_best
    ei_sse_choice1 ="neg_ei"
    ei_sse_choice2 = "sse"
    
    #Set arguments and calculate best and optimal solutions
    #remove theta_mesh from these eventually
    argmts_best = ((train_y, train_p, Xexp, Yexp, theta_mesh, model, likelihood, emulator, sparse_grid, explore_bias, ei_sse_choice1))
    argmts_opt = ((train_y, train_p, Xexp, Yexp, theta_mesh, model, likelihood, emulator, sparse_grid, explore_bias, ei_sse_choice2))
    Best_Solution = optimize.minimize(eval_GP_scipy, theta0_b,bounds=bnds,method = "L-BFGS-B",args=argmts_best)
    Opt_Solution = optimize.minimize(eval_GP_scipy, theta0_o,bounds=bnds,method = "L-BFGS-B",args=argmts_opt)
    
    #save best and optimal values
    theta_b = Best_Solution.x
    theta_o = Opt_Solution.x  
    
    return theta_b, theta_o

# def eval_GP(theta_mesh, train_y, explore_bias, Xexp, Yexp, model, likelihood, verbose, emulator, sparse_grid, set_lengthscale):  
def eval_GP(theta_mesh, train_y, explore_bias, Xexp, Yexp, model, likelihood, verbose, emulator, sparse_grid, set_lengthscale, train_p = None, obj = "obj"):
    """
    Evaluates GP
    
    Parameters:
    -----------
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        train_y: tensor or ndarray, The training y data
        explore_bias: float,int,tensor,ndarray (1 value) The exploration bias parameter
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        verbose: True/False: Determines whether z_term, ei_term_1, ei_term_2, CDF, and PDF terms are saved
        emulator: True/False: Determiens whether GP is an emulator of the function
        sparse_grd: True/False: Determines whether an assumption or sparse grid is used
        set_lengthscale: float/None: Determines whether Hyperparameter values will be set
        train_p: tensor or ndarray, The training parameter space data
        obj: ob or LN_obj: Determines which objective function is used for the 2 input GP
    
    Returns:
    --------
        eval_components: ndarray, The componenets evaluate by the GP. ei, sse, var, stdev, f_best, (z_term, ei_term_1, ei_term_2, CDF, PDF)
    """
    #Find Number of training points
    p = theta_mesh.shape[1]
    
    assert isinstance(train_y, np.ndarray) or torch.is_tensor(train_y) == True, "Train_sse must be ndarray or torch.tensor"
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    ##Set Hyperparameters to 1
    if isinstance(train_y, np.ndarray)==True:
        train_y = torch.tensor(train_y) #1xn
    
    #Set hyperparameters
    if set_lengthscale is not None:
        if verbose == True:
            print("Lengthscale Set To: " + set_lengthscale)
        outputscale = torch.tensor([1])
        lengthscale = torch.tensor([set_lengthscale])
        noise = torch.tensor([0.1])

        model.likelihood.noise = noise
        model.covar_module.base_kernel.lengthscale =lengthscale
        model.covar_module.outputscale = outputscale
    
    model.eval()
    #Puts likelihood in evaluation mode
    likelihood.eval()
    
    #Evaluate GP based on error emulator or property emulator
    if emulator == False:
        eval_components = eval_GP_basic_tot(theta_mesh, train_y, model, likelihood, explore_bias, verbose)
    else:
#         eval_components = eval_GP_emulator_tot(Xexp,Yexp, theta_mesh, model, likelihood, sparse_grid, explore_bias, verbose)
        eval_components = eval_GP_emulator_tot(Xexp,Yexp, theta_mesh, model, likelihood, sparse_grid, explore_bias, verbose, train_p, obj)
    
    return eval_components

def bo_iter(BO_iters,train_p,train_y,theta_mesh,Theta_True,train_iter,explore_bias, Xexp, Yexp, noise_std, obj, run, sparse_grid, emulator, set_lengthscale, verbose = False,save_fig=False, tot_runs = 1, DateTime=None, test_p = None, sep_fact = 0.8):
    """
    Performs BO iterations
    
    Parameters:
    -----------
        BO_iters: integer, number of BO iteratiosn
        train_p: tensor or ndarray, The training parameter space data
        train_y: tensor or ndarray, The training y data
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        Theta_True: ndarray, The array containing the true values of Theta1 and Theta2
        train_iter: int, number of training iterations to run. Default is 300
        explore_bias: float,int,tensor,ndarray (1 value) The exploration bias parameter
        Xexp: ndarray, The list of xs that will be used to generate y
        Yexp: ndarray, The experimental data for y (the true value)
        noise_std: float, int: The standard deviation of the noise
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        run: int, The iteration of the number of times new training points have been picked
        sparse_grid: True/False: Determines whether a sparse grid or approximation is used for the GP emulator
        emulator: True/False, Determines if GP will model the function or the function error
        set_lengthscale: float or None, Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        verbose: True/False, Determines whether z_term, ei_term_1, ei_term_2, CDF, and PDF terms are saved, Default = False
        save_fig: True/False, Determines whether figures will be saved
        tot_runs: The total number of runs to perform
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        test_p: None, tensor, or ndarray, The testing parameter space data. Default None
        sep_fact: float, Between 0 and 1. Determines fraction of all data that will be used to train the GP. Default is 1.
        
        
    Returns:
    --------
        All_Theta_Best: ndarray, Array of all Best Theta values (as determined by max(ei)) for each iteration 
        All_Theta_Opt: ndarray, Array of all Optimal Theta values (as determined by min(sse)) for each iteration
        All_SSE: ndarray, Array of all minimum SSE values (as determined by min(sse)) for each iteration
        All_SSE_abs_min: ndarray, Array of the absolute minimum SSE values (as determined by min(sse)) at each iteration 
        Total_BO_iters: int, The number of BO iteration actually completed    
    """
    #Assert Statments
    assert all(isinstance(i, int) for i in [BO_iters, train_iter]), "BO_iters and train_iter must be integers"
    assert len(train_p) == len(train_y), "Training data must be the same length"
    assert len(Xexp) == len(Yexp), "Experimental data must have the same length"
    assert obj == "obj" or obj == "LN_obj", "Objective function choice, obj, MUST be sse or LN_sse"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    assert emulator==True or emulator==False, "Verbose must be True/False"
    
    #Find parameters
    m = Xexp[0].size #Dimensions of X
    n = len(Xexp) #Length of experimental data
    q = len(Theta_True) #Number of parameters to regress
    p = theta_mesh.shape[1] #Number of points to evaluate the GP at in any dimension of q
    t = int(len(train_p)) + int(len(test_p)) #Original length of all data
    ep0 = explore_bias
    
    #Set arrays to track theta_best, theta_opt, and SSE for every BO iteration
    All_Theta_Best = np.zeros((BO_iters,q)) 
    All_Theta_abs_Opt = np.zeros((BO_iters,q))
    All_Theta_Opt = np.zeros((BO_iters,q)) 
    All_SSE = np.zeros(BO_iters) #Will save ln(SSE) values
    All_SSE_abs_min = np.zeros(BO_iters) #Will save ln(SSE) values  
    All_Max_EI = np.zeros(BO_iters) #Used in stopping criteria
    Total_BO_iters = BO_iters

    #Ensures GP will take correct # of inputs
    if emulator == True:
        GP_inputs = q+m
        assert len(train_p.T) ==q+m, "train_p must have the same number of dimensions as the value of q+m"
    else:
        GP_inputs = q
        assert len(train_p.T) ==q, "train_p must have the same number of dimensions as the value of q"
    
    mean_of_var = 0
    best_error_num = 0
    ep_init = explore_bias
    
    #Loop over # of BO iterations
    for i in range(BO_iters):
        #Converts numpy arrays to tensors
        if torch.is_tensor(train_p) != True:
            train_p = torch.from_numpy(train_p)
        if torch.is_tensor(train_y) != True:
            train_y = torch.from_numpy(train_y)
        if torch.is_tensor(test_p) != True:
            test_p = torch.from_numpy(test_p)
            
        #Redefine likelihood and model based on new training data
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_p, train_y, likelihood)
        
        #Train GP
        train_GP = train_GP_model(model, likelihood, train_p, train_y, train_iter, verbose=False)
        
        #Evaluate GP
#         eval_components = eval_GP(theta_mesh, train_y, explore_bias,Xexp, Yexp, model, likelihood, verbose, emulator, sparse_grid, set_lengthscale)

        #Set Exploration parameter
#         explore_bias = explore_parameter(i, explore_bias, mean_of_var, best_error_num, ep_o = ep_init, ep_method = "Constant") #Defaulting to exp method
        explore_bias = ep_init #Sets ep to the multiplicative scaler between 0.1 and 1 
        
        eval_components = eval_GP(theta_mesh, train_y, explore_bias, Xexp, Yexp, model, likelihood, verbose, emulator, sparse_grid, set_lengthscale, train_p, obj = obj)
        
        #Determines whether debugging parameters are saved for 2 Input GP       
        if verbose == True and emulator == False:
            ei,sse,var,stdev,best_error,z,ei_term_1,ei_term_2,CDF,PDF = eval_components
        else:
            ei,sse,var,stdev,best_error = eval_components
        
        All_Max_EI[i] = np.max(ei)
        
        mean_of_var = np.average(var)
#         print("MOV",mean_of_var)
        best_error_num = best_error
        
        #Use argmax(EI) and argmin(SSE) to find values for Theta_best and theta_opt
        Theta_Best, Theta_Opt_GP = find_opt_and_best_arg(theta_mesh, sse, ei, train_p)
        theta0_b = Theta_Best
        theta0_o = Theta_Opt_GP
#         theta0_b = np.array([0.95,-0.95])
#         theta0_o = np.array([0.95,-0.95])

        #Use argmax/argmin method as initial guesses for use with scipy.minimize
        theta_b, theta_o = find_opt_best_scipy(Xexp, Yexp, theta_mesh, train_y, train_p, theta0_b,theta0_o,sse,ei, model,likelihood,explore_bias, emulator,sparse_grid,obj)
        
        #Save theta_best and theta_opt values for iteration
        All_Theta_Best[i], All_Theta_Opt[i] = theta_b, theta_o
        
        #Calculate values of y given the GP optimal theta values
        y_GP_Opt = gen_y_Theta_GP(Xexp, theta_o)
        
        #Calculate GP SSE and save value
        ln_error_mag = np.log(np.sum((y_GP_Opt-Yexp)**2)) #Should SSE be calculated like this or should we use the GP approximation?
        
#       sse_opt = eval_GP_scipy(theta_o, train_p, Xexp,Yexp, theta_mesh, model, likelihood, emulator, sparse_grid, explore_bias, ei_sse_choice = "sse", verbose = False)
#         ln_error_mag = sse_opt

        All_SSE[i] = ln_error_mag
        
        #Save best value of SSE for plotting 
        if i == 0:
            All_SSE_abs_min[i] = ln_error_mag
            All_Theta_abs_Opt[i] = theta_o
            improvement = False
#             All_SSE_abs_min[i] = sse_opt
        else:
            if All_SSE_abs_min[i-1] >= ln_error_mag:
                All_SSE_abs_min[i] = ln_error_mag
                All_Theta_abs_Opt[i] = theta_o
                improvement = True
            else: 
                All_SSE_abs_min[i] = All_SSE_abs_min[i-1]
                All_Theta_abs_Opt[i] = theta_o
                improvement = False
        
        #Prints certain values at each iteration if verbose is True
        if verbose == True:
            print("BO Iteration = ", i+1)
#             Jas_ep = explore_parameter(i, explore_bias, mean_of_var, best_error_num, ep_o = ep_init, ep_method = "Jasrasaria")
#             print("Jasrasaria EP:", Jas_ep)
#             Boy_ep = explore_parameter(i, explore_bias, mean_of_var, best_error_num, ep_o = ep_init, ep_method = "Boyle", improvement = improvement)
#             print("Boyle EP:", Boy_ep)
#             Exp_ep = explore_parameter(i, explore_bias, mean_of_var, best_error_num, ep_o = ep_init)
#             print("Exp EP:", Exp_ep)
#             print("Exploration Bias = ",explore_bias)
            print("Exploration Bias Factor = ",explore_bias)
            print("Scipy Theta Best = ",theta_b)
            print("Argmax Theta Best = ",Theta_Best)
            print("Scipy Theta Opt = ",theta_o)
            print("Argmin Theta_Opt_GP = ",Theta_Opt_GP)
            print("EI_max =", np.amax(ei), "\n")
        
        #Prints figures if more than 1 BO iter is happening
        if emulator == False:
            titles = ['E(I(\\theta))','log(e(\\theta))','\sigma^2','\sigma','Best_Error','z','EI_term_1','EI_term_2','CDF','PDF']  
            titles_save = ["EI","ln(SSE)","Var","StDev","Best_Error","z","ei_term_1","ei_term_2","CDF","PDF"] 
        else:
            titles = ['E(I(\\theta))','log(e(\\theta))','\sigma^2', '\sigma', 'Best_Error']  
            titles_save = ["EI","ln(SSE)","Var","StDev","Best_Error"] 
        
        #Plot and save figures for all figrues for EI and SSE
        value_plotter(theta_mesh, ei, Theta_True, theta_o, theta_b, train_p, titles[0],titles_save[0], obj, ep0, emulator, sparse_grid, set_lengthscale, save_fig, i, run, BO_iters, tot_runs, DateTime, t, sep_fact = sep_fact)
        
        #Ensure that a plot of SSE (and never ln(SSE)) is drawn
        if obj == "LN_obj" and emulator == False:
            ln_sse = sse
        else:
            ln_sse = np.log(sse)
            
        value_plotter(theta_mesh, ln_sse, Theta_True, theta_o, theta_b, train_p, titles[1], titles_save[1], obj, ep0, emulator, sparse_grid, set_lengthscale, save_fig, i, run, BO_iters, tot_runs, DateTime, t, sep_fact = sep_fact)
        
        #Save other figures
        for j in range(len(eval_components)-2):
            component = eval_components[j+2]
            title = titles[j+2]
            title_save = titles_save[j+2]
            try:
                value_plotter(theta_mesh, component, Theta_True, theta_o, theta_b, train_p, title, title_save, obj, ep0, emulator, sparse_grid, set_lengthscale, save_fig, i, run, BO_iters, tot_runs, DateTime, t, sep_fact = sep_fact)
            except:
                Best_Error_Found = np.round(eval_components[j+2],4)
                if verbose == True:
                    print("Best Error is:", Best_Error_Found)

        ##Append best values to training data 
        #Convert training data to numpy arrays to allow concatenation to work
        train_p = train_p.numpy() #(q x t)
        train_y = train_y.numpy() #(1 x t)
        
        if  i > 0:
            #Change to 1e-7
            if abs(All_Max_EI[i-1]) <= 1e-10 and abs(All_Max_EI[i]) <= 1e-10:
                Total_BO_iters = i+1
                break
        
        if emulator == False:   
            #Call the expensive function and evaluate at Theta_Best
            sse_Best = create_sse_data(q,theta_b, Xexp, Yexp, obj) #(1 x 1)
            #create_sse_data(theta_b, Xexp, Yexp, Constants, obj)
            #Add Theta_Best to train_p and y_best to train_y
            train_p = np.concatenate((train_p, [theta_b]), axis=0) #(q x t)
            train_y = np.concatenate((train_y, [sse_Best]),axis=0) #(1 x t)
            
        else:
            #Loop over experimental data
            for k in range(n):
                Best_Point = theta_b
                Best_Point = np.append(Best_Point, Xexp[k])
                #Create y-value/ experimental data ---- #Should use calc_y_exp correct?
                y_Best = calc_y_exp(theta_b, Xexp[k], noise_std, noise_mean=0,random_seed=6)
                #y_Best = calc_y_exp(Constants_True, Xexp[k], noise_std)
                train_p = np.append(train_p, [Best_Point], axis=0) #(q x t)
                train_y = np.append(train_y, [y_Best]) #(1 x t)
        
        if verbose == True:
            print("Magnitude of ln(SSE) given Theta_Opt = ",theta_o, "is", "{:.4e}".format(ln_error_mag))
    
    #Plots a single line of objective/theta values vs BO iteration if there are no runs
    if tot_runs == 1 and verbose == True:
        #Plot X vs Y for Yexp and Y_GP
        X_line = np.linspace(np.min(Xexp),np.max(Xexp),100)
        y_true = calc_y_exp(Theta_True, X_line, noise_std = noise_std, noise_mean=0)
        #y_true = calc_y_exp(Constants_True, X_line, noise_std)
        y_GP_Opt_100 = gen_y_Theta_GP(X_line, theta_o)   
        plot_xy(X_line,Xexp, Yexp, y_GP_Opt,y_GP_Opt_100,y_true)
              
    return All_Theta_Best, All_Theta_Opt, All_SSE, All_SSE_abs_min, Total_BO_iters

def bo_iter_w_runs(BO_iters,all_data_doc,t,theta_mesh,Theta_True,train_iter,explore_bias, Xexp, Yexp, noise_std, obj, runs, sparse_grid, emulator,set_lengthscale, verbose = True,save_fig=False, shuffle_seed = None, DateTime=None, sep_fact = 1):
    """
    Performs BO iterations with runs. A run contains of choosing different initial training data.
    
    Parameters:
    -----------
        BO_iters: integer, number of BO iterations
        all_data_doc: csv name as a string, contains all training data for GP
        t: int, Number of total points to use
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        Theta_True: ndarray, The array containing the true values of Theta1 and Theta2
        train_iter: int, number of training iterations to run. Default is 300
        explore_bias: float,int,tensor,ndarray (1 value) The initial exploration bias parameter
        Xexp: ndarray, The list of xs that will be used to generate y
        Yexp: ndarray, The experimental data for y (the true value)
        noise_std: float, int: The standard deviation of the noise
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        runs: int, The number of times to choose new training points
        sparse_grid: Determines whether a sparse grid or approximation is used for the GP emulator
        set_lengthscale: float or None, Value of the lengthscale hyperparameter - None if hyperparameters will be updated during training
        emulator: True/False, Determines if GP will model the function or the function error
        verbose: True/False, Determines whether z_term, ei_term_1, ei_term_2, CDF, and PDF terms are saved, Default = False
        save_fig: True/False, Determines whether figures will be saved
        shuffle_seed, int, number of seed for shuffling training data. Default is None. 
        DateTime: str or None, Determines whether files will be saved with the date and time for the run, Default None
        sep_fact: float, Between 0 and 1. Determines fraction of all data that will be used to train the GP. Default is 1.
        
    Returns:
    --------
        bo_opt: int, The BO iteration at which the lowest SSE occurs
        run_opt: int, The run at which the lowest SSE occurs
        Theta_Opt_all: ndarray, the theta values/parameter set that maps to the lowest SSE
        SSE_abs_min: float, the absolute minimum SSE found
        Theta_Best_all: ndarray, the theta values/parameter set that maps to the highest EI
    
    """
    #Assert statements
    assert all(isinstance(i, int) for i in [BO_iters, t,runs,train_iter]), "BO_iters, t, runs, and train_iter must be integers"
    assert BO_iters > 0, "Number of BO Iterations must be greater than 0!"
    assert len(Xexp) == len(Yexp), "Experimental data must have the same length"
    assert obj == "obj" or obj == "LN_obj", "Objective function choice, obj, MUST be sse or LN_sse"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    assert emulator==True or emulator==False, "Verbose must be True/False"
    assert isinstance(runs, int) == True, "Number of runs must be an integer"
    
    #Find constants
#     m = Xexp[0].size #Dimensions of X
    q = len(Theta_True) #Number of parameters to regress
    p = theta_mesh.shape[1] #Number of training points to evaluate in each dimension of q
    ep0 = explore_bias
    
    dim = m+q #dimensions in a CSV
    #Read data from a csv
    all_data = np.array(pd.read_csv(all_data_doc, header=0,sep=",")) 
    
    #Initialize Theta and SSE matricies
    Theta_Opt_matrix = np.zeros((runs,BO_iters,q))
    Theta_Best_matrix = np.zeros((runs,BO_iters,q))
    SSE_matrix = np.zeros((runs,BO_iters)) #Saves ln(SSE) values
    EI_matrix = np.zeros((runs,BO_iters)) #Saves ln(SSE) values
    SSE_matrix_abs_min = np.zeros((runs,BO_iters)) #Saves ln(SSE) values
    Total_BO_iters_matrix = np.zeros(runs)
    
    #Set theta mesh grids
#     theta1_mesh = theta_mesh[0]
#     theta2_mesh = theta_mesh[1]
    
    #Loop over # runs
    for i in range(runs):
#         print("Run Number: ",i+1)
        if verbose == True or save_fig == False:
            print("Run Number: ",i+1)
        #Note: sep_fact can be used to use less training data points
        train_data, test_data = test_train_split(all_data, runs = runs, sep_fact = sep_fact, shuffle_seed=shuffle_seed)
        if emulator == True:
            train_p = train_data[:,1:(q+m+1)]
            test_p = test_data[:,1:(q+m+1)]
        else:
            train_p = train_data[:,1:(q+1)]
            test_p = test_data[:,1:(q+1)]
            
        train_y = train_data[:,-1]
        assert len(train_p) == len(train_y), "Training data must be the same length"
        
        if emulator == True:
            assert len(train_p.T) ==q+m, "train_p must have the same number of dimensions as the value of q+m"
        else:
            assert len(train_p.T) ==q, "train_p must have the same number of dimensions as the value of q"
        
        #Split data based on # of training points to be used. Will delete later, theoretically, all data wll be either training or testing
#         print(train_p)
#         train_p = train_p[0:t]
#         train_y = train_y[0:t]
#         print("test_p",test_p)
#         print("train_p",train_p)
#         plot_org_train(theta_mesh,train_p,Theta_True)
        plot_org_train(theta_mesh,train_p, test_p, Theta_True, emulator, sparse_grid, obj, ep0, set_lengthscale, i, save_fig, BO_iters, runs, DateTime, verbose, sep_fact = sep_fact)

        #Run BO iteration
        BO_results = bo_iter(BO_iters,train_p,train_y,theta_mesh,Theta_True,train_iter,explore_bias, Xexp, Yexp, noise_std, obj, i, sparse_grid, emulator, set_lengthscale, verbose, save_fig, runs, DateTime, test_p, sep_fact = sep_fact)
        
        #Add all SSE/theta results at each BO iteration for that run
        Theta_Best_matrix[i,:,:] = BO_results[0]
        Theta_Opt_matrix[i,:,:] = BO_results[1]
        SSE_matrix[i,:] = BO_results[2]
        SSE_matrix_abs_min[i] = BO_results[3]
        Total_BO_iters_matrix[i] = BO_results[4]
        
#         print(Theta_Best_matrix)
    #Plot all SSE/theta results for each BO iteration for all runs
    if runs >= 1:
        plot_obj(SSE_matrix, t, obj, ep0, emulator, sparse_grid, set_lengthscale, save_fig, BO_iters, runs, DateTime, sep_fact = sep_fact)
        plot_Theta(Theta_Opt_matrix, Theta_True, t, BO_iters, obj,ep0, emulator, sparse_grid,  set_lengthscale, save_fig, BO_iters, runs, DateTime, sep_fact = sep_fact)
        plot_obj_abs_min(SSE_matrix_abs_min, emulator, ep0, sparse_grid, set_lengthscale, t, obj, save_fig, BO_iters, runs, DateTime, sep_fact = sep_fact)
    
    #Find point corresponding to absolute minimum SSE and max(-ei) at that point
    argmin = np.array(np.where(np.isclose(SSE_matrix, np.amin(SSE_matrix),atol=np.amin(SSE_matrix)*1e-6)==True))
    
    #Not sure how to generalize this last part
    if len(argmin) != q: #How to generalize next line?
        argmin = np.array([[argmin[0]],[argmin[1]]])
#     print(argmin)
    #Find theta value corresponding to argmin(SSE) and corresponding argmax(ei) at which run and theta value they occur
    Theta_Best_all = np.array(Theta_Best_matrix[argmin[0],argmin[1]])
    Theta_Opt_all = np.array(Theta_Opt_matrix[argmin[0],argmin[1]])
    SSE_abs_min = np.amin(SSE_matrix)
    run_opt = int(argmin[0,0]+1)
    bo_opt = int(argmin[1,0]+1)
    
    return bo_opt, run_opt, Theta_Opt_all, SSE_abs_min, Theta_Best_all, SSE_matrix