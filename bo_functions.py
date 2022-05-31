import numpy as np
import math
from scipy.stats import norm
import torch
import csv
import gpytorch
import scipy.optimize as optimize
from bo_plotters import ei_plotter
from bo_plotters import y_plotter

def LHS_Design(csv_file):
    """
    Creates LHS Design based on a CSV
    Parameters
    ----------
        csv_file: str, the name of the file containing the LHS design from Matlab. Values should not be scaled between 0 and 1.
    Returns
    -------
        param_space: ndarray , the parameter space that will be used with the GP
    """
    #Asserts that the csv filename is a string
    assert isinstance(csv_file, str)==True, "csv_file must be a sting containing the name of the file"
    
    reader = csv.reader(open(csv_file), delimiter=",") #Reads CSV containing nx3 LHS design
    lhs_design = list(reader) #Creates list from CSV
    param_space = np.array(lhs_design).astype("float") #Turns LHS design into a useable python array (nx3)
    return param_space

def calc_y_exp(Theta_True, x, noise_std, noise_mean=0):
    """
    Creates y_data for the 2 input GP function
    
    Parameters
    ----------
        Theta_True: ndarray, The array containing the true values of Theta1 and Theta2
        x: ndarray, The list of xs that will be used to generate y
        noise_std: float, int: The standard deviation of the noise
        noise_mean: float, int: The mean of the noise
        
    Returns:
        y_exp: ndarray, The expected values of y given x data
    """   
    
    #Asserts that test_T is a tensor with 2 columns
    assert isinstance(noise_std,(float,int)) == True, "The standard deviation of the noise must be an integer ot float."
    assert isinstance(noise_mean,(float,int)) == True, "The mean of the noise must be an integer ot float."
    assert len(Theta_True) ==2, "This function only has 2 unknowns, Theta_True can only contain 2 values."
    
    #Seed Random Noise (For Bug Testing)
    np.random.seed(6)
    #Creates noise values with a certain stdev and mean from a normal distribution
    noise = np.random.normal(size=len(x),loc = noise_mean, scale = noise_std) #1x n_x
    # True function is y=T1*x + T2*x^2 + x^3 with Gaussian noise
    y_exp =  Theta_True[0]*x + Theta_True[1]*x**2 +x**3 + noise #1x n_x #Put this as an input
  
    return y_exp

def create_sse_data(q,train_T, x, y_exp, obj = "obj"):
    #Tested for correctness 5/19/22
    """
    Creates y_data for the 2 input GP function
    
    Parameters
    ----------
        q: int, Number of GP inputs
        train_T: ndarray, The array containing the training data for Theta1 and Theta2
        x: ndarray, The list of xs that will be used to generate y
        y_exp: ndarray, The experimental data for y (the true value)
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        
    Returns:
        sum_error_sq: ndarray, The SSE values that the GP will be trained on
    """   
    
    #Asserts that test_T is a tensor with 2 columns (May delete this)
    assert isinstance(q, int), "Number of inputs must be an integer"
#     print(train_T.T)
    assert len(train_T.T) ==q, str("This is a "+str(q)+" input GP, train_T can only contain 2 columns of values.")
    assert len(x) == len(y_exp), "Xexp and Yexp must be the same length"
    assert obj == "obj" or obj == "LN_obj", "Objective function choice, obj, MUST be sse or LN_sse"
    
    if len(train_T)!= q:
        #Creates an array for train_sse that will be filled with the for loop
        sum_error_sq = torch.tensor(np.zeros(len(train_T))) #1 x n_train^2

        #Iterates over evey combination of theta to find the SSE for each combination
        for i in range(len(train_T)):
            theta_1 = train_T[i,0] #n_train^2x1 
            theta_2 = train_T[i,1] #n_train^2x1
            y_sim = theta_1*x + theta_2*x**2 +x**3 #n_train^2 x n_x
            if obj == "obj":
                sum_error_sq[i] = sum((y_sim - y_exp)**2) #Scaler
            else:
                sum_error_sq[i] = np.log(sum((y_sim - y_exp)**2)) #Scaler
    else:
         #Creates a value for train_sse that will be filled with the for loop
        sum_error_sq = 0 #1 x n_train^2

        #Iterates over x to find the SSE for each combination
        theta_1 = train_T[0] #n_train^2x1 
        theta_2 = train_T[1] #n_train^2x1
        y_sim = theta_1*x + theta_2*x**2 +x**3 #n_train^2 x n_x
        if obj == "obj":
            sum_error_sq = torch.tensor(sum((y_sim - y_exp)**2)) #Scaler 
        else:
            sum_error_sq = torch.tensor(np.log(sum((y_sim - y_exp)**2))) #Scaler 
    
    return sum_error_sq

def create_y_data(q, param_space):
    """
    Creates y_data (training data) based on the function theta_1*x + theta_2*x**2 +x**3
    Parameters
    ----------
        q: int, Number of GP inputs needed for direct calculation of the objective function
        param_space: (nx3) ndarray or tensor, parameter space over which the GP will be run
    Returns
    -------
        y_data: ndarray, The simulated y training data
    """
    #Assert statements check that the types defined in the doctring are satisfied
    assert isinstance(q, int), "Number of inputs must be an integer"
#     print(param_space.T)
    assert len(param_space.T) ==q, str("This is a "+str(q)+" input GP, train_T can only contain 2 columns of values.")
    
    #Converts parameters to numpy arrays if they are tensors
    if torch.is_tensor(param_space)==True:
        param_space = param_space.numpy()
        
    #Creates an array for train_data that will be filled with the for loop
    y_data = np.zeros(len(param_space)) #1 x n (row x col)
    
    if len(param_space)!=q:
        #Iterates over evey combination of theta to find the expected y value for each combination
        for i in range(len(param_space)):
            theta_1 = param_space[i,0] #nx1 
            theta_2 = param_space[i,1] #nx1
            x = param_space[i,2] #nx1 
            y_data[i] = theta_1*x + theta_2*x**2 +x**3 #Scaler
            #Returns all_y
    else:
        theta_1 = param_space[0] #nx1 
        theta_2 = param_space[1] #nx1
        x = param_space[2] #nx1 
        y_data = theta_1*x + theta_2*x**2 +x**3 #Scaler
    return y_data

def test_train_split(param_space, y_data, sep_fact=0.8):
    """
    Splits y data into training and testing data
    
    Parameters
    ----------
        param_space: ndarray or tensor, The parameter space over which the GP will be run
        y_data: ndarray or tensor, The simulated y data
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
    Returns:
        train_param: tensor, The training parameter space data
        train_data: tensor, The training y data
        test_param: tensor, The testing parameter space data
        test_data: tensor, The testing y data
    
    """
    #Assert statements check that the types defined in the doctring are satisfied and sep_fact is between 0 and 1 
    assert isinstance(sep_fact, (float, int))==True, "Separation factor must be a float or integer"
    assert 0 < sep_fact< 1, "Separation factor must be between 0 and 1"
    
    #Asserts length od param_space and y_data are equal
    assert len(param_space) == len(y_data), "The length of param_space and y_data must be the same"
    
    #Converts data and parameters to tensors if they are numpy arrays
    if isinstance(param_space, np.ndarray)==True:
        param_space = torch.tensor(param_space) #1xn
    if isinstance(y_data, np.ndarray)==True:
        y_data = torch.tensor(y_data) #1xn
    
    #Creates the index on which to split data
    train_split = int(np.round(len(y_data))*sep_fact)-1 
    
    #Training and testing data are created and converted into tensors
    train_data =y_data[:train_split] #1x(n*sep_fact)
    test_data = y_data[train_split:] #1x(n-n*sep_fact)
    train_param = param_space[:train_split,:] #1x(n*sep_fact)
    test_param = param_space[train_split:,:] #1x(n-n*sep_fact)
    return train_param, train_data, test_param, test_data

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


def calc_ei_advanced(error_best,pred_mean,pred_var,y_target):
    """ 
    Calculates the expected improvement of the 3 input parameter GP
    Parameters
    ----------
        error_best: float, the best predicted error encountered
        pred_mean: tensor or ndarray, model mean
        pred_var: tensor or ndarray, model variance
        y_target: tensor or ndarray, the expected value of the function from data or other source
    
    Returns
    -------
        ei: ndarray, the expected improvement for one term of the GP model
    """
    #Asserts that f_pred is a float, and y_target is an ndarray
    assert isinstance(error_best, (float,int))==True, "error_best must be a float or integer"
    
    #Coverts any tensors given as inputs to ndarrays
    if torch.is_tensor(pred_mean)==True:
        pred_mean = pred_mean.numpy() #1xn
    if torch.is_tensor(pred_var)==True:
        pred_var = pred_var.detach().numpy() #1xn
    if torch.is_tensor(y_target)==True:
        y_target = y_target.numpy() #1xn
        
    #Checks for equal lengths
    assert isinstance(y_target, float)==True, "y_target, pred_mean, and pred_var must be the same length"
    assert isinstance(pred_mean, float)==True, "y_target, pred_mean, and pred_var must be the same length"
    assert isinstance(pred_var, float)==True, "y_target, pred_mean, and pred_var must be the same length"
    
    #Defines standard devaition
    pred_stdev = np.sqrt(pred_var) #1xn
    
    #If variance is zero this is important
    with np.errstate(divide = 'warn'):
        #Creates upper and lower bounds and described by Alex Dowling's Derivation
        bound_a = ((y_target - pred_mean) +np.sqrt(error_best))/pred_stdev #1xn
        bound_b = ((y_target - pred_mean) -np.sqrt(error_best))/pred_stdev #1xn
        bound_lower = np.min([bound_a,bound_b])
        bound_upper = np.max([bound_a,bound_b])
        
#         print("Upper bound is", bound_upper)
#         print("Lower bound is", bound_lower)
#         print("pdf upper is", norm.pdf(bound_upper))
#         print("cdf upper is", norm.cdf(bound_upper))
#         print("pdf lower is", norm.pdf(bound_lower))
#         print("cdf lower is", norm.cdf(bound_lower))
        

        #Creates EI terms in terms of Alex Dowling's Derivation
        ei_term1_comp1 = norm.cdf(bound_upper) - norm.cdf(bound_lower) #1xn
        ei_term1_comp2 = error_best - (y_target - pred_mean)**2 #1xn

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
        ei = ei_term1 + ei_term2 + ei_term3 #1xn
          
    return ei

def eval_GP_components(p,n,Xexp,Yexp, theta_mesh, model, likelihood):
    """ 
    Calculates the expected improvement of the 3 input parameter GP
    Parameters
    ----------
        p: integer, the length of Theta vectors
        n: integer, the number of experimental data points
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
    
    Returns
    -------
        EI: ndarray, the expected improvement of the GP model
        Error_tot: ndarray, Errors associated with each value of Theta
    """
    #Asserts that inputs are correct
    assert isinstance(p, int)==True, "Number of Theta1 and Theta2 values, p, must be an integer"
    assert isinstance(n, int)==True, "Number of experimental points, n, must be an integer"
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert len(Xexp)==len(Yexp)==n, "Number of data points, n, must be same length as experimental data"
    
    #Will compare the rigorous solution and approximation later (multidimensional integral over each experiment using a sparse grid)
    #Create theta1 and theta2 mesh grids
    theta1_mesh = theta_mesh[0]
    assert len(theta1_mesh)==p, "theta_mesh must be dim, pxp arrays"
    theta2_mesh = theta_mesh[1]
    assert len(theta2_mesh)==p, "theta_mesh must be dim, pxp arrays"
    
    #Create an array in which to store expected improvement values
    EI = np.zeros((p,p)) #(p1 x p2)
    SSE = np.zeros((p,p))
    SSE_var_GP = np.zeros((p,p))
    y_GP = np.zeros((p,p,n))
    stdev_GP = np.zeros((p,p,n))
    error_sq_GP = np.zeros((p,p,n))
    # Loop over theta 1
    for i in range(p):
        #Loop over theta2
        for j in range(p):
            ## Caclulate Best Error
            #Create array to store error values
            sse = np.zeros(n)
            #Loop over Xexp
            for k in range(n):
                #Evaluate GP at a point p = [Theta1,Theta2,Xexp]
                point = [theta1_mesh[i,j],theta2_mesh[i,j],Xexp[k]]
                eval_point = np.array([point])
#                 print(eval_point)
                GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
                model_mean = GP_Outputs[3].numpy()[0] #1xn
                model_variance= GP_Outputs[1].numpy()[0] #1xn
                y_GP[i,j,k] = model_mean
                stdev_GP[i,j,k] = np.sqrt(model_variance)
                
                #Compute error for that point
                error_point = (Yexp[k] - model_mean)
                SSE_var_GP[i,j] += 2*error_point*model_variance
                sse_mag = -(error_point)**2
                sse[k] = sse_mag
                error_sq_GP[i,j,k] = (error_point)**2
                SSE[i,j] += sse_mag

            #Define best_error as the maximum value in the error array and multiply by -1 to get positive number
            #This is the minimum error value
            best_error = -max(sse)
            
            #Calculate EI of a training point
            TP = np.array([1.85665734819319,0.76582815814491,1.24174844363624])
            Output_TP = calc_GP_outputs(model, likelihood, np.array([TP]))
            Output_TP_mean = Output_TP[1].numpy()[0]
            Output_TP_var = Output_TP[3].numpy()[0]
            EI_TP = calc_ei_advanced(best_error, Output_TP_mean, Output_TP_var, 5.401062426299215)
            
            #Loop over Xexp
            ##Calculate EI
            for k in range(n):
                #Caclulate EI for each value n given the best error
                point = [theta1_mesh[i,j],theta2_mesh[i,j],Xexp[k]]
                eval_point = np.array([point])
                GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
                model_mean = GP_Outputs[3].numpy()[0] #1xn
                model_variance= GP_Outputs[1].numpy()[0] #1xn
                EI[i,j] += calc_ei_advanced(best_error, model_mean, model_variance, Yexp[k])
                #IS this calculated right?
                

            
    #Makes Error values all positive, allows amin to work correctly            
    SSE = -SSE
#     print(Eval_points)
#                 print(EI[i,j])
    return EI,SSE, y_GP, stdev_GP, error_sq_GP, SSE_var_GP,EI_TP

def calc_ei_point(p,n,Xexp,Yexp, theta_mesh, model, likelihood):
    """ 
    Calculates the expected improvement of the 3 input parameter GP
    Parameters
    ----------
        p: integer, the length of Theta vectors
        n: integer, the number of experimental data points
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
    
    Returns
    -------
        EI: ndarray, the expected improvement of the GP model
    """
    #Asserts that inputs are correct
    assert isinstance(p, int)==True, "Number of Theta1 and Theta2 values, p, must be an integer"
    assert isinstance(n, int)==True, "Number of experimental points, n, must be an integer"
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert len(Xexp)==len(Yexp)==n, "Number of data points, n, must be same length as experimental data"
    
    #Will compare the rigorous solution and approximation later (multidimensional integral over each experiment using a sparse grid)
    #Create theta1 and theta2 mesh grids
    theta1_mesh = theta_mesh[0]
    assert len(theta1_mesh)==p, "theta_mesh must be dim, pxp arrays"
    theta2_mesh = theta_mesh[1]
    assert len(theta2_mesh)==p, "theta_mesh must be dim, pxp arrays"
    
    #Create an array in which to store expected improvement values
    EI_Point = np.zeros((n,p,p)) #(p1 x p2)
    # Loop over theta 1
    for i in range(p):
        #Loop over theta2
        for j in range(p):
            ## Caclulate Best Error
            #Create array to store error values
            error = np.zeros(n)
            #Loop over Xexp
            for k in range(n):
                #Evaluate GP at a point p = [Theta1,Theta2,Xexp]
                point = [theta1_mesh[i,j],theta2_mesh[i,j],Xexp[k]]
                eval_point = np.array([point])
#                 print(eval_point)
                GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
                model_mean = GP_Outputs[3].numpy()[0] #1xn
                model_variance= GP_Outputs[1].numpy()[0] #1xn
                #Compute error for that point
                error_mag = -(Yexp[k] - model_mean)**2
                error[k] = error_mag

            #Define best_error as the maximum value in the error array and multiply by -1 to get positive number
            #This is the minimum error value
            best_error = -max(error)

            #Loop over Xexp
            ##Calculate EI
            for k in range(n):
                #Caclulate EI for each value n given the best error
                point = [theta1_mesh[i,j],theta2_mesh[i,j],Xexp[k]]
                eval_point = np.array([point])
                GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
                model_mean = GP_Outputs[3].numpy()[0] #1xn
                model_variance= GP_Outputs[1].numpy()[0] #1xn
                EI_Point[k,i,j] = calc_ei_advanced(best_error, model_mean, model_variance, Yexp[k])
                
#     print(Eval_points)
#                 print(EI[i,j])
    return EI_Point

def calc_ei_total_test(p,n,Xexp,Yexp, theta_mesh, model, likelihood):
    #Unused function - 5/19/22
    """ 
    Calculates the expected improvement of the 3 input parameter GP
    Parameters
    ----------
        p: integer, the length of Theta vectors
        n: integer, the number of experimental data points
        Xexp: ndarray, experimental x values
        Yexp: ndarray, experimental y values
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
    
    Returns
    -------
        EI: ndarray, the expected improvement of the GP model
    """
    #Asserts that inputs are correct
    assert isinstance(p, int)==True, "Number of Theta1 and Theta2 values, p, must be an integer"
    assert isinstance(n, int)==True, "Number of experimental points, n, must be an integer"
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert len(Xexp)==len(Yexp)==n, "Number of data points, n, must be same length as experimental data"
    
    
    #Define f_bar and f(x)
    #Will compare the rigorous solution and approximation later (multiensional integral over each experiment using a sparse grid)
    #Create theta1 and theta2 mesh grids
    theta1_mesh = theta_mesh[0]
    assert len(theta1_mesh)==p, "theta_mesh must be dim, pxp arrays"
    theta2_mesh = theta_mesh[1]
    assert len(theta2_mesh)==p, "theta_mesh must be dim, pxp arrays"
    #Create an array in which to store expected improvement values
    EI = np.zeros((p,p)) #(p1 x p2)
    EI_sing = np.zeros((n,p,p))
#     print(EI_sing)
    Error = np.zeros((p,p))
    # Loop over theta 1
    for i in range(p):
        #Loop over theta2
        for j in range(p):
            ## Caclulate Best Error
            #Create array to store error values
            error = np.zeros(n)
            #Loop over Xexp
            for k in range(n):
                #Evaluate GP at a point p = [Theta1,Theta2,Xexp]
                eval_point = []
                eval_point.append([theta1_mesh[i,j],theta2_mesh[i,j],Xexp[k]])
                eval_point = np.array(eval_point)
#                 print(eval_point)
                GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
                model_mean = GP_Outputs[3].numpy()[0] #1xn
                model_variance= GP_Outputs[1].numpy()[0] #1xn
                #Compute error for that point
                error[k] = -(Yexp[k] - model_mean)**2

            #Define best_error as the maximum value in the error array and multiply by -1 to get positive number
            #This is the minimum error value
            best_error = -max(error)
            Error[i,j] = best_error

            #Loop over Xexp
            ##Calculate EI
            for k in range(n):
                #Caclulate EI for each value n given the best error
                eval_point = []
                eval_point.append([theta1_mesh[i,j],theta2_mesh[i,j],Xexp[k]])
                eval_point = np.array(eval_point)
                GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
                model_mean = GP_Outputs[3].numpy()[0] #1xn
                model_variance= GP_Outputs[1].numpy()[0] #1xn
                ei = calc_ei_advanced(best_error, model_mean, model_variance, Yexp[k])
#                 print(ei)
                EI[i,j] += ei
                EI_sing[k,i,j] += ei
    return EI_sing,Error


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
        IF verbose == True
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
    if torch.is_tensor(pred_var)==True:
        pred_var = pred_var.detach().numpy() #1xn
    pred_stdev = np.sqrt(pred_var) #1xn_test
#     print("stdev",pred_stdev)
    #Checks that all standard deviations are positive
    if pred_stdev > 0:
        #Calculates z-score based on Ke's formula
        z = (pred_mean - f_best - explore_bias)/pred_stdev #scaler
        
        #Calculates ei based on Ke's formula
        #Explotation term
        
        #Should we be assuming Mean(z) =0 and stdv(z) =1?
        ei_term_1 = (pred_mean - f_best - explore_bias)*norm.cdf(z) #scaler
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

def eval_GP_basic_tot(p,theta_mesh, train_sse, model, likelihood, explore_bias=0.0, verbose = False):
    """ 
    Calculates the expected improvement of the 2 input parameter GP
    Parameters
    ----------
        p: integer, the length of Theta vectors
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        train_sse: ndarray (1 x t), Training data for sse
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        verbose: True/False: Determines whether z and ei terms are printed
    
    Returns
    -------
        ei: ndarray, the expected improvement of the GP model
        sse: ndarray, the sse of the GP model
        var: ndarray, the variance of the GP model
        stdev: ndarray, the standard deviation of the GP model
        f_best: ndarray, the best value so far
    """
        #Asserts that inputs are correct
    assert isinstance(p, int)==True, "Number of Theta1 and Theta2 values, p, must be an integer"
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(train_sse, np.ndarray) or torch.is_tensor(train_sse) == True, "Train_sse must be ndarray or torch.tensor"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    
    ei = np.zeros((p,p))
    sse = np.zeros((p,p))
    var = np.zeros((p,p))
    stdev = np.zeros((p,p))
    f_best = np.zeros((p,p))
    if verbose == True:
        z_term = np.zeros((p,p))
        ei_term_1 = np.zeros((p,p))
        ei_term_2 = np.zeros((p,p))
        CDF = np.zeros((p,p))
        PDF = np.zeros((p,p))
    
    theta1_mesh = theta_mesh[0]
    theta2_mesh = theta_mesh[1]
    
    assert len(theta1_mesh)==p, "theta_mesh must be dim, pxp arrays"
    assert len(theta2_mesh)==p, "theta_mesh must be dim, pxp arrays"
    
    for i in range(p):
        #Loop over Theta_2
        for j in range(p):
            point = [theta1_mesh[i,j],theta2_mesh[i,j]]
            eval_point = np.array([point])
            GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
            model_sse = GP_Outputs[3].numpy()[0] #1xn
#             print(model_sse)
            model_variance= GP_Outputs[1].numpy()[0] #1xn
#             if verbose == True:
#                 print("Point",eval_point)
#                 print("Model Mean",model_sse)
#                 print("Model Var", model_variance)
            sse[i,j] = model_sse
            var[i,j] = model_variance
            stdev[i,j] = np.sqrt(model_variance)
            best_error = max(-train_sse) #Are we sure this is right
            f_best[i,j] = best_error
            #Negative sign because -max(-train_sse) = min(train_sse)
#             print(best_error)
            if verbose == True:
                ei[i,j] = calc_ei_basic(best_error,-model_sse,model_variance,explore_bias,verbose)[0] #-max(-train_sse) = min(train_sse)
                z_term[i,j] = calc_ei_basic(best_error,-model_sse,model_variance,explore_bias,verbose)[1]
                ei_term_1[i,j] = calc_ei_basic(best_error,-model_sse,model_variance,explore_bias,verbose)[2]
                ei_term_2[i,j] = calc_ei_basic(best_error,-model_sse,model_variance,explore_bias,verbose)[3]
                CDF[i,j] = calc_ei_basic(best_error,-model_sse,model_variance,explore_bias,verbose)[4]
                PDF[i,j] = calc_ei_basic(best_error,-model_sse,model_variance,explore_bias,verbose)[5]
            else:
                ei[i,j] = calc_ei_basic(best_error,-model_sse,model_variance,explore_bias,verbose)
    if verbose == False:
        return ei, sse, var, stdev, f_best
    else:
        return ei, sse, var, stdev, f_best, z_term, ei_term_1, ei_term_2, CDF, PDF

##FOR USE WITH SCIPY##################################################################
def eval_GP_basic_tot_scipy(theta_guess, train_sse, model, likelihood, explore_bias=0.0, ei_sse_choice = "ei", verbose = False):
    """ 
    Calculates the expected improvement of the 2 input parameter GP
    Parameters
    ----------
        theta_guess: ndarray (1xp), The theta value that will be guessed to optimize 
        p: integer, the length of Theta vectors
        train_sse: ndarray (1 x t), Training data for sse
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        ei_sse_choice: "neg_ei" or "sse" - Choose which one to optimize
    
    Returns
    -------
        ei: ndarray, the expected improvement of the GP model
        sse: ndarray, the sse of the GP model
        
    """
        #Asserts that inputs are correct
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(train_sse, np.ndarray) or torch.is_tensor(train_sse) == True, "Train_sse must be ndarray or torch.tensor"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert ei_sse_choice == "neg_ei" or ei_sse_choice == "sse", "ei_sse_choice must be string 'ei' or 'sse'"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    
    theta1_guess = theta_guess[0]
    theta2_guess = theta_guess[1]
    

    point = [theta1_guess,theta2_guess]
    eval_point = np.array([point])
    GP_Outputs = calc_GP_outputs(model, likelihood, eval_point[0:1])
    model_sse = GP_Outputs[3].numpy()[0] #1xn 
#     print(model_sse)
    model_variance= GP_Outputs[1].numpy()[0] #1xn
    best_error = max(-train_sse) #Are we sure this is right
    #Negative sign because -max(-train_sse) = min(train_sse)
#             print(best_error)
    if verbose == True:
        ei = calc_ei_basic(best_error,-model_sse,model_variance,explore_bias,verbose)[0]
    else:
        ei = calc_ei_basic(best_error,-model_sse,model_variance,explore_bias,verbose)
    
    sse = model_sse
    
    if ei_sse_choice == "neg_ei":
#         print("EI chosen")
        return -ei #Because we want to maximize EI and scipy.optimize is a minimizer by default
    else:
#         print("sse chosen")
        return sse #We want to minimize sse

def eval_GP(p, theta_mesh, train_y, explore_bias, model, likelihood, verbose):    
    """
    Evaluates GP
    
    Parameters:
    -----------
        p: integer, the length of Theta vectors
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        train_y: tensor or ndarray, The training y data
        explore_bias: float,int,tensor,ndarray (1 value) The exploration bias parameter
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        verbose: True/False: Determines whether z_term, ei_term_1, ei_term_2, CDF, and PDF terms are saved
    
    Returns:
    --------
        eval_components: ndarray, The componenets evaluate by the GP. ei, sse, var, stdev, f_best, (z_term, ei_term_1, ei_term_2, CDF, PDF)
    """
    
    assert isinstance(train_y, np.ndarray) or torch.is_tensor(train_y) == True, "Train_sse must be ndarray or torch.tensor"
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    ##Set Hyperparameters to 1
    if isinstance(train_y, np.ndarray)==True:
        train_y = torch.tensor(train_y) #1xn
        
    outputscale = torch.tensor([1])
    lengthscale = torch.tensor([1])
    noise = torch.tensor([0.1])

    model.likelihood.noise = noise
    model.covar_module.base_kernel.lengthscale =lengthscale
    model.covar_module.outputscale = outputscale
    
    model.eval()
    #Puts likelihood in evaluation mode
    likelihood.eval()

    #Same point keeps being selected, should I remove that point by force?
    eval_components = eval_GP_basic_tot(p,theta_mesh, train_y, model, likelihood, explore_bias, verbose)
    
    return eval_components

def find_opt_and_best_arg(theta_mesh, sse, ei):
    """
    Finds the Theta value where min(sse) or min(-ei) is true using argmax and argmin
    
    Parameters:
    -----------
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        sse: ndarray (d, p x p), meshgrid of sse values for all points in theta_mesh
        ei: ndarray (d, p x p), meshgrid of ei values for all points in theta_mesh
    
    Returns:
    --------
        Theta_Opt_GP: ndarray, The point where the objective function is minimized in theta_mesh
        Theta_Best: ndarray, The point where the ei is maximized in theta_mesh
    """
    theta1_mesh = theta_mesh[0]
    theta2_mesh = theta_mesh[1]
    
    argmin = np.array(np.where(np.isclose(sse, np.amin(sse),atol=np.amin(sse)*1e-6)==True))
    
    if len(argmin[0]) != 1:
        argmin = np.array([[argmin[0,1]],[argmin[1,1]]])
        
    Theta_1_Opt = float(theta1_mesh[argmin[0],argmin[1]])
    Theta_2_Opt = float(theta2_mesh[argmin[0],argmin[1]])
    Theta_Opt_GP = np.array((Theta_1_Opt,Theta_2_Opt))
    
    #calculates best theta value
    argmax = np.array(np.where(np.isclose(ei, np.amax(ei),atol=np.amax(ei)*1e-6)==True))

    if len(argmax[0]) != 1:
        argmax = np.array([[argmax[0,1]],[argmax[1,1]]])
    
    Theta_1_Best = float(theta1_mesh[argmax[0],argmax[1]])
    Theta_2_Best = float(theta2_mesh[argmax[0],argmax[1]])
    Theta_Best = np.array((Theta_1_Best,Theta_2_Best))  
    
    return Theta_Best, Theta_Opt_GP

def find_opt_best_scipy(theta_mesh, train_y, theta0_b,theta0_o, sse, ei, model, likelihood, explore_bias):
    """
    Finds the Theta value where min(sse) or min(-ei) is true using scipy.minimize and the L-BFGS-B method
    
    Parameters:
    -----------
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        train_y: tensor or ndarray, The training y data
        theta0_b: Initial guess of the Theta value where ei is maximized
        theta0_o: Initial guess of the Theta value where sse is minimized
        sse: ndarray (d, p x p), meshgrid of sse values for all points in theta_mesh
        ei: ndarray (d, p x p), meshgrid of ei values for all points in theta_mesh
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        explore_bias: float,int,tensor,ndarray (1 value) The exploration bias parameter
    
    Returns:
    --------
        theta_b: ndarray, The point where the objective function is minimized in theta_mesh
        theta_o: ndarray, The point where the ei is maximized in theta_mesh
    """
    assert isinstance(train_y, np.ndarray) or torch.is_tensor(train_y) == True, "Train_sse must be ndarray or torch.tensor"
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert len(theta0_b) == len(theta0_o), "Initial guesses must be the same length."
    
    theta1_mesh = theta_mesh[0]
    theta2_mesh = theta_mesh[1]
    bnds = [[np.amin(theta1_mesh), np.amax(theta1_mesh)], [np.amin(theta2_mesh), np.amax(theta2_mesh)]]

    ei_sse_choice1 ="neg_ei"
    argmts_best = ((train_y, model, likelihood, explore_bias, ei_sse_choice1))
#     Best_Solution = optimize.minimize(eval_GP_basic_tot_scipy, theta0_b,bounds=bnds, method='Nelder-Mead',args=argmts_best)
    Best_Solution = optimize.minimize(eval_GP_basic_tot_scipy, theta0_b,bounds=bnds,method = "L-BFGS-B",args=argmts_best)
    theta_b = Best_Solution.x
    
    ei_sse_choice2 = "sse"
    argmts_opt = ((train_y, model, likelihood, explore_bias, ei_sse_choice2))
    Opt_Solution = optimize.minimize(eval_GP_basic_tot_scipy, theta0_o,bounds=bnds, method = "L-BFGS-B",args= argmts_opt)
    theta_o = Opt_Solution.x  
    
    return theta_b, theta_o

def bo_iter(BO_iters,train_p,train_y,p,q,theta_mesh,Theta_True,train_iter,explore_bias, Xexp, Yexp, obj, verbose = False):
    """
    Performs BO iterations
    
    Parameters:
    -----------
        BO_iters: integer, number of BO iteratiosn
        train_p: tensor or ndarray, The training parameter space data
        train_y: tensor or ndarray, The training y data
        p: integer, the length of Theta vectors
        q: integer, Number of GP inputs
        theta_mesh: ndarray (d, p x p), meshgrid of Theta1 and Theta2
        Theta_True: ndarray, The array containing the true values of Theta1 and Theta2
        train_iter: int, number of training iterations to run. Default is 300
        explore_bias: float,int,tensor,ndarray (1 value) The exploration bias parameter
        Xexp: ndarray, The list of xs that will be used to generate y
        Yexp: ndarray, The experimental data for y (the true value)
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        verbose: True/False, Determines whether z_term, ei_term_1, ei_term_2, CDF, and PDF terms are saved, Default = False
        
    Returns:
    --------
        theta_b: The predicted theta where ei is maximized after all BO iteration
        theta_o: The predicted theta where objective function is minimized after all BO iterations
    
    """
    assert all(isinstance(i, int) for i in [BO_iters, p,q,train_iter]), "BO_iters, p,q,and train_iter must be integers"
    assert len(train_p.T) ==q, "train_p must have the same number of dimensions as the value of q"
    assert len(train_p) == len(train_y), "Training data must be the same length"
    assert len(Theta_True)==2, "Theta True must be 2-dimensional"
    assert len(Xexp) == len(Yexp), "Experimental data must have the same length"
    assert obj == "obj" or obj == "LN_obj", "Objective function choice, obj, MUST be sse or LN_sse"
    assert verbose==True or verbose==False, "Verbose must be True/False"
    
    for i in range(BO_iters):
        if torch.is_tensor(train_p) != True:
            train_p = torch.from_numpy(train_p)
        if torch.is_tensor(train_y) != True:
            train_y = torch.from_numpy(train_y)
            
        #Redefine likelihood and model based on new training data
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_p, train_y, likelihood)
        #Train GP
        train_GP = train_GP_model(model, likelihood, train_p, train_y, train_iter, verbose=False)
        #Evaluate GP
        eval_components = eval_GP(p, theta_mesh,train_y, explore_bias, model, likelihood, verbose)
        
        if verbose == False:
            ei,sse,var,stdev,best_error = eval_components
        
        if verbose == True:
            ei,sse,var,stdev,best_error,z,ei_term_1,ei_term_2,CDF,PDF = eval_components
        
#         print(sse)
        
        Theta_Best, Theta_Opt_GP = find_opt_and_best_arg(theta_mesh, sse, ei)
        theta0_b = Theta_Best
        theta0_o = Theta_Opt_GP
#         theta0_b = np.array([0.95,-0.95])
#         theta0_o = np.array([0.95,-0.95])
        theta_b, theta_o = find_opt_best_scipy(theta_mesh, train_y, theta0_b,theta0_o, sse, ei,model, likelihood, explore_bias)
       
        if verbose == True:
            print("BO Iteration = ", i+1)
            print("Exploration Bias = ",explore_bias)
            print("Scipy Theta Best = ",theta_b)
            print("Argmax Theta Best = ",Theta_Best)
            print("Scipy Theta Opt = ",theta_o)
            print("Argmin Theta_Opt_GP = ",Theta_Opt_GP, "\n")
#             print("Best Error =", best_error, "\n")
        
        sse_title = "SSE"
        ei_plotter(theta_mesh, ei, Theta_True, theta_o, theta_b,train_p,plot_train=True)
        if obj == "LN_obj":
            y_plotter(theta_mesh, np.exp(sse), Theta_True, theta_o, theta_b, train_p,sse_title,plot_train=True)
        else:
            y_plotter(theta_mesh, sse, Theta_True, theta_o, theta_b, train_p,sse_title,plot_train=True)
        titles = ["ei","sse","var","stdev","Best_Error","z","ei_term_1","ei_term_2","CDF","PDF"]
        if verbose == True:
            for j in range(len(titles)-2):
                y_plotter(theta_mesh, eval_components[j+2], Theta_True, theta_o, theta_b, train_p,titles[j+2],plot_train=True)
    #     ei_plotter(theta_mesh, ei, Theta_True, Theta_Opt_GP, Theta_Best,train_T,plot_train=True)
    #     y_plotter(theta_mesh, sse, Theta_True, Theta_Opt_GP, Theta_Best, train_T,sse_title,plot_train=True)
        ##Append best values to training data 
        #Convert training data to numpy arrays to allow concatenation to work
        train_p = train_p.numpy() #(q x t)
        train_y = train_y.numpy() #(1 x t)

        #Call the expensive function and evaluate at Theta_Best
        sse_Best = create_sse_data(q,theta_b, Xexp, Yexp, obj) #(1 x 1)
    #     sse_Best = create_sse_data(q,Theta_Best, Xexp, Yexp) #(1 x 1)

        #Add Theta_Best to train_p and y_best to train_y
        train_p = np.concatenate((train_p, [theta_b]), axis=0) #(q x t)
    #     train_T = np.concatenate((train_T, [Theta_Best]), axis=0) #(q x t)
        train_y = np.concatenate((train_y, [sse_Best]),axis=0) #(1 x t)
    return theta_b, theta_o 
        
def create_dicts(i,ei_components,verbose =False):
    """
    Creates dictionaries for noteable parameters
    """
    train_T_dict = {}
    train_sse_dict = {}
    ei_dict = {}
    sse_dict ={}
    var_dict ={}
    GP_mean_best_dict = {}
    GP_var_best_dict = {}
    GP_mean_min_dict ={}
    GP_var_min_dict = {}
    Theta_Opt_dict = {}
    Theta_Best_dict = {}
    Best_Error_dict = {}
    if verbose == True:
        z_dict = {}
        ei_term_1_dict = {}
        ei_term_2_dict = {}
        CDF_dict = {}
        PDF_dict = {}
        
    ei = ei_components[0]
    sse = ei_components[1]
    var = ei_components[2]
    stdev = ei_components[3]
    best_error = ei_components[4]
    if verbose == True:
        z = ei_components[5]
        ei_term_1 = ei_components[6]
        ei_term_2 = ei_components[7]
        CDF = ei_components[8]
        PDF = ei_components[9]
        
    ei_dict[i+1] = ei
    sse_dict[i+1] = sse
    var_dict[i+1] = var
    z_dict[i+1]=z
    Best_Error_dict[i+1] = best_error
    ei_term_1_dict[i+1] = ei_term_1
    ei_term_2_dict[i+1] = ei_term_2
    CDF_dict[i+1] = CDF
    PDF_dict[i+1] = PDF
    train_T_dict[i+1] = train_T
    train_sse_dict[i+1] = train_sse
    Theta_Opt_dict[i+1] = Theta_Opt_GP
    Theta_Best_dict[i+1] = Theta_Best
    GP_mean_best_dict[i+1] = GP_mean_best
    GP_var_best_dict[i+1] = GP_var_best
    GP_mean_min_dict[i+1] = GP_mean_min
    GP_var_min_dict[i+1] = GP_var_min
        
    return None ##May change this    