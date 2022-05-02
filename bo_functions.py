import numpy as np
import math
from scipy.stats import norm
import torch
import csv
import gpytorch

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
    assert len(Theta_True) ==2, "This is a 2 input GP, Theta_True can only contain 2 values."
    
    #Creates noise values with a certain stdev and mean from a normal distribution
    noise = np.random.normal(size=len(x),loc = noise_mean, scale = noise_std) #1x n_x
    # True function is y=T1*x + T2*x^2 + x^3 with Gaussian noise
    y_exp =  Theta_True[0]*x + Theta_True[1]*x**2 +x**3 + noise #1x n_x #Put this as an input
  
    return y_exp

def create_sse_data(train_T, x, y_exp):
    """
    Creates y_data for the 2 input GP function
    
    Parameters
    ----------
        train_T: ndarray, The array containing the training data for Theta1 and Theta2
        x: ndarray, The list of xs that will be used to generate y
        y_exp: ndarray, The experimental data for y (the true value)
        
    Returns:
        sum_error_sq: ndarray, The SSE values that the GP will be trained on
    """   
    
    #Asserts that test_T is a tensor with 2 columns (May delete this)
    assert len(train_T.T) ==2 or len(train_T.T)==3, "This is a 2 input GP, train_T can only contain 2 columns of values."

    #Creates an array for train_sse that will be filled with the for loop
    sum_error_sq = torch.tensor(np.zeros(len(train_T))) #1 x n_train^2

    #Iterates over evey combination of theta to find the SSE for each combination
    for i in range(len(train_T)):
        theta_1 = train_T[i,0] #n_train^2x1 
        theta_2 = train_T[i,1] #n_train^2x1
        y_sim = theta_1*x + theta_2*x**2 +x**3 #n_train^2 x n_x
        sum_error_sq[i] = sum((y_sim - y_exp)**2) #Scaler
    return sum_error_sq

def create_y_data(param_space):
    """
    Creates y_data (training data) based on the function theta_1*x + theta_2*x**2 +x**3
    Parameters
    ----------
        param_space: (nx3) ndarray or tensor, parameter space over which the GP will be run
    Returns
    -------
        y_data: ndarray, The simulated y training data
    """
    #Assert statements check that the types defined in the doctring are satisfied
    assert len(param_space.T) ==3, "Only 3 input parameter space can be taken, param_space must be an nx3 array"
    
    #Converts parameters to numpy arrays if they are tensors
    if torch.is_tensor(param_space)==True:
        param_space = param_space.numpy()

    #Creates an array for train_data that will be filled with the for loop
    y_data = np.zeros(len(param_space)) #1 x n (row x col)

    #Iterates over evey combination of theta to find the expected y value for each combination
    for i in range(len(param_space)):
        theta_1 = param_space[i,0] #nx1 
        theta_2 = param_space[i,1] #nx1
        x = param_space[i,2] #nx1 
        y_data[i] = theta_1*x + theta_2*x**2 +x**3 #Scaler
    #Returns all_y
    return y_data

def test_train_split(param_space, y_data, sep_fact=0.8):
    """
    Splits y data into training and testing data
    
    Parameters
    ----------
        param_space: (nx3) ndarray or tensor, The parameter space over which the GP will be run
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
        optimizer.step(): Updates the value of parameters using the gradient x.grad
    """
    #Assert statements check that inputs are the correct types and lengths
    assert isinstance(model,ExactGPModel) == True, "Model must be the class ExactGPModel"
    assert isinstance(likelihood, gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood) == True, "Likelihood must be Gaussian"
    assert isinstance(iterations, int)==True, "Number of training iterations must be an integer" 
    assert len(train_param) == len(train_data), "training data must be the same length as each other"
    
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
        if verbose == True:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                 model.likelihood.noise.item()
            ))
        #optimizer.step updates the value of x using the gradient x.grad. For example, the SGD optimizer performs:
        #x += -lr * x.grad
        optimizer.step()
    return

def calc_GP_outputs(model,likelihood,test_param):
    """
    Calculates the GP model's approximation of y, and its mean, variance and standard devaition.
    
    Parameters
    ----------
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be a Gaussian likelihood
        test_param: tensor or ndarray, The testing parameter space data
    
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
def calc_ei_total(p,n,Xexp,Yexp, theta_mesh, model, likelihood):
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
    f_bar = Yexp #(1xn)
    #Will compare the rigorous solution and approximation later (multidimensional integral over each experiment using a sparse grid)
    #Create theta1 and theta2 mesh grids
    theta1_mesh = theta_mesh[0]
    assert len(theta1_mesh)==p, "theta_mesh must be dim, pxp arrays"
    theta2_mesh = theta_mesh[1]
    assert len(theta2_mesh)==p, "theta_mesh must be dim, pxp arrays"
    #Create an array in which to store expected improvement values
    EI = np.zeros((p,p)) #(p1 x p2)
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
                error[k] = -(f_bar[k] - model_mean)**2

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
                EI[i,j] += calc_ei_advanced(best_error, model_mean, model_variance, Yexp[k])
#                 print(EI[i,j])
    return EI,Error

def calc_ei_basic(f_best,pred_mean,pred_var, explore_bias=0.0):
    """ 
    Calculates the expected improvement of the 2 input parameter GP
    Parameters
    ----------
        f_best: float, the best predicted sse encountered
        pred_mean: tensor, model mean
        pred_var, tensor, model variance
        explore_bias: float, the numerical bias towards exploration, zero is the default
    
    Returns
    -------
    Calling this model will return the posterior of the latent Gaussian process when conditioned
    on the training data. The output will be a :obj:`~gpytorch.distributions.MultivariateNormal`.
    """
    assert len(pred_mean) == len(pred_var), "GP predicted means and variances must be the same length"
    assert torch.is_tensor(pred_mean)==True and torch.is_tensor(pred_var)==True, "GP predicted means and variances must be tensors"
    
    #Creates empty list to store ei values
    ei = np.zeros(len(pred_var)) #1xn_test
    
    #Converts tensors to np arrays and defines standard deviation
    pred_mean = pred_mean.numpy() #1xn_test
    pred_var = pred_var.detach().numpy()   #1xn_test
    pred_stdev = np.sqrt(pred_var) #1xn_test

    
    #Loops over every standard deviation values
    for i in range(len(pred_var)):
        #Checks that all standard deviations are positive
        if pred_stdev[i] > 0:
            #Calculates z-score based on Ke's formula
            z = (pred_mean[i] - f_best - explore_bias)/pred_stdev[i] #scaler
            #Calculates ei based on Ke's formula
            #Explotation term
            ei_term_1 = (pred_mean[i] - f_best - explore_bias)*norm.cdf(z) #scaler
            #Exploration Term
            ei_term_2 = pred_stdev[i]*norm.pdf(z) #scaler
            ei[i] = ei_term_1 +ei_term_2 #scaler
        else:
            #Sets ei to zero if standard deviation is zero
            ei[i] = 0
    return ei
