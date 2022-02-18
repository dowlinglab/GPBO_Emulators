import numpy as np
from scipy.stats import norm
import torch
import csv
import gpytorch

def calc_y_expected(test_p, noise_stdev, noise_mean=0):
    """
    Creates y_data based on the actual function theta_1*x + theta_2*x**2 +x**3 + noise
    Parameters
    ----------
        test_p: (n_trainx3) ndarray, The parameter space over which the GP will be tested
        noise_std: float or int, The standard deviation of the nosie
        noise_mean: float or int, The mean of the noise. Default is zero.
    Returns
    -------
        y_data: ndarray, The simulated y data
    """
    #Assert statements check that the types defined in the doctring are satisfied
    assert isinstance(noise_mean, (float, int))==True, "noise parameters must be floats or integers"
    assert isinstance(noise_stdev, (float, int))==True, "noise parameters must be floats or integers"
    assert torch.is_tensor(test_p)==True, "Test parameter space must be a tensor"
    assert len(test_p.T) ==3, "Only 3 input Test parameter space can be taken, test_p must be an n_trainx3 array"
    
    noise = np.random.normal(size=1,loc = noise_mean, scale = noise_stdev) #Scaler
    
    #Separates parameters for use
    p_1 = test_p[:,0].numpy() #Theta1 #1 x n_test
    p_2 = test_p[:,1].numpy() #Theta2 #1 x n_test
    p_3 = test_p[:,2].numpy() #x #1 x n_test

    #Calculates expected y for each parameter space parameter
    y_expected = p_1*p_3 + p_2*p_3**2 + p_3**3 + noise #1 x n_test
    return y_expected
    
    #Creates noise values with a certain stdev and mean from a normal distribution
    noise = np.random.normal(size=1,loc = noise_mean, scale = noise_std) #Scaler
    
    
def calc_GP_outputs(model,likelihood,test_p):
    """
    Calculates the GP model's approximation of y, and its mean, variance and standard devaition.
    
    Parameters
    ----------
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be Gaussian
        train_p: tensor, The training parameter space data
    
    Returns
    -------
        model_mean: tensor, The GP model's mean
        model_variance: tensor, The GP model's variance
        model_stdev: tensor, The GP model's standard deviation
        model_y: tensor, The GP model's approximation of y
    """
    #How would I even write an assert statement for the likelihood and model?
    assert torch.is_tensor(test_p)==True, "Test parameter space must be a tensor"
    
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
    #torch.no_grad() 
        #Disabling gradient calculation is useful for inference, 
        #when you are sure that you will not call Tensor.backward(). It will reduce memory consumption
        #Note: Can't use np operations on tensors where requires_grad = True
    #gpytorch.settings.fast_pred_var() 
        #Use this for improved performance when computing predictive variances. 
        #Good up to 10,000 data points
    #Predicts data points for model (sse) by sending the model through the likelihood
        observed_pred = likelihood(model(test_p)) #1 x n_test

    #Calculates model mean  
    model_mean = observed_pred.mean #1 x n_test
    #Calculates the variance of each data point
    model_variance = observed_pred.variance #1 x n_test
    #Calculates the standard deviation of each data point
    model_stdev = np.sqrt(observed_pred.variance)
    y_model = observed_pred.loc #1 x n_test
    return model_mean, model_variance, model_stdev, y_model

def train_GP_model(model, likelihood, train_p, train_y, iterations=300, verbose=False):
    #This function calculates hyperparameters differently than what's in the code right now and I can't figure out why. Is this even an issue to worry about?
    """
    Trains the GP model and finds hyperparameters with the Adam optimizer with an lr =0.1
    
    Parameters
    ----------
        model: bound method, The model that the GP is bound by
        likelihood: bound method, The likelihood of the GP model. In this case, must be Gaussian
        train_p: tensor, The training parameter space data
        train_y: tensor, The training y data
        iterations: float or int, number of training iterations to run. Default is 300
        verbose: Set verbose to "True" to view the associated loss and hyperparameters for each training iteration. False by default
    
    Returns
    -------
        optimizer.step(): Updates the value of parameters using the gradient x.grad
    """
    #How would I even write an assert statement for the likelihood and model?
    assert isinstance(iterations, (float, int))==True, "Number of training iterations must be a float or integer" 
    assert torch.is_tensor(train_p)==True, "Train parameter space must be a tensor"
    assert torch.is_tensor(train_y)==True, "Train y data must be a tensor"
    assert len(train_p) == len(train_y), "training data must be the same length as each other"
    
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
        output = model(train_p) # A multivariate norm of a 1 x n_train^2 tensor
        # Calc loss and backprop gradients
        #Minimizing -logMLL lets us fit hyperparameters
        loss = -mll(output, train_y) #A number (tensor)
        #computes dloss/dx for every parameter x which has requires_grad=True. 
        #These are accumulated into x.grad for every parameter x
        loss.backward()
    #     print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
    #         i + 1, training_iter, loss.item(),
    #         model.covar_module.base_kernel.lengthscale.item(),
    #          model.likelihood.noise.item()
    #     ))
        #optimizer.step updates the value of x using the gradient x.grad. For example, the SGD optimizer performs:
        #x += -lr * x.grad
        optimizer.step()
    return
    
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

    def __init__(self, train_p, train_y, likelihood):
        """
        Initializes the model
        
        Parameters
        ----------
        self : A class,The model itself. In this case, gpytorch.models.ExactGP
        train_T : tensor, The inputs of the training data
        train_y : tensor, the output of the training data
        likelihood : bound method, the lieklihood of the model. In this case, it must be Gaussian
        
        """
        #Initializes the GP model with train_p, train_y, and the likelihood
        ##Calls the __init__ method of parent class
        super(ExactGPModel, self).__init__(train_p, train_y, likelihood)
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

def test_train_split(param_space, y_data, sep_fact=0.8):
    """
    Splits y data into training and testing data
    
    Parameters
    ----------
        param_space: (nx3) ndarray, The parameter space over which the GP will be run
        y_data: ndarray, The simulated y data
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
    Returns:
        train_p: tensor, The training parameter space data
        train_y: tensor, The training y data
        test_p: tensor, The testing parameter space data
        test_y: tensor, The testing y data
    
    """
    
    #Assert statements check that the types defined in the doctring are satisfied and sep_fact is between 0 and 1 
    assert isinstance(param_space, np.ndarray) == True, "parameter space must be a numpy array"
    assert isinstance(sep_fact, (float, int))==True, "Separation factor must be a float or integer"
    assert 0<= sep_fact<=1, "Separation factor must be between 0 and 1"
    
    #Creates the index on which to split data
    train_split = int(np.round(len(y_data))*sep_fact)-1 
    
    #Training and testing data are created and converted into tensors
    train_y =torch.tensor(y_data[:train_split]) #1x(n*sep_fact)
    test_y = torch.tensor(y_data[train_split:]) #1x(n-n*sep_fact)
    train_p = torch.tensor(param_space[:train_split,:]) #1x(n*sep_fact)
    test_p = torch.tensor(param_space[train_split:,:]) #1x(n-n*sep_fact)
    return train_p, train_y, test_p, test_y

def create_y_data(param_space, noise_std,noise_mean=0):
    """
    Creates y_data based on the actual function theta_1*x + theta_2*x**2 +x**3 + noise
    Parameters
    ----------
        param_space: (nx3) ndarray, The parameter space over which the GP will be run
        noise_std: float or int, The standard deviation of the nosie
        noise_mean: float or int, The mean of the noise. Default is zero.
    Returns
    -------
        y_data: ndarray, The simulated y data
    """
    #Assert statements check that the types defined in the doctring are satisfied
    assert isinstance(noise_mean, (float, int))==True, "noise parameters must be floats or integers"
    assert isinstance(noise_std, (float, int))==True, "noise parameters must be floats or integers"
    assert isinstance(param_space, np.ndarray) == True, "parameter space must be a numpy array"
    assert len(param_space.T) ==3, "Only 3 input parameter space can be taken, param_space must be an nx3 array"
    
    #Creates noise values with a certain stdev and mean from a normal distribution
    noise = np.random.normal(size=1,loc = noise_mean, scale = noise_std) #Scaler

    #Creates an array for train_y that will be filled with the for loop
    y_data = np.zeros(len(param_space)) #1 x n (row x col)

    #Iterates over evey combination of theta to find the expected y value for each combination
    for i in range(len(param_space)):
        theta_1 = param_space[i,0] #nx1 
        theta_2 = param_space[i,1] #nx1
        x = param_space[i,2] #nx1
        y_exp = theta_1*x + theta_2*x**2 +x**3 + noise #Scaler
        y_data[i] = y_exp #Scaler
    #Returns all_y
    return y_data

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
    assert isinstance(csv_file, str)==True, "csv_file must be a sting containing the name of the file"
    reader = csv.reader(open(csv_file), delimiter=",") #Reads CSV containing nx3 LHS design
    lhs_design = list(reader) #Creates list from CSV
    param_space = np.array(lhs_design).astype("float") #Turns LHS design into a useable python array (nx3)
    return param_space

def best_error_advanced(y_model, y_target):
    
    """
    Calculates the best error in the 3 input GP model
    
    Parameters
    ----------
        y_model: tensor: The y values that the GP model predicts 
        y_target: ndarray, the expected value of the function from data or other source
    
    Returns:
    --------
        best_error: float, the value of the best error encountered 
    """
    #Asserts that y_model is a tensor, y_target is an ndarray
    assert torch.is_tensor(y_model)==True, "GP predicted y values must be tensors"
    assert isinstance(y_target, np.ndarray)==True, "y_target must be an ndarray size 1xn"
    
    #Changes y_model type from torch tensor to numpy array
    y_model = y_model.numpy() #1xn
    
    #Asserst that that these y_model and y_target have equal lengths.
    assert len(y_target)==len(y_model), "y_target and y_model must be the same length"
    
    #Calculates best error as the maximum of the -error
    error = (y_target-y_model)**2 #1xn
    best_error = np.max(-error) #A number
    
    #Finds the best x index
    best_x = np.argmax(-error) #1xlen(Z.T)
    
    #Makes best_error positive again
    best_error = -best_error
    
    return best_error, best_x #1x2


def calc_ei_advanced(f_best,pred_mean,pred_var,y_target):
    """ 
    Calculates the expected improvement of the 3 input parameter GP
    Parameters
    ----------
        f_best: float, the best predicted error encountered
        pred_mean: tensor, model mean
        pred_var: tensor, model variance
        y_target: ndarray, the expected value of the function from data or other source
    
    Returns
    -------
        ei: ndarray, the expected improvement of the GP model
    """
    #Asserts that pred_mean and pred_var are tensors, f_pred is a float, and y_target is an dparray
    assert torch.is_tensor(pred_mean)==True and torch.is_tensor(pred_var)==True, "GP predicted means and variances must be tensors"
    assert isinstance(y_target, np.ndarray)==True, "y_target must be an ndarray size 1xn"
    assert isinstance(f_best, (float,int))==True, "f_best must be a float or integer"
    
    
    #Coverts tensor to np arrays
    pred_mean = pred_mean.numpy() #1xn
    pred_var = pred_var.numpy() #1xn
    
    #Checks for equal lengths
    assert len(y_target)==len(pred_mean)==len(pred_var), "y_target, pred_mean, and pred_var must be the same length"
    
    #Defines standard devaition
    pred_stdev = np.sqrt(pred_var) #1xn
    
    #If variance is zero this is important
    with np.errstate(divide = 'warn'):
        #Creates upper and lower bounds and described by Nilay's word doc
        bound_upper = ((y_target - pred_mean) +np.sqrt(f_best))/pred_var #1xn
        bound_lower = ((y_target - pred_mean) -np.sqrt(f_best))/pred_var #(STDEV or VAR?) #1xn

        #Creates EI terms in terms of Nilay's code / word doc
        ei_term1_comp1 = norm.pdf(bound_upper) - norm.pdf(bound_lower) #Why is this a CDF and not a PDF? #1xn
        ei_term1_comp2 = f_best - (y_target - pred_mean)**2 #1xn

        ei_term2_comp1 = 2*(y_target - pred_mean)*pred_stdev #(STDEV or VAR?) #1xn
        ei_term2_comp2 = (norm.pdf(bound_upper) - norm.pdf(bound_lower)) #This gives a large negative number when tested #1xn

        ei_term3_comp1 = (1/2)*norm.pdf(bound_upper/np.sqrt(2)) #(CDF or PDF) #1xn
        ei_term3_comp2 = -norm.pdf(bound_upper)*bound_upper #1xn
        ei_term3_comp3 = (1/2)*norm.pdf(bound_lower/np.sqrt(2)) #(CDF or PDF) #1xn
        ei_term3_comp4 = -norm.pdf(bound_lower)*bound_lower #1xn

        ei_term3_psi_upper = ei_term3_comp1 + ei_term3_comp2 #1xn
        ei_term3_psi_lower = ei_term3_comp3 + ei_term3_comp4 #1xn

        ei_term1 = ei_term1_comp1*ei_term1_comp2 #1xn
        ei_term2 = ei_term2_comp1*ei_term2_comp2 #1xn
        ei_term3 = -pred_var*(ei_term3_psi_upper-ei_term3_psi_lower) #1xn

        ei = ei_term1 + ei_term2 + ei_term3 #1xn
    return ei