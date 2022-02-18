import numpy as np
from scipy.stats import norm
import torch
import gpytorch

def create_y_data_basic(Theta_True, train_T, x, noise_std, noise_mean=0):
    """
    Creates y_data for the 2 input GP function
    
    Parameters
    ----------
        Theta_True: ndarray, The array containing the true values of Theta1 and Theta2
        train_T: ndarray, The array containing the training data for Theta1 and Theta2
        x: ndarray, The list of xs that will be used to generate y
        
    Returns:
        train_y: ndarray, The SSE values that the GP will be trained on
    """    
    assert len(train_T.T) ==2, "This is a 2 input GP, train_T can only contain 2 columns of values."
    assert len(Theta_True) ==2, "This is a 2 input GP, Theta_True can only contain 2 values."
    
    #Creates noise values with a certain stdev and mean from a normal distribution
    noise = torch.tensor(np.random.normal(size=len(x),loc = noise_mean, scale = noise_std)) #1x n_x
    # True function is y=T1*x + T2*x^2 + x^3 with Gaussian noise
    y_true =  Theta_True[0]*x + Theta_True[1]*x**2 +x**3 + noise #1x n_x

    #Creates an array for train_y that will be filled with the for loop
    train_y = torch.tensor(np.zeros(len(train_T))) #1 x n^2

    #Iterates over evey combination of theta to find the SSE for each combination
    for i in range(len(train_T)):
        theta_1 = train_T[i,0] #n^2x1 
        theta_2 = train_T[i,1] #n^2x1
        y_exp = theta_1*x + theta_2*x**2 +x**3 + noise #n^2 x n_x
        train_y[i] = sum((y_true - y_exp)**2) #Scaler
    return train_y

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

    def __init__(self, train_T, train_y, likelihood):
        """
        Initializes the model
        
        Parameters
        ----------
        self : A class,The model itself. In this case, gpytorch.models.ExactGP
        train_T : tensor, The inputs of the training data
        train_y : tensor, the output of the training data
        likelihood : bound method, the lieklihood of the model. In this case, it must be Gaussian
        
        """
        #Initializes the GP model with train_Y, train_y, and the likelihood
        ##Calls the __init__ method of parent class
        super(ExactGPModel, self).__init__(train_T, train_y, likelihood)
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
        mean_x = self.mean_module(x) #1x100
        #Defines the covariance matrix based off of x
        covar_x = self.covar_module(x) #100 x 100 covariance matrix
        #Constructs a multivariate normal random variable, based on mean and covariance. 
            #Can be multivariate, or a batch of multivariate normals
            #Returns multivariate normal distibution gives the mean and covariance of the GP        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) #Multivariate dist based on 1x100 tensor
    
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
        ei: ndarray, the expected improvement of the GP model
    
    """
    assert len(pred_mean) == len(pred_var), "GP predicted means and variances must be the same length"
    assert torch.is_tensor(pred_mean)==True and torch.is_tensor(pred_var)==True, "GP predicted means and variances must be tensors"
    
    #Creates empty list to store ei values
    ei = np.zeros(len(pred_var)) #1xn
    
    #Converts tensors to np arrays and defines standard deviation
    pred_mean = pred_mean.numpy() #1xn
    pred_var = pred_var.numpy()    #1xn
    pred_stdev = np.sqrt(pred_var) #1xn
  
    
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