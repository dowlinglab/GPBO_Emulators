import numpy as np
from scipy.stats import norm
import torch

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