import numpy as np
from scipy.stats import norm

def calc_ei_basic(f_best,pred_mean,pred_var,explore_bias):
    """ 
    Calculates the expected improvement of the 2 input parameter GP
    Parameters
    ----------
        f_best: float, the best predicted sse encountered
        pred_mean: tensor, model mean
        pred_var, tensor, model variance
        explore_bias: float, the numerical bias towards exploration
    
    Returns
    -------
        ei: ndarray, the expected improvement of the GP model
    
    """
    ei = np.zeros(len(pred_var)) #1x25
    pred_mean = pred_mean.numpy() #1x25
    pred_var = pred_var.numpy()    #1x25
    pred_stdev = np.sqrt(pred_var) #1x25
    for i in range(len(pred_var)):
        if pred_stdev[i] > 0:
            z = (pred_mean[i] - f_best - explore_bias)/pred_stdev[i] #number
            ei_term_1 = (pred_mean[i] - f_best - explore_bias)*norm.cdf(z) #number
            ei_term_2 = pred_stdev[i]*norm.pdf(z) #number
            ei[i] = ei_term_1 +ei_term_2
        else:
            ei[i] = 0
    return ei