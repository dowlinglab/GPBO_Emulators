import numpy as np
from scipy.stats import norm

def best_error_advanced(test_p, y_model, y_target):
    
    """
    Calculates the best error in the 3 input GP model
    
    Parameters
    ----------
        test_p: ndarray: The parameter space for which the best error is being calculated
        y_model: tensor: The y values that the GP model predicts 
        y_target: ndarray, the expected value of the function from data or other source
    
    Returns:
    --------
        best_error: float, the value of the best error encountered 
    """
    assert 
    y_model = y_model.numpy()
    
    #Calculates best error as the maximum of the -error
    error = (y_target-y_model)**2 #1x6
    best_error = np.max(-error) #A number
    
    #Finds the best x index
    best_x = np.argmax(-error) #1x3
    
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
    #Coverts tensor to np arrays
    pred_mean = pred_mean.numpy()
    pred_var = pred_var.numpy()
    
    #Defines standard devaition
    pred_stdev = np.sqrt(pred_var)
    
    #If variance is zero this is important
    with np.errstate(divide = 'warn'):
        #Creates upper and lower bounds and described by Nilay's word doc
        bound_upper = ((y_target - pred_mean) +np.sqrt(f_best))/pred_var
        bound_lower = ((y_target - pred_mean) -np.sqrt(f_best))/pred_var #(STDEV or VAR?)

        #Creates EI terms in terms of Nilay's code / word doc
        ei_term1_comp1 = norm.pdf(bound_upper) - norm.pdf(bound_lower) #Why is this a CDF and not a PDF?
        ei_term1_comp2 = f_best - (y_target - pred_mean)**2

        ei_term2_comp1 = 2*(y_target - pred_mean)*pred_stdev #(STDEV or VAR?)
        ei_term2_comp2 = (norm.pdf(bound_upper) - norm.pdf(bound_lower)) #This gives a large negative number when tested

        ei_term3_comp1 = (1/2)*norm.pdf(bound_upper/np.sqrt(2)) #(CDF or PDF)
        ei_term3_comp2 = -norm.pdf(bound_upper)*bound_upper
        ei_term3_comp3 = (1/2)*norm.pdf(bound_lower/np.sqrt(2)) #(CDF or PDF)
        ei_term3_comp4 = -norm.pdf(bound_lower)*bound_lower

        ei_term3_psi_upper = ei_term3_comp1 + ei_term3_comp2
        ei_term3_psi_lower = ei_term3_comp3 + ei_term3_comp4

        ei_term1 = ei_term1_comp1*ei_term1_comp2
        ei_term2 = ei_term2_comp1*ei_term2_comp2
        ei_term3 = -pred_var*(ei_term3_psi_upper-ei_term3_psi_lower)

        ei = ei_term1 + ei_term2 + ei_term3
    return ei