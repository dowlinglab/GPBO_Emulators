# import numpy as np
# from scipy.stats import norm

# def best_error_basic(test_p, x, y_model, noise):
#     """
#     Calculates the best error in the 2 input GP model
    
#     Parameters
#     ----------
#         test_p: ndarray: The parameter space for which the best error is being calculated
#         y_model: ndarray: The y values that the GP model predicts 
#         noise: ndarray: The noise associated with the experimental value of the model
    
#     Returns:
#     --------
#         best_error: float, the value of the best error encountered  
#      """ 
#     for i in range(len(test_p)):
#         test_p_1 = test_p[i,0] #5x1 
#         test_p_2 = test_p[i,1] #5x1
#         #Calculates actual y value for each parameter space combination
#         best_error = np.argmax(y_model)
#     return best_error

# def calc_ei_basic(f_best,pred_mean,pred_var, explore_bias):
#     """ 
#     Calculates the expected improvement of the 2 input parameter GP
#     Parameters
#     ----------
#         f_best: float, the best predicted sse encountered
#         pred_mean: tensor, model mean
#         pred_var, tensor, model variance
#         explore_bias: float, the numerical bias towards exploration
    
#     Returns
#     -------
#         ei: ndarray, the expected improvement of the GP model
    
#     """
#     pred_mean = pred_mean.numpy() #1x25
#     pred_var = pred_var.numpy()    #1x25
#     pred_stdev = np.sqrt(pred_var) #1x25
#     for i in range(len(pred_mean)):
#         if pred_stdev[i] > 0:
#             z = (pred_mean - f_best - explore_bias)/pred_stdev #1x25
#             ei_term_1 = (pred_mean[i] - f_best - explore_bias)*norm.cdf(z) #1 x 25
#             ei_term_2 = pred_stdev*norm.pdf(z) #1 x 25
#             ei = ei_term_1 +ei_term_2
#         else:
#             ei = 0
#     return ei


# def best_error_advanced(test_p, y_model, y_true, noise):
    
#     """
#     Calculates the best error in the 3 input GP model
    
#     Parameters
#     ----------
#         test_p: ndarray: The parameter space for which the best error is being calculated
#         y_model: ndarray: The y values that the GP model predicts 
#         noise: ndarray: The noise associated with the experimental value of the model
    
#     Returns:
#     --------
#         best_error: float, the value of the best error encountered 
#     """
#     error = np.zeros(len(test_p)) #1 x 25
#     for i in range(len(test_p)):
#         test_p_1 = test_p[i,0] #5x1 (Theta 1) 
#         test_p_2 = test_p[i,1] #5x1(Theta 2)
#         test_p_3 = test_p[i,2] #5x1 (x)
#         #Calculates expected y value for each parameter space combination
#         y_exp = test_p_1*test_p_3 + test_p_2*test_p_3**2 +test_p_3**3 + noise #100 x5 (may need to redefine noise)
#         #Calculates error of model
#         error[i] = (y_true[i]-y_model[i])**2 #A number
#         if i == 0:
#             #Sets first error to be the best error
#             best_error = -error[i]
#             best_p = test_p[np.argmax(-error[i])]
#         else:
#             #Overwrites a new value of best error as error at each theta combo is updated
#             best_error = np.max(-error)
#             best_p = test_p[np.argmax(-error)]
#         #Makes best_error positive again
#         best_error = -best_error
#     return best_error
  


# def calc_ei_advanced(f_best,pred_mean,pred_var, y_true):
#     ei = np.zeros(len(pred_mean)) # 1 x 6
#     #If variance is zero this is important
#     for i in range(len(pred_mean)):
#         with np.errstate(divide = 'warn'):
#             #Creates upper and lower bounds and described by Nilay's word doc
#             bound_upper = ((y_true[i] - pred_mean[i]) +np.sqrt(f_best))/pred_var[i]
#             bound_lower = ((y_true[i] - pred_mean[i]) -np.sqrt(f_best))/pred_var[i]

#             #Creates EI terms in terms of Nilay's word doc
#             ei_term1_comp1 = norm.cdf(bound_upper) - norm.cdf(bound_lower)
#             ei_term1_comp2 = f_best - (y_true[i] - pred_mean[i])**2

#             ei_term2_comp1 = (y_true[i] - pred_mean[i])*np.sqrt(pred_var[i])
#             ei_term2_comp2 = norm.pdf(bound_upper) - norm.pdf(bound_lower)

#             ei_term3_comp1 = (1/2)*norm.pdf(bound_upper/np.sqrt(2))
#             ei_term3_comp2 = -norm.pdf(bound_upper)*bound_upper
#             ei_term3_comp3 = (1/2)*norm.pdf(bound_lower/np.sqrt(2))
#             ei_term3_comp4 = -norm.pdf(bound_lower)*bound_lower

#             ei_term3_psi_upper = ei_term3_comp1 + ei_term3_comp2
#             ei_term3_psi_lower = ei_term3_comp3 + ei_term3_comp4

#             ei_term1 = ei_term1_comp1 + ei_term1_comp2
#             ei_term2 = ei_term2_comp1 + ei_term2_comp2
#             ei_term3 = -pred_var[i]*(ei_term3_psi_upper-ei_term3_psi_lower)

#             ei[i] = ei_term1 + ei_term2 + ei_term3
#         return ei

import numpy as np
from scipy.stats import norm

def best_error_basic(test_p, x, y_model, noise):
    """
    Calculates the best error in the 2 input GP model
    
    Parameters
    ----------
        test_p: ndarray: The parameter space for which the best error is being calculated
        y_model: ndarray: The y values that the GP model predicts 
        noise: ndarray: The noise associated with the experimental value of the model
    
    Returns:
    --------
        best_error: float, the value of the best error encountered  
     """ 
    for i in range(len(test_p)):
        test_p_1 = test_p[i,0] #5x1 
        test_p_2 = test_p[i,1] #5x1
        #Calculates actual y value for each parameter space combination
        best_error = np.argmax(y_model)
    return best_error

def calc_ei_basic(f_best,pred_mean,pred_var, explore_bias):
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
    pred_mean = pred_mean.numpy() #1x25
    pred_var = pred_var.numpy()    #1x25
    pred_stdev = np.sqrt(pred_var) #1x25
    for i in range(len(pred_mean)):
        if pred_stdev[i] > 0:
            z = (pred_mean - f_best - explore_bias)/pred_stdev #1x25
            ei_term_1 = (pred_mean[i] - f_best - explore_bias)*norm.cdf(z) #1 x 25
            ei_term_2 = pred_stdev*norm.pdf(z) #1 x 25
            ei = ei_term_1 +ei_term_2
        else:
            ei = 0
    return ei


def best_error_advanced(test_p, y_model, y_true, noise):
    
    """
    Calculates the best error in the 3 input GP model
    
    Parameters
    ----------
        test_p: ndarray: The parameter space for which the best error is being calculated
        y_model: ndarray: The y values that the GP model predicts 
        noise: ndarray: The noise associated with the experimental value of the model
    
    Returns:
    --------
        best_error: float, the value of the best error encountered 
    """
    #Separates parameters for use
    test_p_1 = test_p[:,0] #Theta1 #1 x 6
    test_p_2 = test_p[:,1] #Theta 2 #1 x 6
    test_p_3 = test_p[:,2] #x #1 x 6
    
    #Calculates error for each parameter space combination
    y_exp = (test_p_1*test_p_3 + test_p_2*test_p_3**2 +test_p_3**3 + noise).numpy() #1 x6 (may need to redefine noise)
    error = (y_exp-y_model)**2 #1x6
    best_error = np.max(-error) #A number
    best_p = test_p[np.argmax(-error)] #1x3
    
    #Makes best_error positive again
    best_error = -best_error
    
    return best_error, best_p #1x2


def calc_ei_advanced(f_best,pred_mean,pred_var, y_true):
        """ 
    Calculates the expected improvement of the 3 input parameter GP
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
    #ei = np.zeros(len(pred_mean)) # 1 x 6
    #If variance is zero this is important
    pred_mean = pred_mean.numpy()
    pred_var = pred_var.numpy()
    with np.errstate(divide = 'warn'):
        #Creates upper and lower bounds and described by Nilay's word doc
        bound_upper = ((y_true - pred_mean) +np.sqrt(f_best))/pred_var
        bound_lower = ((y_true - pred_mean) -np.sqrt(f_best))/pred_var

        #Creates EI terms in terms of Nilay's word doc
        ei_term1_comp1 = norm.cdf(bound_upper) - norm.cdf(bound_lower)
        ei_term1_comp2 = f_best - (y_true - pred_mean)**2

        ei_term2_comp1 = (y_true - pred_mean)*np.sqrt(pred_var)
        ei_term2_comp2 = norm.pdf(bound_upper) - norm.pdf(bound_lower)

        ei_term3_comp1 = (1/2)*norm.pdf(bound_upper/np.sqrt(2))
        ei_term3_comp2 = -norm.pdf(bound_upper)*bound_upper
        ei_term3_comp3 = (1/2)*norm.pdf(bound_lower/np.sqrt(2))
        ei_term3_comp4 = -norm.pdf(bound_lower)*bound_lower

        ei_term3_psi_upper = ei_term3_comp1 + ei_term3_comp2
        ei_term3_psi_lower = ei_term3_comp3 + ei_term3_comp4

        ei_term1 = ei_term1_comp1 + ei_term1_comp2
        ei_term2 = ei_term2_comp1 + ei_term2_comp2
        ei_term3 = -pred_var*(ei_term3_psi_upper-ei_term3_psi_lower)

        ei = ei_term1 + ei_term2 + ei_term3
    return ei