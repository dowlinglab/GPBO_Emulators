import numpy as np

def calc_best_error(test_p, x, y_model, noise):
    """
    Calculates the best error in the GP model
    
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

def calc_ei_basic(f_best,pred_mean,pred_var,explore_bias):
    """ 
    Calculates the expected improvement of the 2 input parameter GP
    Parameters
    ----------
        f_best: float, the best predicted sse encountered
        pred_mean: ndarray, model mean
        pred_var, ndarray, model variance
        explore_bias: float, the numerical bias towards exploration
    
    Returns
    -------
        ei: ndarray, the expected improvement of the GP model
    
    """
    return 

# def cacl_ei_advanced():
#     EI = np.zeros(len(test_T)) # 1 x 25
#     #If variance is zero this is important 
#     with np.errstate(divide = 'warn'):
#         #Creates upper and lower bounds and described by Nilay's word doc
#         bound_upper = ((sse_true[j] - model_mean[j]) +np.sqrt(best_error))/model_variance[j]
#         bound_lower = ((sse_true[j] - model_mean[j]) -np.sqrt(best_error))/model_variance[j]
        
#         #Creates EI terms in terms of Nilay's word doc
#         ei_term1_comp1 = norm.cdf(bound_upper) - norm.cdf(bound_lower)
#         ei_term1_comp2 = best_error - (sse_true[j] - model_mean[j])**2
        
#         ei_term2_comp1 = (sse_true[j] - model_mean[j])*model_stdev[j]
#         ei_term2_comp2 = norm.pdf(bound_upper) - norm.pdf(bound_lower)
        
#         ei_term3_comp1 = (1/2)*norm.pdf(bound_upper/np.sqrt(2))
#         ei_term3_comp2 = -norm.pdf(bound_upper)*bound_upper
#         ei_term3_comp3 = (1/2)*norm.pdf(bound_lower/np.sqrt(2))
#         ei_term3_comp4 = -norm.pdf(bound_lower)*bound_lower
        
#         ei_term3_psi_upper = ei_term3_comp1 + ei_term3_comp2
#         ei_term3_psi_lower = ei_term3_comp3 + ei_term3_comp4
        
#         ei_term1 = ei_term1_comp1 + ei_term1_comp2
#         ei_term2 = ei_term2_comp1 + ei_term2_comp2
#         ei_term3 = -model_variance[j]*(ei_term3_psi_upper-ei_term3_psi_lower)
        
#         EI[j] = ei_term1 + ei_term2 + ei_term3
#     return EI

    
#     error = np.zeros(len(test_p)) #1 x 25
#     y_real = np.zeros(len(test_p)) #1 x 25
#     for i in range(len(test_p)):
#         test_p_1 = test_p[i,0] #5x1 
#         test_p_2 = test_p[i,1] #5x1
#         #Calculates actual y value for each parameter space combination
#         y_exp = test_p_1*x + test_p_2*x**2 +x**3 + noise #100 x5
#         #Ccalculates the actual y for each p combo
#         y_real[i] = sum((y_true - y_exp)**2) # A number 
#         #Calculates error of model
#         error[i] = (y_real[i]-y_model[i])**2 #A number
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