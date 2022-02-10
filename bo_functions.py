def calc_best_error(test_p, y_true, x, sse_model):
    """
    Calculates the best error in the GP model
    
    Parameters
    ----------
        test_p: ndarray: The parameter space for which the best error is being calculated
    
    Returns:
    --------
        best_error: float, the value of the best error encountered  
     """ 
    for i in range(len(test_x)):
    test_p_1 = test_p[i,0] #5x1 
    test_p_2 = test_p[i,1] #5x1
    #Calculates actual y value for each parameter space combination
    y_exp = test_p_1*x + test_p_2*x**2 +x**3 + noise #100 x5
    #Ccalculates the actual sse for each theta combo
    sse_true[i] = sum((y_true - y_exp)**2) # A number
    #Calculates error of sse and sse_model
    error[i] = (sse_true[i]-sse_model[i])**2 #A number
    if j == 0:
        #Sets first error to be the best error
        best_error = -error[i]
        best_theta = test_T[np.argmax(-error[i])]
    else:
        #Overwrites a new value of best error as error at each theta combo is updated
        best_error = np.max(-error)
        best_theta = test_T[np.argmax(-error)]
    #Makes best_error positive again
    best_error = -best_error
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
    
    for i in range(len(pred_mean))