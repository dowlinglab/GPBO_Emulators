import numpy as np
from scipy.stats import norm
import torch
import csv
def create_y_data(param_space, noise_std,noise_mean=0):
    """
    Creates y_data based on the actual function theta_1*x + theta_2*x**2 +x**3 + noise 
    param_space: ndarray, The parameter space over which the GP will be run
    noise_std: float or int, The standard deviation of the nosie
    noise_mean: float or int, The mean of the noise. Default is zero.
    """
    assert isinstance(noise_mean, (float, int))==True, "noise parameters must be floats or integers"
    assert isinstance(noise_std, (float, int))==True, "noise parameters must be floats or integers"
    assert isinstance(param_space, np.ndarray) == True, "parameter space must be a numpy array"
    assert len(param_space.T) ==3, "Only 3 input parameter space can be taken, param_space must be an nx3 array"
    #Creates noise values with a certain stdev and mean from a normal distribution
    noise = np.random.normal(size=1,loc = noise_mean, scale = noise_std) #Scaler

    #Creates an array for train_y that will be filled with the for loop
    y_data = np.zeros(len(param_space)) #1 x 25 (row x col)

    #Iterates over evey combination of theta to find the expected y value for each combination
    for i in range(len(param_space)):
        theta_1 = param_space[i,0] #25x1 
        theta_2 = param_space[i,1] #25x1
        x = param_space[i,2] #25x1
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