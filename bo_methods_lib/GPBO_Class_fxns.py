import numpy as np

def clean_1D_arrays(array):
    """
    Turns arrays that are shape (n,) into (n, 1) arrays
    
    Parameters:
        array: ndarray, 1D array
    Returns:
        array: ndarray, 2D array with shape (n,1)
    """
    #If array is not 2D, give it shape (len(array), 1)
    if not len(array.shape) > 1:
        array == array.reshape(-1,1)
    return array

def cs1_calc_y_exp(Theta_True, x, noise_std, noise_mean=0,random_seed=6):
    """
    Creates y_data for the 2 input GP function
    
    Parameters
    ----------
        Theta_True: ndarray, The array containing the true values of Theta1 and Theta2
        x: ndarray, The list of xs that will be used to generate y
        noise_std: float, int: The standard deviation of the noise
        noise_mean: float, int: The mean of the noise
        random_seed: int: The random seed
        
    Returns:
        y_exp: ndarray, The expected values of y given x data
    """   
    
    #Asserts that test_T is a tensor with 2 columns
    assert isinstance(noise_std,(float,int)) == True, "The standard deviation of the noise must be an integer ot float."
    assert isinstance(noise_mean,(float,int)) == True, "The mean of the noise must be an integer ot float."
    assert len(Theta_True) ==2, "This function only has 2 unknowns, Theta_True can only contain 2 values."
    
    #Seed Random Noise (For Bug Testing)
    if random_seed != None:
        assert isinstance(random_seed,int) == True, "Seed number must be an integer or None"
        np.random.seed(random_seed)
        
    #Creates noise values with a certain stdev and mean from a normal distribution
    noise = np.random.normal(size=x.shape[1],loc = noise_mean, scale = noise_std) #1x n_x
    #     if isinstance(x, np.ndarray):
#         noise = np.random.normal(size=len(x),loc = noise_mean, scale = noise_std) #1x n_x
#     else:
#         noise = np.random.normal(size=1,loc = noise_mean, scale = noise_std) #1x n_x
    # True function is y=T1*x + T2*x^2 + x^3 with Gaussian noise
    y_exp =  Theta_True[0]*x + Theta_True[1]*x**2 +x**3 + noise #1x n_x #Put this as an input
  
    return y_exp


def calc_muller(x, model_coefficients, noise = 0):
    """
    Caclulates the Muller Potential
    
    Parameters
    ----------
        x: ndarray, Values of X
        model_coefficients: ndarray, The array containing the values of Muller constants
        noise: ndarray, Any noise associated with the model calculation
    
    Returns:
    --------
        y_mul: float, value of Muller potential
    """
    #Reshape x to matrix form
    x = clean_1D_arrays(x)
    X1, X2 = x
    
    #Separate all model parameters into their appropriate pieces
    sublists = []
    chunk_size = len(model_coefficients) // 6 # Calculate the size of each sublist, 6 different parameters

    for i in range(6):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size
        sublist = model_coefficients[start_index:end_index]
        sublists.append(sublist)
        
    #Calculate Muller Potential
    A, a, b, c, x0, y0 = sublists
    Term1 = a*(X1 - x0)**2
    Term2 = b*(X1 - x0)*(X2 - y0)
    Term3 = c*(X2 - y0)**2
    y_mul = np.sum(A*np.exp(Term1 + Term2 + Term3) ) + noise
    
    return y_mul

def cs2_calc_y_exp(true_model_coefficients, x, noise_std, noise_mean=0,random_seed=9):
    """
    Creates y_data (Muller Potential) for the 2 input GP function
    
    Parameters
    ----------
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
        x: ndarray, The list of xs that will be used to generate y
        noise_std: float, int: The standard deviation of the noise
        noise_mean: float, int: The mean of the noise
        random_seed: int: The random seed
        
    Returns:
        y_exp: ndarray, The expected values of y given x data
    """   
   #Asserts that test_T is a tensor with 2 columns
    assert isinstance(noise_std,(float,int)) == True, "The standard deviation of the noise must be an integer ot float."
    assert isinstance(noise_mean,(float,int)) == True, "The mean of the noise must be an integer ot float."
    
    x = clean_1D_arrays(x)
    len_x = x.shape[0]
    
#     print(len_x)
    #Seed Random Noise (For Bug Testing)
    if random_seed != None:
        assert isinstance(random_seed,int) == True, "Seed number must be an integer or None"
        np.random.seed(random_seed)
        
    #Creates noise values with a certain stdev and mean from a normal distribution
    noise = np.random.normal(size= 1 ,loc = noise_mean, scale = noise_std) #1x n_x
    
    # True function is Muller Potential
    
    y_exp = np.zeros(len_x)
    
    for i in range(len_x):
#         print(true_model_coefficients.shape)
        y_exp[i] = calc_muller(x[i], true_model_coefficients, noise)
  
    return y_exp



##FIX THESE ################################################################

def create_sse_data(q,train_T, x, y_exp, obj = "obj"):
    """
    Creates y_data for the 2 input GP function
    
    Parameters
    ----------
        q: int, Number of parameters to be regressed
        train_T: ndarray, The array containing the training data for Theta1 and Theta2
        x: ndarray, The list of xs that will be used to generate y
        y_exp: ndarray, The experimental data for y (the true value)
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        
    Returns:
        sum_error_sq: ndarray, The SSE or ln(SSE) values that the GP will be trained on
    """   
    #Asserts that test_T is a tensor with 2 columns (May delete this)
    assert isinstance(q, int), "Number of inputs must be an integer"
#     print(train_T.T)    
    if torch.is_tensor(train_T)==True:
        assert len(train_T.permute(*torch.arange(train_T.ndim -1, -1, -1))) >=q, str("This is a "+str(q)+" input GP, train_T must have at least q columns of values.")
    else:
        assert len(train_T.T) >=q, str("This is a "+str(q)+" input GP, train_T must have at least q columns of values.")
    assert len(x) == len(y_exp), "Xexp and Yexp must be the same length"
    assert obj == "obj" or obj == "LN_obj", "Objective function choice, obj, MUST be sse or LN_sse"
    
    #Clean train_T to shape (1, len(train_T)) and y_exp to (len(y_exp),1)
    train_T = clean_1D_arrays(train_T, param_clean = True)
#     print(train_T.shape)
    y_exp = clean_1D_arrays(y_exp)    

    #Creates an array for train_sse that will be filled with the for loop
    sum_error_sq = np.zeros((train_T.shape[0]))

    #Iterates over evey combination of theta to find the SSE for each combination
    #For each point in train_T
    for i in range(len(train_T)):
        #Theta 1 and theta 2 represented by columns for this case study
        theta_1 = train_T[i,0] #n_train^2x1 
        theta_2 = train_T[i,1] #n_train^2x1
        #Calc y_sim
        y_sim = theta_1*x + theta_2*x**2 +x**3 #n_train^2 x n_x
        #Clean y_sim and calculate sse or log(sse)
        y_sim = clean_1D_arrays(y_sim)
#         print(type(y_sim))
        if obj == "obj":
            sum_error_sq[i] = sum((y_sim - y_exp)**2) #Scaler
        else:
            sum_error_sq[i] = np.log(sum((y_sim - y_exp)**2)) #Scaler
    sum_error_sq = torch.from_numpy(sum_error_sq)
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
    assert len(param_space.T) >= 3, "Parameter space must have at least 3 parameters"
    
    #Converts parameters to numpy arrays if they are tensors
    if torch.is_tensor(param_space)==True:
        param_space = param_space.numpy()
    
    #clean 1D_arrays to shape (1, len(param_space))
    param_space = clean_1D_arrays(param_space)
    
    #Creates an array for train_data that will be filled with the for loop
    y_data = np.zeros(param_space.shape[0]) #1 x n (row x col)
    
    #Used when multiple values of y are being calculated
    #Iterates over evey combination of theta to find the expected y value for each combination
    for i in range(len(param_space)):
        #Theta1, theta2, and xexp are defined as coulms of param_space
        theta_1 = param_space[i,0] #nx1 
        theta_2 = param_space[i,1] #nx1
        x = param_space[i,2] #nx1 
        #Calculate y_data
        y_data[i] = theta_1*x + theta_2*x**2 +x**3 #Scaler
        #Returns all_y
    
    #Flatten y_data
    y_data = y_data.flatten()
    return y_data

def create_sse_data(param_space, x, y_exp, true_model_coefficients, obj = "obj", skip_param_types = 0):
    """
    Creates y_data for the 2 input GP function
    
    Parameters
    ----------
        param_space: ndarray, The array containing the data for Theta1 and Theta2
        x: ndarray, The list of xs that will be used to generate y
        y_exp: ndarray, The experimental data for y (the true value)
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed
        
    Returns:
        sum_error_sq: ndarray, The SSE or ln(SSE) values that the GP will be trained on
    """  
#     print(x)
#     print(skip_param_types)
    if isinstance(param_space, pd.DataFrame):
        param_space = param_space.to_numpy()

    #Will need assert statement
    assert obj == "obj" or obj == "LN_obj", "Objective function choice, obj, MUST be sse or LN_sse"
    
    x = clean_1D_arrays(x)
    param_space = clean_1D_arrays(param_space, param_clean = True)
    len_x, dim_x = x.shape[0], x.shape[1]
    
#     len_data, dim_data = param_space.shape[1], param_space.shape[0] 
#     print(len_data, dim_data)
#     print(param_space.shape)
    len_data, dim_data = param_space.shape[0], param_space.shape[1]
#     print(len_data, dim_data)
    dim_param = dim_data
#     print(true_model_coefficients)
    try:
        num_constant_type, len_constants = true_model_coefficients.shape[0], true_model_coefficients.shape[1] # 6,4
    except:
        true_model_coefficients = clean_1D_arrays(true_model_coefficients)
        num_constant_type, len_constants = true_model_coefficients.shape[0], true_model_coefficients.shape[1]
#         print(true_model_coefficients, true_model_coefficients.shape)
    num_param_type_guess = int(dim_param/len_constants)
        
    #For the case where more than 1 point is geing generated
    #Creates an array for train_sse that will be filled with the for loop
    sum_error_sq = torch.tensor(np.zeros(len_data)) #1 x n_train^2 
    model_coefficients = true_model_coefficients.copy()
    
    #Iterates over evey combination of theta to find the SSE for each combination
    for i in range(len_data):
        #Set dig out values of a from train_p
        #Set constants to change the a row to the index of the first loop
        for j in range(num_param_type_guess):
            j_model = skip_param_types + j
            model_coefficients[j_model] = param_space[i][len_constants*j: len_constants*(j+1)]
        
        y_sim = np.zeros(len_x)
        #Loop over state points (5)
        for k in range(len_x):
#             print(x[k])
            y_sim[k] = calc_muller(x[k], model_coefficients)
                
        if obj == "obj":
            sum_error_sq[i] = sum((y_sim - y_exp)**2) #Scaler
#                 print(sum_error_sq[i])
        else:
            sum_error_sq[i] = np.log(sum((y_sim - y_exp)**2)) #Scaler
    
    return sum_error_sq


def create_y_data(param_space, true_model_coefficients, x, skip_param_types = 0, noise_std=None, noise_mean=0,random_seed=9):
    """
    Creates y_data (training data) based on the function theta_1*x + theta_2*x**2 +x**3
    Parameters
    ----------
        param_space: (nx3) ndarray or tensor, parameter space over which the GP will be run
        true_model_coefficients: ndarray, The array containing the true values of Muller constants
        x: ndarray, Array containing x data
        skip_param_types: The offset of which parameter types (A - y0) that are being guessed
        noise_std: float, int: The standard deviation of the noise
        noise_mean: float, int: The mean of the noise
        random_seed: int: The random seed
    Returns
    -------
        y_sim: ndarray, The simulated y training data
    """
    #Assert statements check that the types defined in the doctring are satisfied
    
    #Converts parameters to numpy arrays if they are tensors
    if torch.is_tensor(param_space)==True:
        param_space = param_space.numpy()
        
    if isinstance(param_space, pd.DataFrame):
        param_space = param_space.to_numpy()
    
    if random_seed != None:
        assert isinstance(random_seed,int) == True, "Seed number must be an integer or None"
        np.random.seed(random_seed)
        
    #Creates noise values with a certain stdev and mean from a normal distribution
    if noise_std != None:
        noise = np.random.normal(size= 1 ,loc = noise_mean, scale = noise_std) #1x n_x
    else:
        noise = np.random.normal(size= 1 ,loc = noise_mean, scale = 0) #1x n_x
    
    param_space = clean_1D_arrays(param_space) 
    x = clean_1D_arrays(x) 
    len_data, dim_data = param_space.shape[0], param_space.shape[1] #300, 10
#     print(len_data, dim_data)
    dim_x = x.shape[1] # 2
    dim_param = dim_data - dim_x
#     print(len_data, dim_data, dim_x, dim_param)
    
    num_constant_type, len_constants = true_model_coefficients.shape[0], true_model_coefficients.shape[1] # 6,4
    num_param_type_guess = int(dim_param/len_constants)
#     print(num_param_type_guess)
        
    #For the case where more than 1 point is geing generated
    #Creates an array for train_sse that will be filled with the for loop
    #Initialize y_sim
    y_sim = np.zeros(len_data) #1 x n_train^2
    model_coefficients = true_model_coefficients.copy()

    #Iterates over evey data point to find the y for each combination
    for i in range(len_data):
        #Set dig out values of a from train_p
        #Set constants to change the a row to the index of the first loop

        #loop over number of param types (A, a, b c, x0, y0)
        for j in range(num_param_type_guess):
            j_model = skip_param_types + j
            model_coefficients[j_model] = param_space[i][len_constants*j: len_constants*(j+1)]
#         print(model_coefficients)
#         print(model_coefficients)
        A, a, b, c, x0, y0 = model_coefficients         
        #Calculate y_sim
        x = param_space[i][dim_param:dim_data]
#         print(x,x.shape)
        y_sim[i] = calc_muller(x, model_coefficients, noise)
   
    return y_sim