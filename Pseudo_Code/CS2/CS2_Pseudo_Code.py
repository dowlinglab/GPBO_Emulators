#CS2 Pseudo Code

# Loop over number of runs

    # 1) Read in all LHS generated data + experimental value data, and set theta mesh grids
         # Reading in is the same as before
        
    # 2) Separate into training and testing data
        # Same as before
        #Could use - sklearn.model_selection import train_test_split instead of test_train split I created and then turn them to tensors and stack them manually
        #Plot training data: Should I even do this? If so can we write efficient pseudo code for this?
        

    # 3) Normalize values
        # Not exactly sure when to do this. Should I just normalize the inputs or should I normalize X and Y data? 
        #When is it appropriate to switch back to non-normalized units?

    # 4) Train GP
        #Same as before

    # 5) Evaluate GP
        #Get model mean, variance, etc,
            # Same as before
        # Calculate things like EI, SSE, stdev, etcs.
            #DIFFERENT 
            

    # 6) Use grid search to find initial guesses for parameters

    # 7) Use Scipy to find the true values of max(EI) and min(SSE) parameter sets

    # 8) Plots and save important values
    
    # 9) Check for convergence
    
    # 10) Add max(EI) to training set
    
    