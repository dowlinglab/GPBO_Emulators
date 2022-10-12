#CS2 Pseudo Code

# Loop over number of runs

    # 1) Read in all LHS generated data + experimental value data, and set theta mesh grids
         # Reading in is the same as before
        
    # 2) Separate into training and testing data
        # Same as before - Use test_train_split()
        #Could use - sklearn.model_selection import train_test_split instead of test_train split I created and then turn them to tensors and stack them manually
        #Plot training data: Should I even do this? If so can we write efficient pseudo code for this?
        

    # 3) Normalize values
        # Not exactly sure when to do this. Should I just normalize the inputs or should I normalize X and Y data? 
        #When is it appropriate to switch back to non-normalized units?

    # 4) Train GP
        #Same as before
        #train_GP_model()

    # 5) Evaluate GP
        #Get model mean, variance, etc,
            # calc_GP_outputs()
        # Calculate things like EI, SSE, stdev, etcs.
            #Need to modify eval_GP_emulator_tot() and eval_GP_basic_tot() to efficiently enumerate all points,
                #Itertools.combinations()
            #Need to save SSE reults in an 8d pxp array , otherwise, it's the same
            #Not sure how to do this well
            

    # 6) Use grid search to find initial guesses for parameters
        #Same except not sure how to expand to multiple dimensions efficiently
            #Option 1: Grid Search w/ #Itertools.combinations()
            #Option 2: LHS
        # Need to Modify find_opt_and_best_arg() & argmax_multiple()

    # 7) Use Scipy to find the true values of max(EI) and min(SSE) parameter sets
        #Need to modify eval_GP_scipy() to multiple dimensions #Itertools.combinations()
        #point = [theta_guess]
        #OR for emulator versions
        #point = [theta_guess]
        #point.append(Xexp[k])
        
        #Need to modify bounds for find_opt_best_scipy() efficiently in multiple dimensions
        

    # 8) Plots and save important values
        #Need to figure out how to do this efficiently for multiple values
        
    # 9) Check for convergence
        #Same as before
        
    # 10) Add max(EI) to training set
        #Same as before. Turn this part into a function    