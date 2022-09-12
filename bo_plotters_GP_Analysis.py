from matplotlib import pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import pandas as pd
import os
import matplotlib.pyplot as plt

def save_fig(path, ext='png', close=True, verbose=True):
    """Save a figure from pyplot.
    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.
    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.
    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.
    """
    
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath),

    # Actually save the figure
    plt.savefig(savepath)
    
    # Close it
    if close:
        plt.close()

    if verbose:
        print("Done")
             
def path_name(emulator, fxn, set_lengthscale, t, obj, sep_fact, DateTime = None):
    """
    names a path
    
    Parameters
    ----------
        emulator: True/False, Determines if GP will model the function or the function error
        fxn: str, The name of the function whose file path name will be created
        set_lengthscale: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        t: int, int, Number of initial training points to use
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
    Returns:
        path: str, The path to which the file is saved
    
    """

    obj_str = "/"+str(obj)
    len_scl = "/len_scl_varies"
    sep_fact_str = ""
    run_str = "/Single_Run"
    org_TP = "/TP_"+str(t)
    
    if emulator == False:
        Emulator = "/GP_Error_Emulator"
    else:
        Emulator = "/GP_Emulator"
            
    fxn_dict = {"plot_3GP_performance":"/3_Input_GP_Analysis" , "LSO_LOO_Analysis":"/Sep_Fact_SA" , "plot_org_train":"/org_TP"}
    plot = fxn_dict[fxn]
    
    if sep_fact is not None:
        sep_fact_str = "/Sep_Fact_"+str(np.round(float(sep_fact),3))
    
    if set_lengthscale is not None:
        len_scl = "/len_scl_"+ str(set_lengthscale)           
        
    if DateTime is not None:
        path_org = DateTime+"/Figures" 
    else:
        path_org = "Test_Figs"+"/Figures"
#         path_org = "Test_Figs"+"/Sep_Analysis2"+"/Figures"
        
    path_end = Emulator + obj_str + org_TP + len_scl + sep_fact_str + plot    

    path = path_org + "/GP_Analysis_Figs" + path_end 
       
    return path


def plot_3GP_performance(X_space, Y_sim, GP_mean, GP_stdev, Theta, Xexp, emulator, len_scl, t, obj, sep_fact, test_p = None, test_y = None, verbose = True, save_figure = True, DateTime = None):
    """
    """
#     if verbose == True:
#         print("GP Mean",GP_mean)
#         print("GP Stdev",GP_stdev)
#         print("SSE",sum(GP_mean-Y_sim)**2)
#         plt.close()
    fxn = "plot_3GP_performance"
    fig, ax = plt.subplots()
#     print(Theta)
    ax.plot(X_space, GP_mean, lw=2, label="GP_mean")
    ax.plot(X_space, Y_sim, color = "green", label = "Y_sim")
#     if train_p != None:
#         ax.scatter(train_p[:,-1], train_y, color = "black", label = "Training")
    if test_p != None:
        ax.scatter(Xexp, test_y, color = "red", label = "Testing")
    
    ax.fill_between(
        X_space,
        GP_mean - 1.96 * GP_stdev,
        GP_mean + 1.96 * GP_stdev,
        alpha=0.3
    )
#     ax.set_title("GP Mean + confidence interval at"+ str(Theta))
    ax.set_xlabel("Xexp")
    ax.set_ylabel("Function Value")
    ax.set_title("GP Analysis Plot for Theta =" + str(Theta))
    ax.legend()
    
    if save_figure == True:
        path = path_name(emulator, fxn, len_scl, t, obj, sep_fact, DateTime = DateTime)
        save_fig(path, ext='png', close=True, verbose=False)  
    
    if verbose == True and save_figure == False:
        plt.show()
    
    
    return plt.show()

# def plot_2GP_performance(X_space, Y_sim, y_mean, GP_stdev, Theta, Xexp, train_p = None, train_y = None, test_p = None, test_y = None, verbose = True):
#     """
#     """
# #     if verbose == True:
# #         print("GP Mean",GP_mean)
# #         print("GP Stdev",GP_stdev)
# #         print("SSE",sum(GP_mean-Y_sim)**2)
# #         plt.close()
#     fig, ax = plt.subplots()
    
#     ax.plot(X_space, GP_mean, lw=2, label="GP_mean")
#     ax.plot(X_space, Y_sim, color = "green", label = "Y_sim")
# #     if train_p != None:
# #         ax.scatter(train_p[:,-1], train_y, color = "black", label = "Training")
#     if test_p != None:
#         ax.scatter(Xexp, test_y, color = "red", label = "Testing")
    
#     ax.fill_between(
#         X_space,
#         GP_mean - 1.96 * GP_stdev,
#         GP_mean + 1.96 * GP_stdev,
#         alpha=0.3
#     )
# #     ax.set_title("GP Mean + confidence interval at"+ str(Theta))
#     ax.set_xlabel("Xexp")
#     ax.set_ylabel("Function Value")
#     ax.legend()
    
#     return plt.show()

def plot_hyperparams(iterations, hyperparam, title):
    '''
    Plots Hyperparameters
    Parameters
    ----------
        Iterations: Float, number of training iterations
        hyperparam: ndarray, array of hyperparameter value at each training iteration 
        title: string, title of the graph
     
    Returns
    -------
        plt.show(), A plot of iterations and hyperparameter
    '''
    #Assert Statements and making an axis for training iterations
    assert isinstance(title, str)==True, "Title must be a string"
    iters_axis = np.linspace(0,iterations, iterations)
    assert len(iters_axis) == len(hyperparam), "Hyperparameter array must have length of # of training iterations"
    
    #Plotting hyperparameters vs training iterations
    plt.figure()
    plt.plot(iters_axis, hyperparam)
    plt.grid(True)
    plt.xlabel('Iterations',weight='bold')
    plt.ylabel('Hyperparameter Value',weight='bold')
    plt.title("Plot of "+title, weight='bold',fontsize = 16)
    return plt.show()

#Rework this
def plot_org_train(test_mesh,train_p, test_p, p_true, emulator, obj, len_scl, save_figure, DateTime=None, verbose = True, sep_fact = None):
    '''
    Plots original training data with true value
    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        train_p: tensor or ndarray, The training parameter space data
        test_p: tensor or ndarray, The training parameter space data
        p_true: ndarray, A 2x1 containing the true input parameters
        emulator: True/False, Determines if GP will model the function or the function error
        obj: str, Must be either obj or LN_obj. Determines whether objective fxn is sse or ln(sse)
        len_scl: float or None, The value of the lengthscale hyperparameter or None if hyperparameters will be updated at training
        save_figure: True/False, Determines whether figures will be saved
        sep_fact: float or int, The separation factor that decides what percentage of data will be training data. Between 0 and 1.
     
    Returns
    -------
        plt.show(), A plot of the original training data points and the true value
    '''
    fxn = "plot_org_train"
#     t = int(len(train_p))
    t = int(len(train_p)) + int(len(test_p))
    #xx and yy are the values of the parameter sets
    xx,yy = test_mesh
    if emulator == False:
        plt.figure()
        #plot training data and true values
        plt.scatter(train_p[:,0],train_p[:,1], color="green",s=50, label = "Training Data", marker = "x")
        try:
            plt.scatter(test_p[:,0],test_p[:,1], color="red",s=25, label = "Testing Data", marker = "x")
        except:
            plt.scatter(test_p[0],test_p[1], color="red",s=25, label = "Testing Data", marker = "x")
        plt.scatter(p_true[0],p_true[1], color="blue", label = "True Optimal Value", s=100, marker = (5,1))
        #Set plot details
        plt.legend(loc = "best")
        plt.xlabel("$\Theta_1$")
        plt.ylabel("$\Theta_2$")
        #Set axis limits based on the maximum and minimum of the parameter search space
        plt.xlim((np.amin(xx), np.amax(xx)))
        plt.ylim((np.amin(yy), np.amax(yy)))
        plt.title("Starting Training Data")
        plt.grid(True)
        
    else:
        # Create the figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the values
        ax.scatter(train_p[:,0], train_p[:,1], train_p[:,2], color = "green", s=50, label = "Training Data", marker='o')
        try:
            ax.scatter(test_p[:,0],test_p[:,1], test_p[:,2], color="red", s=25, label = "Testing Data", marker = "x")
        except:
            ax.scatter(test_p[0],test_p[1], test_p[2], color="red",s=25, label = "Testing Data", marker = "x")
        ax.set_xlabel('$\Theta_1$')
        ax.set_ylabel('$\Theta_2$')
        ax.set_zlabel('X-Value')
        ax.legend(loc = "best")
        plt.xlim((np.amin(xx), np.amax(xx)))
        plt.ylim((np.amin(yy), np.amax(yy)))
        plt.title("Starting Training Data")
        plt.grid(True)
        plt.title("Starting Training Data")
        
    if save_figure == True:
        path = path_name(emulator, fxn, len_scl, t, obj, sep_fact, DateTime)
        save_fig(path, ext='png', close=True, verbose=False)  
    
    if verbose == True and save_figure == False:
        plt.show()
        
    plt.close()
    
    return 
