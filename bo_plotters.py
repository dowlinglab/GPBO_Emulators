from matplotlib import pyplot as plt

def basic_plotter(test_mesh, z, p_true, p_GP_opt,title):
    '''
    Plots heat maps for 2 input GP
    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        z: ndarray or tensor, An NxN Array containing all points that will be plotted
        p_true: ndarray, A 2x1 containing the true input parameters
        p_GP_Opt: ndarray, A 2x1 containing the optimal input parameters predicted by the GP
        title: str, A string containing the title of the plot
     
    Returns
    -------
        plt.show(), A heat map of test_mesh and z
    '''
    #Defines the x and y coordinates that will be used to generate the heat map, this step isn't
    #necessary, but streamlines the process
    xx , yy = test_mesh #NxN, NxN
    
    #Assert that test_mesh and z are NxN, that p_true and p_GP_opt are 2x1, and the title is a string
    assert isinstance(z, np.ndarray)==True or assert torch.is_tensor(z)==True, "Values in the heat map must be np arrays or torch tensors
    assert xx.shape==yy.shape, "Test_mesh must be 2 NxN arrays"
    assert z.shape==xx.shape, "Array z must be NxN"
    assert len(p_true) ==len(p_GP_opt)==2, "p_true and p_GP_opt must be 2x1 for a 2 input GP"
    assert isinstance(title, str)==True, "Title must be a string"
    
    #Plots Theta1 vs Theta 2 with sse on the z axis and plots the color bar
    #Plot z.T because test_mesh.T was used to calculate z
    plt.contourf(xx, yy,z.T)
    plt.colorbar()

    #Plots the true optimal value and the GP value
    plt.scatter(p_true[0],p_true[1], color="red", label = "True Optimal Value", s=50, marker = (5,1))
    plt.scatter(p_GP_opt[0],p_GP_opt[1], color="orange", label = "GP Optimal Value", marker = ".")

    #Plots axes such that they are scaled the same way (eg. circles look like circles)
    plt.axis('scaled')

    #Plots grid and legend
    plt.grid()
    plt.legend(loc = 'best')

    #Creates axis labels and title
    plt.xlabel('$\\theta_1$',weight='bold')
    plt.ylabel('$\\theta_2$',weight='bold')
    plt.title("Heat Map of "+title, weight='bold',fontsize = 16)

    #Shows plot
    return plt.show()

def y_plotter_basic(test_mesh, z, p_true, p_GP_opt,title="y"):
    '''
    Helper function for basic_plotter. Calls basic_plotter specifically for plotting y values.

    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        z: ndarray, An NxN Array containing all points that will be plotted. Y-values
        p_true: ndarray, A 2x1 containing the true input parameters
        p_GP_Opt: ndarray, A 2x1 containing the optimal input parameters predicted by the GP
        title: str, A string containing the title of the plot
     
    Returns
    -------
        plt.show(), A heat map of test_mesh and z (y values)
    '''
    return basic_plotter(test_mesh, z, p_true, p_GP_opt,title)

def stdev_plotter_basic(test_mesh, z, p_true, p_GP_opt):
    '''
    Helper function for basic_plotter. Calls basic_plotter specifically for plotting standard deviation values.

    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        z: ndarray, An NxN Array containing all points that will be plotted. Y-values
        p_true: ndarray, A 2x1 containing the true input parameters
        p_GP_Opt: ndarray, A 2x1 containing the optimal input parameters predicted by the GP
     
    Returns
    -------
        plt.show(), A heat map of test_mesh and z (standard deviation values)
    '''
    title = "Standard Deviation"
    return basic_plotter(test_mesh, z, p_true, p_GP_opt,title)

def ei_plotter_basic(test_mesh, z, p_true, p_GP_opt):
    """
    Helper function for basic_plotter. Calls basic_plotter specifically for plotting expected improvement values.

    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        z: ndarray, An NxN Array containing all points that will be plotted. Y-values
        p_true: ndarray, A 2x1 containing the true input parameters
        p_GP_Opt: ndarray, A 2x1 containing the optimal input parameters predicted by the GP
     
    Returns
    -------
        plt.show(), A heat map of test_mesh and z (expected improvement values)
    """
    title = "Expected Improvement"
    return basic_plotter(test_mesh, z, p_true, p_GP_opt,title)

                        
    