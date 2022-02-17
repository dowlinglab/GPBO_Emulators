from matplotlib import pyplot as plt

def basic_plotter(test_mesh, z, p_true, p_GP_opt,title):
    '''
    Plots heat maps for 2 input GP
    Parameters
    ----------
        test_mesh: ndarray, 2 NxN uniform arrays containing all values of the 2 input parameters. Created with np.meshgrid()
        z: ndarray, An NxN Array containing all points that will be plotted
        p_true: ndarray, A 2x1 containing the true input parameters
        p_GP_Opt: ndarray, A 2x1 containing the optimal input parameters predicted by the GP
        title: str, A string containing the title of the plot
     
    Returns
    -------
        plt.show(), A heat map of test_mesh and z
    '''
    #Defines the x and y coordinates that will be used to generate the heat map, this step isn't
    #necessary, but streamlines the process
    xx , yy = test_mesh

    #Plots Theta1 vs Theta 2 with sse on the z axis and plots the color bar
    #Plot sse.T because test_mesh.T was used to calculate sse
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