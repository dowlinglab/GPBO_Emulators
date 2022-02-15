from matplotlib import pyplot as plt
import numpy as np
def y_plotter2(parameter_space, data):
    """
    Plots the y_values of the GP
    Parameters
    ----------
        parameter_space: nparray, meshgrid of 3 input parameters 
        data: NOT SURE - y-model at each point defined by the meshgrid
    
    Returns
    -------
        A 3D Heat map of the values predicted by the GP
    """
    X,Y,Z = parameter_space
    
    # Create data
    ##Not sure how to do this
    data = X+Y+Z #(filler for testing)
#     with gpytorch.settings.fast_pred_var(), torch.no_grad():
#         data = likelihood(model(parameter_space)).loc.numpy()

    kw = {
        'vmin': data.min(),
        'vmax': data.max(),
        'levels': np.linspace(data.min(), data.max(), 10),
    }

    # Create a figure with 3D ax
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Plot contour surfaces
    _ = ax.contourf(
        X[:, :, 0], Y[:, :, 0], data[:, :, 0],
        zdir='z', offset=0, **kw
    )
    _ = ax.contourf(
        X[0, :, :], data[0, :, :], Z[0, :, :],
        zdir='y', offset=0, **kw
    )
    C = ax.contourf(
        data[:, -1, :], Y[:, -1, :], Z[:, -1, :],
        zdir='x', offset=X.max(), **kw
    )
    # --


    # Set limits of the plot from coord limits
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    zmin, zmax = Z.min(), Z.max()
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

    # Plot edges
    edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
    ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
    ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
    ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

    # Set labels and zticks
    ax.set(
        xlabel='X [km]',
        ylabel='Y [km]',
        zlabel='Z [m]',
    )

    # Set distance and angle view
    ax.view_init(40, -30)
    ax.dist = 11

    # Colorbar
    fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='y value')

    # Show Figure
    plt.show()

def y_plotter(test_mesh, z, p_true, p_GP_opt):
    # Plot Heat Map for SSE

    #Defines the x and y coordinates that will be used to generate the heat map, this step isn't
    #necessary, but streamlines the process
    xx , yy = test_mesh

    #Plots Theta1 vs Theta 2 with sse on the z axis and plots the color bar
    #Plot sse.T because test_mesh.T was used to calculate sse
    plt.contourf(xx, yy,z.T)
    plt.colorbar()

    #Plots the true optimal value and the GP value
    plt.scatter(p_true[0],p_true[1], color="red", label = "True", s=50)
    plt.scatter(p_GP_opt[0],p_GP_opt[1], color="orange", label = "GP")

    #Plots axes such that they are scaled the same way (eg. circles look like circles)
    plt.axis('scaled')

    #Plots grid and legend
    plt.grid()
    plt.legend(loc = 'best')

    #Creates axis labels and title
    plt.xlabel('Theta 1',weight='bold')
    plt.ylabel('Theta 2',weight='bold')
    plt.title('Heat Map of SSE', weight='bold',fontsize = 16)

    #Shows plot
    return plt.show()

def stdev_plotter(test_mesh, z, p_true, p_GP_opt):

    xx , yy = test_mesh

    #Plots Theta1 vs Theta 2 with sse on the z axis and plots the color bar
    #Plot stdev.T because test_mesh.T was used to calculate stdev
    plt.contourf(xx,yy,z.T)
    plt.colorbar()

    #Plots the true optimal value and the GP value
    plt.scatter(p_true[0],p_true[1], color="red", label = "True", s=50)
    plt.scatter(p_GP_opt[0],p_GP_opt[1], color="orange", label = "GP")

    #Plots axes such that they are scaled the same way (eg. circles look like circles)
    plt.axis('scaled')

    #Plots grid and legend
    plt.grid()
    plt.legend(loc = 'best')

    #Creates axis labels and title
    plt.xlabel('Theta 1',weight='bold')
    plt.ylabel('Theta 2',weight='bold')
    plt.title('Heat Map of Standard Deviation', weight='bold',fontsize = 16)

    #Shows plot
    return plt.show()

def ei_plotter(test_mesh, z, p_true, p_GP_opt):
    xx , yy = test_mesh
    
    #Plots EI
    plt.contourf(xx, yy,z.T)
    plt.colorbar()

    #Plots axes such that they are scaled the same way (eg. circles look like circles)
    plt.axis('scaled')
    
    #Plots the true optimal value and the GP value
    plt.scatter(p_true[0],p_true[1], color="red", label = "True", s=50)
    plt.scatter(p_GP_opt[0],p_GP_opt[1], color="orange", label = "GP")
    
    #Plots grid and legend
    plt.grid()
    plt.legend(loc = 'best')

    #Creates axis labels and title
    plt.xlabel('Theta 1',weight='bold')
    plt.ylabel('Theta 2',weight='bold')
    plt.title('Heat Map of Expected Improvement', weight='bold',fontsize = 16)

    #Shows plot
    return plt.show()


def y_plotter_adv(test_p, z):
    # Plot Heat Map for SSE

    #Defines the x and y coordinates that will be used to generate the heat map, this step isn't
    #necessary, but streamlines the process
    p1 = test_p[:,0]
    p2 = test_p[:,1]
    p3 = test_p[:,2]

    #Plots Theta1 vs Theta 2 with sse on the z axis and plots the color bar
    #Plot sse.T because test_mesh.T was used to calculate sse
    plt.scatter(p1,z, label = "Theta_1")
    plt.scatter(p2,z, label = "Theta_2")
    plt.axis('scaled')
    plt.grid()
    plt.legend(loc = 'best')
    plt.xlabel('Theta Values',weight='bold')
    plt.ylabel('y_model',weight='bold')
    plt.title('y_model plotted', weight='bold',fontsize = 16)
    plt.show()
    
    plt.scatter(p3,z)
    plt.axis('scaled')
    plt.grid()
    plt.xlabel('x',weight='bold')
    plt.ylabel('y_model',weight='bold')
    plt.title('y_model plotted', weight='bold',fontsize = 16)
    plt.show()
#     plt.contourf(xx,yy,z) #How would I make a heat map for something that isn't 2D?
#     plt.colorbar()
    #Plots the true optimal value and the GP value
#     plt.scatter(p_true[0],p_true[1], color="red", label = "True", s=50)
#     plt.scatter(p_GP_opt[0],p_GP_opt[1], color="orange", label = "GP")
    #Plots axes such that they are scaled the same way (eg. circles look like circles)

    #Shows plot
    return print("")

# def stdev_plotter_adv(test_mesh, z):

#     xx , yy = test_mesh

#     #Plots Theta1 vs Theta 2 with sse on the z axis and plots the color bar
#     #Plot stdev.T because test_mesh.T was used to calculate stdev
#     plt.contourf(xx,yy,z)
#     plt.colorbar()

#     #Plots the true optimal value and the GP value
# #     plt.scatter(p_true[0],p_true[1], color="red", label = "True", s=50)
# #     plt.scatter(p_GP_opt[0],p_GP_opt[1], color="orange", label = "GP")

#     #Plots axes such that they are scaled the same way (eg. circles look like circles)
#     plt.axis('scaled')

#     #Plots grid and legend
#     plt.grid()
# #     plt.legend(loc = 'best')

#     #Creates axis labels and title
#     plt.xlabel('Theta 1',weight='bold')
#     plt.ylabel('Theta 2',weight='bold')
#     plt.title('Heat Map of Standard Deviation', weight='bold',fontsize = 16)

#     #Shows plot
#     return plt.show()

# def ei_plotter_adv(test_mesh, z):
#     xx , yy = test_mesh
    
#     #Plots EI
#     plt.contourf(xx, yy,z.T)
#     plt.colorbar()

#     #Plots axes such that they are scaled the same way (eg. circles look like circles)
#     plt.axis('scaled')
    
#     #Plots the true optimal value and the GP value
# #     plt.scatter(p_true[0],p_true[1], color="red", label = "True", s=50)
# #     plt.scatter(p_GP_opt[0],p_GP_opt[1], color="orange", label = "GP")
    
#     #Plots grid and legend
#     plt.grid()
# #     plt.legend(loc = 'best')

#     #Creates axis labels and title
#     plt.xlabel('Theta 1',weight='bold')
#     plt.ylabel('Theta 2',weight='bold')
#     plt.title('Heat Map of Expected Improvement', weight='bold',fontsize = 16)

#     #Shows plot
#     return plt.show()