from matplotlib import pyplot as plt
import numpy as np
# import torch
# import gpytorch

def plotter_adv(parameter_space, z, title="Model Output"):
    """
    Plots the y_values of the GP
    Parameters
    ----------
        parameter_space: ndarray, meshgrid of 3 input parameters, Theta1, Theta2, and x
        z:  ndarray, nx1 array of values
        title: str, The title for the graph
    
    Returns
    -------
        A 3D Heat map of the values of z predicted by the GP
    """
    p_1 = parameter_space[:,0].numpy() #Theta1 #1xn
    p_2 = parameter_space[:,1].numpy() #Theta2 #1xn
    p_3 = parameter_space[:,2].numpy() #x #1xn

    color = z

    # creating figures
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # setting color bar
    color_map = cm.ScalarMappable(cmap=cm.Greens_r)
    color_map.set_array(color)

    # creating the heatmap
    img = ax.scatter(p_1, p_2, p_3, marker='s',
                     s=200, color='green')
    plt.colorbar(color_map)

    # adding title and labels
    ax.set_title("3D Heatmap of"+title)
    ax.set_xlabel('$\\theta_1$')
    ax.set_ylabel('$\\theta_2$')
    ax.set_zlabel('x coordinate')
  
    # displaying plot
    return plt.show()

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