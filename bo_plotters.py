from matplotlib import pyplot as plt
import numpy as np
# import torch
# import gpytorch

def plotter_adv(test_p, z, title):
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
    test_p_1 = test_p[:,0].numpy() #Theta1 #1 x 6
    test_p_2 = test_p[:,1].numpy() #Theta 2 #1 x 6
    test_p_3 = test_p[:,2].numpy() #x #1 x 6

    color = y_model

    # creating figures
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # setting color bar
    color_map = cm.ScalarMappable(cmap=cm.Greens_r)
    color_map.set_array(color)

    # creating the heatmap
    img = ax.scatter(test_p_1, test_p_2, test_p_3, marker='s',
                     s=200, color='green')
    plt.colorbar(color_map)

    # adding title and labels
    ax.set_title("3D Heatmap")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
  
# displaying plot
plt.show()

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