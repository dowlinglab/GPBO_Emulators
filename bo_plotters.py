from matplotlib import pyplot as plt

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