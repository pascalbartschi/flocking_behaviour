import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def initialize_random(param):
    '''
    This function initializes the agents, as well as the plotting window

    Parameters
    ----------
    param : dict
        Holds the necessary parameters.

    Raises
    ------
    ValueError
        Dimension is not supplied or invalid.

    Returns
    -------
    agent_old : np.ndarray
        Zeros.
    agent_now : np.ndarray
        Initial positions of agents.
    fig : figure object
        Figure to plot in.
    ax : artist object
        Artist axis to plot in.

    '''
    n = param["n"]
    d = param["d"]
    low, high = param["init_coord"]
    lower_lim, upper_lim = param["ax_lim"]
    

    # initialize n agents in d dimensions
    
    
    agent_now = np.random.uniform(low=low, high=high, size=(n, d))
    agent_old = np.zeros_like(agent_now)
    agent_temp = np.zeros_like(agent_now)
    
    if d == 2: 
        # some fig to plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(lower_lim, upper_lim)
        ax.set_ylim(lower_lim, upper_lim)
    elif d == 3: 
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection = "3d")
        ax.set_xlim(lower_lim, upper_lim)
        ax.set_ylim(lower_lim, upper_lim)
        ax.set_zlim(lower_lim, upper_lim)
    else:
        raise ValueError("Incorrect number of dimensions. Choose d=2 or d=3.")

    return agent_old, agent_now, fig, ax, param

def initialize_predator(param):
    '''
    This function initializes the agents, as well as the plotting window

    Parameters
    ----------
    param : dict
        Holds the necessary parameters.

    Raises
    ------
    ValueError
        Dimension is not supplied or invalid.

    Returns
    -------
    agent_old : np.ndarray
        Zeros.
    agent_now : np.ndarray
        Initial positions of agents.
    fig : figure object
        Figure to plot in.
    ax : artist object
        Artist axis to plot in.

    '''
    n = param["n"]
    d = param["d"]
    low, high = param["init_coord"]
    lower_lim, upper_lim = param["ax_lim"]
    

    # initialize n agents in d dimensions
    
    
    agent_now = np.random.uniform(low=low, high=high, size=(n, d))
    agent_old = np.zeros_like(agent_now)
    param["predator_xy"] = np.random.choice([lower_lim + 1, upper_lim - 1], size = 2, replace = True)
    agent_temp = np.zeros_like(agent_now)
    
    if d == 2: 
        # some fig to plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(lower_lim, upper_lim)
        ax.set_ylim(lower_lim, upper_lim)
    elif d == 3: 
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection = "3d")
        ax.set_xlim(lower_lim, upper_lim)
        ax.set_ylim(lower_lim, upper_lim)
        ax.set_zlim(lower_lim, upper_lim)
    else:
        raise ValueError("Incorrect number of dimensions. Choose d=2 or d=3.")

    return agent_old, agent_now, fig, ax, param

def inline_plot_2D_basic(agent_now, ax, param):
    '''
    Plots in 2D after clearing axis object

    Parameters
    ----------
    agent_now : array (n, d)
    ax : axis object

    Returns
    -------
    ax : axis object

    '''
    lower_lim, upper_lim = param["ax_lim"]
    # plot 2D
    ax.clear()
    ax.scatter(agent_now[:, 0], agent_now[:, 1], s = param["pointsize"])
    ax.set_xlim(lower_lim, upper_lim)
    ax.set_ylim(lower_lim, upper_lim)
    plt.pause(0.01)
    
    return ax
    
def inline_plot_2D_predator(agent_now, ax, param):
    '''
    Plots in 2D after clearing axis object

    Parameters
    ----------
    agent_now : array (n, d)
    ax : axis object

    Returns
    -------
    ax : axis object

    '''
    lower_lim, upper_lim = param["ax_lim"]
    # plot 2D
    ax.clear()
    ax.scatter(agent_now[:, 0], agent_now[:, 1], s = param["pointsize"])
    ax.scatter(param["predator_xy"][0], param["predator_xy"][1], s = param["pointsize"] * 4, c = "red")
    ax.set_xlim(lower_lim, upper_lim)
    ax.set_ylim(lower_lim, upper_lim)
    plt.legend()
    plt.pause(0.1)
    
    return ax

def inline_plot_3D_basic(agent_now, ax, param):
    '''
    Plots in 3D after clearing axis object

    Parameters
    ----------
    agent_now : array (n, d)
    ax : axis object

    Returns
    -------
    ax : axis object

    '''
    lower_lim, upper_lim = param["ax_lim"]
    ax.clear()
    ax.scatter(agent_now[:, 0], agent_now[:, 1], agent_now[:, 2], s = param["pointsize"])
    ax.set_xlim(lower_lim, upper_lim)
    ax.set_ylim(lower_lim, upper_lim)
    ax.set_zlim(lower_lim, upper_lim)
    plt.pause(0.01)
    
    return ax

def euclidian_dist(array, axis):

    return np.sqrt(np.sum(array**2, axis = axis))

def update(agent_now, agent_old, param):
    '''
    Update of the postion of the agents, with the assumption that their acceleretion is computed from
    their pull towards the center of mass and the delta between their new and old position
    Parameters
    ----------
    agent_now : array (n, d)
        DESCRIPTION.
    agent_old : array (n, d)
        DESCRIPTION.
    center_pull : int, optional
        DESCRIPTION. The default is 1.5.

    Returns
    -------
    agent_temp : updated positon of agents

    '''
    d = param["d"]
    center_pull = param["center_pull"]
    lower_lim, upper_lim = param["ax_lim"]
    
    agent_temp = np.zeros_like(agent_now)
    # calculate center of mass
    C = np.mean(agent_now, axis=0)

    # update the agent position according to acceleration to center
    for j in range(d):
        agent_temp[:, j] = 2 * agent_now[:, j] - agent_old[:, j] + \
            center_pull * (C[j] - agent_now[:, j]) / euclidian_dist((C - agent_now), axis = 1)
            
    # periodic boundary conditions by calculating delta in bracket and adding it to opposite mean
    agent_plot = np.where(agent_temp < lower_lim, 
                          upper_lim + (agent_temp + upper_lim), 
                          np.where(agent_temp > upper_lim, lower_lim + (agent_temp + lower_lim), agent_temp)
                          )
    # returning an array for plotting a one with accurate positions, such that periodic boundaries do not intetfere with CoM calculations
    return agent_temp, agent_plot, param

def update_predator(agent_now, agent_old, param):
    '''
    Update of the postion of the agents, with the assumption that their acceleretion is computed from
    their pull towards the center of mass and the delta between their new and old position
    Parameters
    ----------
    agent_now : array (n, d)
        DESCRIPTION.
    agent_old : array (n, d)
        DESCRIPTION.
    center_pull : int, optional
        DESCRIPTION. The default is 1.5.

    Returns
    -------
    agent_temp : updated positon of agents

    '''
    d = param["d"]
    center_pull = param["center_pull"]
    lower_lim, upper_lim = param["ax_lim"]
    predator_push = param["predator_push"]
    predator_pull = param["predator_pull"]
    
    agent_temp = np.zeros_like(agent_now)
    # calculate center of mass
    C = np.mean(agent_now, axis=0)

    # update the agent position according to acceleration to center
    for j in range(d):
        agent_temp[:, j] = 2 * agent_now[:, j] - agent_old[:, j] + \
            center_pull * (C[j] - agent_now[:, j]) / euclidian_dist((C - agent_now), axis = 1) \
            + predator_push * (param["predator_xy"][j] - agent_now[:, j]) / euclidian_dist((param["predator_xy"] - agent_now), axis = 1)
            
        param["predator_xy"][j] =  predator_pull * (param["predator_xy"][j] - C[j]) / euclidian_dist((param["predator_xy"] - C), axis = 0)
            
    # periodic boundary conditions by calculating delta in bracket and adding it to opposite mean
    agent_plot = np.where(agent_temp < lower_lim, 
                          upper_lim + (agent_temp + upper_lim), 
                          np.where(agent_temp > upper_lim, lower_lim + (agent_temp + lower_lim), agent_temp)
                          )
    # returning an array for plotting a one with accurate positions, such that periodic boundaries do not intetfere with CoM calculations
    return agent_temp, agent_plot, param
    

def simulate_flocking(mode = "basic",
                      inline_plotting = True,
                      d = 2,
                      param = {"n" : 100,
                               "init_coord":(-1, 1),
                               "ax_lim": (-50, 50),
                               "steps": 500,
                               "center_pull": 1.5, 
                               "pointsize": 2, 
                               "predator_push": 1.5,
                               "predator_pull": 1.5}): 
    '''

    Parameters
    ----------
    n : int , optional
        number of agents. The default is 100.
    init_coord : tuple, optional
        Range of initialization of agents. The default is (-1, 1).
    steps : int, optional
        number of simulation steps. The default is 100.
    center_pull : float, optional
        pull factor towards center of mass. The default is 1.5.

    Returns
    -------
    None.

    '''
    # use an interactive backend
    matplotlib.use('Qt5Agg') # or 'Qt5Agg' or 'WXAgg'
    
    steps = param["steps"]
    n = param["n"]
    param["d"] = d
    
    
    
    
    if mode == "basic": 
        inline_plot_2D = inline_plot_2D_basic
        inline_plot_3D = inline_plot_3D_basic
        initialize_func = initialize_random
        update_func = update
    elif mode == "predator": 
        inline_plot_2D = inline_plot_2D_predator
        # inline_plot_3D = inline_plot_3D_basic
        initialize_func = initialize_predator
        update_func = update_predator
        
    agent_old, agent_now, fig, ax, param = initialize_func(param)
        
    # inline plotting to explore
    if inline_plotting:
        if d == 2: 
            inline_plotting_func = inline_plot_2D
        elif d == 3: 
            inline_plotting_func = inline_plot_3D
        else:
            raise ValueError("Please simulate in 2 or 3 dimensions to plot inline!")
        
        # simulate
        for i in range(steps):
        
            # print(param)
            agent_temp, agent_plot, param = update_func(agent_now, agent_old, param)
            # store updated and this position for next acceleration
            agent_old = agent_now.copy()
            agent_now = agent_temp.copy()
            
            # plot
            inline_plotting_func(agent_plot, ax, param)
        
        # close the plotting window
        plt.close()
    
    # return animation for saving
    else:
        plt.close()
        # storage
        positions = np.zeros((steps+1, n, d))
        positions[0, :, :] = agent_now.copy()
        
        for i in range(steps):
        
        
            agent_temp, agent_plot = update_func(agent_now, agent_old, param)
            # store updated and this position for next acceleration
            agent_old = agent_now.copy()
            agent_now = agent_temp.copy()
            
            # insert in storage
            positions[i+1, :, :] = agent_plot.copy()
            
        return positions
    
def animate_simulations(simulation_list, titles, filename, ax_lims = 50, pointsize = 2, directory = "animations"):
    # use non-GUI backend
    matplotlib.use('Agg')
    
    if not all(len(x) == len(simulation_list[0]) for x in simulation_list):
        raise RuntimeError("All simulation must have been simulated with same number of steps")
        
    
    # check if directory exists
    if not os.path.exists(directory):
        # create directory if it does not exist
        os.makedirs(directory)
        
    
    # extraxt dimension    
    d = simulation_list[0].shape[2]
    
    if d == 2:

        # Create a list of plot objects
        lines = []
        ncol = len(simulation_list)
        fig, axs = plt.subplots(nrows=1, ncols=ncol, figsize = (ncol * 6,6))
    
        # distinguish between single and multi plotting
        if ncol == 1:
            # create PathCollection
            line = axs.scatter([], [], s = pointsize)
            axs.set_title = titles[0]
            axs.set_xlim([-ax_lims, ax_lims])
            axs.set_ylim([-ax_lims, ax_lims])
    
            def update(frame):
                # reformat data
                xy = np.column_stack(tup=(simulation_list[0][frame, :, 0], simulation_list[0][frame, :, 1]))
                line.set_offsets(xy)
                return (line,)
    
        else:
            # for row in axs
            for i, ax in enumerate(axs):
                line = ax.scatter([], [], s = pointsize)
                ax.set_title(titles[i])
                ax.set_xlim([-ax_lims, ax_lims])
                ax.set_ylim([-ax_lims, ax_lims])
                lines.append(line)
    
            # define the animation function
            def update(frame):
                for i, line in enumerate(lines):
                    # reformat data
                    xy = np.column_stack(tup=(simulation_list[i][frame, :, 0], simulation_list[i][frame, :, 1]))
                    line.set_offsets(xy)
                return tuple(lines)
    
        anim = FuncAnimation(
            fig=fig,
            func=update,
            frames=len(simulation_list[0]),
            interval=50,
            blit=True,
        )
    
        anim.save(directory + "/" + filename + ".mp4")
        
    elif d == 3:

        # create a list of plot objects
        lines = []
        ncol = len(simulation_list)
        fig = plt.figure(figsize=(ncol*6, 6))
        axs = [fig.add_subplot(1, ncol, i+1, projection='3d') for i in range(ncol)]
    
        if ncol == 1:
            line = axs.scatter([], [], [], s = pointsize)
            axs.set_title(titles[0])
            axs.set_xlim3d([-ax_lims, ax_lims])
            axs.set_ylim3d([-ax_lims, ax_lims])
            axs.set_zlim3d([-ax_lims, ax_lims])
    
            def update(frame):
                xyz = simulation_list[0][frame]
                line._offsets3d = (xyz[:,0], xyz[:,1], xyz[:,2])
                return (line,)
    
        else:
            # for row in axs
            for i, ax in enumerate(axs):
                line = ax.scatter([], [], [], s = pointsize)
                ax.set_title(titles[i])
                ax.set_xlim3d([-ax_lims, ax_lims])
                ax.set_ylim3d([-ax_lims, ax_lims])
                ax.set_zlim3d([-ax_lims, ax_lims])
                lines.append(line)
    
            # define the animation function
            def update(frame):
                for i, line in enumerate(lines):
                    xyz = simulation_list[i][frame]
                    line._offsets3d = (xyz[:,0], xyz[:,1], xyz[:,2])
                return tuple(lines)
    
        anim = FuncAnimation(
            fig=fig,
            func=update,
            frames=len(simulation_list[0]),
            interval=50,
            blit=True,
        )
    
        anim.save(directory + "/" + filename + ".mp4")


        

        

        
if __name__ == "__main__":
    simulate_flocking(d=2, 
                      mode = "predator")
    # simulate in 2D
    # simulate_flocking(d = 2)
    # simulate in 3D
    #simulate_flocking(d = 3)
    # store a simulation
    # pos = simulate_flocking(d = 3, inline_plotting = False)
    # animate_simulations([pos, pos], ["p1", "p2"], "example")



