import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
matplotlib.use('Agg')


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

    return agent_old, agent_now, fig, ax

def inline_plot_2D(agent_now, ax, param):
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
    ax.scatter(agent_now[:, 0], agent_now[:, 1])
    ax.set_xlim(lower_lim, upper_lim)
    ax.set_ylim(lower_lim, upper_lim)
    plt.pause(0.01)
    
    return ax
    

def inline_plot_3D(agent_now, ax, param):
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
    ax.scatter(agent_now[:, 0], agent_now[:, 1], agent_now[:, 2])
    ax.set_xlim(lower_lim, upper_lim)
    ax.set_ylim(lower_lim, upper_lim)
    ax.set_zlim(lower_lim, upper_lim)
    plt.pause(0.01)
    
    return ax

def euclidian_dist(array):

    return np.sqrt(np.sum(array**2, axis = 1))

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
    
    agent_temp = np.zeros_like(agent_now)
    # calculate center of mass
    C = np.mean(agent_now, axis=0)

    # update the agent position according to acceleration to center
    for j in range(d):
        agent_temp[:, j] = 2 * agent_now[:, j] - agent_old[:, j] + \
            center_pull * (C[j] - agent_now[:, j]) / euclidian_dist((C - agent_now))
    
    return agent_temp
    

def simulate_flocking(initialize_func = initialize_random,
                      update_func = update,
                      inline_plotting = True,
                      d = 2,
                      param = {"n" : 100,
                               "init_coord":(-1, 1),
                               "ax_lim": (-50, 50),
                               "steps": 100,
                               "center_pull": 1.5}): 
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
    steps = param["steps"]
    n = param["n"]
    param["d"] = d
    
    agent_old, agent_now, fig, ax = initialize_func(param)
    
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
        
        
            agent_temp = update_func(agent_now, agent_old, param)
            # store updated and this position for next acceleration
            agent_old = agent_now.copy()
            agent_now = agent_temp.copy()
            
            # plot
            inline_plotting_func(agent_now, ax, param)
        
        # close the plotting window
        plt.close()
    
    # return animation for saving
    else:
        plt.close()
        # storage
        positions = np.zeros((steps+1, n, d))
        positions[0, :, :] = agent_now.copy()
        
        for i in range(steps):
        
        
            agent_temp = update_func(agent_now, agent_old, param)
            # store updated and this position for next acceleration
            agent_old = agent_now.copy()
            agent_now = agent_temp.copy()
            
            # insert in storage
            positions[i+1, :, :] = agent_now.copy()
            
        return positions
    
def animate_simulations(simulation_list, titles, filename, ax_lims = 50, directory = "animations"):
    
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
            line = axs.scatter([], [])
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
                line = ax.scatter([], [])
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
            line = axs.scatter([], [], [])
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
                line = ax.scatter([], [], [])
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
    # simulate in 2D
    #simulate_flocking(d = 2)
    # simulate in 3D
    #simulate_flocking(d = 3)
    # store a simulation
    pos = simulate_flocking(d = 3, inline_plotting = False)
    animate_simulations([pos, pos], ["p1", "p2"], "example")



