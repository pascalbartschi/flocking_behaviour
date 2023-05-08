import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def initialize_random(param, pred=False):
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
    double_agent: np array, initial position of predator
    fig : figure object
        Figure to plot in.
    ax : artist object
        Artist axis to plot in.

    '''
    n = param["n"]
    d = param["d"]
    low, high = param["init_coord"]
    lower_lim, upper_lim = param["ax_lim"]
    lower_lim, upper_lim = 2*lower_lim, 2*upper_lim

    # initialize n agents in d dimensions
    
    np.random.seed(18)
    agent_now = np.random.uniform(low=low, high=high, size=(n, d))
    agent_old = np.zeros_like(agent_now)
    agent_temp = np.zeros_like(agent_now)
    if pred:
        double_agent_now = np.random.uniform(low=low-10,high=high-10, size=(1,d))  #start predator off in lower corner
        double_agent_old = np.random.uniform(low=low-10,high=high-10, size=(1,d))  #start predator off in lower corner
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
    if pred:
        return agent_old, agent_now, fig, ax, double_agent_now, double_agent_old
    return agent_old, agent_now, fig, ax

def inline_plot_2D(agent_now, ax, param,double_agent_now='not here'):
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
    if (type(double_agent_now)!=str):
        ax.scatter(double_agent_now[0,0],double_agent_now[0,1],marker='D',s=100)
    ax.set_xlim(lower_lim, upper_lim)
    ax.set_ylim(lower_lim, upper_lim)
    plt.pause(0.01)
    
    return ax
    

def inline_plot_3D(agent_now, ax, param,double_agent_now='not here'):
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
    if (type(double_agent_now)!=str):
        ax.scatter(double_agent_now[0,0],double_agent_now[0,1],double_agent_now[0, 2],marker='D',s=50)
    ax.scatter(agent_now[:, 0], agent_now[:, 1], agent_now[:, 2])
    ax.set_xlim(lower_lim, upper_lim)
    ax.set_ylim(lower_lim, upper_lim)
    ax.set_zlim(lower_lim, upper_lim)
    plt.pause(0.01)
    
    return ax

def euclidian_dist(array):

    return np.sqrt(np.sum(array**2, axis = 1))

def update(agent_now, agent_old, param,double_agent_now='not here', double_agent_old=None):
    '''
    Update of the postion of the agents, with the assumption that their acceleretion is computed from
    their pull towards the center of mass and the delta between their new and old position
    Parameters
    ----------
    agent_now : array (n, d)
        DESCRIPTION.
    agent_old : array (n, d)
        DESCRIPTION.
    double_agent_old : array (1,d) (or None)
        DESCRIPTION.
    double_agent_now : array (1,d) (or string not here)
        DESCRIPTION.
    center_pull : int, optional
        DESCRIPTION. The default is 1.5.
    predator_pull : int, optional
        Description. The default is 1.5
    predator_push : int, optional
        Description. The default is -1.5
    Returns
    -------
    agent_temp : updated positon of agents
    optional :  double_agent_now, updated position of double agent
    '''
    d = param["d"]
    center_pull = param["center_pull"]

    agent_temp = np.zeros_like(agent_now)
    double_agent_temp = np.zeros_like(double_agent_now)
    # calculate center of mass
    C = np.mean(agent_now, axis=0)
    if (type(double_agent_now) != str):
        predator_pull = param["predator_pull"]
        predator_push = param["predator_push"]
        double_agent_temp = np.zeros_like(double_agent_now)
        n=len(agent_old)
        predator_position = np.tile(double_agent_old, (n, 1)) #old because assume the bird's reaction times are delayed

    # update the agent position according to acceleration to center
    for j in range(d):
        if (type(double_agent_now) == str):
            agent_temp[:, j] = 2 * agent_now[:, j] - agent_old[:, j] + \
            center_pull * (C[j] - agent_now[:, j]) / euclidian_dist((C - agent_now))
        else:
            agent_temp[:,j] = 2 * agent_now[:, j] - agent_old[:, j] + \
            center_pull * (C[j] - agent_now[:, j]) / euclidian_dist((C - agent_now)) +\
            predator_push * (predator_position[:,j]-agent_now[:,j]) / euclidian_dist((double_agent_old-agent_now))**3
            double_agent_temp[0,j] = 2 * double_agent_now[0,j] - double_agent_old[0,j] + \
            predator_pull * (C[j] - double_agent_now[0,j]) / euclidian_dist((C - double_agent_now))    #note that C is calculated from last iteration, showing reaction time of predator also not 0
    if (type(double_agent_now) != str):
        return agent_temp, double_agent_temp
    return agent_temp
    

def simulate_flocking(initialize_func = initialize_random,
                      update_func = update,
                      inline_plotting = True,
                      d = 2,
                      param = {"n" : 100,
                               "init_coord":(-1, 1),
                               "ax_lim": (-50, 50),
                               "steps": 100,
                               "center_pull": 1.5,"predator_pull": 1.5,"predator_push": -0.5}, pred=False):
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
    predator_pull : int, optional
        pull factor for predator towards center of mass of birds. The default is 1.5
    predator_push : int, optional
        push factor for predator towards center of mass of birds. The default is -1.5

    Returns
    -------
    None.

    '''
    steps = param["steps"]
    n = param["n"]
    param["d"] = d


    if pred:
        agent_old, agent_now, fig, ax, double_agent_now, double_agent_old= initialize_func(param, pred=True)
    else:
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
        
            if pred:
                agent_temp, double_agent_temp = update_func(agent_now,agent_old,param,double_agent_now,double_agent_old)
                double_agent_old = double_agent_now.copy()
                double_agent_now = double_agent_temp.copy()
                agent_old = agent_now.copy()
                agent_now = agent_temp.copy()
                inline_plotting_func(agent_now, ax, param, double_agent_now)
            else:
                agent_temp = update_func(agent_now, agent_old, param)
                agent_old = agent_now.copy()
                agent_now = agent_temp.copy()
                # store updated and this position for next acceleration
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
            
        

        

        
if __name__ == "__main__":
    # simulate in 2D
    simulate_flocking(d = 2, pred=True,param = {"n" : 100,
                               "init_coord":(-1, 1),
                               "ax_lim": (-100, 100),
                               "steps": 100,
                               "center_pull": 1,"predator_pull": 1.5,"predator_push": -0.6})
    # simulate in 3D
    simulate_flocking(d = 3,pred=True,param = {"n" : 100,
                               "init_coord":(-1, 1),
                               "ax_lim": (-100, 100),
                               "steps": 100,
                               "center_pull": 1,"predator_pull": 1.5,"predator_push": -1})
    # store a simulation
    pos = simulate_flocking(d = 2, inline_plotting = False)



