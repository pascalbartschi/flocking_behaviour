import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


def initialize(n = 100, d = 2, init_coord = (-1, 1)):
    '''
    This function initializes the agents, as well as the plotting window

    Parameters
    ----------
    n : int, optional
        Number of Agents. The default is 100.
    d : int, optional
        Dimensions of Movement. The default is 2.
    init_coord : tuple, optional
        Range where Agents are intialized. The default is (-1, 1).

    Returns
    -------
    agent_old, agents_now, , agent_temp, figure, ax

    '''

    # initialize n agents in d dimensions
    low, high = init_coord
    
    agent_now = np.random.uniform(low=low, high=high, size=(n, d))
    agent_old = np.zeros_like(agent_now)
    agent_temp = np.zeros_like(agent_now)
    
    if d == 2: 
        # some fig to plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
    elif d == 3: 
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection = "3d")
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(-50, 50)
    else:
        raise ValueError("Incorrect number of dimensions. Choose d=2 or d=3.")

    return agent_old, agent_now, fig, ax

def plot_2D(agent_now, ax):
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
    # plot 2D
    ax.clear()
    ax.scatter(agent_now[:, 0], agent_now[:, 1])
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    plt.pause(0.01)
    
    return ax
    

def plot_3D(agent_now, ax):
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
    ax.clear()
    ax.scatter(agent_now[:, 0], agent_now[:, 1], agent_now[:, 2])
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-50, 50)
    plt.pause(0.01)
    
    return ax

def euclidian_dist(array):

    return np.sqrt(np.sum(array**2, axis = 1))

def update(agent_now, agent_old, d = 3, cpull_f = 1.5):
    '''
    Update of the postion of the agents, with the assumption that their acceleretion is computed from
    their pull towards the center of mass and the delta between their new and old position
    Parameters
    ----------
    agent_now : array (n, d)
        DESCRIPTION.
    agent_old : array (n, d)
        DESCRIPTION.
    cpull_f : int, optional
        DESCRIPTION. The default is 1.5.

    Returns
    -------
    agent_temp : updated positon of agents

    '''
    agent_temp = np.zeros_like(agent_now)
    # calculate center of mass
    C = np.mean(agent_now, axis=0)

    # update the agent position according to acceleration to center
    for j in range(d):
        agent_temp[:, j] = 2 * agent_now[:, j] - agent_old[:, j] + \
            cpull_f * (C[j] - agent_now[:, j]) / euclidian_dist((C - agent_now))
    
    return agent_temp
    

def simulate_flocking_2D(n = 100, init_coord = (-1, 1), steps = 100, cpull_f = 1.5): 
    
    d = 2
    
    agent_old, agent_now, fig, ax = initialize(d = d)
    
    # simulate
    for i in range(steps):
    
    
        agent_temp= update(agent_now, agent_old, d, cpull_f)
        # store updated and this position for next acceleration
        agent_old = agent_now.copy()
        agent_now = agent_temp.copy()
        
        # plot
        plot_2D(agent_now, ax)
        
def simulate_flocking_3D(n = 100, init_coord = (-1, 1), steps = 100, cpull_f = 1.5): 
    
    d = 3
    
    agent_old, agent_now, fig, ax = initialize(d = d)
    
    # simulate
    for i in range(steps):
    
    
        agent_temp= update(agent_now, agent_old, d, cpull_f)
        # store updated and this position for next acceleration
        agent_old = agent_now.copy()
        agent_now = agent_temp.copy()
        
        # plot
        plot_3D(agent_now, ax)
        

simulate_flocking_2D()
simulate_flocking_2D()

