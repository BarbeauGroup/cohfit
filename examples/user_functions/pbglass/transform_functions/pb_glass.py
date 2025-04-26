import numpy as np

def energy_efficiency(x):
    """
    Calculate the Pb-glass detector efficiency.

    Parameters
    ----------
    x : float
        Energy in MeV.
    
    Returns
    -------
    float
        The efficiency.
    """
    return np.ones_like(x)

def time_efficiency(t):
    """
    Calculate the efficiency of the Pb-glass detector as a function of time.

    Parameters
    ----------
    t: float
        The time at which the efficiency is calculated in ns.
    
    Returns
    -------
    float
        The efficiency.
    """


    return np.ones_like(t)