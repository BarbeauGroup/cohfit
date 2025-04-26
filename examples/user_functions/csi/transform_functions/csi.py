import numpy as np

def energy_efficiency(x):
    """
    Calculate the CsI detector efficiency.

    Parameters
    ----------
    x : float
        Energy in PE.
    
    Returns
    -------
    float
        The efficiency.
    """
    eps = (1.32045 / (1 + np.exp(-0.285979*(x - 10.8646)))) - 0.333322
    return np.where(x > 60, 0, np.where(eps < 0, 0, eps))

def time_efficiency(t):
    """
    Calculate the efficiency of the CsI detector as a function of time.

    Parameters
    ----------
    t: float
        The time at which the efficiency is calculated in ns.
    
    Returns
    -------
    float
        The efficiency.
    """
    a = 0.52 * 1000
    b = 0.0494 / 1000

    return np.where(t < 0, 0, np.where(t < a, 1, np.where(t < 6000, np.exp(-b*(t - a)), 0)))