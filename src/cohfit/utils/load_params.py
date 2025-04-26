# L, target mass, atomic numbers of isotopes, nus/POT, POT
# paths to flux, detector and time smearing matrices
# filename to flux root file

import json
import numpy as np

def load_params(filename) -> dict:
    """
    Load the parameters from a json file.

    Parameters
    ----------

    filename : str
        The name of the json file containing the parameters.
    
    Returns
    -------
    dict
        The parameters.
    """

    with open(filename, 'r') as f:
        params = json.load(f)

    return params