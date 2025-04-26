import numpy as np
import json

def make_flat_ssb(config_file):
    with open(config_file, 'r') as f:
        params = json.load(f)
    
    observable_bin_arr = np.asarray(params["analysis"]["energy_bins"])
    t_bin_arr = np.asarray(params["analysis"]["time_bins"])

    ones = np.ones((len(observable_bin_arr) - 1, len(t_bin_arr) - 1))
    hist = ones * params["detector"]["norms"]["ssb"] / np.sum(ones)

    return ((observable_bin_arr, t_bin_arr), hist)