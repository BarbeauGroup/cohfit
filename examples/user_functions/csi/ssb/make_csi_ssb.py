import json

import numpy as np

from utils.histograms import centers_to_edges, rebin_histogram

def make_csi_ssb(config_file: str):
    """
    params: dictionary with SSB information
    """
    with open(config_file, 'r') as f:
        if config_file.endswith(".json"):
            params = json.load(f)
        else:
            raise ValueError("Only JSON config files are supported right now")

    # Energy PDF is just the AC data
    energy_bins = np.asarray(params["analysis"]["energy_bins"])
    time_bins = np.asarray(params["analysis"]["time_bins"])

    dataBeamOnAC = np.loadtxt(params["detector"]["beam_ac_data_file"])
    AC_PE = dataBeamOnAC[:,0]
    AC_t = dataBeamOnAC[:,1]

    # Only filter the energy pdf because time is constructed
    # must filter the events that have t > time roi (e.g. 6)
    AC_high_time_idx = np.where(AC_t > params["analysis"]["time_roi"][1])[0]
    AC_PE_cut = np.delete(AC_PE, AC_high_time_idx)

    # and energy > energy roi (e.g. 60)
    AC_high_energy_idx = np.where(AC_PE_cut > params["analysis"]["energy_roi"][1])[0]
    AC_PE_cut = np.delete(AC_PE_cut, AC_high_energy_idx)

    AC_PE_hist, _ = np.histogram(AC_PE_cut, bins=energy_bins)
    AC_PE_hist = np.divide(AC_PE_hist, np.sum(AC_PE_hist))

    # Time PDF is an analytic function
    k = 0.0494 / 1000.
    def exp_decay(t, k):
        return k * np.exp(-k*t)

    t = np.linspace(0, 6000, 100)
    y = exp_decay(t, k)
    y /= np.sum(y)
    t_edges = centers_to_edges(t)
    time_values = rebin_histogram(y, t_edges, time_bins)
    time_values /= np.sum(time_values)

    total_histogram = params["detector"]["norms"]["ssb"] * np.outer(AC_PE_hist, time_values)

    return (energy_bins, time_bins), total_histogram



