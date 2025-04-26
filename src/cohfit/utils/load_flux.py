import numpy as np
import uproot

from utils.histograms import rebin_histogram

def read_flux_from_root(params: dict, new_t_edges=None) -> dict:
    """
    
    params: dictionary with flux_file key and other info

    returns: dictionary with SNS flux information
    
    """

    filename = params["flux_file"]
    paper_rf = uproot.open(filename)

    convolved_energy_and_time_of_nu_mu = paper_rf["convolved_energy_time_of_nu_mu;1"]
    convolved_energy_and_time_of_nu_mu_bar = paper_rf["convolved_energy_time_of_anti_nu_mu;1"]
    convolved_energy_and_time_of_nu_e = paper_rf["convolved_energy_time_of_nu_e;1"]
    convolved_energy_and_time_of_nu_e_bar = paper_rf["convolved_energy_time_of_anti_nu_e;1"]

    # TODO: put in config file
    # new_t_edges = np.arange(0, 15125, 125)
    
    # nu e
    NuE = convolved_energy_and_time_of_nu_e.values()[:, 1:60]
    NuE /= np.sum(NuE)
    NuE *= params["nus_per_pot"]["nuE"]
    NuE_e_edges = convolved_energy_and_time_of_nu_e.axis(1).edges()[1:60]
    NuE_t_edges = convolved_energy_and_time_of_nu_e.axis(0).edges()

    # nu e bar
    NuEBar = convolved_energy_and_time_of_nu_e_bar.values()[:, 1:60]
    NuEBar /= np.sum(NuEBar)
    NuEBar *= params["nus_per_pot"]["nuEBar"]
    NuEBar_e_edges = convolved_energy_and_time_of_nu_e_bar.axis(1).edges()[1:60]
    NuEBar_t_edges = convolved_energy_and_time_of_nu_e_bar.axis(0).edges()

    # nu mu
    NuMu = convolved_energy_and_time_of_nu_mu.values()[:, 1:60]
    NuMu /= np.sum(NuMu)
    NuMu *= params["nus_per_pot"]["nuMu"]
    NuMu_e_edges = convolved_energy_and_time_of_nu_mu.axis(1).edges()[1:60]
    NuMu_t_edges = convolved_energy_and_time_of_nu_mu.axis(0).edges()

    # nu mu bar
    NuMuBar = convolved_energy_and_time_of_nu_mu_bar.values()[:, 1:60]
    NuMuBar /= np.sum(NuMuBar)
    NuMuBar *= params["nus_per_pot"]["nuMuBar"]
    NuMuBar_e_edges = convolved_energy_and_time_of_nu_mu_bar.axis(1).edges()[1:60]
    NuMuBar_t_edges = convolved_energy_and_time_of_nu_mu_bar.axis(0).edges()

    if new_t_edges is not None:
        new_hist_arr = [np.zeros((len(new_t_edges)-1, hist.shape[1])) for hist in [NuE, NuEBar, NuMu, NuMuBar]]
        for i, hist in enumerate([NuE, NuEBar, NuMu, NuMuBar]):
            for col in range(hist.shape[1]):
                new_hist_arr[i][:, col] = rebin_histogram(hist[:, col], NuE_t_edges, new_t_edges)
    else:
        new_hist_arr = [NuE, NuEBar, NuMu, NuMuBar]
        new_t_edges = NuE_t_edges

    # Make a tuple of the flux information for each neutrino type
    return {
        'nuE': ((new_t_edges, NuE_e_edges), new_hist_arr[0]),
        'nuEBar': ((new_t_edges, NuEBar_e_edges), new_hist_arr[1]),
        'nuMu': ((new_t_edges, NuMu_e_edges), new_hist_arr[2]),
        'nuMuBar': ((new_t_edges, NuMuBar_e_edges), new_hist_arr[3]),
        'nuTau': ((new_t_edges, NuMu_e_edges), np.zeros_like(new_hist_arr[0])),
        'nuTauBar': ((new_t_edges, NuMu_e_edges), np.zeros_like(new_hist_arr[0])),
        'nuS': ((new_t_edges, NuMu_e_edges), np.zeros_like(new_hist_arr[0])),
        'nuSBar': ((new_t_edges, NuMu_e_edges), np.zeros_like(new_hist_arr[0]))
    }