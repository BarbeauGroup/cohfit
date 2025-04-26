import json

import numpy as np

from utils.histograms import centers_to_edges


def make_csi_brn_histogram(config_file: str):
    with open(config_file, 'r') as f:
        if config_file.endswith(".json"):
            params = json.load(f)
        else:
            raise ValueError("Only JSON config files are supported right now")

    # BRNs
    brn_pe = np.loadtxt(params["beam"]["brn_energy_file"])
    brn_t = np.loadtxt(params["beam"]["brn_time_file"])

    brn_pe_bins_centers = brn_pe[:60, 0]
    brn_pe_bins_edges = centers_to_edges(brn_pe_bins_centers)
    brn_pe_counts = brn_pe[:60, 1]
    brn_pe_counts /= np.sum(brn_pe_counts)

    brn_t_bins_centers = brn_t[:, 0]
    brn_t_bins_edges = centers_to_edges(brn_t_bins_centers)
    brn_t_counts = brn_t[:, 1]
    brn_t_counts /= np.sum(brn_t_counts)

    hist = params["detector"]["norms"]["brn"] * np.outer(brn_pe_counts, brn_t_counts)

    return (brn_pe_bins_edges, brn_t_bins_edges), hist
