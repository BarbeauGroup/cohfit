import json

import numpy as np

from utils.histograms import centers_to_edges


def make_csi_nin_histogram(config_file: str):
    with open(config_file, 'r') as f:
        if config_file.endswith(".json"):
            params = json.load(f)
        else:
            raise ValueError("Only JSON config files are supported right now")

    # NIN
    nin_pe = np.loadtxt(params["beam"]["nin_energy_file"])
    nin_t = np.loadtxt(params["beam"]["nin_time_file"])

    nin_pe_bins_centers = nin_pe[:60, 0]
    nin_pe_bins_edges = centers_to_edges(nin_pe_bins_centers)
    nin_pe_counts = nin_pe[:60, 1]
    nin_pe_counts /= np.sum(nin_pe_counts)

    nin_t_bins_centers = nin_t[:, 0]
    nin_t_bins_edges = centers_to_edges(nin_t_bins_centers)
    nin_t_counts = nin_t[:, 1]
    nin_t_counts /= np.sum(nin_t_counts)

    hist = params["detector"]["norms"]["nin"] * np.outer(nin_pe_counts, nin_t_counts)

    return (nin_pe_bins_edges, nin_t_bins_edges), hist
