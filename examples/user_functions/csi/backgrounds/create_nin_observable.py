from utils.histograms import rebin_histogram2d

# TODO: add cache
def create_nin_observables(experiment, nuisance_params):
    """
    This takes in an experiment (which has a nin_hist set already) and nuisance parameters 
    that may modify the brn histogram non-trvially (that is not just a scaling factor).
    It modifies the brn_hist and returns it.

    Parameters
    ----------
    experiment : Experiment
        The experiment object to use for the calculation
    nuisance_params : dict
        The nuisance parameters to use for the calculation

    Returns
    -------
    2d histogram
    """

    nuoff = nuisance_params.get("nu_time_offset", 0.0)
    offset = nuoff #+ nuisance_params.get("nin_time_offset_csi", 0.0) # TODO: make experiment independent

    counts = rebin_histogram2d(
        experiment.base_hists["nin"][1],
        experiment.base_hists["nin"][0][0],
        experiment.base_hists["nin"][0][1] + offset,
        experiment.observable_energy_bins,
        experiment.observable_time_bins,
    )

    return (experiment.base_hists["nin"][0][0], experiment.base_hists["nin"][0][1]), counts