import numpy as np

def loglike_stat(experiment, predicted, observed=None) -> float:
    if observed is None:
        observed = experiment.data_hist[1]

    with np.errstate(all="ignore"):
        logterm = np.where(observed > 0, observed * np.log(observed / predicted), 0) # replace with 0 when observed bins are 0 (see pdg)

    with np.errstate(all='raise'):
        try:
            return np.sum(-predicted + observed - logterm)
        except FloatingPointError:
            return -np.inf

def loglike_sys(nuisance_params: dict, nuisance_param_priors: dict) -> float:
    loglike = 0
    for k in nuisance_param_priors.keys():
        if hasattr(nuisance_param_priors[k], "__len__"):
            continue # skip the nuisance parameters with uniform priors
        loglike += -0.5*(nuisance_params[k]/nuisance_param_priors[k])**2
    return loglike