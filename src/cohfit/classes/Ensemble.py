import pyjson5 as json

import numpy as np

from ..stats.likelihood import loglike_stat, loglike_sys
from ..utils.load_flux import read_flux_from_root


class Ensemble:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            if config_file.endswith(".json"):
                self.params = json.load(f)
            else:
                raise ValueError("Only JSON config files are supported right now")
        
        time_edges = np.arange(*self.params["time_edges"])

        self.flux = read_flux_from_root(self.params, time_edges)

        self.experiments = []
        self.nuisance_params = []
    
    def add_experiment(self, experiment):
        self.experiments.append(experiment)

    def set_nuisance_params(self, nuisance_params):
        self.nuisance_params = nuisance_params
    
    def analysis_hists(self, x, mass=None, ue4=None, umu4=None, set_data_hist=False, plot_hists=False):
        fit_params = dict(zip(self.nuisance_params, x))
        if mass is None:
            mass = fit_params["mass"]
        if ue4 is None:
            ue4 = fit_params["ue4"]
        if umu4 is None:
            umu4 = fit_params["umu4"]

        ret_arr = []
        for experiment in self.experiments:
            ret_arr.append(experiment.calculate_predicted(self.flux, mass, ue4, umu4, fit_params, set_data_hist=set_data_hist, plot_hists=plot_hists))
        return ret_arr

    # TODO :
    # If there's no time offset, pull create observables out
    def __call__(self, x, mass=None, ue4=None, umu4=None, observed_hists=None, include_sys=True):
        # Extract the parameters from x
        if mass is None:
            mass = x[0]
            x = x[1:]
        if ue4 is None:
            ue4 = x[0]
            x = x[1:]
        if umu4 is None:
            umu4 = x[0]
            x = x[1:]

        # Make the model
        predicted_hists = self.analysis_hists(x, mass, ue4, umu4)

        # Need to do this for zip to work
        if observed_hists is None:
            observed_hists = [None] * len(self.experiments)

        fit_param_priors = {}
        fit_params = dict(zip(self.nuisance_params, x))
        
        ll_stat = 0
        for experiment, predicted, observed in zip(self.experiments, predicted_hists, observed_hists):
            for k in experiment.params["detector"]["systematics"].keys():
                if k not in self.nuisance_params:
                    continue # skip unused nuisance parameters
                v = fit_param_priors.get(k)
                if v is not None and v != experiment.params["detector"]["systematics"][k]:
                    raise ValueError("Mismatch in priors for shared nuisance parameters")
                fit_param_priors[k] = experiment.params["detector"]["systematics"][k]

            else:
                ll_stat += loglike_stat(experiment, predicted, observed)

        if include_sys:
            ll_sys = loglike_sys(fit_params, fit_param_priors)
        else:
            # ll_sys = 0.0
            ll_sys = loglike_sys(fit_params, fit_param_priors)

        return -2 * (ll_stat + ll_sys)