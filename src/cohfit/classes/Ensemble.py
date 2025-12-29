import pyjson5 as json

import numpy as np

from ..stats.likelihood import loglike_stat, loglike_sys
from ..utils.load_flux import read_flux_from_root


class Ensemble:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            if config_file.endswith(".json") or config_file.endswith(".json5"):
                self.params = json.load(f)
            else:
                raise ValueError("Only JSON config files are supported right now")
        
        time_edges = np.arange(*self.params["time_edges"])

        self.flux = read_flux_from_root(self.params, time_edges)

        self.experiments = []
        self.nuisance_params = []
    
    def add_experiment(self, experiment):
        self.experiments.append(experiment)

    def set_model_params(self, model_params):
        self.model_params = model_params

    def set_nuisance_params(self, nuisance_params):
        self.nuisance_params = nuisance_params
    
    def analysis_hists(self, x, model_params, set_data_hist=False, plot_hists=False):
        nuisance_params = dict(zip(self.nuisance_params, x))

        ret_arr = []
        for experiment in self.experiments:
            ret_arr.append(experiment.calculate_predicted(self.flux, model_params, nuisance_params, set_data_hist=set_data_hist, plot_hists=plot_hists))
        return ret_arr

    # TODO :
    # If there's no time offset, pull create observables out
    def __call__(self, x, observed_hists=None, include_sys=True):

        num_model_params = len(self.model_params)
        model_params = dict(zip(self.model_params, x[:num_model_params]))

        nuisance_x = x[num_model_params:]
        nuisance_param_priors = {}
        nuisance_params = dict(zip(self.nuisance_params, nuisance_x))

        # Make the model
        predicted_hists = self.analysis_hists(nuisance_x, model_params)

        for i, pred in enumerate(predicted_hists):
            if not np.all(np.isfinite(pred)):
                print("NON-FINITE prediction histogram")
                print("  experiment index:", i)
                print("  model params:", model_params)
                print("  nuisance params:", nuisance_params)
                print("  min(pred):", np.nanmin(pred))
                raise RuntimeError("Invalid prediction")

            if np.any(pred <= 0):
                idx = np.where(pred <= 0)[0][:5]
                print("NON-POSITIVE bins in prediction")
                print("  experiment index:", i)
                print("  bins:", idx)
                print("  values:", pred[idx])
                print("  model params:", model_params)
                print("  nuisance params:", nuisance_params)
                raise RuntimeError("Invalid prediction")


        # Need to do this for zip to work
        if observed_hists is None:
            observed_hists = [None] * len(self.experiments)
        
        ll_stat = 0
        for experiment, predicted, observed in zip(self.experiments, predicted_hists, observed_hists):
            for k in experiment.params["detector"]["systematics"].keys():
                if k not in self.nuisance_params:
                    continue # skip unused nuisance parameters
                v = nuisance_param_priors.get(k)
                if v is not None and v != experiment.params["detector"]["systematics"][k]:
                    raise ValueError("Mismatch in priors for shared nuisance parameters")
                nuisance_param_priors[k] = experiment.params["detector"]["systematics"][k]

            else:
                ll_stat += loglike_stat(experiment, predicted, observed)

        if include_sys:
            ll_sys = loglike_sys(nuisance_params, nuisance_param_priors)
        else:
            # ll_sys = 0.0
            ll_sys = loglike_sys(nuisance_params, nuisance_param_priors)

        return -2 * (ll_stat + ll_sys)