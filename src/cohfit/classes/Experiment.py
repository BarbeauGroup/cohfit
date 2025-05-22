from copy import deepcopy

import numpy as np
import pyjson5 as json

from ..flux.nuflux import oscillate_flux
from ..utils.num_atoms import num_atoms


class Experiment:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            if config_file.endswith(".json") or config_file.endswith(".json5"):
                self.params = json.load(f)
            else:
                raise ValueError("Only JSON config files are supported right now")

        for isotope in self.params["detector"]["isotopes"]:
            isotope["flux_matrix"] = np.load(isotope["flux_matrix"])
            isotope["num_atoms"] = num_atoms(self.params, isotope)
        
        self.params["detector"]["detector_matrix"] = np.load(self.params["detector"]["detector_matrix"])

        # TODO: make matrix loading generic or a function or something
        if "f90_matrix" in self.params["detector"]:
            self.params["detector"]["f90_matrix"] = np.load(self.params["detector"]["f90_matrix"])

        # self.observable_energy_bins = np.asarray(self.params["analysis"]["energy_bins"])
        # self.observable_time_bins = np.asarray(self.params["analysis"]["time_bins"])

        # self.base_hists = {}
        self.data_hist = None
        self.total_predicted_hist = None

        self.plot_hists = {}
        self.unosc_plot_hists = {}

        self.signal_function = None
        self.background_functions = {}

    def set_signal_function(self, signal_function):
        self.signal_function = signal_function

    def set_background_function(self, key, background_function):
        self.background_functions[key] = background_function

    def set_data_hist(self, hist):
        self.data_hist = hist

    def calculate_predicted(self, flux, mass, ue4, umu4, fit_params, set_data_hist=False, plot_hists=False):
        ll_hists = {}

        # Modify the neutrino spectrum before detector effects
        osc_params = [self.params["detector"]["distance"], mass, ue4, umu4, 0.0]
        osc_flux = oscillate_flux(flux=flux, oscillation_params=osc_params) # TODO: make it take fit_params too

        # Create the signal
        # nu_obs = create_neutrino_observables(flux=osc_flux, experiment=self, nuisance_params=fit_params, flavorblind=True, detector_effects=True)
        nu_obs = self.signal_function(flux=osc_flux, params=self.params, nuisance_params=fit_params, flavorblind=True)
        ll_hists["signal"] = nu_obs["combined"]

        for k in self.background_functions.keys():
            ll_hists[k] = self.background_functions[k](fit_params)

        ## Calculate the predicted histogram
        flux_nuisance = 1
        for k in fit_params.keys():
            if k.startswith("flux"):
                flux_nuisance += fit_params[k]

        predicted = 0
        predicted += ll_hists["signal"] * flux_nuisance

        for k in ll_hists.keys():
            if k == "signal":
                continue
            predicted += ll_hists[k] * (
                1 + fit_params.get("{}_{}".format(k, self.params['name']), 0)
            )

        # Calculate the plotting histograms if required # TODO: pull out into separate function
        if plot_hists:
            self.plot_hists = deepcopy(ll_hists)
            self.unosc_plot_hists = deepcopy(ll_hists)

            # If we want to create the plot hists, we create an oscillated and unoscillated neutrino flavor spectrum
            nu_obs = self.signal_function(flux=osc_flux, params=self.params, nuisance_params=fit_params, flavorblind=False)
            unosc_nu_obs = self.signal_function(flux=flux, params=self.params, nuisance_params=fit_params, flavorblind=False)
            self.plot_hists["signal"] = nu_obs
            self.unosc_plot_hists["signal"] = unosc_nu_obs

            for k in ll_hists.keys():
                if k == "signal":
                    # both oscillated and unoscillated histograms get multiplied by the flux nuisance
                    for nu in self.plot_hists["signal"].keys():
                        if nu not in self.params["detector"]["observable_flavors"]:
                            continue
                        self.plot_hists[nu] = self.plot_hists["signal"][nu] * flux_nuisance
                    for nu in self.unosc_plot_hists["signal"].keys():
                        if nu not in self.params["detector"]["observable_flavors"]:
                            continue
                        self.unosc_plot_hists[nu] = self.unosc_plot_hists["signal"][nu] * flux_nuisance

                    # remove the signal from the plot hists
                    del self.plot_hists["signal"]
                    del self.unosc_plot_hists["signal"]
                    continue

                self.plot_hists[k] =  self.plot_hists[k] * (1 + fit_params.get("{}_{}".format(k, self.params['name']), 0))
                self.unosc_plot_hists[k] = self.unosc_plot_hists[k] * (1 + fit_params.get("{}_{}".format(k, self.params['name']), 0))

            # DEBUG
            # total_counts = 0
            # for k in self.plot_hists.keys():
            #     total_counts += np.sum(self.plot_hists[k][1])
            #     print(k, np.sum(self.plot_hists[k][1]))
            # print("total predicted", total_counts)
            # print("total observed", np.sum(self.data_hist[1]))

        if set_data_hist:
            # this is for if we're calculating a new model/dataset
            self.data_hist = predicted

        return predicted

