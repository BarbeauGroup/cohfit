import json
from copy import deepcopy

import numpy as np

from flux.nuflux import oscillate_flux
from user_functions.csi.backgrounds.create_brn_observable import create_brn_observables
from user_functions.csi.backgrounds.create_nin_observable import create_nin_observables
from user_functions.csi.signal.create_neutrino_observables import create_neutrino_observables

from utils.num_atoms import num_atoms


class Experiment:
    def __init__(self, config_file, e_eff, t_eff, form_factor=None):
        with open(config_file, 'r') as f:
            if config_file.endswith(".json"):
                self.params = json.load(f)
            else:
                raise ValueError("Only JSON config files are supported right now")
            if self.params["analysis"]["unbinned"]:
                raise NotImplementedError("Unbinned analysis deprecated")

        for isotope in self.params["detector"]["isotopes"]:
            isotope["flux_matrix"] = np.load(isotope["flux_matrix"])
            isotope["num_atoms"] = num_atoms(self.params, isotope)
        
        self.detector_matrix = np.load(self.params["detector"]["detector_matrix"])

        self.observable_energy_bins = np.asarray(self.params["analysis"]["energy_bins"])
        self.observable_time_bins = np.asarray(self.params["analysis"]["time_bins"])

        self.energy_efficiency = e_eff
        self.time_efficiency = t_eff

        # form factor stuff
        self.form_factor = form_factor

        self.base_hists = {}
        self.data_hist = None
        self.total_predicted_hist = None

        self.plot_hists = {}
        self.unosc_plot_hists = {}

    def calculate_predicted(self, flux, mass, ue4, umu4, fit_params, set_data_hist=False, plot_hists=False):
        ll_hists = {}

        # Modify the neutrino spectrum before detector effects
        osc_params = [self.params["detector"]["distance"], mass, ue4, umu4, 0.0]
        osc_flux = oscillate_flux(flux=flux, oscillation_params=osc_params) # TODO: make it take fit_params too

        # Create the signal
        nu_obs = create_neutrino_observables(flux=osc_flux, experiment=self, nuisance_params=fit_params, flavorblind=True, detector_effects=True)
        ll_hists["signal"] = nu_obs["combined"]

        # Create backgrounds
        if "brn" in self.base_hists.keys():
            brn_obs = create_brn_observables(self, fit_params)
            ll_hists["brn"] = brn_obs

        if "nin" in self.base_hists.keys():
            nin_obs = create_nin_observables(self, fit_params)
            ll_hists["nin"] = nin_obs

        if "ssb" in self.base_hists.keys():
            ## ssb_obs trivial
            ll_hists["ssb"] = self.base_hists["ssb"]

        ## Calculate the predicted histogram
        flux_nuisance = 1
        for k in fit_params.keys():
            if k.startswith("flux"):
                flux_nuisance += fit_params[k]

        predicted = 0
        predicted += ll_hists["signal"][1] * flux_nuisance

        for k in ll_hists.keys():
            if k == "signal":
                continue
            predicted += ll_hists[k][1] * (
                1 + fit_params.get("{}_{}".format(k, self.params['name']), 0)
            )

        # Calculate the plotting histograms if required # TODO: pull out into separate function
        if plot_hists:
            self.plot_hists = deepcopy(ll_hists)
            self.unosc_plot_hists = deepcopy(ll_hists)

            # If we want to create the plot hists, we create an unoscillated neutrino spectrum
            unosc_nu_obs = create_neutrino_observables(flux=flux, experiment=self, nuisance_params=fit_params, flavorblind=True)
            self.unosc_plot_hists["signal"] = unosc_nu_obs["combined"]

            for k in ll_hists.keys():
                if k == "signal":
                    # both oscillated and unoscillated histograms get multiplied by the flux nuisance
                    self.plot_hists["signal"] = (
                        self.plot_hists["signal"][0],
                        self.plot_hists["signal"][1] * flux_nuisance,
                    )
                    self.unosc_plot_hists["signal"] = (
                        self.unosc_plot_hists["signal"][0],
                        self.unosc_plot_hists["signal"][1] * flux_nuisance,
                    )
                    continue

                self.plot_hists[k] = (
                    self.plot_hists[k][0],
                    self.plot_hists[k][1]
                    * (1 + fit_params.get("{}_{}".format(k, self.params['name']), 0)),
                )
                self.unosc_plot_hists[k] = (
                    self.unosc_plot_hists[k][0],
                    self.unosc_plot_hists[k][1]
                    * (1 + fit_params.get("{}_{}".format(k, self.params['name']), 0)),
                )

            # DEBUG
            # total_counts = 0
            # for k in self.plot_hists.keys():
            #     total_counts += np.sum(self.plot_hists[k][1])
            #     print(k, np.sum(self.plot_hists[k][1]))
            # print("total predicted", total_counts)
            # print("total observed", np.sum(self.data_hist[1]))

        if set_data_hist:
            # this is for if we're calculating a new model/dataset
            self.data_hist = ((np.asarray(self.params["analysis"]["energy_bins"]), np.asarray(self.params["analysis"]["time_bins"])), predicted)

        return predicted

