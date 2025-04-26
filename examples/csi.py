
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from classes.Ensemble import Ensemble
from classes.Experiment import Experiment
from minimization.feldmancousins import feldmancousins, global_best_fit

from user_functions.csi.data.make_csi_data_histogram import make_csi_data_histogram
from user_functions.csi.ssb.make_csi_ssb import make_csi_ssb
from user_functions.csi.backgrounds.make_csi_brn_histogram import make_csi_brn_histogram
from user_functions.csi.backgrounds.make_csi_nin_histogram import make_csi_nin_histogram

from user_functions.csi.transform_functions import csi, form_factors

import matplotlib.colors as mcolors

import scienceplots

plt.style.use('science')


def main():

    ensemble = Ensemble("config/ensemble.json")
    csi_real = Experiment("config/csi.json", csi.energy_efficiency, csi.time_efficiency, form_factors.helm)
    
    ensemble.set_nuisance_params([
        "flux",
        # "flux_qf_csi",
        # "flux_eef_csi",

        # "brn_csi",
        # "nin_csi",
        "ssb_csi",

        "nu_time_offset",
        #"brn_time_offset_csi",
        #"nin_time_offset_csi"
    ])

    x0 = [0, 0, 0, # mass, ue4, umu4
          0, #0, 0, # flux, flux_qf_csi, flux_eef_csi
          0, #0, 0, # brn_csi, nin_csi, ssb_csi
          0, #0, 0 # nu_time_offset, brn_time_offset_csi, nin_time_offset_csi
    ]

    csi_real.base_hists["ssb"] = make_csi_ssb("config/csi.json")
    csi_real.base_hists["brn"] = make_csi_brn_histogram("config/csi.json")
    csi_real.base_hists["nin"] = make_csi_nin_histogram("config/csi.json")
    csi_real.data_hist = make_csi_data_histogram("config/csi.json")

    ensemble.add_experiment(csi_real)

    bounds_l = [0, 0, 0, # mass, ue4, umu4
                -np.inf, #-np.inf, -np.inf,  # flux, flux_qf_csi, flux_eef_csi
                -np.inf, #-np.inf, -np.inf, # brn_csi, nin_csi, ssb_csi
                -250, #50, -50 # nu_time_offset, brn_time_offset_csi, nin_time_offset_csi
    ]

    bounds_u = [100, 0.5, 0.5, # mass, ue4, umu4
                np.inf, #np.inf, np.inf,  # flux, flux_qf_csi, flux_eef_csi
                np.inf, #np.inf, np.inf, # brn_csi, nin_csi, ssb_csi
                250, #300, 100 # nu_time_offset, brn_time_offset_csi, nin_time_offset_csi
    ]

    # find best fit
    # res_global = global_best_fit(ensemble, x0, bounds_l, bounds_u)
    # print(res_global)
    # return

    ue4_bins = np.linspace(0,0.5,11,True)
    um4_bins = [0] #np.array([res_global.x[2]]) #
    mass_bins = np.linspace(1,13,13,True) #[0]

    res_global, alpha_grid, lstar_grid, alpha_conv_grid, lambda_crit_grid, phi_grid = feldmancousins(ensemble, x0, ue4_bins, um4_bins, mass_bins, bounds_l, bounds_u)

    np.savez_compressed("output/bisset_coarse",
             mass_bins=mass_bins,
             ue4_bins=ue4_bins,
             um4_bins=um4_bins,
             res_global=res_global.x,
             alpha_grid=alpha_grid,
             lstar_grid=lstar_grid,
             alpha_conv_grid=alpha_conv_grid,
             lambda_crit_grid=lambda_crit_grid,
             phi_grid=phi_grid)

    return

if __name__ == "__main__":
    # cProfile.run("main()", "output.prof")
    main()