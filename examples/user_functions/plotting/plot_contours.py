import iminuit
from scipy.optimize import Bounds
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.colors as mcolors
import scienceplots

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from classes.Ensemble import Ensemble
from classes.Experiment import Experiment
from plot_histograms import plot_1d_histograms

from user_functions.csi.data.make_csi_data_histogram import make_csi_data_histogram
from user_functions.csi.ssb.make_csi_ssb import make_csi_ssb
from user_functions.csi.backgrounds.make_csi_brn_histogram import make_csi_brn_histogram
from user_functions.csi.backgrounds.make_csi_nin_histogram import make_csi_nin_histogram

from user_functions.csi.transform_functions import csi, form_factors

plt.style.use('science')

def main():
    # set up ensemble for plotting histograms:
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

    csi_real.base_hists["ssb"] = make_csi_ssb("config/csi.json")
    csi_real.base_hists["brn"] = make_csi_brn_histogram("config/csi.json")
    csi_real.base_hists["nin"] = make_csi_nin_histogram("config/csi.json")
    csi_real.data_hist = make_csi_data_histogram("config/csi.json")

    ensemble.add_experiment(csi_real)

    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()

    def on_click(event):
        if event.inaxes not in ax:
            return

        # Grab the x/y data coordinates where the click occurred
        ue4, mass = event.xdata, event.ydata
        print(f"Clicked at ue4={ue4:.2f}, mass={mass:.2f}")
        event.inaxes.scatter(ue4, mass, c="k", marker="+", s=15, alpha=0.5)

        x0 = [0, 0, 0] # flux, ssb_csi, nu_time_offset

        # have to calculate the best fit np for the new point (this will be saved for future fits)
        res_data = iminuit.minimize(ensemble, x0, (mass, ue4, 0), bounds=Bounds([-np.inf,-np.inf,-250], [np.inf,np.inf,250], keep_feasible=True))
        phi = res_data.x
        print(f"Best fit params: {phi}")
        print(f"Best fit chi2: {res_data.fun}")

        ensemble.analysis_hists(phi, mass, ue4, 0, False, True)

        # Plot the new histogram
        plot_1d_histograms(csi_real)


    arrays = np.load("output/bisset.npz")
    mass_bins = arrays["mass_bins"]
    ue4_bins = arrays["ue4_bins"]
    um4_bins = arrays["um4_bins"]
    res_global = arrays["res_global"]
    alpha_grid = arrays["alpha_grid"]
    lstar_grid = arrays["lstar_grid"]
    alpha_conv_grid = arrays["alpha_conv_grid"]
    lambda_crits = arrays["lambda_crit_grid"]

    X, Y = np.meshgrid(ue4_bins, mass_bins)

    lambda_crit_slice = lambda_crits[:, 0, :]
    fc_correction = 2.3 - lambda_crit_slice # 2d
    pvalue_slice = 1 - alpha_grid[:, 0, :]
    wilks_slice = stats.chi2.cdf(lstar_grid[:, 0, :], 2) # 2d

    if np.min(fc_correction) < 0 and np.max(fc_correction) > 0:
        norm = mcolors.TwoSlopeNorm(vmin=np.min(fc_correction), vcenter=0, vmax=np.max(fc_correction))
        cmap = "bwr"
    else:
        norm = mcolors.Normalize(vmin=np.min(fc_correction), vmax=np.max(fc_correction))
        if np.min(fc_correction) < 0:
            cmap = "Blues"
        else:
            cmap = "Reds"
    # contour = ax[0].contourf(X, Y, fc_correction.T, alpha=0.3, cmap=cmap, norm=norm)  # set colormap here
    # ax[0].contour(X, Y, pvalue_slice.T, colors='k', levels=[0.68], linewidths=1)
    # ax[0].contour(X, Y, wilks_slice.T, colors='k', levels=[0.68], linewidths=1, linestyles='dashed')
    contour = ax[0].contourf(X, Y, wilks_slice.T, alpha=0.5)  # set colormap here
    ax[0].contour(X, Y, wilks_slice.T, levels=[0.68], colors='k', linewidths=0.5, linestyles='dashed')

    ax[0].scatter(res_global[1], res_global[0], c="red", marker="x", label="Best fit")
    ax[0].set_xlabel(r"$|U_{e 4}|^2$")
    ax[0].set_ylabel(r"$\Delta m_{41}^2 [\textrm{eV}^2]$")
    ax[0].set_title("Wilks Likelihood")
    # ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlim(1E-2, 0.5)
    ax[0].set_xscale("log")

    plt.colorbar(contour, label=r"p-value", ax=ax[0])  # link colorbar to contour

    contour = ax[1].contourf(X, Y, pvalue_slice.T, alpha=0.5)  # set colormap here
    ax[1].contour(X, Y, pvalue_slice.T, levels=[0.68], colors='k', linewidths=0.5, linestyles='dashed')

    ax[1].scatter(res_global[1], res_global[0], c="red", marker="x", label="Best fit")
    ax[1].set_xlabel(r"$|U_{e 4}|^2$")
    ax[1].set_ylabel(r"$\Delta m_{41}^2 [\textrm{eV}^2]$")
    ax[1].set_title("Feldman-Cousins Likelihood")
    # ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(1E-2, 0.5)
    ax[1].set_xscale("log")

    plt.colorbar(contour, label="p-value", ax=ax[1])  # link colorbar to contour

    ############

    arrays = np.load("output/bisset_fake.npz")
    mass_bins = arrays["mass_bins"]
    ue4_bins = arrays["ue4_bins"]
    um4_bins = arrays["um4_bins"]
    res_global = arrays["res_global"]
    alpha_grid = arrays["alpha_grid"]
    lstar_grid = arrays["lstar_grid"]
    alpha_conv_grid = arrays["alpha_conv_grid"]
    lambda_crits = arrays["lambda_crit_grid"]

    X, Y = np.meshgrid(ue4_bins, mass_bins)

    lambda_crit_slice = lambda_crits[:, 0, :]
    fc_correction = 2.3 - lambda_crit_slice # 2d
    pvalue_slice = 1 - alpha_grid[:, 0, :]
    wilks_slice = stats.chi2.cdf(lstar_grid[:, 0, :], 2) # 2d

    if np.min(fc_correction) < 0 and np.max(fc_correction) > 0:
        norm = mcolors.TwoSlopeNorm(vmin=np.min(fc_correction), vcenter=0, vmax=np.max(fc_correction))
        cmap = "bwr"
    else:
        norm = mcolors.Normalize(vmin=np.min(fc_correction), vmax=np.max(fc_correction))
        if np.min(fc_correction) < 0:
            cmap = "Blues"
        else:
            cmap = "Reds"
    # contour = ax[0].contourf(X, Y, fc_correction.T, alpha=0.3, cmap=cmap, norm=norm)  # set colormap here
    # ax[0].contour(X, Y, pvalue_slice.T, colors='k', levels=[0.68], linewidths=1)
    # ax[0].contour(X, Y, wilks_slice.T, colors='k', levels=[0.68], linewidths=1, linestyles='dashed')
    contour = ax[2].contourf(X, Y, wilks_slice.T, alpha=0.5)  # set colormap here
    ax[2].contour(X, Y, wilks_slice.T, levels=[0.68], colors='k', linewidths=0.5, linestyles='dashed')

    ax[2].scatter(res_global[1], res_global[0], c="red", marker="x", label="Best fit")
    ax[2].set_xlabel(r"$|U_{e 4}|^2$")
    ax[2].set_ylabel(r"$\Delta m_{41}^2 [\textrm{eV}^2]$")
    ax[2].set_title("Wilks Likelihood (new)")
    # ax[0].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlim(1E-2, 0.5)
    ax[2].set_xscale("log")

    plt.colorbar(contour, label=r"p-value", ax=ax[2])  # link colorbar to contour

    contour = ax[3].contourf(X, Y, pvalue_slice.T, alpha=0.5)  # set colormap here
    ax[3].contour(X, Y, pvalue_slice.T, levels=[0.68], colors='k', linewidths=0.5, linestyles='dashed')

    ax[3].scatter(res_global[1], res_global[0], c="red", marker="x", label="Best fit")
    ax[3].set_xlabel(r"$|U_{e 4}|^2$")
    ax[3].set_ylabel(r"$\Delta m_{41}^2 [\textrm{eV}^2]$")
    ax[3].set_title("Feldman-Cousins Likelihood (new)")
    # ax[1].set_xscale("log")
    ax[3].set_yscale("log")
    ax[3].set_xlim(1E-2, 0.5)
    ax[3].set_xscale("log")

    plt.colorbar(contour, label="p-value", ax=ax[3])  # link colorbar to contour

    # register the callback
    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

if __name__ == "__main__":
    main()