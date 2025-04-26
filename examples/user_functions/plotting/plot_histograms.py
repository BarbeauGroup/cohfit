import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from classes.Experiment import Experiment

plt.style.use(['science'])

def plot_1d_histograms(experiment: Experiment) -> None:
    params = experiment.params
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    colors = ["#d73027", "#f46d43", "#fdae61", "#fee090", "#e0f3f8", "#abd9e9", "#74add1", "#4575b4"]
    label_dict = {
        "brn": "BRNs",
        "nin": "NINs",
        "ssb": "Steady-State Bkg.",
        "signal": r"Predicted $\nu$s",
        "nuE": r"$\nu_e$",
        "nuMu": r"$\nu_\mu$",
        "nuTau": r"$\nu_\tau$",
        "nuEBar": r"$\bar{\nu}_e$",
        "nuMuBar": r"$\bar{\nu}_\mu$",
        "nuTauBar": r"$\bar{\nu}_\tau$",
        "nuS": r"$\nu_s$",
        "nuSBar": r"$\bar{\nu}_s$"
    }

    # stacked histogram
    x = []
    t = []
    e_weights = []
    t_weights = []
    labels = []

    observable_bin_arr = np.asarray(params["analysis"]["energy_bins"])
    t_bin_arr = np.asarray(params["analysis"]["time_bins"])

    # data
    e_hist = np.sum(experiment.data_hist[1] - experiment.plot_hists["ssb"][1], axis=1) / np.diff(observable_bin_arr)
    t_hist = np.sum(experiment.data_hist[1] - experiment.plot_hists["ssb"][1], axis=0) / np.diff(t_bin_arr) * 100.

    # 3+1 model
    for k in experiment.plot_hists.keys():
        if k == "ssb":
            continue
        x.append(observable_bin_arr[:-1])
        t.append(t_bin_arr[:-1])
        e_weights.append(np.sum(experiment.plot_hists[k][1], axis=1) / np.diff(observable_bin_arr))
        t_weights.append(np.sum(experiment.plot_hists[k][1], axis=0) / np.diff(t_bin_arr) * 100.)

        labels.append(k)

    # transform labels
    labels = [label_dict[label] for label in labels]

    ax[0].hist(x, bins=observable_bin_arr, weights=e_weights,
                stacked=True, 
                histtype='step',
                edgecolor='black')
    
    ax[0].hist(x, bins=observable_bin_arr, weights=e_weights,
            stacked=True, 
            label=labels,
            alpha=1,
            color=colors[:len(labels)])      

    ax[1].hist(t, bins=t_bin_arr, weights=t_weights,
                   stacked=True, 
                    histtype='step',
                    edgecolor='black')

    ax[1].hist(t, bins=t_bin_arr, weights=t_weights,
                stacked=True, 
                label=labels,
                alpha=1,
                color=colors[:len(labels)])
    
    # Unoscillated

    e_weights = 0
    t_weights = 0

    for k in experiment.unosc_plot_hists.keys():
        if k == "ssb":
            continue
        e_weights += (np.sum(experiment.unosc_plot_hists[k][1], axis=1) / np.diff(observable_bin_arr))
        t_weights += (np.sum(experiment.unosc_plot_hists[k][1], axis=0) / np.diff(t_bin_arr) * 100.)

    ax[0].hist(observable_bin_arr[:-1], bins=observable_bin_arr, weights=e_weights, histtype='step', linestyle='dashed', label="No Oscillation", color="grey")
    ax[1].hist(t_bin_arr[:-1], bins=t_bin_arr, weights=t_weights, histtype='step', linestyle='dashed', label="No Oscillation", color="grey")

    # data
    ax[0].errorbar(x=(observable_bin_arr[1:] + observable_bin_arr[:-1])/2, y=e_hist, ls="none", label="Data", color="black", marker="x", markersize=5)
    ax[1].errorbar(x=(t_bin_arr[1:] + t_bin_arr[:-1])/2, y=t_hist, ls="none", label="Data", color="black", marker="x", markersize=5)
    
    ax[0].set_xlabel(f"Energy [{params['analysis']['_energy_units']}]")
    ax[0].set_ylabel(f"Counts / {params['analysis']['_energy_units']}")

    ax[1].set_xlabel(r"Time [$\mu$s]")
    ax[1].set_ylabel(r"Counts / 10 $\mu$s")
    ax[1].yaxis.set_tick_params(which='both', labelleft=True)

    # ax[1].set_xscale('function', functions=(lambda x: np.where(x < 1, x, (x - 1) / 4 + 1),
    #                                         lambda x: np.where(x < 1, x, 4 * (x - 1) + 1)))
    
    # ax[0].legend(*map(reversed, ax[0].get_legend_handles_labels()))
    # ax[1].legend(*map(reversed, ax[1].get_legend_handles_labels()))

    plt.plot()
    plt.show()

    return

# def plot_observables2d(params: dict, histograms_unosc: dict, histograms_osc: dict, alpha) -> None:
#     fig, ax = plt.subplots(2, 2, figsize=(14, 10))
#
#     observable_bin_arr = np.asarray(params["analysis"]["energy_bins"])
#     t_bin_arr = np.asarray(params["analysis"]["time_bins"])
#
#     # x, y = np.meshgrid(observable_bin_arr[:-1], t_bin_arr[:-1])
#     x = np.tile(observable_bin_arr[:-1], len(t_bin_arr[:-1]))
#     y = np.repeat(t_bin_arr[:-1], len(observable_bin_arr[:-1]))
#
#     # 3 + 1 model
#     weights = 0
#
#     # for bkd in ["brn", "nin"]:
#     #     if histograms_osc.get(bkd) is None: continue
#     #     if bkd == "brn": scale = alpha[1]
#     #     if bkd == "nin": scale = alpha[2]
#     #     weights += histograms_osc[bkd] * (1 + scale) / np.outer(np.diff(observable_bin_arr), np.diff(t_bin_arr))
#
#     for flavor in histograms_osc["neutrinos"].keys():
#         if flavor not in params["detector"]["observable_flavors"]:
#             continue
#         print(flavor)
#         scale = alpha[0]
#         weights += histograms_osc["neutrinos"][flavor] * (1 + scale) / np.outer(np.diff(observable_bin_arr), np.diff(t_bin_arr))
#
#     ax[0, 0].set_title("3+1 Model Observables")
#     ax[0,0].hist2d(x, y, bins=[observable_bin_arr, t_bin_arr], weights=weights.T.flatten(), cmap="Greys")
#     cbar = plt.colorbar(ax[0,0].collections[0], ax=ax[0,0])
#     cbar.set_label(r"Counts / PE / $\mu$s")
#
#     model_weights = weights
#
#     # disappearance
#     weights = 0
#     for flavor in histograms_unosc["neutrinos"].keys():
#         if flavor not in params["detector"]["observable_flavors"]:
#             continue
#         scale = alpha[0]
#         weights += histograms_unosc["neutrinos"][flavor] * (1 + scale) / np.outer(np.diff(observable_bin_arr), np.diff(t_bin_arr) * 100.)
#
#     for flavor in histograms_osc["neutrinos"].keys():
#         if flavor not in params["detector"]["observable_flavors"]:
#             continue
#         scale = alpha[0]
#         weights -= histograms_osc["neutrinos"][flavor] * (1 + scale) / np.outer(np.diff(observable_bin_arr), np.diff(t_bin_arr) * 100.)
#
#     # ax[0, 1].set_title("3+1 Model Steriles")
#     # ax[0,1].hist2d(x, y, bins=[observable_bin_arr, t_bin_arr], weights=weights.T.flatten(), cmap="Greys")
#     # cbar = plt.colorbar(ax[0,1].collections[0], ax=ax[0,1])
#     # cbar.set_label(r"Counts / PE / $\mu$s")
#
#     # data residual
#     weights = (histograms_unosc["beam_state"]["C"] - histograms_unosc["ssb"] * (1 + alpha[3])) / np.outer(np.diff(observable_bin_arr), np.diff(t_bin_arr))
#     weights -= model_weights
#     ax[1, 0].set_title("Data Residual")
#     ax[1,0].hist2d(x, y, bins=[observable_bin_arr, t_bin_arr], weights=weights.T.flatten(), cmap="bwr", norm=colors.CenteredNorm())
#     cbar = plt.colorbar(ax[1,0].collections[0], ax=ax[1,0])
#     cbar.set_label(r"Counts / PE / $\mu$s")
#
#     print("\"chi^2\"")
#     print(np.sum(weights**2))
#
#     # stat likelihood
#     predicted = 0
#     for flavor in histograms_osc["neutrinos"].keys():
#         if flavor not in params["detector"]["observable_flavors"]:
#             continue
#         predicted += histograms_osc["neutrinos"][flavor] * (1 + alpha[0])
#     if histograms_unosc.get("brn") is not None:
#         predicted += histograms_unosc["brn"] * (1 + alpha[1])
#     if histograms_unosc.get("nin") is not None:
#         predicted += histograms_osc["nin"] * (1 + alpha[2])
#     if histograms_unosc.get("ssb") is not None:
#         predicted += histograms_osc["ssb"] * (1 + alpha[3])
#
#     observed = histograms_unosc["beam_state"]["C"]
#
#     weights = -predicted + observed * np.log(predicted) - gammaln(observed + 1)
#     ax[1, 1].set_title("Log Likelihood")
#     ax[1,1].hist2d(x, y, bins=[observable_bin_arr, t_bin_arr], weights=weights.T.flatten(), cmap="Greys_r")
#     cbar = plt.colorbar(ax[1,1].collections[0], ax=ax[1,1])
#     cbar.set_label("Log Likelihood")
#
#     # settings
#
#     for a in ax.flat:
#         # a.set_yscale('function', functions=(lambda x: np.where(x < 1, x, (x - 1) / 3 + 1),
#         #                                         lambda x: np.where(x < 1, x, 3 * (x - 1) + 1)))
#         # a.set_yticks(np.arange(0, 7.0, 1.0))
#         # a.set_yticks(np.arange(0, 1., 0.125), minor=True)
#         # a.tick_params(axis='y', direction='out')
#         # a.tick_params(axis='y', which='minor', direction='out')
#         # a.tick_params(axis='x', direction='out')
#         a.set_xlabel("Energy [PE]")
#         a.set_ylabel(r"Time [$\mu$s]")
#     plt.plot()
#     plt.show()