
import matplotlib.pyplot as plt
import numpy as np

from classes.Ensemble import Ensemble
from classes.Experiment import Experiment
from minimization.feldmancousins import feldmancousins
from user_functions.pbglass.ssb.make_flat_ssb import make_flat_ssb
from user_functions.pbglass.transform_functions import form_factors, pb_glass


def main():
    ensemble = Ensemble("config/ensemble.json")

    pbglass20 = Experiment("config/pb_glass_20m.json", pb_glass.energy_efficiency, pb_glass.time_efficiency, form_factors.unity)
    
    ssb_hist = make_flat_ssb("config/pb_glass_20m.json")
    pbglass20.base_hists["ssb"] = ssb_hist

    ensemble.add_experiment(pbglass20)
    ensemble.set_nuisance_params(["flux", "ssb_pb_glass_20m"])

    ue4_bins = np.linspace(0, 0.5, num=6, endpoint=True)
    um4_bins =  np.linspace(0, 0.5, num=6, endpoint=True)
    mass_bins =  np.logspace(0, 2, num=5, endpoint=True)

    mass = 10 #mass_bins[2] #np.random.uniform(1,100) #mass_bins[20]
    ue4_2 = 0.3 #ue4_bins[2] #np.random.uniform(0,0.5) #ue4_bins[20]
    um4_2 = 0 #um4_bins[2] # um4_bins[5]
    flux = 0 #0.08 #np.random.gauss(0, 0.1)
    ssb_alpha = 0

    x0 = [0, 0, 0, # mass, ue4_2, um4_2,
          0, # flux
          0] # ssb_pb_glass_20m

    bounds_l = [0, 0, 0, # mass ue4_2, um4_2,
                -np.inf, # flux
                -np.inf] # ssb_pb_glass_20m

    bounds_u = [100, 0.5, 0, # mass ue4_2, um4_2,
                np.inf, # flux
                np.inf] # ssb_pb_glass_20m

    # NOTE I think we have to have something like this here before we call FC
    # To avoid some kind of horrific circular reasoning
    print(mass, ue4_2, um4_2, flux, ssb_alpha)
    ensemble.analysis_hists([flux, ssb_alpha], mass, ue4_2, um4_2, set_data_hist=True) # this would normally be something like load data from text?
    # plot_1d_histograms(pbglass20)
    # return

    # breakpoint()
    # write to file
    with open("output/feldmancousins/fake_pb_truth.txt", "w") as f:
        f.write(f"mass^2: {mass}\n")
        f.write(f"U_e4^2: {ue4_2}\n")
        f.write(f"U_m4^2: {um4_2}\n")
        f.write(f"flux: {flux}\n")
        f.write(f"ssb_alpha: {ssb_alpha}\n")

    # ensemble.analysis_hists([0.02887356, -0.00037322], 13.1622776601683795, 0.3, 0.01, set_data_hist=False, plot_hists=True)
    # # ensemble.analysis_hists([0.02887356, -0.00037322], 0, 0.00, 0.00, set_data_hist=False, plot_hists=True)
    # plot_1d_histograms(pbglass20)
    #
    # breakpoint()

    x0 = [mass, ue4_2, 0, 0, 0]
    res_global, alpha_grid, lstar_grid = feldmancousins(ensemble, x0, ue4_bins, um4_bins, mass_bins, bounds_l, bounds_u)

    np.save("output/alphas", alpha_grid)
    breakpoint()

    # best_cost = res_global.fun
    # best_params = res_global.x

    # alpha_grid = np.load("output/alphas.npy")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    p = 0.3

    for i, ue in enumerate(ue4_bins):
        for j, um in enumerate(um4_bins):
            for k, m in enumerate(mass_bins):
                if 1 - alpha_grid[i, j, k] < p:
                    ax.scatter(ue, um, m, c="b", marker="o")
    
    ax.set_xlabel("Ue4_2")
    ax.set_ylabel("um4_2")
    ax.set_zlabel("mass")

    ax.set_xlim(ue4_bins[0], ue4_bins[-1])
    ax.set_ylim(um4_bins[0], um4_bins[-1])
    ax.set_zlim(mass_bins[0], mass_bins[-1])

    ax.set_title(f"{1-p}% CL region")
    plt.show()


    return

    # pvalue_slice is just ue4_2 and mass
    # pvalue_slice = alpha_grid[:, 0, :]
    # plt.imshow(pvalue_slice.T, origin="lower", extent=(ue4_bins[0], ue4_bins[-1], mass_bins[0], mass_bins[-1]), aspect="auto")
    # plt.colorbar(label="p-value")
    # plt.xlabel("Ue4_2")
    # plt.ylabel("mass")
    # plt.title("p-value grid")
    # plt.show()

    # sigma = np.sqrt(2) * erfinv(1 - pvalue_slice)
    # plt.clf()
    # plt.imshow(sigma.T, origin="lower", extent=(ue4_bins[0], ue4_bins[-1], mass_bins[0], mass_bins[-1]), aspect="auto")
    # plt.colorbar(label="sigma")
    # plt.xlabel("Ue4_2")
    # plt.ylabel("mass")
    # plt.title("sigma grid")
    # plt.show()

    # fig, ax = plt.subplots(1, 1)
    #
    # ax.plot(Ue4_2, mass, "x")
    #
    # for i in range(1000):
    #     # ensemble.create_asimov_dataset([], mass, Ue4_2, Umu4_2) # need to reload ensemble to get the asimov data
    #     parameters = {
    #         "mass": mass,
    #         "ue4": Ue4_2,
    #         "umu4": Umu4_2,
    #         "flux": flux
    #     }
    #     hist_osc_2d, _ = ensemble.histograms(pbglass20, parameters, oned=False)
    #     samples = np.random.poisson(hist_osc_2d["neutrinos"]["nuE"])
    #     np.savetxt(pbglass20.params["detector"]["asimov_data_file"], samples)
    #     print("True")
    #     print(mass, Ue4_2, flux, ensemble([flux], mass, Ue4_2, Umu4_2))
    #     # return
    #
    #     # x0 = [0, 0, 0, 0, 100, 100, 100, 0, 0, 0, 0]
    #
    #     # ensemble.plot_intermediate_steps(x0, 0.0, 0., 0.)
    #     # return
    #
    #     # x0 = []
    #     # print(ensemble(x0, 0.0, 0.0, 0.))
    #
    #     # x0 = [50, 0.3, 0]
    #     x0 = [mass, Ue4_2, 0, 0]
    #     # print(evaluate_gridpoint(4, 4, ensemble=ensemble, x0=x0, sin_bins=np.logspace(-3, 0, num=10, endpoint=True), mass_bins=np.logspace(0, 2, num=10), angle="ee", n=1000))
    #     res = feldmancousins(ensemble, x0, ue4_bins, mass_bins, "output/feldmancousins/fake_pb")
    #     results.append(res)
    #
    #     print("Fitted")
    #     print(res[0], res[1], res[3], ensemble([res[3]], res[0], res[1], res[2]))
    #     print()
    #
    #     ax.plot(res[1], res[0], "o")
    # # ax.set_xlim(ue4_bins[0], ue4_bins[-1])
    # # ax.set_ylim(mass_bins[0], mass_bins[-1])
    #
    # # print(results)
    # plt.show()
    #
    # return
    #
    # # Rejected
    # parameters = {
    #     "mass": 84.4379647354121,
    #     "ue4": 0.30598337097021052,
    #     "umu4": 0
    # }
    #
    # print(parameters["mass"], parameters["ue4"], parameters["umu4"])
    #
    # alpha = [0, 0, 0, 0]
    #
    # hist_osc_1d, hist_unosc_1d = ensemble.histograms(pbglass20, parameters, oned=True)
    # # hist_osc_2d, hist_unosc_2d = ensemble.histograms(pbglass20, parameters, oned=False)
    #
    # # print(hist_osc_1d)
    # # return
    # plot_histograms(pbglass20.params, hist_unosc_1d, hist_osc_1d, alpha)
    # # plot_observables2d(pbglass20.params, hist_unosc_2d, hist_osc_2d, alpha)
    #
    # return
    #
    # # feldmancousins(ensemble, x0, np.logspace(-2, 0, num=4, endpoint=True), np.logspace(0, 2, num=4), "ee", "output/feldmancousins/fake_pb")
    # # return
    #
    # # Accepted
    # # parameters = {
    # #     "mass": 100.0,
    # #     "ue4": 0.5,
    # #     "umu4": 0.24999990680323825,
    # #     "nu_time_offset": 98.09970251985338,
    # #     "brn_time_offset_csi": 1.8703386090163532,
    # #     "nin_time_offset_csi": 149.9997814492828,
    # #     "flux": 0.023039668807789404,
    # #     "flux_qf_csi": 0.0033268934815811058,
    # #     "flux_eef_csi": 0.0038729275011695208,
    # #     "brn_csi": -2.1094237467877974e-13,
    # #     "nin_csi": -1.9984014443252818e-13,
    # #     "ssb_csi": -0.01244740254758847,
    # #     "r_n_csi": -1.4432899320127035e-13
    # # }
    #
    #
    # # Rejected
    # parameters = {
    #     "mass": 2.5119,
    #     "ue4": 0.5,
    #     "umu4": 1.0685896612017132e-15,
    #     "nu_time_offset": 118.14425252412768,
    #     "brn_time_offset_csi": 1.9978684731137397,
    #     "nin_time_offset_csi": 149.9724256651715,
    #     "flux": 0.09307422869777593,
    #     "flux_qf_csi": 0.013423974942794459,
    #     "flux_eef_csi": 0.015634261100538893,
    #     "brn_csi": 0,
    #     "nin_csi": 0,
    #     "ssb_csi": -0.004155567880415045,
    #     "r_n_csi": 0
    # }
    #
    # alpha = [
    #     np.sqrt(parameters["flux"]**2 + parameters["flux_qf_csi"]**2 + parameters["flux_eef_csi"]**2),
    #     parameters["brn_csi"],
    #     parameters["nin_csi"],
    #     parameters["ssb_csi"]
    # ]
    #
    # hist_osc_1d, hist_unosc_1d = ensemble.histograms(csi_real, parameters)
    # plot_histograms(csi_real.params, hist_unosc_1d, hist_osc_1d, alpha)


if __name__ == "__main__":
    # cProfile.run("main()", "output.prof")
    main()