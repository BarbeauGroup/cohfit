from functools import partial
from time import time

import iminuit
import numpy as np
from scipy.optimize import Bounds
from tqdm import tqdm
from multiprocessing import Pool
import os
from scipy.stats import chi2
from scipy.special import gammaincc as gammaQ

def guess_nfce(u, lambda_star, dof=3):
    lambda_star = np.abs(lambda_star)
    return (1 - gammaQ(dof/2, lambda_star/2)) * gammaQ(dof/2, lambda_star/2) / u**2

def global_best_fit(ensemble, x0, bounds_l, bounds_u):
    # Global best fit theta_hat, phi_hat
    bounds = Bounds(bounds_l, bounds_u, keep_feasible=True)

    res_global = iminuit.minimize(ensemble, x0, bounds=bounds) # TODO: think about basinhopping or other stuff.. and bounds maybe. and initial guess.

    print("Global best fit", res_global.x)# ensemble(res_global.x[3:], res_global.x[0], res_global.x[1], res_global.x[2]))
    print("Global best fit cost", res_global.fun)

    return res_global

def feldmancousins(ensemble, x0, ue4_bins, um4_bins, mass_bins, bounds_l, bounds_u, minFCE=2, maxFCE=10000):
    """"
    Feldman-Cousins method for calculating the p-value of a given point in parameter space.
    """
    print("Feldman-Cousins method for calculating the p-value of a given point in parameter space.")

    # Store the original hists? maybe reset them at the end to not get confused
    # original_hists = []
    # for n,experiment in enumerate(ensemble.experiments):
    #     copy_hist = deepcopy(experiment.data_hist)
    #     original_hists.append(copy_hist)

    ncores = int(os.environ.get('FC_NUM_CORES', 1))
    print("Using {} cores".format(ncores))
    # nFCE = 200 # TODO: parameter or calculate smartly ala nova
    
    # param grid is always 3 dimensional mass ue4 umu4
    # param_grid = list(product(range(len(ue4_bins)), range(len(um4_bins)), range(len(mass_bins))))
    alpha_grid = np.zeros((len(ue4_bins), len(um4_bins), len(mass_bins)), dtype=float)
    lambda_crit_grid = np.zeros((len(ue4_bins), len(um4_bins), len(mass_bins)), dtype=float)
    lstar_grid = np.zeros((len(ue4_bins), len(um4_bins), len(mass_bins)), dtype=float)
    # alpha_conv_grid = np.zeros((len(ue4_bins), len(um4_bins), len(mass_bins), maxFCE), dtype=float)
    phi_grid = np.zeros((len(ue4_bins), len(um4_bins), len(mass_bins), len(x0)-3), dtype=float)
    bounds = Bounds(bounds_l, bounds_u, keep_feasible=True)

    times = []

    # Global best fit theta_hat, phi_hat
    res_global = global_best_fit(ensemble, x0, bounds_l, bounds_u)
    global_cost = res_global.fun

    for i in range(len(ue4_bins)):
        for j in range(len(um4_bins)):
            for k in range(len(mass_bins)):
                Ue4_2 = ue4_bins[i]
                Um4_2 = um4_bins[j]
                mass = mass_bins[k]

                grid_point_n = i*len(um4_bins)*len(mass_bins) + j*len(mass_bins) + k
                print()
                print("Grid point", grid_point_n, "of", len(ue4_bins)*len(um4_bins)*len(mass_bins))
                print("ue4", Ue4_2, "umu4", Um4_2, "mass", mass)

                st = time()

                # This fit is the nuisance parameters phi ONLY for a specific grid point (model parameters theta) of the DATA (not FCPE)
                # This makes phihathat_i
                local_bl, local_bu = bounds_l[3:], bounds_u[3:]
                local_x0 = x0[3:]
                # res_data = iminuit.minimize(ensemble, local_x0, (mass, Ue4_2, Um4_2, None, False), bounds=Bounds(local_bl, local_bu, keep_feasible=True)) # Don't pass data here because we're not using FCPE
                # local_cost = res_data.fun
                res_data = iminuit.minimize(ensemble, local_x0, (mass, Ue4_2, Um4_2), bounds=Bounds(local_bl, local_bu, keep_feasible=True)) # Don't pass data here because we're not using FCPE
                local_cost = res_data.fun #ensemble(local_x0, mass, Ue4_2, Um4_2, include_sys=False)
                local_phi = res_data.x


                print("\tphihathat_i (best fit nuisance params for grid point)", local_phi)
                print("\tcost", local_cost)
                lambda_star = local_cost - global_cost
                lstar_grid[i, j, k] = lambda_star
                phi_grid[i, j, k] = local_phi

                if lambda_star > 50:
                    alpha_grid[i, j, k] = 0 # wilks likelihood is big, so we skip FC
                    print("\tSkipping grid point because lambda_star is", lambda_star)
                    continue

                # need to create a dataset for this gridpoint
                local_asimov_data = ensemble.analysis_hists(res_data.x, mass, Ue4_2, Um4_2, set_data_hist=False)

                nfce_guess = guess_nfce(0.03, lambda_star, 3)
                print("\tlambda_star", lambda_star)
                print("\tnFCE guess", nfce_guess)
                nFCE = int(max(min(nfce_guess, maxFCE), minFCE))

                input_arr = np.zeros(nFCE, dtype=float)
                with Pool(ncores) as pool:
                    lambda_arr = np.asarray(list(tqdm(
                        pool.map(partial(
                            run_pseudoexperiment, ensemble=ensemble, x0=x0, local_asimov_data=local_asimov_data, phi_i_hathat=res_data.x, theta_i=(mass, Ue4_2, Um4_2), bounds=bounds, local_bounds=Bounds(local_bl, local_bu, keep_feasible=True)),
                            input_arr, chunksize=nFCE // ncores), total=nFCE)))

                # calculate alpha; the fraction of that have lambda greater than res_data
                print(lambda_arr)
                alpha = np.sum(lambda_arr > lambda_star) / nFCE
                alpha_conv = np.cumsum(lambda_arr > lambda_star) / np.arange(1, nFCE + 1)

                # calculate the FC (1 sigma) critical chi^2
                crit_val = 0.6826894921370859
                nominal_chi2 = chi2.ppf(crit_val, 3)
                lambda_crit = np.percentile(lambda_arr, crit_val*100)

                print("\talpha", alpha, "lambda_star", lambda_star, "lambda_crit", lambda_crit, "FC correction:", lambda_crit - nominal_chi2)

                alpha_grid[i, j, k] = alpha
                lambda_crit_grid[i, j, k] = lambda_crit
                # alpha_conv_grid[i, j, k] = alpha_conv

                et = time()
                times.append(et-st)
                print("\ttime elapsed", et-st, "seconds")

    print("Average time per grid point", np.mean(times), "seconds")
    return res_global, alpha_grid, lstar_grid, lambda_crit_grid, phi_grid

def run_pseudoexperiment(_, /, ensemble, x0, local_asimov_data, phi_i_hathat, theta_i, bounds, local_bounds):
    np.random.seed() # new seed for each thread
    mass, Ue4_2, Um4_2 = theta_i

    # Poisson histograms
    poisson_hists = []
    for i in range(len(local_asimov_data)):
        # bins = experiment.data_hist[0]
        # experiment.set_data_hist((bins, np.random.poisson(local_asimov_data[l])))
        poisson_hists.append(np.random.poisson(local_asimov_data[i])) # annoying we have to rely on secret experiment order here

    # print("Local FCE has this many counts: ", np.sum(ensemble.experiments[0].data_hist))

    # minimize l1 allowing phi to vary (with theta fixed to theta_i)
    initial_guess_phi = phi_i_hathat # phi_i_hathat
    res1 = iminuit.minimize(ensemble, initial_guess_phi, (mass, Ue4_2, Um4_2, poisson_hists, False), bounds=local_bounds)
    # print("\tl1 (theta_i fixed, phi varied)", res1.x, res1.fun)

    # minimize l2 allowing theta AND phi to vary
    res2 = iminuit.minimize(ensemble, x0, (None, None, None , poisson_hists, False), bounds=bounds)
    # print("\tl2 (both varied)", res2.x, res2.fun)

    # calculate lambda_ij = l1 - l2 for a given PE j
    lambda_ij = res1.fun - res2.fun
    # print("\t\tlambda_ij", lambda_ij)
    return lambda_ij
