import numpy as np
from utils.histograms import rebin_histogram2d, centers_to_edges
from threadpoolctl import threadpool_limits

# TODO: add cache
def create_neutrino_observables(flux, experiment, nuisance_params, flavorblind=False, detector_effects=True) -> dict:
    """
    This is the linear algebra step: flux -> true -> reconstructed

    Parameters
    ----------

    Returns
    -------
    dict
        The true observables.
    """
    # flux is normalized to number of neutrinos per POT so it's not counted here
    pot_per_cm2 = experiment.params["beam"]["pot"] / (4 * np.pi * np.power(experiment.params["detector"]["distance"] * 100, 2))

    # Load in energy binning
    dx = experiment.params["detector"]["detector_matrix_dx"]
    det_matrix_energy_bins = np.arange(0, experiment.detector_matrix.shape[0] * dx, dx)

    # Load in time analysis bins
    t_anal_bins = experiment.observable_time_bins
    e_anal_bins = experiment.observable_energy_bins

    # Flux energy bins
    flux_energy_bins = np.arange(0, 60, 1) # TODO add to param file? / flux object

    # Do time offset
    if "nu_time_offset" in nuisance_params:
        new_time_edges = flux["nuE"][0][0][:-1] + nuisance_params["nu_time_offset"]
    else:
        new_time_edges = flux["nuE"][0][0][:-1]

    # Calculate the efficiency arrays
    energy_efficiency = experiment.energy_efficiency(det_matrix_energy_bins)
    time_efficiency = experiment.time_efficiency(new_time_edges)
    
    # Add together isotope fluxes somehow
    if flavorblind:
        observables = {
            "combined": [[e_anal_bins, t_anal_bins], 0]
        }
    else:
        observables = {
            "nuE": [[e_anal_bins, t_anal_bins], 0],
            "nuMu": [[e_anal_bins, t_anal_bins], 0],
            "nuTau": [[e_anal_bins, t_anal_bins], 0],
            "nuEBar": [[e_anal_bins, t_anal_bins], 0],
            "nuMuBar": [[e_anal_bins, t_anal_bins], 0],
            "nuTauBar": [[e_anal_bins, t_anal_bins], 0],
            "nuS": [[e_anal_bins, t_anal_bins], 0],
            "nuSBar": [[e_anal_bins, t_anal_bins], 0],
        }

    for isotope in experiment.params["detector"]["isotopes"]:
        recoil_bins = np.linspace(0, isotope["flux_matrix"].shape[0] * experiment.params["detector"]["flux_matrix_dx"], isotope["flux_matrix"].shape[0])

        ff_2 = experiment.form_factor(isotope, recoil_bins, nuisance_params.get("r_n_{}".format(experiment.params['name']), 0.0))**2
        # print(isotope["name"])
        # print(ff_2)

        @threadpool_limits.wrap(limits=1, user_api='blas') # Limit to 1 thread for BLAS (numpy linalg)
        def calculate(flux_object):
            # Apply time efficiency to flux object
            flux_post_te = flux_object * time_efficiency

            # Rebin in time
            flux_rebinned = rebin_histogram2d(flux_post_te, flux_energy_bins, new_time_edges, flux_energy_bins, t_anal_bins)

            # Load in energy and calculate observable energy before efficiencies
            # observable = experiment.matrix @ flux_object

            recoil_spectrum = isotope["flux_matrix"] @ flux_rebinned 
            recoil_spectrum_post_ff = recoil_spectrum * ff_2[:, None]
            observable = experiment.detector_matrix @ recoil_spectrum_post_ff

            # Rescale counts 
            observable *= isotope["num_atoms"] * pot_per_cm2 

            # Apply energy efficiency
            post_efficiency = observable * energy_efficiency[:, None]

            if detector_effects: 
                return post_efficiency
            else:
                # DEBUG !!!!!!!
                return recoil_spectrum * isotope["num_atoms"] * pot_per_cm2 

        combined_flux = 0
        for flavor in flux.keys():
            if flavorblind:
                if flavor in experiment.params["detector"]["observable_flavors"]:
                    combined_flux += flux[flavor][1]
            else:
                observables[flavor][1] += calculate(flux[flavor][1].T)
        
        if flavorblind:
            observables["combined"][1] += calculate(combined_flux.T)
    
    for flavor in observables.keys():
        # Rebin to final analysis bins
        observables[flavor][1] = rebin_histogram2d(observables[flavor][1], centers_to_edges(det_matrix_energy_bins), t_anal_bins, e_anal_bins, t_anal_bins)
    # e = 0
    # for flavor in observables.keys():
    #     e += np.sum(observables[flavor][1])
    # print("total", e)

    # breakpoint()

    return observables