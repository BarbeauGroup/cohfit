
from flux.probabilities import Pab


def oscillate_flux(flux: dict, oscillation_params: list[float]) -> dict:
    """
    flux: dictionary with SNS flux information

    returns: dictionary with oscillated SNS flux information
    """

    ###

    # Oscillation Parameters

    L = oscillation_params[0]
    deltam41_2 = oscillation_params[1]
    Ue4_2 = oscillation_params[2]
    Umu4_2 = oscillation_params[3]
    Utau4_2 = oscillation_params[4]

    # Make an empty dictionary to store the oscillated flux information
    oscillated_flux_dict = {}

    flavors = ["nuE", "nuMu", "nuTau", "nuS"]
    anti_flavors = ["nuEBar", "nuMuBar", "nuTauBar", "nuSBar"]

    # This for loop fills the dictionary
    for final_i, (final_flavor, final_antiflavor) in enumerate(zip(flavors, anti_flavors)):
        # This for loop does the sum over initial flavors to calculate the oscillated flux
        oscillated_flux = 0
        anti_oscillated_flux = 0
        for initial_i, (initial_flavor, initial_antiflavor) in enumerate(zip(flavors, anti_flavors)):
            oscillated_flux += Pab(flux[initial_flavor][0][1], L, deltam41_2, final_i+1, initial_i+1, Ue4_2, Umu4_2, Utau4_2) * flux[initial_flavor][1]
            anti_oscillated_flux += Pab(flux[initial_antiflavor][0][1], L, deltam41_2, final_i+1,  initial_i+1, Ue4_2, Umu4_2, Utau4_2) * flux[initial_antiflavor][1] # TODO: final and initial should be switched for anti-flavors? (Yes probably)

        oscillated_flux_dict[final_flavor] = (flux[final_flavor][0], oscillated_flux)
        oscillated_flux_dict[final_antiflavor] = (flux[final_antiflavor][0], anti_oscillated_flux)
    
    return oscillated_flux_dict