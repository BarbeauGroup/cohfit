import numpy as np

def make_profile_likelihood(ensemble, fixed_model_params):
    """
    fixed_model_params: dict, e.g. {"eps_u_ee": 0.1, "eps_u_mumu": -0.2}
    """

    model_param_names = ensemble.model_params
    nuisance_param_names = ensemble.nuisance_params

    def pll(x_nuisance):
        # Assemble full x vector in correct order
        x_full = []

        for p in model_param_names:
            if p in fixed_model_params:
                x_full.append(fixed_model_params[p])
            else:
                raise ValueError(f"Model param {p} not fixed in PLL")

        x_full.extend(x_nuisance)
        return ensemble(np.array(x_full))

    return pll
