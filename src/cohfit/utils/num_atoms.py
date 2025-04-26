def num_atoms(params, isotope):
    mass = params["detector"]["mass"]
    molar_mass = params["detector"]["molar_mass"]
    Av = 6.022e23
    
    return (mass * 1000) / molar_mass * Av * isotope["abundance"]