import numpy as np

from .constants import K_B, M_E, M_H, H_BAR, CHI_H, CHI_HEI, CHI_HEII

from ..calculus import complement

def debroglie_wavelength_squared(temperature, mass):
    return 2 * np.pi * H_BAR**2 / mass / K_B / temperature

def approx_saha(temperature, density, mean_mass, ionisation_energy, degeneracy, previous_degeneracy):
    # Eq. 32
    electron_debroglie = debroglie_wavelength_squared(temperature, M_E)
    return (
        2 * degeneracy / previous_degeneracy  # 2 g_i^(r) / g_i^(r-1)
        * mean_mass / density / electron_debroglie**1.5  # m_0 / (\rho \lambda_e^3)
        * np.exp(- ionisation_energy / K_B / temperature)  # \exp(- \chi_i^(r) / (k T))
    )

def delta(i, j):
    return 1.0 if i == j else 0.0

def gamma_i(temperature, ionisation_energy, x, y, i=1):
    z = 1 - y
    return x * y * z * (ionisation_energy / K_B / temperature)**2 / (1 + delta(i, 1) * z)

def partial_free_energy_density_i(temperature, ionisation_energy, x, y, i=1):
    # Eq. 40
    phi = 1.5 + ionisation_energy / K_B / temperature
    z = 1 - y
    return x * y * (1.5 + z * phi**2 / (1 + delta(i, 1) * z))

def adiabatic_depression(temperature, density, helium):
    # Eq. 54
    x_He = helium / (4 - 3 * helium)
    x_H = 1 - x_He
    mean_mass = M_H * (x_H + 4 * x_He) # assume m_He = 4 m_H

    K_H = approx_saha(temperature, density, mean_mass, CHI_H, 1, 2)
    K_HeI = approx_saha(temperature, density, mean_mass, CHI_HEI, 2, 1)
    K_HeII = approx_saha(temperature, density, mean_mass, CHI_HEII, 1, 2)
    
    y_H = 0.5 * (np.sqrt(K_H*(K_H + 4)) - K_H)
    y_HeI = K_HeI / (1 + K_HeI)
    y_HeII = K_HeII / (1 + K_HeII)
    
#     x = 1.0  # since all is hygrogen
#     y = y_H(temperature, density)

#     d2TTf = partial_free_energy_density(temperature, density, CHI_H, x, y)
    d2TTf = (
        1.5 + partial_free_energy_density_i(temperature, CHI_H, x_H, y_H)
        + partial_free_energy_density_i(temperature, CHI_HEI, x_He, y_HeI, i=2)
        + partial_free_energy_density_i(temperature, CHI_HEII, x_He, y_HeII, i=2)
    )
    return (
        1 / d2TTf 
#         * x * y * (1 - y)
#         * (CHI_H / K_B / temperature)**2 / (2 - y)
        * (
            gamma_i(temperature, CHI_H, x_H, y_H)
            + gamma_i(temperature, CHI_HEI, x_He, y_HeI, i=2)
            + gamma_i(temperature, CHI_HEII, x_He, y_HeII, i=2)
        )
    )

def first_adiabatic_exponent(temperature, density, helium):
    # Eq. 53 fro Houdayer et al. (2021)
    return 5/3 - 2/3 * adiabatic_depression(temperature, density, helium)

def sound_speed(gamma, pressure, density):
    return np.sqrt(gamma * pressure / density)

def acoustic_depth(radius, csound, axis=-1):
    return complement(1/csound, radius, axis=axis, initial=0.0)
    # if tau0 is None:
        # tau0 = trapezoid(1/csound, radius)
    # return tau0 - cumulative_trapezoid(1/csound, radius, initial=0)
