import astropy.constants as const
import astropy.units as u

G = const.G.cgs.value

K_B = const.k_B.cgs.value  #.to(u.erg/u.Kelvin).value
M_E = const.m_e.cgs.value  #.to(u.g).value
H_BAR = const.hbar.cgs.value  #.to(u.erg * u.s).value
M_H = 1.00784 * const.u.cgs.value  #.to(u.g).value

# Excitation energies
CHI_H = (13.59844 * u.electronvolt).cgs.value
CHI_HEI = (24.58739 * u.electronvolt).cgs.value
CHI_HEII = (54.41777 * u.electronvolt).cgs.value
