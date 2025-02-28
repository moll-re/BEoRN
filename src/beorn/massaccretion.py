"""
Mass Accretion Model
"""
import numpy as np
from scipy.interpolate import splrep, splev, interp1d
from scipy.integrate import odeint
import logging
logger = logging.getLogger(__name__)

from .cosmo import D, hubble, Hubble
from .halomassfunction import HaloMassFunction
from .parameters import Parameters


def mass_accretion(z_bins, parameters: Parameters):
    if parameters.source.mass_accretion_model == 'EXP':
        halo_mass, halo_mass_derivative = mass_accretion_EXP(z_bins, parameters)
    elif parameters.source.mass_accretion_model == 'EPS':
        halo_mass, halo_mass_derivative = mass_accretion_EPS(z_bins, parameters)
    else:
        raise ValueError(f"Unknown mass accretion model: {parameters.source.mass_accretion_model}")
    return halo_mass, halo_mass_derivative



def mass_accretion_EPS(z_bins, parameters: Parameters):
    """
    Assuming EPS formula
    (see Eq. 6 in 1409.5228)

    mm : array. The initial mass bin at z = zstart (self.M_Bin).
    zz : decreasing array of redshifts.

    Returns :
    Mh and dMh_dt, two 2D arrays of shape (zz, mm)
    """
    z_bins = np.flip(z_bins) # flip the z array so that it increases : zz = 6...25 etc. This way we solve the evolution of h masses backward in time, since M_Bin is defined as the h masses at the final redshift.
    m_bins = parameters.simulation.halo_mass_bins
    aa = 1 / (z_bins + 1)
    Dgrowth = []
    for i in range(len(z_bins)):
        Dgrowth.append(D(aa[i], parameters))  # growth factor
    Dgrowth = np.array(Dgrowth)


    parameters.halo_mass_function.z = [0]  # we just want the linear variance
    # TODO: don't update parameters on the fly. They were set by the user!
    parameters.halo_mass_function.m_min = parameters.simulation.halo_mass_bin_min * 1e-5 ## Need small enough value for the source term below (0.6*M)
    parameters.halo_mass_function.m_max = parameters.simulation.halo_mass_bin_max
    HMF = HaloMassFunction(parameters)
    HMF.generate_HMF(parameters)
    var_tck = splrep(HMF.tab_M, HMF.sigma2)

    # free parameter
    fM = 0.6
    fracM = np.full(len(m_bins), fM)
    frac = interp1d(m_bins, fracM, axis=0, fill_value='extrapolate')

    Dg_tck = splrep(z_bins, Dgrowth)
    D_growth = lambda z: splev(z, Dg_tck)
    dDda = lambda z: splev(z, Dg_tck, der=1)

    Maccr = np.zeros((len(z_bins), len(m_bins)))
    source = lambda M, z: (2 / np.pi) ** 0.5 * M / (splev(frac(M) * M, var_tck, ext=1) - splev(M, var_tck, ext=1)) ** 0.5 * 1.686 / D_growth(z) ** 2 * dDda( z)

    Maccr[:, :] = odeint(source, m_bins, z_bins)
    Maccr = np.nan_to_num(Maccr, nan=0)

    Raccr = Maccr / m_bins[None, :]
    dMaccrdz = np.gradient(Maccr, z_bins, axis=0, edge_order=1)
    dMaccrdt = - dMaccrdz * (1 + z_bins)[:, None] * hubble(z_bins, parameters)[:, None] * sec_per_year / km_per_Mpc

    # remove NaN
    Raccr[np.isnan(Raccr)] = 0.0
    dMaccrdz[np.isnan(dMaccrdz)] = 0.0
    dMaccrdt[np.isnan(dMaccrdt)] = 0.0
    dMaccrdt = dMaccrdt.clip(min=0)

    return np.flip(Raccr * m_bins,axis=0), np.flip(dMaccrdt,axis=0)



def mass_accretion_EXP(z_bins: np.ndarray, parameters: Parameters) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the halo mass and its derivative with respect to time using the exponential model, using a range of initial halo masses and alpha values.
    Args:
        param : dictionary containing all the input parameters
        z_bins  : arr of redshifts

    Returns:
        halo_mass and halo_mass_derivative
        These are two 3d arrays of shape (m_bins, alpha_bins, z_bins)
        halo_mass_derivative is in [Msol/h/yr]
    """
    m_bins = parameters.simulation.halo_mass_bins
    alpha_bins = parameters.source.mass_accretion_alpha_range
    z_initial = z_bins.min()
    logger.debug(f"Using {m_bins.shape=}, {alpha_bins.shape=} and {m_bins.shape=}")

    halo_mass = m_bins[:, None, None] * np.exp(alpha_bins[None, :, None] * (z_initial - z_bins[None, None, :]))
    logger.debug(f"{halo_mass.shape=}")
    halo_mass_derivative = mass_accretion_EXP_derivative(parameters, halo_mass, z_bins)
    logger.debug(f"{halo_mass_derivative.shape=}")

    plot_halo_mass(halo_mass, halo_mass_derivative, m_bins, z_bins, alpha_bins)
    return halo_mass, halo_mass_derivative


def mass_accretion_EXP_derivative(parameters: Parameters, halo_mass: np.ndarray, z_bins: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    param : dictionary containing all the input parameters
    Mh : arr. Halo masss
    z  : arr of redshifts, shape(Mh).

    Returns
    ----------
    Halo mass accretion rate, i.e. time derivative of halo mass (dMh/dt in [Msol/h/yr])
    """
    # by construction halo_mass has an alpha dependence and an initial mass dependence
    # using the function from above we can formulate an analytical expression for the derivative:
    # dMh/dt = Mh * alpha * H(z) * (z+1)
    alpha_bins = parameters.source.mass_accretion_alpha_range
    print(((1 + z_bins) * hubble(z_bins, parameters)).shape)
    return halo_mass * alpha_bins[None, :, None] * ((1 + z_bins) * hubble(z_bins, parameters))[None, None, :]


def plot_halo_mass(halo_mass: np.ndarray, halo_mass_derivative: np.ndarray, m_bins: np.ndarray, z_bins: np.ndarray, alpha_bins: np.ndarray):
    import matplotlib.pyplot as plt

    m_bin = 20
    plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Halo Mass - $M = {m_bins[m_bin]:.2g}$")
    for i in range(len(alpha_bins)):
        # plot a given starting mass for all alpha values
        plt.plot(z_bins, halo_mass[m_bin, i, :], label=f"$\\alpha = {alpha_bins[i]:.2f}$")
    plt.legend()
    plt.show()


    plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Halo Mass derivative - $M = {m_bins[m_bin]:.2g}$")
    for i in range(len(alpha_bins)):
        # plot a given starting mass for all alpha values
        plt.plot(z_bins, halo_mass_derivative[m_bin, i, :], label=f"$\\alpha = {alpha_bins[i]:.2f}$")
    plt.show()
