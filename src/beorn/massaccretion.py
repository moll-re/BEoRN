"""
Mass Accretion Model
"""
import numpy as np
import logging
logger = logging.getLogger(__name__)

from .cosmo import Hubble
from .structs.parameters import Parameters


def mass_accretion(z_bins: np.ndarray, parameters: Parameters) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the halo mass and its derivative with respect to time using an exponential mass accretion model. A range of initial halo masses and alpha values is computed.
    Args:
        param : dictionary containing all the input parameters
        z_bins  : arr of redshifts

    Returns:
        halo_mass and halo_mass_derivative
        These are two 3d arrays of shape (m_bins, alpha_bins, z_bins)
        halo_mass_derivative is in [Msol/h/yr]
    """
    # the z_bins correspond to the mid points of the redshift bins - do the same for the mass and alpha bins
    m_bins = (parameters.simulation.halo_mass_bins[:-1] + parameters.simulation.halo_mass_bins[1:]) / 2
    # m_bins = parameters.simulation.halo_mass_bins[:-1]
    alpha_bins = (parameters.source.mass_accretion_alpha_range[:-1] + parameters.source.mass_accretion_alpha_range[1:]) / 2
    
    z_initial = z_bins.min()
    logger.info(f"Computing mass accretion for a parameter space consisting of: {m_bins.shape=}, {alpha_bins.shape=} and {z_bins.shape=}")

    halo_mass = m_bins[:, None, None] * np.exp(alpha_bins[None, :, None] * (z_initial - z_bins[None, None, :]))
    halo_mass_derivative = mass_accretion_derivative(parameters, halo_mass, z_bins)
    logger.debug(f"{halo_mass.shape=} and {halo_mass_derivative.shape=}")

    return halo_mass, halo_mass_derivative


def mass_accretion_derivative(parameters: Parameters, halo_mass: np.ndarray, z_bins: np.ndarray) -> np.ndarray:
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
    alpha_bins = (parameters.source.mass_accretion_alpha_range[:-1] + parameters.source.mass_accretion_alpha_range[1:]) / 2
    return halo_mass * alpha_bins[None, :, None] * ((1 + z_bins) * Hubble(z_bins, parameters))[None, None, :]


def plot_halo_mass(halo_mass: np.ndarray, halo_mass_derivative: np.ndarray, m_bins: np.ndarray, z_bins: np.ndarray, alpha_bins: np.ndarray):
    import matplotlib.pyplot as plt
    print(f"{halo_mass.shape=}, {halo_mass_derivative.shape=}, {m_bins.shape=}, {z_bins.shape=}, {alpha_bins.shape=}")

    # plot the halo mass for a medium mass:
    m_bin = m_bins.size // 2
    m0_string = fr"$M_0 = {m_bins[m_bin]:.2g} M_{{\odot}}$"
    plt.figure()
    plt.xscale('log')
    plt.xlabel('Redshift z')
    plt.ylabel('Halo Mass [Msol]')
    plt.yscale('log')
    plt.title(f"Halo Mass - {m0_string}")
    for i in range(len(alpha_bins) - 1):
        # plot a given starting mass for all alpha values
        plt.plot(z_bins, halo_mass[m_bin, i, :], label=f"$\\alpha = {alpha_bins[i]:.2f}$")
    plt.legend()
    plt.show()


    plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Halo Mass derivative - {m0_string}")
    for i in range(len(alpha_bins) - 1):
        # plot a given starting mass for all alpha values
        plt.plot(z_bins, halo_mass_derivative[m_bin, i, :], label=f"$\\alpha = {alpha_bins[i]:.2f}$")
    plt.show()
