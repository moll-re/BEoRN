"""""""""
Computes the 1D profiles of Tk, xal, xHII. 
"""""""""

import importlib
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid, odeint
from scipy.interpolate import splrep, splev, interp1d
import logging
logger = logging.getLogger(__name__)


from .cosmo import comoving_distance, hubble
from .global_qty import *
from .cross_sections import alpha_HII
from .massaccretion import mass_accretion
from .astro import *
from .couplings import eps_lyal
from .functions import *
from .constants import *
from .astro import *
from .cross_sections import sigma_HI, sigma_HeI
from .parameters import Parameters
from . import constants


class RadiationProfiles:
    """
    Computes the 1D profiles. Similar to the HM for 21cm (Schneider et al 2021)

    M_bin : Halo masses in [Msol/h]
    Rbubble is in comoving Mpc/h. shape((zz,M_bin))
    r_grid is in cMpc/h
    z_arr should be a decreasing array.
    r_lyal : pMpc/h
    The 1D profiles are arrays of shape (z,r,M)
    Mhistory has shape [zz, Mass]
    """

    def __init__(self, parameters: Parameters):
        self.z_initial = parameters.solver.z_max  # starting redshift
        if self.z_initial < 35:
            # TODO: add this warning as a validator of the parameters dataclass
            print('WARNING : z_start (parameters.solver.zmax) should be larger than 35.  ')

        # TODO remove hardcoded values
        rmin = 1e-2
        rmax = 600
        Nr = 200
        rr = np.logspace(np.log10(rmin), np.log10(rmax), Nr)
        self.r_grid = rr  ##cMpc/h

        self.z_arr = np.flip(np.sort(np.unique(np.concatenate((def_redshifts(parameters),np.arange(6,40,0.5))))))  ## we add up some redshifts to be converged


    def solve(self, parameters: Parameters) -> None:
        halo_mass, halo_mass_derivative = mass_accretion(self.z_arr, parameters)
        # both arrays have shape [M_bins, alpha_bins, z_arr]
        if parameters.solver.fXh == 'constant':
            print('param.solver.fXh is set to constant. We will assume f_X,h = 2e-4**0.225')
            x_e = np.full(len(self.z_arr), 2e-4)
        else:
            zz_, sfrd_ = compute_sfrd(parameters, self.z_arr, halo_mass, halo_mass_derivative)
            sfrd = np.interp(self.z_arr, zz_, sfrd_, right=0)
            Gamma_ion, Gamma_sec_ion = mean_gamma_ion_xray(parameters, sfrd, self.z_arr)
            self.Gamma_ion = Gamma_ion
            self.Gamma_sec_ion = Gamma_sec_ion
            self.sfrd = np.array((zz_,sfrd_))
            x_e = solve_xe(parameters, Gamma_ion, Gamma_sec_ion, self.z_arr)
            print('param.solver.fXh is not set to constant. We will compute the free e- fraction x_e and assume fXh = x_e**0.225.')


        R_bubble_ = R_bubble(parameters, self.z_arr, halo_mass, halo_mass_derivative).clip(min=0) # cMpc/h
        # logger.info(f"Computed R_bubble: {R_bubble_.shape=}")
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(R_bubble_[:, 0, 0, 0])
        # plt.plot(R_bubble_[:, 1, 1, 1])
        # plt.plot(R_bubble_[:, -2, -2, -2])
        # plt.plot(R_bubble_[:, -1, -1, -1])
        # plt.show()
        logger.info(f"Computing profiles for {len(self.z_arr)} redshifts")
        R_bubble_ = R_bubble(parameters, self.z_arr, halo_mass, halo_mass_derivative).clip(min=0) # cMpc/h

        rho_xray_ = rho_xray(parameters, self.z_arr, self.r_grid, halo_mass, halo_mass_derivative, x_e)
        rho_heat_ = rho_heat(parameters, self.z_arr, self.r_grid, rho_xray_)

        r_lyal = np.logspace(-5, 2, 1000, base=10)  ##    physical distance for lyal profile. Never goes further away than 100 pMpc/h (checked)
        rho_alpha = rho_alpha_profile(parameters, r_lyal,halo_mass,halo_mass_derivative, self.z_arr)
        # TODO assert correct shapes here!
        
        self.r_lyal = r_lyal
        self.rho_alpha = rho_alpha

        self.x_e = x_e
        self.rho_xray_ = rho_xray_
       # T_history = {}
       # rhox_history = {}
       # for i in range(len(zz)):
       #     T_history[str(zz[i])] = rho_heat_[i]
       #     rhox_history[str(zz[i])] =rho_xray_[i]

        if parameters.simulation.average_profiles_in_bin:
            halo_mass_HR, halo_mass_derivative_HR = mass_accretion(self.z_arr, parameters)
            self.rho_alpha_HR = rho_alpha_profile(parameters,r_lyal,halo_mass_HR, halo_mass_derivative_HR, self.z_arr)
            self.rho_xray_HR = rho_xray(parameters, self.r_grid,halo_mass_HR, halo_mass_derivative_HR, x_e, self.z_arr)
            self.rho_heat_HR = rho_heat(parameters, self.r_grid, self.rho_xray_HR, self.z_arr)
            self.R_bubble_HR = R_bubble(parameters, self.z_arr, halo_mass_HR, halo_mass_derivative_HR).clip(min=0)  # cMpc/h
            self.halo_mass_HR = halo_mass_HR
            self.halo_mass_derivative_HR = halo_mass_derivative_HR

        #self.rhox_history = rhox_history
        # TODO - rename these to be fit snake_case
        self.Mh_history = halo_mass
        self.dMh_dt = halo_mass_derivative
        self.z_history = self.z_arr
        self.R_bubble = R_bubble_     # cMpc/h (zz,M)
        #self.T_history = T_history    # Kelvins
        self.rho_heat = rho_heat_           #shape (z,r,M)
        self.r_grid_cell = self.r_grid
        self.Ngdot_ion = Ngdot_ion(parameters, self.z_arr, halo_mass,halo_mass_derivative)




def Ngdot_ion(parameters: Parameters, zz, Mh, dMh_dt):
    """
    Parameters
    ----------
    zz : array of redshifts. It enters in the mass accretion rate.
    Mass : array. mass of the halo in Msol/h. Shape is len(zz)
    dMh_dt : array. MAR (Msol/h/yr)

    Returns
    ----------
    Array. Number of ionizing photons emitted per sec [s**-1].
    """
    # TODO zz probably needs to be inflated since the callers don't do it anymore
    Ob, Om, h0 = parameters.cosmology.Ob, parameters.cosmology.Om, parameters.cosmology.h

    if (parameters.source.source_type == 'SED'):
        Ngam_dot_ion = dMh_dt / h0 * f_star_Halo(parameters, Mh) * Ob / Om * f_esc(parameters, Mh) * parameters.source.Nion / sec_per_year / m_H * M_sun
        Ngam_dot_ion[np.where(Mh < parameters.source.halo_mass_min)] = 0
        return Ngam_dot_ion
    elif parameters.source.source_type == 'Ghara':
        print('CAREFUL, Ghara source type is chosen, Nion becomes just a fine tuning multiplicative factor')
        Mh = zz**0 * Mh * parameters.source.Nion # to make sure it has the correct shape
        Ngam_dot_ion = 1.33 * 1e43 * Mh/h0
        Ngam_dot_ion[np.where(Mh < parameters.source.halo_mass_min)] = 0
        return Ngam_dot_ion  # eq (1) from 3D vs 1D RT schemes.

    elif parameters.source.source_type == 'constant':
        print('constant number of ionising photons chosen. Param.source.Nion becomes Ngam_dot_ion.')
        return np.full(len(zz), parameters.source.Nion)

    elif (parameters.source.source_type == 'Ross'):
        return Mhalo / h0 * Ob / Om / (10 * 1e6 * sec_per_year) / m_p_in_Msun
    else:
        print('Source Type not available. Should be SED or Ross.')
        exit()



def R_bubble(parameters: Parameters, z_bins: np.ndarray, halo_mass: np.ndarray, halo_mass_derivative: np.ndarray):
    """
    Args:
        parameters: dictionary containing all the input parameters
        M_accr,dMh_dt: halo mass history, and growth rate as a function of redshift zz. 2d arrary of shape [zz, M_bin]
        zz: redshift. Matters for the mass accretion rate in Ngdot_ion!

    Returns
    ----------
    Comoving size [cMpc/h] of the ionized bubble around the source, as a function of time. 2d array of size (zz,M_bin)
    """

    # Ngam_dot = Ngdot_ion(parameters, z_bins, halo_mass, halo_mass_derivative)  # s-1
    # print(f"{Ngam_dot.shape=}, {halo_mass.shape=}, {halo_mass_derivative.shape=}")
    Ob, Om, h0 = parameters.cosmology.Ob, parameters.cosmology.Om, parameters.cosmology.h

    # \bar{n}^0_H - mean comoving number density of baryons [Mpc/h]**-3
    comoving_baryon_number_density = (Ob * constants.rhoc0) / (constants.m_p_in_Msun * h0)

    aa = 1/(z_bins+1)
    # nb0_z = comoving_baryon_number_density * (1 + z_bins) ** 3 # physical baryon density

    # print(f"{nb0_z.shape=}, {aa.shape=}, {Ngam_dot.shape=}")
    # nb0_interp  = interp1d(aa, nb0_z, fill_value = 'extrapolate')
    # Ngam_interp = interp1d(aa, Ngam_dot, axis = -1, fill_value = 'extrapolate')
    C = parameters.cosmology.clumping #.0 #clumping factor
    #source = lambda r, a: km_per_Mpc / (hubble(1 / a - 1, param) * a) * (Ngam_interp(a) / (4 * np.pi * r ** 2 * nb0) - alpha_HII( 1e4) / cm_per_Mpc ** 3 * a ** -3 * h0 ** 3 * nb0 * r / 3) # nb0 * a**-3 is physical baryon density
    #source = lambda V, t: Ngam_interp(t) / nb0 - alpha_HII(1e4) * C / cm_per_Mpc ** 3 * h0 ** 3 * nb0_interp(t) * V  # eq 65 from barkana and loeb
    def volume_derivative(volume, a):
        z = 1 / a - 1
        photon_number = Ngdot_ion(parameters, z, halo_mass, halo_mass_derivative)
        baryon_number = comoving_baryon_number_density * (1 + z) ** 3
        # print(f"{photon_number.shape=}, {volume.shape=}")
        volume = volume.reshape(photon_number.shape)
        return km_per_Mpc / (hubble(z, parameters) * a) * (photon_number / comoving_baryon_number_density - alpha_HII(1e4) * C / cm_per_Mpc ** 3 * h0 ** 3 * baryon_number * volume).flatten()  # eq 65 from barkana and loeb

    volume_shape = (parameters.simulation.halo_mass_bin_n, len(parameters.source.mass_accretion_alpha_range), len(z_bins), ) 
    v0 = np.zeros(volume_shape)
    logger.debug(f"{v0.shape=}")
    bubble_volume = odeint(volume_derivative, v0.flatten(), aa)
    # since odeint works with 1d arrays we have a flattened version currently
    bubble_volume = bubble_volume.reshape((aa.size, *volume_shape))
    bubble_radius = (bubble_volume * 3 / (4 * np.pi)) ** (1 / 3)

    return bubble_radius



def rho_xray(parameters: Parameters, z_bins: np.ndarray, rr, M_accr, dMdt_accr, xe):
    """
    Args:
        parameters: dictionary containing all the input parameters
        z_bins: redshift in decreasing order.
        rr: comoving distance from source center [cMpc/h]
        M_accr: function of zz, hence should increase. 3D array of shape [M_bins, alpha_bins, z_arr]
        dMdt_accr: Time derivative of halo mass (MAR). 3D array of shape [M_bins, alpha_bins, z_arr]

    Returns:
        X-ray profile, i.e. energy injected as heat by X-rays, in [eV/s], and of shape [M_bins, alpha_bins, z_arr, r_arr]
        (zz,rr,M_bin) (M_accr, dMdt_accr all have same dimension :(zz,M_bin) )
    """

    Om = parameters.cosmology.Om
    Ob = parameters.cosmology.Ob
    h0 = parameters.cosmology.h
    # TODO: remove hardcoded values
    z_star = 35
    Emin = parameters.source.energy_cutoff_min_xray
    Emax = parameters.source.energy_cutoff_max_xray
    NE = 50

    nb0 = rhoc0 * Ob / (m_p_in_Msun * h0)  # [h/Mpc]^3

    # zprime binning
    dz_prime = 0.1

    # define frequency bin
    nu_min = Emin / constants.h_eV_sec
    nu_max = Emax / constants.h_eV_sec
    N_mu = NE
    nu = np.logspace(np.log(nu_min), np.log(nu_max), N_mu, base=np.e)

    f_He_bynumb = 1 - parameters.cosmology.HI_frac
    # hydrogen
    nH0 = (1-f_He_bynumb) * nb0
    # helium
    nHe0 = f_He_bynumb * nb0

    M_star_dot = (Ob / Om) * f_star_Halo(parameters, M_accr) * dMdt_accr
    M_star_dot[np.where(M_accr < parameters.source.halo_mass_min)] = 0

    # compute the N prime array before hand
    # TODO what is N prime?
    # for this computation we consider the maximum redshift to be z_star
    z_range = z_star - z_bins
    N_prime = z_range / dz_prime
    # we cast to int later on because this gives the number of points
    N_prime = np.maximum(N_prime, 4).astype(int) # TODO explain why 4 exactly

    rho_xray = np.zeros((len(rr), len(z_bins), parameters.simulation.halo_mass_bin_n, len(parameters.source.mass_accretion_alpha_range)))
    # logger.debug(f"{rho_xray.shape=}")

    for i, z in enumerate(z_bins):
        # it only makes sense to compute the profile for z < zstar
        if z > z_star:
            continue
        
        # lookback redshift
        z_prime = np.logspace(np.log(z), np.log(z_star), N_prime[i], base=np.e)
        rcom_prime = comoving_distance(z_prime, parameters) * h0  # comoving distance

        # TODO: why do we interpolate here if we have the analytical expression?
        if i == 0: # if zz[0]<zstar, then concatenate two numbers..
            # TODO - this is still broken!
            dMdt_int = interp1d(
                x = np.concatenate(([z_star], z_bins[:i+1])),
                y = np.concatenate(
                    np.zeros(len(M_accr[..., 0])),
                    np.array(M_star_dot[..., :i+1]),
                    axis=0
                ),
                axis=-1,
                fill_value='extrapolate'
            )
        else:
            dMdt_int = interp1d(
                x = z_bins[:i + 1],
                y = M_star_dot[..., :i+1],
                axis = -1,
                fill_value = 'extrapolate'
            )

        # import matplotlib.pyplot as plt
        # plt.figure()
        # logger.debug(f"{dMdt_int(z_prime).shape=}, {z_prime.shape=}, {dMdt_int(z_prime).mean()=} {dMdt_int(z_prime).max()=} {dMdt_int(z_prime).min()=}")
        # plt.plot(z_prime, dMdt_int(z_prime)[0, 0, :])
        # plt.plot(z_prime, dMdt_int(z_prime)[1, 1, :])
        # plt.show()

        # as described in the paper, we express the emission of xrays as a function of distance
        # this is precomputed for a range of parameters: alpha, Mh, z
        # the main component of the emission is given by an integral over the frequency
        # to compute the integral we prepend the nu dependence as the first axis of the flux array (flux[nu, r, Mh, alpha])

        def integrand(nu):
            tau_prime = cum_optical_depth(z_prime, nu * constants.h_eV_sec, parameters)
            eps_X = eps_xray(nu[:, None] * (1 + z_prime)[None, :] / (1 + z), parameters)
            # both quantities are functions of the frequency and the redshift (2d)
            # logger.debug(f"{eps_X.shape=}, {tau_prime.shape=}")

            sigma_i_nu = sigma_HI(nu * constants.h_eV_sec)
            # sigma_i_nu is a function of the frequency (1d)

            # final complication - the integrand is expressed in terms of the radial distance
            # we perform a hack to interpret eps(z) as a function of r
            # the mass also had an M0, alpha dependence, so we add a dimension
            integral_factors = (np.exp(tau_prime) * eps_X)[:,  None, None, :] * dMdt_int(z_prime)[None, ...]
            integral_factors_interpolated = interp1d(rcom_prime, integral_factors, axis=-1, fill_value=0.0, bounds_error=False)
            # but r should be the first axis after nu: 0, 1, 2, 3 -> 0, 3, 1, 2
            integral_factors_r = integral_factors_interpolated(rr)
            integral_factors_r = np.swapaxes(np.swapaxes(integral_factors_r, 1, -1), -1, -2)

            # the final integrand is a function of the frequency and the radial distance
            # logger.debug(f"{nu.shape=}, {sigma_i_nu.shape=}, {np.exp(tau_prime).shape=}, {integral_factors_r.shape=}")
            return ((nu * constants.h_eV_sec - E_HI) * sigma_i_nu)[:, None, None, None] * integral_factors_r
        
        prefactor = ((nH0 / nb0) * sigma_HI(nu * h_eV_sec) * (nu * h_eV_sec - E_HI) + (nHe0 / nb0) * sigma_HeI(nu * h_eV_sec) * (nu * h_eV_sec - E_HeI))   # [cm^2 * eV] 4 * np.pi *
        flux = prefactor[:, None, None, None] * integrand(nu)
        integrated_flux = trapezoid(flux, nu, axis=0)
        heat = integrated_flux
        fXh = f_Xh(parameters, xe[i])
        rho = fXh * 1 / (4 * np.pi * (rr/(1+z)) ** 2)[:, None, None] * heat
        # logger.debug(f"{fXh.shape=}, {rr.shape=}, {nu.shape=}, {rho.shape=}")
        rho_xray[:, i, ...] = rho


    return rho_xray




# TODO : never called => delete
def Gamma_ion_xray(param,rr, M_accr, dMdt_accr, zz):
    """
    Parameters
    ----------
    param : dictionary containing all the input parameters
    M_accr :  function of zz and hence should increase. 2d array of shape (zz,M_bin)
    dMdt_accr :  Time derivative of halo mass (MAR). 2d array of shape (zz,M_bin)
    zz : redshift in decreasing order.
    rr : comoving distance from source center [cMpc/h]

    Returns
    ----------
    Two profiles of the X-ray ionisation rate (primary and secondary). This function is not used. It's here in case we need x_e profile (becomes problematic since heat equation loose additivity...)

    -Gamma_ion : Primary ionisation rate from Xray (arXiv:1406.4120, Eq.9,10) -- similar to Gamma_ion in HM code.
    Gamma-HI, ionisation rate due to xray. It's a profile in [s**-1], that we use to compute x_e ,

    -Gamma_sec_ion : Secondary ionisation rate from xray. We assume fXion only depends on xe (astro-ph/0608032, Eq.69)

    Shape is (zz,rr,M_bin) (M_accr, dMdt_accr all have same dimension (zz,M_bin))
    """

    Om = param.cosmo.Om
    Ob = param.cosmo.Ob
    h0 = param.cosmo.h
    zstar = 35
    Emin = param.source.E_min_xray
    Emax = param.source.E_max_xray
    NE = 50

    nb0 = rhoc0 * Ob / (m_p_in_Msun * h0)  # [h/Mpc]^3

    # zprime binning
    dz_prime = 0.1

    # define frequency bin
    nu_min = Emin / h_eV_sec
    nu_max = Emax / h_eV_sec
    N_mu = NE
    nu = np.logspace(np.log(nu_min), np.log(nu_max), N_mu, base=np.e)

    f_He_bynumb = 1 - param.cosmo.HI_frac
    # hydrogen
    nH0 = (1-f_He_bynumb) * nb0
    # helium
    nHe0 = f_He_bynumb * nb0

    Gamma_ion = np.zeros((len(zz), len(rr),len(M_accr[0]))) # primary ion
    Gamma_sec_ion = np.zeros((len(zz), len(rr),len(M_accr[0]))) # secondary ion

    M_star_dot = (Ob / Om) * f_star_Halo(param, M_accr) * dMdt_accr
    M_star_dot[np.where(M_accr<param.source.M_min)]=0


    for i in range(len(zz)):
        if (zz[i] < zstar):
           # rr_comoving = rr * (1 + zz[i])
            z_max = zstar
            zrange = z_max - zz[i]
            N_prime = int(zrange / dz_prime)

            if (N_prime < 4):
                N_prime = 4
            z_prime = np.logspace(np.log(zz[i]), np.log(z_max), N_prime, base=np.e)
            rcom_prime = comoving_distance(z_prime, param) * h0  # comoving distance

            dMdt_int = interp1d(zz[:i+1], M_star_dot[:i+1,:],axis=0, fill_value='extrapolate')

            flux = np.zeros((len(nu), len(rr),len(M_accr[0])))

            for j in range(len(nu)):
                tau_prime = cum_optical_depth(z_prime, nu[j] * h_eV_sec, param)
                eps_X = eps_xray(nu[j] * (1 + z_prime) / (1 + zz[i]), param)[:,None] * np.exp(-tau_prime)[:,None] * dMdt_int(z_prime)  # [1/s/Hz]
                eps_int = interp1d(rcom_prime, eps_X, axis=0, fill_value=0.0, bounds_error=False)
                flux[j, :,:] = np.array(eps_int(rr)) # [1/s/Hz]


            pref_ion = (sigma_HI(h_eV_sec*nu)*nH0+sigma_HeI(h_eV_sec*nu)* nHe0)/ nb0
            pref_sec_ion = (sigma_HI(h_eV_sec*nu)*nH0*(h_eV_sec*nu-E_HI)/E_HI+sigma_HeI(h_eV_sec*nu)* nHe0*(h_eV_sec*nu-E_HeI)/E_HeI)/ nb0
            integrated_flux         = trapezoid(flux * pref_ion[:,None,None], nu, axis=0)# [cm**2/s]
            integrated_flux_sec_ion = trapezoid(flux*pref_sec_ion[:,None,None], nu, axis=0)# [cm**2/s]


            Gamma_ion[i, :,:] = integrated_flux / (4 * np.pi * (rr/(1+zz[i])) ** 2)[:,None] / (cm_per_Mpc/h0) ** 2  # [1/s] ionisation rate due to Xray
            Gamma_sec_ion[i,:,:] = integrated_flux_sec_ion / (4 * np.pi * (rr/(1+zz[i])) ** 2)[:,None] / (cm_per_Mpc/h0) ** 2  # [1/s] secondary ionisation rate

    return Gamma_ion, Gamma_sec_ion


def mean_gamma_ion_xray(parameters: Parameters, sfrd, zz):
    """
    Parameters
    ----------
    param : dictionary containing all the input parameters
    sfrd : Star formation rate density, in Msol/h/yr/(Mpc/h)^3. Shape is len(zz)
    zz : redshift in decreasing order.

    Returns
    ----------
    Mean X ray ionisation rate (primary and secondary). Used to compute the mean x_e, to then compute f_Xh, the fraction of energy deposited as heat by electrons in the neutral medium.
    Shape is (2,zz)

    -Gamma_ion : Primary ionisation rate from Xray (arXiv:1406.4120, Eq.9,10) -- similar to Gamma_ion in HM code.
    -Gamma_sec_ion : Secondary ionisation rate from xray. We assume fXion only depends on xe (astro-ph/0608032, Eq.69)
    """

    Ob = parameters.cosmology.Ob
    h0 = parameters.cosmology.h
    zstar = 35
    Emin = parameters.source.energy_cutoff_min_xray
    Emax = parameters.source.energy_cutoff_max_xray
    NE = 50

    nb0 = rhoc0 * Ob / (m_p_in_Msun * h0)  # [h/Mpc]^3

    # zprime binning
    dz_prime = 0.1

    # define frequency bin
    nu_min = Emin / h_eV_sec
    nu_max = Emax / h_eV_sec
    N_mu = NE
    nu = np.logspace(np.log(nu_min), np.log(nu_max), N_mu, base=np.e)

    f_He_bynumb = 1 - parameters.cosmology.HI_frac
    # hydrogen
    nH0 = (1 - f_He_bynumb) * nb0
    # helium
    nHe0 = f_He_bynumb * nb0

    Gamma_ion = np.zeros(len(zz))  # primary ion
    Gamma_sec_ion = np.zeros(len(zz))  # secondary ion

    sfrd_interp = interp1d(zz, sfrd, fill_value='extrapolate')

    for i in range(len(zz)):
        J_X_nu_z = np.zeros(len(nu))
        if (zz[i] < zstar):
            # rr_comoving = rr * (1 + zz[i])
            z_max = zstar
            zrange = z_max - zz[i]
            N_prime = int(zrange / dz_prime)

            if (N_prime < 4):
                N_prime = 4
            z_prime = np.logspace(np.log(zz[i]), np.log(z_max), N_prime, base=np.e)

            for j in range(len(nu)):
                tau_prime = cum_optical_depth(z_prime, nu[j] * h_eV_sec, parameters)
                eps_X = eps_xray(nu[j] * (1 + z_prime) / (1 + zz[i]), parameters)  * sfrd_interp(z_prime)  # [1/s/Hz/(Mpc/h)^3]
                itd = c_km_s * h0 / hubble(z_prime, parameters) * eps_X * np.exp(-tau_prime)

                J_X_nu_z[j] = (1 + zz[i]) ** 2 / (4 * np.pi) * trapezoid(itd, z_prime) * (h0/cm_per_Mpc)**2       # [1/s/Hz * (1/cm)^2]


        itlH = nH0 * sigma_HI(nu * h_eV_sec) * J_X_nu_z
        itlHe = nHe0 * sigma_HeI(nu * h_eV_sec) * J_X_nu_z
        Gamma_ion[i] = 4*np.pi * trapezoid((itlH+itlHe),nu) / nb0  # s^-1

        itlH = itlH * (nu * h_eV_sec - E_HI) / E_HI
        itlHe = itlHe * (nu * h_eV_sec - E_HeI) / E_HeI
        Gamma_sec_ion[i] = 4 * np.pi * trapezoid((itlH + itlHe), nu) / nb0  # [1/s]

    return Gamma_ion, Gamma_sec_ion


def T_gas_fit(zz):
    """
    Aproximative fit for gas temperature
    (see Eq.2 in arXiv:1005.2416)
    """
    a = 1/(1+zz)
    a1 = 1.0/119.0
    a2 = 1.0/115.0
    Tgas = Tcmb0/a/(1.0+(a/a1)/(1.0+(a2/a)**(3.0/2.0)))
    return Tgas

# TODO: never called
def rho_x_e(param,rr,G_ion,G_sec_ion,zz):
    """
    Compute the profile of  x_e : free electron fraction of (largely) neutral IGM
    See  arXiv:1406.4120 (eq. 12, 13)

    Parameters
    ----------
    zz : redshift array in decreasing order.

    G_ion,G_sec_ion : Energy deposition from first and second ionisation. Output of Gamma_ion_xray.
    """
    h0 = param.cosmo.h
    Ob = param.cosmo.Ob
    f_He_bynumb = 1 - param.cosmo.HI_frac

    aa = list((1 / (1 + zz)))

    nb0 = rhoc0 * Ob / (m_p_in_Msun * h0)  # [h/Mpc]^3

    # fit from astro-ph/9909275
    tt = T_gas_fit(zz) / 1e4
    alB = 1.14e-19 * 4.309 * tt ** (-0.6166) / (1 + 0.6703 * tt ** 0.53) * 1e6  # [cm^3/s]
    alB = alB * (h0 / cm_per_Mpc) ** 3
    alB_tck = splrep(aa, alB)
    alphaB = lambda a: splev(a, alB_tck)

    x_e_profile = np.zeros((len(zz), len(rr), len(G_ion[0, 0, :]))) ## (zz,rr,Mh_bin)
    for j in range(len(rr)):
        print(j)
        # Energy deposition from first ionisation, see astro-ph/060723 (Eq.12) or 1509.07868 (Eq.3)
        Gamma_HI = interp1d(aa, G_ion[:, j, :], axis=0, fill_value="extrapolate")
        fXion = lambda xe: (1 - xe) / 2.5  # approx from Fig.4 of 0910.4410

        G_sec_ion_tck = interp1d(aa, G_sec_ion[:, j, :], axis=0, fill_value="extrapolate")
        gamma_HI = lambda a, xe: G_sec_ion_tck(a) * fXion(xe)

        nH = lambda a: (1 - f_He_bynumb) * nb0 / a ** 3

        # x_e
        source = lambda xe, a: (Gamma_HI(a) + gamma_HI(a, xe)) * (1 - xe) / (a * hubble(1 / a - 1, param) / km_per_Mpc) - \
                           alphaB(a) * nH(a) * xe ** 2 / (a * hubble(1 / a - 1, param) / km_per_Mpc)

        x_e = odeint(source, np.full(len(G_ion[0,0,:]),0), aa)

        x_e_profile[:,j,:] = x_e

    return x_e_profile


def solve_xe(parameters: Parameters, mean_G_ion, mean_Gsec_ion, zz):
    """
    Parameters
    ----------
    param : dictionary containing all the input parameters
    mean_G_ion,mean_Gsec_ion : output of Mean_Gamma_ion_xray
    zz : redshift in decreasing order.

    Returns
    ----------
    Mean free electron fraction in the neutral medium. We use it to compute the fraction of energy deposited as heat by e- originating from ionisation by xray: fXh = xe**0.225
    """
    print('Computing x_e(z) from the sfrd, including first and secondary ionisations....')
    h0 = parameters.cosmology.h
    Ob = parameters.cosmology.Ob
    f_He_bynumb = 1 - parameters.cosmology.HI_frac

    xe0 = 2e-4
    aa = list((1 / (1 + zz)))

    nb0 = rhoc0 * Ob / (m_p_in_Msun * h0)  # [h/Mpc]^3

    # fit from astro-ph/9909275
    tt = T_gas_fit(zz) / 1e4
    alB = 1.14e-19 * 4.309 * tt ** (-0.6166) / (1 + 0.6703 * tt ** 0.53) * 1e6  # [cm^3/s]
    alB = alB * (h0 / cm_per_Mpc) ** 3
    alB_tck = splrep(aa, alB)
    alphaB = lambda a: splev(a, alB_tck)

    # Energy deposition from first ionisation, see astro-ph/060723 (Eq.12) or 1509.07868 (Eq.3)
   # Gamma_HI = np.interp(zz, aa, mean_G_ion, right=0)
   # G_sec_ion_tck = np.interp(zz, aa, mean_G_ion, right=0)

    fXion = lambda xe: (1 - xe) / 2.5  # approx from Fig.4 of 0910.4410
    gamma_HI = lambda a, xe: np.interp(a, aa, mean_Gsec_ion, right=0) * fXion(xe)
    nH = lambda a: (1 - f_He_bynumb) * nb0 / a ** 3

    # x_e
    source = lambda xe, a: (np.interp(a, aa, mean_G_ion, right=0)  + gamma_HI(a, xe)) * (1 - xe) / (a * hubble(1 / a - 1, parameters) / km_per_Mpc) - \
                           alphaB(a) * nH(a) * xe ** 2 / (a * hubble(1 / a - 1, parameters) / km_per_Mpc)

    x_e = odeint(source,xe0, aa)
    print('.....done computing x_e(z).')
    return x_e


def rho_heat(parameters: Parameters, zz, rr, rho_xray):
    """
    Parameters
    ----------
    parameters : dictionary containing all the input parameters
    rho_xray :  output of rho_xray.
    zz : redshift in decreasing order.
    rr : comoving distance from source center [cMpc/h]

    Returns
    ----------
    Solve the temperature equation, to go from a heating rate to a Temperature in [K].
    Array of shape (zz,rr, M_bin)
    We assume 0K initial conditions (background adiabatic temperature is added afterward at the map level.)
    """

    # decoupling redshift as ic
    z0 = parameters.cosmology.z_decoupling
    zz = np.concatenate((np.array([z0]),zz))
    # scale factor
    aa = 1 / (1 + zz)
    # prepend 0 to the rho_xray array to account for the additional z bin
    # logger.debug(f"{rho_xray.shape=}, {np.zeros_like(rho_xray[0, ...]).shape=}")
    rho_xray = np.concatenate((np.zeros_like(rho_xray[:, 0, ...])[:, None, ...], rho_xray), axis=1)


    # perturbations
    rho_heat = np.zeros((len(rr), len(zz), parameters.simulation.halo_mass_bin_n, len(parameters.source.mass_accretion_alpha_range)))
    # logger.debug(f"{rho_xray.shape=}, {rho_heat.shape=}")

    # the shape of the xray profile at a given redshift is:
    # (rr, M_bin, alpha_bin)
    single_rho_xray_shape = rho_xray[:, 0, ...].shape

    # allow us to query the xray profile at any scale factor (<-> redshift)
    rho_interpolated = interp1d(aa, rho_xray, axis=1, fill_value="extrapolate")

    def right_hand_side(y, a):
        # since odeint works with 1d arrays we have a flattened version currently
        gamma_heat = 2 * rho_interpolated(a) / (3 * kb_eV_per_K * a * hubble(1 / a - 1, parameters)) * km_per_Mpc  #rho_intp(a)
        return gamma_heat.flatten() - 2 * y / a

    y0 = np.zeros(single_rho_xray_shape)
    source_in_time = odeint(right_hand_side, y0.flatten(), aa)
    # don't keep the initial condition
    rho_heat = source_in_time[1:, ...].reshape((*single_rho_xray_shape, aa.size - 1))
    # logger.debug(f"{rho_heat.shape=}")

    return rho_heat





def cum_optical_depth(zz, E, parameters: Parameters):
    """
    Cumulative optical optical depth of array zz.
    See e.g. Eq. 6 of 1406.4120

    We use it for the xray heating and xray ion rate calculations.
    """
    Ob = parameters.cosmology.Ob
    h0 = parameters.cosmology.h

    # Energy of a photon observed at (zz[0], E) and emitted at zz
    if type(E) == np.ndarray:
        Erest = np.outer(E,(1 + zz)/(1 + zz[0]))
    else:
        Erest = E * (1 + zz)/(1 + zz[0])

    #hydrogen and helium cross sections
    sHI   = sigma_HI(Erest)*(h0/cm_per_Mpc)**2   #[Mpc/h]^2
    sHeI  = sigma_HeI(Erest)*(h0/cm_per_Mpc)**2  #[Mpc/h]^2



    nb0   = rhoc0*Ob/(m_p_in_Msun*h0)                    # [h/Mpc]^3

    f_He_bynumb = 1 - parameters.cosmology.HI_frac

    #H and He abundances
    nHI   = (1-f_He_bynumb)*nb0 *(1+zz)**3       # [h/Mpc]^3
    nHeI  = f_He_bynumb * nb0 *(1+zz)**3

    #proper line element
    dldz = c_km_s*h0/hubble(zz, parameters)/(1+zz) # [Mpc/h]

    #integrate
    tau_int = dldz * (nHI*sHI + nHeI*sHeI)

    if type(E) == np.ndarray:
        tau = cumulative_trapezoid(tau_int,x=zz,axis=1,initial=0.0)
    else:
        tau = cumulative_trapezoid(tau_int,x=zz,initial=0.0)

    return tau



def rho_alpha_profile(parameters: Parameters, r_grid: np.ndarray, halo_mass: np.ndarray, halo_mass_derivative: np.ndarray, z_bins: np.ndarray):
    """
    Ly-al coupling profile
    of shape (r_grid)
    - r_grid : physical distance around halo center in [pMpc/h]
    - zz  : redshift
    - MM  : halo mass

    Return rho_alpha : shape is (zz,rr,MM). Units : [pcm-2.s-1.Hz-1]
    """
    # TODO: remove hardcoded values
    z_star = 35
    h0 = parameters.cosmology.h
    rectrunc = 23

    # rec fraction
    names = 'n, f'
    path_to_file = Path(importlib.util.find_spec('beorn').origin).parent / 'input_data' / 'recfrac.dat'
    rec = np.genfromtxt(path_to_file, usecols=(0, 1), comments='#', dtype=float, names=names)

    # line frequencies
    nu_n = nu_LL * (1 - 1 / rec['n'][2:] ** 2)
    nu_n = np.insert(nu_n, [0, 0], np.inf)

    rho_alpha = np.zeros((len(r_grid), parameters.simulation.halo_mass_bin_n, len(parameters.source.mass_accretion_alpha_range), len(z_bins)))

    # compute the N prime array before hand
    # TODO what is N prime?
    # for this computation we consider the maximum redshift to be z_star
    dz_prime = 0.01
    logger.debug(f"Computing rho_alpha_profile for {len(z_bins)} redshifts")
    for i, z in enumerate(z_bins):
        # TODO - handle the cutoff more efficiently
        if z > z_star:
            continue

        flux = []
        # TODO - handle this in a vectorized way
        for k in range(2, rectrunc):
            z_max = (1 - (rec['n'][k] + 1) ** (-2)) / (1 - (rec['n'][k]) ** (-2)) * (1 + z) - 1
            z_range = np.minimum(z_max, z_star) - z
            N_prime = z_range / dz_prime
            # we cast to int later on because this gives the number of points
            N_prime = np.maximum(N_prime, 4).astype(int) # TODO explain why 4 exactly
            # logger.debug(f"{N_prime=}")
            # the lookback time to TODO what?
            z_prime = np.logspace(np.log(z), np.log(z_max), N_prime, base=np.e)
            rcom_prime = comoving_distance(z_prime, parameters) * h0  # comoving distance in [cMpc/h]

            # since we require slightly altered valus of the halo mass at z' (instead of z) we interpolate the values
            halo_mass_interpolated = interp1d(z_bins[:i + 1], halo_mass[..., :i+1], axis=-1, fill_value='extrapolate')
            halo_mass_derivative_interpolated = interp1d(z_bins[:i + 1], halo_mass_derivative[..., :i+1], axis=-1, fill_value='extrapolate')
            # What follows is the emissivity of the source at z_prime (such that at z the photon is at rcom_prime)
            # We then interpolate to find the correct emissivity such that the photon is at r_grid*(1+z) (in comoving unit)

            dMdt_star = halo_mass_derivative[..., :i+1] * f_star_Halo(parameters, halo_mass[..., :i+1]) * parameters.cosmology.Ob / parameters.cosmology.Om  # SFR Msol/h/yr
            # logger.debug(f"{dMdt_star.shape=}")
            dMdt_star_int = interp1d(
                # TODO - original code prepends a value of zero at zstar
                z_bins[:i + 1],
                dMdt_star[..., :i+1],
                axis=-1,
                fill_value='extrapolate'
            )

            eps_al = eps_lyal(nu_n[k] * (1 + z_prime) / (1 + z), parameters) * dMdt_star_int(z_prime)  # [photons.yr-1.Hz-1]
            # we are now using the lookback time interchangebly as travel distance r (comoving)
            # this should be the first axis 0 1 2 -> 2 0 1
            eps_al = np.swapaxes(np.swapaxes(eps_al, 0, 2), 1, 2)
            eps_int = interp1d(rcom_prime, eps_al, axis=0, fill_value=0.0, bounds_error=False)

            flux_m = eps_int(r_grid * (1 + z)) * rec['f'][ k]  # want to find the z' corresponding to comoving distance r_grid * (1 + z).
            flux += [np.array(flux_m)]
        logger.debug(f"{len(flux)=}")
        flux = np.array(flux)
        logger.debug(f"{flux.shape=}")
        flux_of_r = np.sum(flux, axis=0)  # shape is (r_grid, Mbin, alpha_bin)
        logger.debug(f"{flux_of_r.shape=}, {r_grid.shape=}")
        rho_alpha_ = flux_of_r / (4 * np.pi * r_grid ** 2)[:, None, None]  ## physical flux in [(pMpc/h)-2.yr-1.Hz-1]
        logger.debug(f"{rho_alpha_.shape=}")

        rho_alpha[..., i] = rho_alpha_ * (h0 / cm_per_Mpc) ** 2 / sec_per_year  # [pcm-2.s-1.Hz-1]

    return rho_alpha