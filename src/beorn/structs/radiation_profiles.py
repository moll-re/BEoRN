from .base_struct import BaseStruct
from .parameters import Parameters
import numpy as np
from dataclasses import dataclass


@dataclass(slots = True)
class RadiationProfiles(BaseStruct):
    # redshift range for which the profiles have been computed
    # TODO - same as parameters.?.Nz
    z_history: np.ndarray
    # evolution of halo mass with dependence on final mass, redshift and accretion rate alpha
    Mh_history: np.ndarray
    dMh_dt: np.ndarray
    # the core profiles:
    # - radiation volume/radius
    # - heating
    # - lyman alpha photon spreading
    rho_xray: np.ndarray
    # heating depends on xray profile
    rho_heat: np.ndarray
    rho_alpha: np.ndarray
    R_bubble: np.ndarray
    r_lyal: np.ndarray
    
    # radial component of the profiles
    r_grid_cell: np.ndarray

    def profiles_of_halo_bin(self, z_index: int, alpha_index: slice, mass_index:slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the radiation profile for a given halo bin (z, alpha, mass).
        """
        return(
            self.R_bubble[mass_index, alpha_index, z_index],
            self.rho_alpha[:, mass_index, alpha_index, z_index],
            self.rho_heat[:, mass_index, alpha_index, z_index]
        )

    #TODO
    def Ngdot_ion(self, parameters: Parameters) -> np.ndarray:
        """
        Number of ionizing photons emitted per sec [s**-1].
        """
        Ob, Om, h0 = parameters.cosmology.Ob, parameters.cosmology.Om, parameters.cosmology.h

        if (parameters.source.source_type == 'SED'):
            Ngam_dot_ion = self.dMh_dt / h0 * f_star_Halo(parameters, Mh) * Ob / Om * f_esc(parameters, Mh) * parameters.source.Nion / sec_per_year / m_H * M_sun
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
            # TODO - this has been broken for a while
            return Mhalo / h0 * Ob / Om / (10 * 1e6 * sec_per_year) / m_p_in_Msun
        else:
            raise ValueError('Source Type not available. Should be SED, Ross, Ghara or constant.')
