import time
import logging
from itertools import product
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, wait
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

global_logger = logging.getLogger(__name__)

from ..cosmo import T_adiab_fluctu, dTb_factor, dTb_fct
from ..couplings import x_coll, S_alpha
from ..io.handler import Handler
from ..io.load import load_delta_b
from ..structs.radiation_profiles import RadiationProfiles
from ..structs.parameters import Parameters
from ..structs.snapshot_profiles import GridData
from ..structs.global_profiles import GridDataMultiZ
from ..structs.halo_catalog import HaloCatalog
from .. import constants
from ..profiles_on_grid import average_profile, cumulated_number_halos, profile_to_3Dkernel, put_profiles_group, stacked_lyal_kernel, stacked_T_kernel, spreading_excess_fast


class Painter:
    """
    The Painter class responsible for applying the 1D profiles onto the DM haloes and create 3D maps and the quantities associated with them.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, parameters: Parameters, cache_handler: Handler = None, output_handler: Handler = None):
        """
        Initialize the Painter class with the given parameters.

        Args:
            parameters (Parameters): The parameters object containing cosmological and simulation parameters.
            cache_handler (Handler, optional): The handler for loading and saving cache data. Defaults to None.
            output_handler (Handler, optional): The handler for saving the painted output data. Defaults to None.
        """
        self.parameters = parameters
        self.output_handler = output_handler
        self.cache_handler = cache_handler

    
    def paint_full(self, radiation_profiles: RadiationProfiles) -> GridDataMultiZ:
        """Starts the painting process for the full redshift range."""
        multi_z_data = GridDataMultiZ()

        self.logger.info(f"Painting profiles onto grid for {radiation_profiles.z_history.size} redshift snapshots.")
        # TODO - this loop could be parallelized using MPI
        for ii, z in enumerate(radiation_profiles.z_history):
            # TODO - griddata at indivdual snapshots is not really necessary anymore
            try:
                GridData.read(directory=self.cache_handler.file_root, parameters=self.parameters, z=z)
                self.logger.debug(f"Found painted output in cache for {z=}. Skipping.")
                continue
            except FileNotFoundError:
                self.logger.debug("Painted output not found in cache. Processing now")

            # there is no cache or the cache does not contain the halo catalog - compute it fresh
            grid_data = self.paint_single(z_index=ii, grid_model=radiation_profiles)

            grid_data.write(directory=self.cache_handler.file_root, parameters=self.parameters, z=z)
            multi_z_data.append(grid_data, directory=self.output_handler.file_root, parameters=self.parameters)


        # reinitialize the grid data to create the attributes that are mapped from the hdf5 fields
        del multi_z_data
        multi_z_data = GridDataMultiZ.read(directory=self.output_handler.file_root, parameters=self.parameters)
        return multi_z_data


    def paint_single(self, z_index: int, grid_model: RadiationProfiles) -> GridData:
        """Paints the halo properties for a single redshift."""

        start_time = time.process_time()
        zero_grid = np.zeros((self.parameters.simulation.Ncell, self.parameters.simulation.Ncell, self.parameters.simulation.Ncell))

        # TODO - these should be a parameter method
        halo_catalog = HaloCatalog.load(self.parameters.simulation.halo_catalogs[z_index], self.parameters)
        delta_b = load_delta_b(self.parameters, z_index)

        # find matching redshift between solver output and simulation snapshot.
        # this will raise an error if the needed profiles are not available
        # The loading is left in this function to allow for the possibility of parallelizing the painting
        zgrid = grid_model.z_history[z_index]
        mass_range = grid_model.Mh_history[..., z_index]

        # TODO - what exactly is coef
        coef = constants.rhoc0 * self.parameters.cosmology.h ** 2 * self.parameters.cosmology.Ob * (1 + zgrid) ** 3 * constants.M_sun / constants.cm_per_Mpc ** 3 / constants.m_H

        quantity = halo_catalog.M
        self.logger.debug(f"halo_catalog.M: {quantity.mean()=:.2e} {quantity.std()=:.2e} {quantity.min()=:.2e} {quantity.max()=:.2e}")
        # since we want to paint the halo profiles in grouped mass bins, we need to know which halos are in which mass bin
        # but there are a few short-circuits:
        # 1. if there are no halos at all -> skip the painting
        # 2. if there are halos but they lie outside the mass range -> raise an error

        if halo_catalog.M.max() > mass_range.max() or halo_catalog.M.min() < mass_range.min():
            raise RuntimeError(f"The current halo catalog at z={zgrid} has a higher masse range than the mass range of the precomputed profiles. You need to adjust your parameters: either increase the mass range of the profile simulation (parameters.simulation) or decrease the mass range of star forming halos (parameters.source).")

        self.logger.info(f'Painting {halo_catalog.M.size} halos at zgrid={zgrid:.2f}.')
        
        # Shortcut if there are no halos
        if halo_catalog.M.size == 0:
            self.logger.debug("No halos to paint. Returning empty grid.")
            factor = dTb_factor(self.parameters)

            Grid_Temp = T_adiab_fluctu(zgrid, self.parameters, delta_b)
            Grid_xHII = zero_grid

            Grid_xal = zero_grid
            Grid_xcoll = x_coll(z=zgrid, Tk=Grid_Temp, xHI=(1 - Grid_xHII), rho_b=(delta_b + 1) * coef)
            Grid_dTb = factor * np.sqrt(1 + zgrid) * (1 - constants.Tcmb0 * (1 + zgrid) / Grid_Temp) * (1 - Grid_xHII) * (delta_b + 1) * Grid_xcoll / (1 + Grid_xcoll)
            Grid_dTb_no_reio = factor * np.sqrt(1 + zgrid) * (1 - constants.Tcmb0 * (1 + zgrid) / Grid_Temp) * (delta_b + 1) * Grid_xcoll / (1 + Grid_xcoll)
            Grid_dTb_RSD = zero_grid
            Grid_dTb_T_sat = factor * np.sqrt(1 + zgrid) * (1 - Grid_xHII) * (delta_b + 1) * Grid_xcoll / (1 + Grid_xcoll)
            xcoll_mean = np.mean(Grid_xcoll)
            
            return GridData(
                z = zgrid,
                delta_b = delta_b,
                Grid_Temp = Grid_Temp,
                Grid_dTb = Grid_dTb,
                Grid_dTb_no_reio = Grid_dTb_no_reio,
                Grid_dTb_T_sat = Grid_dTb_T_sat,
                Grid_dTb_RSD = Grid_dTb_RSD,
                Grid_xHII = Grid_xHII,
                Grid_xal = Grid_xal,
                # TODO - what value for xtot?
                Grid_xtot = Grid_xcoll,
            )




        # TODO - i think this speedup is negligible
        # Ionized_vol = xHII_approx(
        #     self.parameters,
        #     grid_model.Mh_history[..., 5, z_index],
        #     grid_model.R_bubble[..., 5, z_index],
        #     halo_catalog
        #     )
        # if Ionized_vol > 1:
        #     self.logger.info('Universe is fully ionized. Returning [1] for xHII, T and [0] for dTb.')
        #     Grid_xHII = zero_grid + 1
        #     Grid_Temp = zero_grid + 1
        #     Grid_dTb = zero_grid
        #     Grid_dTb_no_reio = zero_grid
        #     Grid_dTb_T_sat = zero_grid
        #     Grid_xal = zero_grid
        #     return GridData(
        #         parameters = self.parameters,
        #         delta_b = delta_b,
        #         Grid_Temp = Grid_Temp,
        #         Grid_dTb = Grid_dTb,
        #         Grid_dTb_no_reio = Grid_dTb_no_reio,
        #         Grid_dTb_T_sat = Grid_dTb_T_sat,
        #         # TODO - what should the value be?
        #         Grid_dTb_RSD = zero_grid,
        #         Grid_xHII = Grid_xHII,
        #         Grid_xal = Grid_xal,
        #         # TODO - what value for xtot?
        #         Grid_xtot = zero_grid
        #     )


        # initialise the "main" grids here. Since they will be filled in place by multiple parallel processes, we need to use shared memory
        # get the size of the grids
        size = zero_grid.size * np.dtype(np.float64).itemsize
        if "bubbles" in self.parameters.simulation.store_grids:
            buffer_xHII = shared_memory.SharedMemory(create=True, size=size)
        else:
            buffer_xHII = None

        if "Tk" in self.parameters.simulation.store_grids:
            buffer_Temp = shared_memory.SharedMemory(create=True, size=size)
        else:
            buffer_Temp = None

        if "lyal" in self.parameters.simulation.store_grids:
            buffer_xal = shared_memory.SharedMemory(create=True, size=size)
        else:
            buffer_xal = None

        with ProcessPoolExecutor(max_workers=self.parameters.simulation.cores) as executor:
            # if only one process is used, we won't make use of the executor, but it allows us to keep the code more concise
            self.logger.info(f"Using {self.parameters.simulation.cores} processes for painting.")

            futures = []
            total_halos = 0

            ## iterate over the range of mass and alpha bins that the profiles are available for
            # the alpha bins are constant so we can use the ones from the parameters
            # the mass bins are more tricky - they follow the mass accretion history i.e. they shift with each redshift step
            alpha_indices = range(len(self.parameters.source.mass_accretion_alpha_range) - 1)
            mass_indices = range(len(self.parameters.simulation.halo_mass_bins) - 1)

            # now each profile was computed for a precise mass/alpha value that we set to be the center points of the bins
            # => in the actual profile the shape is (l-1)x(m-1)x(n-1) where l,m,n are the number of bins in mass, alpha and redshift
            # For each profile we have a range of mass and alpha values where we can pick haloes from
            # We just need to ensure that all haloes are considered in the end (hence the total_halos check)
            
            for alpha_index, mass_index in product(alpha_indices, mass_indices):
                loop_alpha_range = [self.parameters.source.mass_accretion_alpha_range[alpha_index], self.parameters.source.mass_accretion_alpha_range[alpha_index + 1]]
                # as mentioned above, the mass range is not constant but depends on the redshift
                m_range_base = [self.parameters.simulation.halo_mass_bins[mass_index], self.parameters.simulation.halo_mass_bins[mass_index + 1]]
                loop_mass_range = np.array(m_range_base) * np.exp(self.parameters.source.mass_accretion_alpha_range[alpha_index] * (self.parameters.solver.Nz.min() - zgrid))

                halo_indices = halo_catalog.get_halo_indices(loop_alpha_range, loop_mass_range)
                total_halos += halo_indices.size

                # yet another shortcut: don't copy any memory if there are no halos to begin with
                if halo_indices.size == 0:
                    continue

                kwargs = {
                    "radiation_profiles": grid_model,
                    "halo_catalog": halo_catalog.at_indices(halo_indices),
                    "z_index": z_index,
                    "alpha_index": alpha_index,
                    "mass_index": mass_index,
                    "buffer_lyal": buffer_xal,
                    "buffer_temp": buffer_Temp,
                    "buffer_xHII": buffer_xHII
                }
                
                if self.parameters.simulation.cores > 1:
                    # use the multiprocessing approach and submit the task to the executor
                    f = executor.submit(
                        self.paint_single_mass_bin,
                        **kwargs
                    )
                    futures.append(f)
                else:
                    # use the single process approach and call the function directly
                    self.paint_single_mass_bin(**kwargs)

            # wait for all futures to complete
            completed, uncompleted = wait(futures)
            assert len(uncompleted) == 0, "Not all painting subprocesses completed successfully"
            assert total_halos == halo_catalog.M.size, f"Total painted halos {total_halos} do not match the halo catalog size {halo_catalog.M.size}. This is a bug."

        # clean up the shared memory buffers - but keep the data in the buffer
        buffer_array = np.ndarray(zero_grid.shape, dtype=np.float64, buffer=buffer_xHII.buf)
        Grid_xHII = buffer_array.copy()
        buffer_array = np.ndarray(zero_grid.shape, dtype=np.float64, buffer=buffer_Temp.buf)
        Grid_Temp = buffer_array.copy()
        buffer_array = np.ndarray(zero_grid.shape, dtype=np.float64, buffer=buffer_xal.buf)
        Grid_xal = buffer_array.copy()

        quantity = Grid_xHII
        self.logger.debug(f"Grid_xHII: {quantity.mean()=:.2e} {quantity.std()=:.2e} {quantity.min()=:.2e} {quantity.max()=:.2e}")
        quantity = Grid_Temp
        self.logger.debug(f"Grid_Temp: {quantity.mean()=:.2e} {quantity.std()=:.2e} {quantity.min()=:.2e} {quantity.max()=:.2e}")
        quantity = Grid_xal
        self.logger.debug(f"Grid_xal: {quantity.mean()=:.2e} {quantity.std()=:.2e} {quantity.min()=:.2e} {quantity.max()=:.2e}")


        buffer_xHII.close()
        buffer_xHII.unlink()
        buffer_Temp.close()
        buffer_Temp.unlink()
        buffer_xal.close()
        buffer_xal.unlink()

        self.logger.info('Profile painting done. Redistributing excess photons from the overlapping regions.')
        start_time = time.process_time()
        if np.sum(Grid_xHII) < self.parameters.simulation.Ncell ** 3:
            Grid_xHII = spreading_excess_fast(self.parameters, Grid_xHII)
        else:
            Grid_xHII = zero_grid + 1

        self.logger.info(f'Overlap processing done. Took {time.process_time() - start_time:.2} seconds.')



        ## Post processing on the already filled grids
        # take into account the background temperature
        Grid_Temp += T_adiab_fluctu(zgrid, self.parameters, delta_b)

        # Enforce a minimum ionization fraction
        # TODO is this necessary?
        Grid_xHII[Grid_xHII < self.parameters.source.min_xHII_value] = self.parameters.source.min_xHII_value

        # TODO
        S_al = True
        if S_al:
            self.logger.debug('Including Salpha fluctuations in dTb')
            Grid_xal *= S_alpha(zgrid, Grid_Temp, 1 - Grid_xHII) / (4 * np.pi)
            # We divide by 4pi to go to sr**-1 units
        else:
            self.logger.debug('NOT including Salpha fluctuations in dTb')
            Grid_xal *= S_alpha(zgrid, np.mean(Grid_Temp), 1 - np.mean(Grid_xHII)) / (4 * np.pi)

        #TODO
        xcoll = True
        if xcoll:
            self.logger.debug('Including xcoll fluctuations in dTb')
            Grid_xcoll = x_coll(z = zgrid, Tk = Grid_Temp, xHI = (1 - Grid_xHII), rho_b = (delta_b + 1) * coef)
            xcoll_mean = np.mean(Grid_xcoll)
            Grid_xtot = Grid_xcoll + Grid_xal
        else:
            self.logger.debug('NOT including xcoll fluctuations in dTb')
            xcoll_mean = x_coll(z = zgrid, Tk = np.mean(Grid_Temp), xHI = (1 - np.mean(Grid_xHII)), rho_b = coef)
            Grid_xtot = Grid_xal + xcoll_mean


        # if Rsmoothing > 0:
        #     self.logger.info(f'Smoothing the fields with {Rsmoothing=}')
        #     Grid_xal = smooth_field(Grid_xal, Rsmoothing, LBox, nGrid)
        #     Grid_Temp = smooth_field(Grid_Temp, Rsmoothing, LBox, nGrid)
        #     #Grid_xHII = smooth_field(Grid_xHII, Rsmoothing, LBox, nGrid)
        #     #delta_b   = smooth_field(delta_b, Rsmoothing, LBox, nGrid)
        #     # TODO - why are the other fields not smoothed?


        if "dTb" in self.parameters.simulation.store_grids:
            Grid_dTb = dTb_fct(z=zgrid, Tk=Grid_Temp, xtot=Grid_xtot, delta_b=delta_b, x_HII=Grid_xHII, parameters=self.parameters)
            Grid_dTb_no_reio = dTb_fct(z=zgrid, Tk=Grid_Temp, xtot=Grid_xtot, delta_b=delta_b, x_HII=np.array([0]), parameters=self.parameters)
            Grid_dTb_T_sat = dTb_fct(z=zgrid, Tk=1e50, xtot = 1e50, delta_b=delta_b, x_HII=Grid_xHII, parameters=self.parameters)
        else:
            Grid_dTb = zero_grid
            Grid_dTb_no_reio = zero_grid
            Grid_dTb_T_sat = zero_grid

        return GridData(
            z = zgrid,
            delta_b = delta_b,
            Grid_Temp = Grid_Temp,
            Grid_dTb = Grid_dTb,
            Grid_dTb_no_reio = Grid_dTb_no_reio,
            Grid_dTb_T_sat = Grid_dTb_T_sat,
            Grid_dTb_RSD = zero_grid,
            Grid_xHII = Grid_xHII,
            Grid_xal = Grid_xal,
            Grid_xtot = zero_grid,
            # Grid_xtot = Grid_xtot,
        )




    def paint_single_mass_bin(
        self,
        radiation_profiles: RadiationProfiles,
        halo_catalog: HaloCatalog,
        z_index: int,
        alpha_index: slice,
        mass_index: slice,

        buffer_lyal: np.ndarray = None,
        buffer_temp: np.ndarray = None,
        buffer_xHII: np.ndarray = None,
    ):
        self.logger.info("Called")
        zgrid = radiation_profiles.z_history[z_index]
        radial_grid = radiation_profiles.r_grid_cell / (1 + zgrid)  # pMpc/h
        nGrid = self.parameters.simulation.Ncell
        output_shape = (nGrid, nGrid, nGrid)
        LBox = self.parameters.simulation.Lbox
        # TODO
        # truncate = self.parameters.simulation.truncate_radius
        truncate = False
        cic = False

        R_bubble, rho_alpha_, Temp_profile = radiation_profiles.profiles_of_halo_bin(z_index, alpha_index, mass_index)

        # This is the position of halos in base "nGrid". We use this to speed up the code.
        # We count with np.unique the number of halos in each cell. Then we do not have to loop over halo positions in --> profiles_on_grid/put_profiles_group
        # TODO - I Hate this
        unique_base_nGrid_poz, nbr_of_halos = cumulated_number_halos(self.parameters, halo_catalog.X, halo_catalog.Y, halo_catalog.Z, cic=cic)
        ZZ_indice = unique_base_nGrid_poz // (nGrid ** 2)
        YY_indice = (unique_base_nGrid_poz - ZZ_indice * nGrid ** 2) // nGrid
        XX_indice = (unique_base_nGrid_poz - ZZ_indice * nGrid ** 2 - YY_indice * nGrid)

        # Every halo in the mass bin i is assumed to have the mass M_bin[i].
        if buffer_xHII:
            # initialize the output grid over the shared memory buffer
            output_grid_xHII = np.ndarray(output_shape, dtype=np.float64, buffer=buffer_xHII.buf)
            x_HII_profile = np.zeros((len(radial_grid)))
            x_HII_profile[np.where(radial_grid < R_bubble / (1 + zgrid))] = 1

            # modify Grid_xHII in place
            self.paint_ionization_profile(
                output_grid_xHII, radial_grid, x_HII_profile, nGrid, LBox, zgrid, XX_indice, YY_indice, ZZ_indice, nbr_of_halos
            )
        if buffer_lyal:
            # initialize the output grid over the shared memory buffer
            output_grid_lyal = np.ndarray(output_shape, dtype=np.float64, buffer=buffer_lyal.buf)
            x_alpha_prof = 1.81e11 * (rho_alpha_) / (1 + zgrid)
            # We add up S_alpha(zgrid, T_extrap, 1 - xHII_extrap) later, a the map level.

            # TODO - document how r_lyal is the physical distance for lyal profile. Never goes further away than 100 pMpc/h (checked)
            # modify Grid_xal in place
            self.paint_alpha_profile(
                output_grid_lyal, radiation_profiles.r_lyal, x_alpha_prof, nGrid, LBox, zgrid, XX_indice, YY_indice, ZZ_indice, truncate, nbr_of_halos
            )
        if buffer_temp:
            # initialize the output grid over the shared memory buffer
            output_grid_temp = np.ndarray(output_shape, dtype=np.float64, buffer=buffer_temp.buf)
            # modify Grid_Temp in place
            self.paint_temperature_profile(
                output_grid_temp, radial_grid, Temp_profile, nGrid, LBox, zgrid, XX_indice, YY_indice, ZZ_indice, truncate, nbr_of_halos
            )



    def paint_ionization_profile(
        self,
        output_grid: np.ndarray,
        radial_grid,
        x_HII_profile,
        nGrid, LBox,
        z,
        XX_indice,
        YY_indice,
        ZZ_indice,
        nbr_of_halos
        ):
        # TODO - describe how this modifies the output_grid in place
        profile_xHII = interp1d(
            x = radial_grid * (1 + z),
            y = x_HII_profile,
            bounds_error = False,
            fill_value = (1, 0)
        )
        kernel_xHII = profile_to_3Dkernel(profile_xHII, nGrid, LBox)
        if not np.any(kernel_xHII > 0):
            ### if the bubble volume is smaller than the grid size,we paint central cell with ion fraction value
            # kernel_xHII[int(nGrid / 2), int(nGrid / 2), int(nGrid / 2)] = np.trapz(x_HII_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (LBox / nGrid / (1 + z)) ** 3
            output_grid[XX_indice, YY_indice, ZZ_indice] += trapezoid(x_HII_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (LBox / nGrid / (1 + z)) ** 3 * nbr_of_halos

        else:
            renorm = trapezoid(x_HII_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (LBox / (1 + z)) ** 3 / np.mean(kernel_xHII)
            # extra_ion = put_profiles_group(Pos_Halos_Grid[indices], kernel_xHII * 1e-7 / np.sum(kernel_xHII)) * np.sum(kernel_xHII) / 1e-7 * renorm
            output_grid += put_profiles_group(
                np.array((XX_indice, YY_indice, ZZ_indice)),
                nbr_of_halos,
                kernel_xHII * 1e-7 / np.sum(kernel_xHII)
            ) * np.sum(kernel_xHII) / 1e-7 * renorm
            # bubble_volume = trapezoid(4 * np.pi * radial_grid ** 2 * x_HII_profile, radial_grid)
            # print('bubble volume is ', len(indices) * bubble_volume,'pMpc, grid volume is', np.sum(extra_ion)* (LBox /nGrid/ (1 + z)) ** 3 )
            # Grid_xHII_i += extra_ion


    def paint_alpha_profile(
        self,
        output_grid: np.ndarray,
        r_lyal,
        x_alpha_prof,
        nGrid,
        LBox,
        z,
        XX_indice,
        YY_indice,
        ZZ_indice,
        truncate,
        nbr_of_halos
        ):

        ### We use this stacked_kernel functions to impose periodic boundary conditions when the lyal or T profiles extend outside the box size. Very important for Lyman-a.
        if isinstance(truncate, float):
            # truncate below a certain radius
            x_alpha_prof[r_lyal * (1 + z)< truncate] = x_alpha_prof[r_lyal * (1 + z) < truncate][-1]

        quantity = x_alpha_prof
        self.logger.debug(f"x_alpha_prof: {quantity.mean()=:.2e} {quantity.std()=:.2e} {quantity.min()=:.2e} {quantity.max()=:.2e}")
        kernel_xal = stacked_lyal_kernel(
            r_lyal * (1 + z),
            x_alpha_prof,
            LBox,
            nGrid,
            nGrid_min = self.parameters.simulation.nGrid_min_lyal
        )
        renorm = trapezoid(x_alpha_prof * 4 * np.pi * r_lyal ** 2, r_lyal) / (LBox / (1 + z)) ** 3 / np.mean(kernel_xal)

        if np.any(kernel_xal > 0):
            # Grid_xal += put_profiles_group(Pos_Halos_Grid[indices], kernel_xal * 1e-7 / np.sum(kernel_xal)) * renorm * np.sum( kernel_xal) / 1e-7  # we do this trick to avoid error from the fft when np.sum(kernel) is too close to zero.
            output_grid += put_profiles_group(
                np.array((XX_indice, YY_indice, ZZ_indice)),
                nbr_of_halos,
                kernel_xal * 1e-7 / np.sum(kernel_xal)
            ) * renorm * np.sum(kernel_xal) / 1e-7
            # we do this trick to avoid error from the fft when np.sum(kernel) is too close to zero.



    def paint_temperature_profile(
        self,
        output_grid: np.ndarray,
        radial_grid,
        Temp_profile,
        nGrid,
        LBox,
        z,
        XX_indice,
        YY_indice,
        ZZ_indice,
        truncate,
        nbr_of_halos
    ):

        # TODO - truncation should not be handled by the PAINT function
        if isinstance(truncate, float):
            # truncate below a certain radius
            Temp_profile[radial_grid * (1 + z) < truncate] = Temp_profile[radial_grid * (1 + z) < truncate][-1]

        quantity = Temp_profile
        self.logger.debug(f"Temp_profile: {quantity.mean()=:.2e} {quantity.std()=:.2e} {quantity.min()=:.2e} {quantity.max()=:.2e}")
        kernel_T = stacked_T_kernel(
            radial_grid * (1 + z),
            Temp_profile,
            LBox,
            nGrid,
            nGrid_min = self.parameters.simulation.nGrid_min_heat
        )
        renorm = trapezoid(Temp_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (LBox / (1 + z)) ** 3 / np.mean(kernel_T)

        if np.any(kernel_T > 0):
            output_grid += put_profiles_group(
                np.array((XX_indice, YY_indice, ZZ_indice)),
                nbr_of_halos,
                kernel_T * 1e-7 / np.sum(kernel_T)
            ) * np.sum(kernel_T) / 1e-7 * renorm
