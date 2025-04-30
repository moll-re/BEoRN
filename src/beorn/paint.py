import time
import logging
import numpy as np
import tools21cm as t2c
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
logger = logging.getLogger(__name__)

from .structs.parameters import Parameters
from .io import Handler
from .structs import GridData, GridDataMultiZ, SnapshotQuantities

from .functions import def_redshifts, z_string_format, load_halo, load_delta_b, delta_fct, def_k_bins, pixel_position, smooth_field
from .cosmo import dTb_factor, dTb_fct, T_adiab_fluctu, Tspin_fct
from . import constants
from .computing_profiles import ProfileSolver
from .profiles_on_grid import bin_edges_log, log_binning
from .couplings import x_coll, S_alpha
from .run import dTb_RSD
from .global_qty import xHII_approx
from .profiles_on_grid import average_profile, cumulated_number_halos, profile_to_3Dkernel, put_profiles_group, stacked_lyal_kernel, stacked_T_kernel, spreading_excess_fast




def paint_boxes(
    parameters: Parameters,
    cache_handler: Handler = None,
    output_handler: Handler = None,
    temp = True,
    lyal = True,
    ion = True,
    dTb = True,
    RSD = True,
    xcoll = True,
    S_al = True,
    cross_corr = False,
    third_order = False,
    fourth_order = False,
    cic = False,
    variance = False,
    Rsmoothing = 0,
    truncate = False
    ):
    """
    Parameters
    ----------
    TODO

    Returns
    -------
    Does not return anything. Loop over all snapshots in param.sim.halo_catalogs and calls paint_profile_single_snap.
    """
    start_time = time.process_time()
    
    LBox = parameters.simulation.Lbox  # Mpc/h
    nGrid = parameters.simulation.Ncell  # number of grid cells
    catalog_dir = parameters.simulation.halo_catalogs

    if catalog_dir is None:
        # TODO - this should be a warning while initialising the parameters
        raise FileNotFoundError('You should specify param.sim.halo_catalogs. Should be a file containing the halo catalogs.')



    # comm, rank, size = initialise_mpi4py(parameters)

    z_arr = def_redshifts(parameters)
    logger.info(f'Painting profiles for {len(z_arr)} z snapshots. Using a grid with {nGrid=} (pixels per dim) and a box size of {LBox=} cMpc/h.')

    multi_z_data = GridDataMultiZ(parameters=parameters, file_root=output_handler.file_root)
    # TODO - this loop could be parallelized using MPI
    for ii, z in enumerate(z_arr):
        z = np.round(z, 2)
        # if rank == ii % size:
        # print('Core nbr', rank, 'is taking care of z = ', z)
        if cache_handler is not None and output_handler is not None:
            try:
                SnapshotQuantities.read(output_handler.file_root, parameters, z=z)
                GridData.read(cache_handler.file_root, parameters, z=z)
                logger.debug(f"Found painted output in cache for {z=}. Skipping.")
                continue
            except FileNotFoundError:
                logger.debug("Painted output not found in cache. Processing now")
        else:
            logger.debug("No cache or output handler provided. Processing now")

        # there is no cache or the cache does not contain the halo catalog - compute it fresh
        logger.debug(f'Painting 3D map for {z=}')
        grid_data = paint_profiles_single_z(
            ii,
            z,
            parameters,
            cache_handler=cache_handler,
            temp=temp,
            lyal=lyal,
            ion=ion,
            dTb=dTb,
            xcoll=xcoll,
            S_al=S_al,
            cic=cic,
            Rsmoothing=Rsmoothing,
            truncate=truncate
        )

        grid_quantities = compute_quantities_single_z(
            z,
            parameters,
            grid_data,
            RSD=RSD,
            cross_corr=cross_corr,
            third_order=third_order,
            fourth_order=fourth_order,
            truncate=truncate,
            variance=variance,
        )

        if cache_handler is not None:
            grid_data.write(cache_handler.file_root, z=z)
        if output_handler is not None:
            grid_quantities.write(output_handler.file_root, z=z)
    
        multi_z_data.append(grid_data, z=z)

    logger.info(f"Paining finished! Elapsed time: {time.process_time() - start_time:.2} seconds")
    return multi_z_data


def compute_quantities_single_z(
    z: float,
    parameters: Parameters,
    data: GridData,
    RSD = False,
    cross_corr = False,
    third_order = False,
    fourth_order = False,
    variance = False,
    truncate = False
) -> SnapshotQuantities:
    T_spin = np.mean(Tspin_fct(constants.Tcmb0 * (1 + z), data.Grid_Temp, data.Grid_xtot))


    PS_dTb, k_bins = t2c.power_spectrum.power_spectrum_1d(
        delta_fct(data.Grid_dTb),
        box_dims = parameters.simulation.Lbox,
        kbins = def_k_bins(parameters)
        )
    PS_dTb_no_reio, k_bins = t2c.power_spectrum.power_spectrum_1d(
        delta_fct(data.Grid_dTb_no_reio),
        box_dims = parameters.simulation.Lbox,
        kbins = def_k_bins(parameters)
        )
    PS_dTb_T_sat, k_bins = t2c.power_spectrum.power_spectrum_1d(
        delta_fct(data.Grid_dTb_T_sat),
        box_dims = parameters.simulation.Lbox,
        kbins = def_k_bins(parameters)
        )


    if not RSD:
        dTb_RSD_mean = 0
        PS_dTb_RSD = 0
        Grid_dTb_RSD = data.Grid_dTb
    else:
        logger.debug('Computing RSD for snapshot...')
        Grid_dTb_RSD = dTb_RSD(parameters, z, data.delta_b, data.Grid_dTb)
        delta_Grid_dTb_RSD = Grid_dTb_RSD / np.mean(Grid_dTb_RSD) - 1
        PS_dTb_RSD = t2c.power_spectrum.power_spectrum_1d(
            delta_Grid_dTb_RSD,
            box_dims = data.LBox,
            kbins = def_k_bins(parameters)
            )[0]
        dTb_RSD_mean = np.mean(Grid_dTb_RSD)




    # TODO - make it more clear what values are stored (and why)
    # GS_PS_dict = {'z': z, 'dTb': np.mean(data.Grid_dTb), 'Tk': np.mean(data.Grid_Temp), 'x_HII': np.mean(data.Grid_xHII),
    #               'PS_dTb': PS_dTb, 'k': k_bins, 'Tspin':T_spin,
    #               'PS_dTb_RSD': PS_dTb_RSD, 'dTb_RSD': dTb_RSD_mean, 'x_al': np.mean(data.Grid_xal),
    #               'x_coll': data.xcoll_mean,'PS_dTb_no_reio':PS_dTb_no_reio,'dTb_no_reio': np.mean(data.Grid_dTb_no_reio),
    #               'PS_dTb_T_sat':PS_dTb_T_sat,'dTb_T_sat': np.mean(data.Grid_dTb_T_sat)}


    # if cross_corr:
    #     q1, q2, q3 = compute_cross_correlations(
    #         parameters,
    #         data.Grid_Temp,
    #         data.Grid_xHII,
    #         data.Grid_xal,data.Grid_dTb,
    #         data.delta_b,
    #         third_order = third_order,
    #         fourth_order = fourth_order,
    #         truncate = truncate
    #     )

    # if variance:
    #     # we do this since in compute_var we change the kbins to go to smaller scales.
    #     # TODO - why was it deepcopied?
    #     # param_copy = copy.deepcopy(parameters)
    #     # TODO - DO NOT EXPORT THIS AS A SEPARATE FILE!!
    #     compute_var_single_z(parameters, z, data.Grid_xal, data.Grid_xHII, data.Grid_Temp, data.delta_b, k_bins)

    return SnapshotQuantities(
        parameters = parameters,
        z = z,
        k_bins = k_bins,
        PS_dTb = PS_dTb,
        PS_dTb_RSD = PS_dTb_RSD,
        PS_dTb_no_reio = PS_dTb_no_reio,
        PS_dTb_T_sat = PS_dTb_T_sat,
        dTb_RSD_mean = dTb_RSD_mean,
        # TODO
        T_spin = T_spin
    )



def paint_profiles_single_z(
    redshift_index: int,
    z: float,
    parameters: Parameters,
    cache_handler: Handler = None,
    temp = True,
    lyal = True,
    ion = True,
    dTb = True,
    xcoll = True,
    S_al = True,
    cic = False,
    Rsmoothing = 0,
    truncate = False
) -> GridData:
    """
    Paint the Tk, xHII and Lyman alpha profiles on a grid for a single halo catalog named filename.

    Parameters
    ----------
    TODO
    param : dictionnary containing all the input parameters
    z_str : the redshift of the snapshot.
                    filename : the name of the halo catalog, contained in param.sim.halo_catalogs.
    temp, lyal, ion, dTb : which map to paint.
    S_alpha : if equals to False, then we write xal = rhoal*Salpha(mean quantity). If true, we take the full inhomogeneous Salpha.

    Returns
    -------
    Does not return anything. Paints and stores the grids int the directory grid_outputs.
    """

    start_time = time.process_time()

    LBox = parameters.simulation.Lbox  # Mpc/h
    nGrid = parameters.simulation.Ncell  # number of grid cells
    zeros = np.zeros((nGrid, nGrid, nGrid))

    z_string = z_string_format(z)
    # try:
    halo_catalog = load_halo(parameters, redshift_index)
    # except:
    #     # TODO - I am convinced this never worked
    #     halo_catalog = parameters.simulation.halo_catalogs[float(z_string)]

    H_Masses, H_X, H_Y, H_Z = halo_catalog['M'], halo_catalog['X'], halo_catalog['Y'], halo_catalog['Z']

    ### To later add up the adiabatic Tk fluctuations at the grid level.
    if temp or dTb:
        # try:
        delta_b = load_delta_b(parameters, redshift_index)  # rho/rhomean-1
        # except:
        #     # TODO - same here???
        #     delta_b = parameters.simulation.dens_field[float(z_string)] #param.sim.dens_fields[z_str]
    else:
        delta_b = zeros

    coef = constants.rhoc0 * parameters.cosmology.h ** 2 * parameters.cosmology.Ob * (1 + z) ** 3 * constants.M_sun / constants.cm_per_Mpc ** 3 / constants.m_H

    # find matching redshift between solver output and simulation snapshot.
    # this will raise an error if the needed profiles are not available
    # The loading is left in this function to allow for the possibility of parallelizing the painting
    grid_model: ProfileSolver = cache_handler.load_file(parameters, ProfileSolver)
    ind_z = np.argmin(np.abs(grid_model.z_history - z))
    zgrid = grid_model.z_history[ind_z]

    # if H_Masses is digitized to bin_edges_log[i] it means it should take the value of M_Bin[i-1] (bin 0 is what's on the left...)
    # M_Bin                0.   1.    2.    3.  ...
    # bin_edges_log     0.  | 1. |  2. |  3. |  4. ....
    # TODO - adapt this for variable alpha values
    Indexing = log_binning(H_Masses, bin_edges_log(grid_model.Mh_history[..., 5, ind_z]))
    Indexing = Indexing - 1
    # same result as if you do np.argmin(np.abs(np.log10(H_Masses[:,None]/grid_model.Mh_history[..., ind_z]),axis=1), but faster

    if any(Indexing < 0):
        raise ValueError('Need lower Mmin!')

    logger.debug(f'There are {H_Masses.size} halos at {z=}')
    logger.debug('Looping over halo mass bins and painting profiles on 3D grid .... ')

    if H_Masses.size == 0:
        logger.info(f'There are no sources at {z=}')

        factor = dTb_factor(parameters)

        Grid_Temp = T_adiab_fluctu(z, parameters, delta_b)
        Grid_xHII = zeros

        Grid_xal = zeros
        Grid_xcoll = x_coll(z=z, Tk=Grid_Temp, xHI=(1 - Grid_xHII), rho_b=(delta_b + 1) * coef)
        Grid_dTb = factor * np.sqrt(1 + z) * (1 - constants.Tcmb0 * (1 + z) / Grid_Temp) * (1 - Grid_xHII) * (delta_b + 1) * Grid_xcoll / (1 + Grid_xcoll)
        Grid_dTb_no_reio = factor * np.sqrt(1 + z) * (1 - constants.Tcmb0 * (1 + z) / Grid_Temp) * (delta_b + 1) * Grid_xcoll / (1 + Grid_xcoll)
        Grid_dTb_RSD = zeros
        Grid_dTb_T_sat = factor * np.sqrt(1 + z) * (1 - Grid_xHII) * (delta_b + 1) * Grid_xcoll / (1 + Grid_xcoll)
        xcoll_mean = np.mean(Grid_xcoll)
        
        return GridData(
            parameters = parameters,
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


    # TODO - should these trigger a failure?
    if np.max(H_Masses) > np.max(grid_model.Mh_history[..., ind_z]):
        print('Max Mh_bin is :', np.max(grid_model.Mh_history[..., ind_z]), 'while the largest halo in catalog is',
                np.max(H_Masses))
        print('WARNING!!! You should use a larger value for param.sim.Mh_bin_max')
    if np.min(H_Masses) < np.min(grid_model.Mh_history[..., ind_z]):
        print('WARNING!!! You should use a smaller value for param.sim.Mh_bin_min')

    Ionized_vol = xHII_approx(
        parameters,
        grid_model.Mh_history[..., 5, ind_z],
        grid_model.R_bubble[..., 5, ind_z],
        halo_catalog
        )
    logger.info(f'Quick calculation from the profiles predicts xHII = {Ionized_vol:.4}')
    if Ionized_vol > 1:
        logger.info('Universe is fully ionized. Returning [1] for xHII, T and [0] for dTb.')
        Grid_xHII = zeros + 1
        Grid_Temp = zeros + 1
        Grid_dTb = zeros
        Grid_dTb_no_reio = zeros
        Grid_dTb_T_sat = zeros
        Grid_xal = zeros
        return GridData(
            parameters = parameters,
            delta_b = delta_b,
            Grid_Temp = Grid_Temp,
            Grid_dTb = Grid_dTb,
            Grid_dTb_no_reio = Grid_dTb_no_reio,
            Grid_dTb_T_sat = Grid_dTb_T_sat,
            # TODO - what should the value be?
            Grid_dTb_RSD = zeros,
            Grid_xHII = Grid_xHII,
            Grid_xal = Grid_xal,
            # TODO - what value for xtot?
            Grid_xtot = zeros
        )


    # we don't want Pos_Halos_Grid==nGrid. This only happens if Pos_Bubbles=LBox
    Pos_Halos_Grid = pixel_position(H_X, H_Y, H_Z, LBox, nGrid)

    # initialise the "main" grids here and iteratively add the profiles
    Grid_xHII = np.zeros((nGrid, nGrid, nGrid))
    Grid_Temp = np.zeros((nGrid, nGrid, nGrid))
    Grid_xal = np.zeros((nGrid, nGrid, nGrid))
    
    radial_grid = grid_model.r_grid_cell / (1 + zgrid)  # pMpc/h

    for i in range(len(parameters.simulation.halo_mass_bins)):
        # indices in H_Masses of halos that have an initial mass at z=z_start between M_Bin[i-1] and M_Bin[i]
        indices = np.where(Indexing == i)[0]
        # TODO - pick alpha flexibly
        Mh_ = grid_model.Mh_history[i, 5, ind_z]
        logger.debug(f'Processing {len(indices)} halos in mass bin {i} with {Mh_=:0.2e}')
        
        # nothing to process - skip to next loop iteration
        if len(indices) == 0 or Mh_ < parameters.source.halo_mass_min:
            continue


        # for sanity
        R_bubble, rho_alpha_, Temp_profile = average_profile(parameters, grid_model, H_Masses[indices], ind_z, i)
        logger.debug(
            "Computing average profiles: "
            f"{grid_model.R_bubble.shape=} -> {R_bubble=}, "
            f"{grid_model.rho_heat.shape=} -> {Temp_profile.shape}, "
            f"{grid_model.rho_alpha.shape=} -> {rho_alpha_.shape}"
        )


        # This is the position of halos in base "nGrid". We use this to speed up the code.
        # We count with np.unique the number of halos in each cell. Then we do not have to loop over halo positions in --> profiles_on_grid/put_profiles_group
        unique_base_nGrid_poz, nbr_of_halos = cumulated_number_halos(parameters, H_X[indices], H_Y[indices], H_Z[indices], cic=cic)
        ZZ_indice = unique_base_nGrid_poz // (nGrid ** 2)
        YY_indice = (unique_base_nGrid_poz - ZZ_indice * nGrid ** 2) // nGrid
        XX_indice = (unique_base_nGrid_poz - ZZ_indice * nGrid ** 2 - YY_indice * nGrid)

        # Every halo in the mass bin i is assumed to have the mass M_bin[i].
        if ion:
            x_HII_profile = np.zeros((len(radial_grid)))
            x_HII_profile[np.where(radial_grid < R_bubble / (1 + zgrid))] = 1

            # modify Grid_xHII in place
            paint_ionization_profile(
                parameters, Grid_xHII, radial_grid, x_HII_profile, nGrid, LBox, z, XX_indice, YY_indice, ZZ_indice, nbr_of_halos
            )
        if lyal:
            x_alpha_prof = 1.81e11 * (rho_alpha_) / (1 + zgrid)
            # We add up S_alpha(zgrid, T_extrap, 1 - xHII_extrap) later, a the map level.

            # TODO - document how r_lyal is the physical distance for lyal profile. Never goes further away than 100 pMpc/h (checked)
            # modify Grid_xal in place
            paint_alpha_profile(
                parameters, Grid_xal, grid_model.r_lyal, x_alpha_prof, nGrid, LBox, z, XX_indice, YY_indice, ZZ_indice, truncate, nbr_of_halos
            )
        if temp:
            # modify Grid_Temp in place
            paint_temperature_profile(
                parameters, Grid_Temp, radial_grid, Temp_profile, nGrid, LBox, z, XX_indice, YY_indice, ZZ_indice, truncate, nbr_of_halos
            )

        end_time = time.process_time()
        logger.debug(f'Painted profiles in {end_time - start_time:.2} seconds.')
        start_time = end_time

    logger.info('Profile painting done.')

    logger.info('Dealing with the overlap of ionised bubbles. Redistributing excess photons from the overlapping regions.')
    start_time = time.process_time()
    if ion and np.sum(Grid_xHII) < nGrid ** 3:
        Grid_xHII = spreading_excess_fast(parameters, Grid_xHII)
    else:
        Grid_xHII = zeros + 1

    logger.info(f'Overlap processing done. Took {time.process_time() - start_time:.2} seconds.')

    # TODO - why not just leave the full array?
    # if np.all(Grid_xHII == 0):
    #     Grid_xHII = np.array([0])
    # if np.all(Grid_xHII == 1):
    #     logger.info('Universe is fully ionized. Returning [1] for Grid_xHII.')
    #     Grid_xHII = np.array([1])

    Grid_Temp += T_adiab_fluctu(z, parameters, delta_b)


    # TODO - why is this here??
    # TODO - this is super error prone.
    # if read_temp:
    #     Grid_Temp = load_grid(parameters, z=z, type='Tk')

    # if read_ion:
    #     Grid_xHII = load_grid(parameters, z=z, type='bubbles')

    # if read_lyal:
    #     Grid_xal = load_grid(parameters, z=z, type='lyal')
    # else:
    if S_al:
        logger.debug('Including Salpha fluctuations in dTb')
        Grid_xal = Grid_xal * S_alpha(z, Grid_Temp, 1 - Grid_xHII) / (4 * np.pi)
        # We divide by 4pi to go to sr**-1 units
    else:
        logger.debug('NOT including Salpha fluctuations in dTb')
        Grid_xal = Grid_xal * S_alpha(z, np.mean(Grid_Temp), 1 - np.mean(Grid_xHII)) / (4 * np.pi)



    if Rsmoothing > 0:
        logger.info(f'Smoothing the fields with {Rsmoothing=}')
        Grid_xal = smooth_field(Grid_xal, Rsmoothing, LBox, nGrid)
        Grid_Temp = smooth_field(Grid_Temp, Rsmoothing, LBox, nGrid)
        #Grid_xHII = smooth_field(Grid_xHII, Rsmoothing, LBox, nGrid)
        #delta_b   = smooth_field(delta_b, Rsmoothing, LBox, nGrid)
        # TODO - why are the other fields not smoothed?


    if xcoll:
        logger.info('Including xcoll fluctuations in dTb')
        Grid_xcoll = x_coll(z = z, Tk = Grid_Temp, xHI = (1 - Grid_xHII), rho_b = (delta_b + 1) * coef)
        xcoll_mean = np.mean(Grid_xcoll)
        Grid_xtot = Grid_xcoll + Grid_xal
        # del Grid_xcoll
    else:
        logger.info('NOT including xcoll fluctuations in dTb')
        xcoll_mean = x_coll(z = z, Tk = np.mean(Grid_Temp), xHI = (1 - np.mean(Grid_xHII)), rho_b = coef)
        Grid_xtot = Grid_xal + xcoll_mean


    if dTb:
        Grid_dTb = dTb_fct(z=z, Tk=Grid_Temp, xtot=Grid_xtot, delta_b=delta_b, x_HII=Grid_xHII, parameters=parameters)
        Grid_dTb_no_reio = dTb_fct(z=z, Tk=Grid_Temp, xtot=Grid_xtot, delta_b=delta_b, x_HII=np.array([0]), parameters=parameters)
        Grid_dTb_T_sat = dTb_fct(z=z, Tk=1e50, xtot = 1e50, delta_b=delta_b, x_HII=Grid_xHII, parameters=parameters)
    else:
        Grid_dTb = np.array([0])
        Grid_dTb_no_reio = np.array([0])
        Grid_dTb_T_sat = np.array([0])

    return GridData(
        parameters = parameters,
        delta_b = delta_b,
        Grid_Temp = Grid_Temp,
        Grid_dTb = Grid_dTb,
        Grid_dTb_no_reio = Grid_dTb_no_reio,
        Grid_dTb_T_sat = Grid_dTb_T_sat,
        Grid_dTb_RSD = np.array([0]),
        Grid_xHII = Grid_xHII,
        Grid_xal = Grid_xal,
        Grid_xtot = Grid_xtot,
    )





def paint_ionization_profile(parameters: Parameters, output_grid: np.ndarray, radial_grid, x_HII_profile, nGrid, LBox, z, XX_indice, YY_indice, ZZ_indice, nbr_of_halos):
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

    # fill in empty pixels with the min xHII
    # TODO - only do this once! in a separate function
    output_grid[output_grid < parameters.source.min_xHII_value] = parameters.source.min_xHII_value



def paint_alpha_profile(parameters: Parameters, output_grid: np.ndarray, r_lyal, x_alpha_prof, nGrid, LBox, z, XX_indice, YY_indice, ZZ_indice, truncate, nbr_of_halos):

    ### We use this stacked_kernel functions to impose periodic boundary conditions when the lyal or T profiles extend outside the box size. Very important for Lyman-a.
    if isinstance(truncate, float):
        # truncate below a certain radius
        x_alpha_prof[r_lyal * (1 + z)< truncate] = x_alpha_prof[r_lyal * (1 + z) < truncate][-1]

    kernel_xal = stacked_lyal_kernel(
        r_lyal * (1 + z),
        x_alpha_prof,
        LBox,
        nGrid,
        nGrid_min = parameters.simulation.nGrid_min_lyal
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



def paint_temperature_profile(parameters: Parameters, output_grid: np.ndarray, radial_grid, Temp_profile, nGrid, LBox, z, XX_indice, YY_indice, ZZ_indice, truncate, nbr_of_halos):

    # TODO - truncation should not be handled by the PAINT function
    if isinstance(truncate, float):
        # truncate below a certain radius
        Temp_profile[radial_grid * (1 + z) < truncate] = Temp_profile[radial_grid * (1 + z) < truncate][-1]

    kernel_T = stacked_T_kernel(
        radial_grid * (1 + z),
        Temp_profile,
        LBox,
        nGrid,
        nGrid_min = parameters.simulation.nGrid_min_heat
    )
    renorm = trapezoid(Temp_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (LBox / (1 + z)) ** 3 / np.mean(kernel_T)

    if np.any(kernel_T > 0):
        output_grid += put_profiles_group(
            np.array((XX_indice, YY_indice, ZZ_indice)),
            nbr_of_halos,
            kernel_T * 1e-7 / np.sum(kernel_T)
        ) * np.sum(kernel_T) / 1e-7 * renorm

