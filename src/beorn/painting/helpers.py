"""Helpers for the painting module."""
import numpy as np
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, wait
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import interp1d
from skimage.measure import label
import logging
logger = logging.getLogger(__name__)

from ..structs.parameters import Parameters



def profile_to_3Dkernel(profile: callable, nGrid: int, LB: float) -> np.ndarray:
    """
    Put profile_1D on a grid

    Parameters
    ----------
    profile  : profile_1D(r, c1=2, c2=5).
    nGrid, LB  : number of grids and boxsize (in cMpc/h) respectively

    Returns
    -------
    meshgrid of size (nGrid,nGrid,nGrid), with the profile at the center.
    """
    x = np.linspace(-LB / 2, LB / 2, nGrid)
    # y, z will be the same.
    rx, ry, rz = np.meshgrid(x, x, x, sparse=True)
    rgrid = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    kern = profile(rgrid)
    assert np.all(np.isfinite(kern)), "Profile function returned non-finite values."
    return kern



def spread_excess_ionization(parameters: Parameters, input_grid: np.ndarray, show_plot=False):
    """
    This function spreads the excess ionizing photons in the grid. It uses a distance transform to find the closest ionized region and spread the excess photons to it, enlarging the space of the ionized region.

    Parameters
    ----------
        parameters : Parameters
        input_grid : The meshgrid containing the ionizing fractions
        show_plot : If True, will show the plot of the grid before and after spreading.

    Returns
    -------
        Grid : The meshgrid with the excess ionizing photons spread out.

    A word regarding the elements of this function :
        - Binary_Grid: contains 1 where Grid_input>=1 and 0 elsewhere. This grid is used as input for scipy.measure.label. Format is (X,X,X).
        - Connected_regions: (X,X,X). Output of skimage.measure.label. Each pixel of it is labeled according to the ionized clump it belongs to.
        - x_ion_tot_i: total sum of ionizing fraction.
        - region_nbr, size_of_region: region label, and size of it . We use it to idenfity the very small regions (small_regions) with less than "pix_thresh" pixels. We treat them all together to speed up the process
        - Spread_Single:  spread the excess photons.
    """

    output_grid = np.copy(input_grid)

    if parameters.simulation.spreading_pixel_threshold is None:
        # TODO - what are these values???
        nGrid = input_grid.shape[0]
        pix_thresh = 80 * (nGrid / 256) ** 3
    else:
        pix_thresh = parameters.simulation.spreading_pixel_threshold

    binary_grid = np.zeros_like(output_grid)

    binary_grid[np.where(output_grid >= 0.9999999)] = 1

    # The first region (i=0) is the still neutral IGM, in between the bubbles
    label_image = label(binary_grid)

    # Periodic boundary conditions for label_image
    # assign  same label to ionised regions that are connected through left/right, up/down, front/back box boundaries

    total_ionizing_fraction_initial = np.sum(output_grid)
    logger.debug(f'Current ionization fraction: {total_ionizing_fraction_initial:.3f}')

    if total_ionizing_fraction_initial > output_grid.size:
        logger.info('Universe is fully ionized.')
        # return a final grid with all pixels set to 1
        return np.ones_like(input_grid)

    logger.debug(f'Universe not fully ionized : xHII is {total_ionizing_fraction_initial / output_grid.size:.4e}.')

    region_nbr, size_of_region = np.unique(label_image, return_counts=True)
    logger.debug(f"Found {len(region_nbr)} connected regions.")
    label_max = np.max(label_image)

    small_regions = np.where(np.isin(label_image, region_nbr[np.where(size_of_region < pix_thresh)[0]]))
    # Gridmesh indices gathering all the connected regions that have less than 10 pixels
    small_regions_labels = region_nbr[np.where(size_of_region < pix_thresh)[0]]
    # labels of the small regions. Use this to exclude them from the for loop

    initial_excess = np.sum(output_grid[small_regions] - 1)
    excess_ion = initial_excess

    logger.debug(f'there are {len(small_regions_labels)} small connected regions with less than {pix_thresh} pixels. They contain a fraction {excess_ion / total_ionizing_fraction_initial:.4e} of the total ionisation fraction.')

    # Do the spreading for the small regions
    spread_excess_ionization_single_region(parameters, output_grid, small_regions)
    assert not np.any(output_grid[small_regions] > 1), 'Small regions not correctly spread.'

    # the remaining larges overlapping ionized regions
    all_regions_labels = np.arange(1, label_max + 1, dtype=int)
    large_regions_labels = all_regions_labels[np.where(np.isin(all_regions_labels, small_regions_labels) == False)[0]]
    # large_regions_labels = all_regions_labels[np.where(not np.isin(all_regions_labels, small_regions_labels))]
    # indices of regions that have more than pix_thresh pixels


    # Do the spreading individually for large regions, but parallelize the process
    if large_regions_labels.size > 0:
        logger.debug(f"Spreading {len(large_regions_labels)} large regions.")
        size = input_grid.size * np.dtype(np.float64).itemsize
        buffer = shared_memory.SharedMemory(create=True, size=size)
        output_grid_shared = np.ndarray(input_grid.shape, dtype=np.float64, buffer=buffer.buf)
        output_grid_shared[:] = input_grid[:]

        futures = []

        with ProcessPoolExecutor(max_workers=parameters.simulation.cores) as executor:
            for i, l in enumerate(large_regions_labels):
                connected_indices = np.where(label_image == l)
                f = executor.submit(spread_excess_ionization_single_region, parameters, buffer, connected_indices)
                futures.append(f)


            completed, uncompleted = wait(futures)
            assert len(uncompleted) == 0, "Not all spreading subprocesses completed successfully"


        # clean up the shared memory buffers - but keep the data in the buffer
        output_grid = np.ndarray(input_grid.shape, dtype=np.float64, buffer=buffer.buf)
        output_grid = output_grid_shared.copy()


        buffer.close()
        buffer.unlink()

    logger.debug(f"{np.any(output_grid > 1)}")
    # assert not np.any(output_grid > 1), 'Some grid pixels are still in excess.'

    total_ionization_fraction_final = np.sum(output_grid)
    logger.debug(f'Final xion sum: {total_ionization_fraction_final:.3f}')
    if int(total_ionization_fraction_final) != int(total_ionizing_fraction_initial):
        raise RuntimeError('Spreading procedure did not conserve the photons. This should not happen.')

    return output_grid




def spread_excess_ionization_single_region(parameters: Parameters, Grid: np.ndarray | shared_memory.SharedMemory, connected_indices):
    """
    Spreads the excess ionization_fraction for a given region by modifying the grid in place.
    Input :
    - Grid : The meshgrid containing the ionizing fractions
    - Connected_indices : The indices of the ionized region from which you want to spread the overlaps. (excess_ion)


    Trick : we run distance_transform only for a sub-box centered on the connected region. This is particularly important for high resolution grids, when distance_transform_edt starts to take time (~s, but multilplied by the number of connected regions >1e4, starts taking time...)
            the size of the subbox is N_subgrid. It is called Sub_Grid.
    """

    if isinstance(Grid, shared_memory.SharedMemory):
        # If Grid is a shared memory object, create a numpy array from it
        Grid = np.ndarray((Grid.shape[0], Grid.shape[1], Grid.shape[2]), dtype=np.float64, buffer=Grid.buf)

    nGrid = Grid.shape[0]

    initial_excess = np.sum(Grid[connected_indices] - 1)
    Grid[connected_indices] = np.where(Grid[connected_indices] > 1, 1, Grid[connected_indices])
    excess_ion = initial_excess

    if initial_excess < 1e-8:
        # If the excess is too small, just return the grid
        # TODO - does this really improve the performance?
        return Grid

    ## take sub grid with only the connected region, find pixels where xion>1, sum the excess, and set these pixels to 1.
    Inverted_grid = np.full(((nGrid, nGrid, nGrid)), 1)
    Inverted_grid[connected_indices] = 0
    sum_distributed_xion = 0

    # TODO - what are these???
    Delta_pixel = int(excess_ion ** (1. / 3) / 2) + 1

    Min_X, Max_X = np.min(connected_indices[0]), np.max(connected_indices[0])
    Min_Y, Max_Y = np.min(connected_indices[1]), np.max(connected_indices[1])
    Min_Z, Max_Z = np.min(connected_indices[2]), np.max(connected_indices[2])
    Delta_max = np.max((Max_X - Min_X + 0, Max_Y - Min_Y + 0, Max_Z - Min_Z + 0))
    Center_X, Center_Y, Center_Z = int((Min_X + Max_X) / 2), int((Min_Y + Max_Y) / 2), int((Min_Z + Max_Z) / 2)

    if parameters.simulation.spreading_subgrid_approximation:
        # Is this flag is True, set the subgrid size
        N_subgrid = 2 * (Delta_max + 2 * Delta_pixel)
        # length of subgrid embedding the connected region
        N_subgrid += N_subgrid % 2
        # Ensure N_subgrid is even

    else:
        # If the flag is False, then set N_subgrid > nGrid, so that we never do the subbox trick (this is to check if the trick gives good results compared to the full)
        N_subgrid = nGrid + 1


    if N_subgrid > nGrid:
        dist_from_boundary = distance_transform_edt(Inverted_grid)
        dist_from_boundary[np.where(dist_from_boundary == 0)] = 2 * nGrid  ### eliminate pixels inside boundary
        dist_from_boundary[np.where(
            Grid > 1)] = 2 * nGrid  ### eliminate pixels that already have excess x_ion (belonging to another connected regions..)
        minimum = np.min(dist_from_boundary)
        boundary = np.where(dist_from_boundary == minimum)  # np.where((dist_from_boundary == minimum )& ( Grid<1))

        while np.sum(1 - Grid[boundary]) < excess_ion:
            sum_distributed_xion += np.sum(1 - Grid[boundary])
            excess_ion = excess_ion - np.sum(1 - Grid[boundary])
            Grid[boundary] = 1
            dist_from_boundary[boundary] = nGrid * 2  ### exclude this layer for next step
            minimum = np.min(dist_from_boundary)
            boundary = np.where(dist_from_boundary == minimum)  ### new closest region to fill with excess ion
        # you go out of the *while* when np.sum(1 - Grid[boundary]) > excess_ion
        residual_excess = (1 - Grid[boundary]) * excess_ion / np.sum(1 - Grid[boundary])
        Grid[boundary] += residual_excess
        sum_distributed_xion += excess_ion


    else:

        Sub_Grid = np.full(((N_subgrid, N_subgrid, N_subgrid)), 0)

        Sub_Grid = Sub_Grid.astype('float64')

        Sub_Grid[:] = Grid[np.max((Center_X - int(N_subgrid / 2), 0)) - np.max(
            (0, Center_X + int(N_subgrid / 2) + 0 - nGrid)): np.min(
            (nGrid, Center_X + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_X)),
                        np.max((Center_Y - int(N_subgrid / 2), 0)) - np.max(
                            (0, Center_Y + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                            (nGrid, Center_Y + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_Y)),
                        np.max((Center_Z - int(N_subgrid / 2), 0)) - np.max(
                            (0, Center_Z + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                            (nGrid, Center_Z + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_Z))]

        while np.sum(1 - Sub_Grid) < excess_ion:
            # just check if Sub_Grid has enough room for excess_ion. If not, increase its size N_subgrid.
            # Here be dragons
            # TODO - what a great piece of code.
            N_subgrid = N_subgrid + 2
            Sub_Grid = np.full(((N_subgrid, N_subgrid, N_subgrid)), 0)
            Sub_Grid = Sub_Grid.astype('float64')
            Sub_Grid[:] = Grid[np.max((Center_X - int(N_subgrid / 2), 0)) - np.max(
                (0, Center_X + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                (nGrid, Center_X + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_X)),
                            np.max((Center_Y - int(N_subgrid / 2), 0)) - np.max(
                                (0, Center_Y + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                                (nGrid, Center_Y + int(N_subgrid / 2) + 0)) + np.max(
                                (0, int(N_subgrid / 2) - Center_Y)),
                            np.max((Center_Z - int(N_subgrid / 2), 0)) - np.max(
                                (0, Center_Z + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                                (nGrid, Center_Z + int(N_subgrid / 2) + 0)) + np.max(
                                (0, int(N_subgrid / 2) - Center_Z))]

        Sub_Inverted_Grid = np.full(((N_subgrid, N_subgrid, N_subgrid)), 1)
        Sub_Inverted_Grid = Sub_Inverted_Grid.astype('float64')
        Sub_Inverted_Grid[:] = Inverted_grid[np.max((Center_X - int(N_subgrid / 2), 0)) - np.max(
            (0, Center_X + int(N_subgrid / 2) + 0 - nGrid)): np.min(
            (nGrid, Center_X + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_X)),
                                np.max((Center_Y - int(N_subgrid / 2), 0)) - np.max(
                                    (0, Center_Y + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                                    (nGrid, Center_Y + int(N_subgrid / 2) + 0)) + np.max(
                                    (0, int(N_subgrid / 2) - Center_Y)),
                                np.max((Center_Z - int(N_subgrid / 2), 0)) - np.max(
                                    (0, Center_Z + int(N_subgrid / 2) + 0 - nGrid)): np.min(
                                    (nGrid, Center_Z + int(N_subgrid / 2) + 0)) + np.max(
                                    (0, int(N_subgrid / 2) - Center_Z))]

        Sub_Grid_Initiale = np.copy(Sub_Grid)

        dist_from_boundary = distance_transform_edt(Sub_Inverted_Grid)
        dist_from_boundary[np.where(dist_from_boundary == 0)] = 2 * N_subgrid  ### eliminate pixels inside boundary
        dist_from_boundary[np.where(
            Sub_Grid >= 1)] = 2 * N_subgrid  ### eliminate pixels that already have excess x_ion (belonging to another connected regions..)
        minimum = np.min(dist_from_boundary)
        boundary = np.where(dist_from_boundary == minimum)

        excess_ion_i = excess_ion
        while np.sum(1 - Sub_Grid[boundary]) < excess_ion:
            sum_distributed_xion += np.sum(1 - Sub_Grid[boundary])
            excess_ion = excess_ion - np.sum(1 - Sub_Grid[boundary])
            Sub_Grid[boundary] = 1
            dist_from_boundary[boundary] = N_subgrid * 2  ### exclude this layer for nex
            minimum = np.min(dist_from_boundary)
            boundary = np.where(
                dist_from_boundary == minimum)  ### new closest region to fill # you go out of the *while* when np.sum(1 - Grid[boundary]) > excess_ion

        residual_excess = (1 - Sub_Grid[boundary]) * excess_ion / np.sum(1 - Sub_Grid[boundary])

        Sub_Grid[boundary] = np.add(Sub_Grid[boundary], residual_excess)
        sum_distributed_xion += excess_ion

        Grid[
        np.max((Center_X - int(N_subgrid / 2), 0)) - np.max((0, Center_X + int(N_subgrid / 2) + 0 - nGrid)): np.min(
            (nGrid, Center_X + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_X)),
        np.max((Center_Y - int(N_subgrid / 2), 0)) - np.max((0, Center_Y + int(N_subgrid / 2) + 0 - nGrid)): np.min(
            (nGrid, Center_Y + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_Y)),
        np.max((Center_Z - int(N_subgrid / 2), 0)) - np.max((0, Center_Z + int(N_subgrid / 2) + 0 - nGrid)): np.min(
            (nGrid, Center_Z + int(N_subgrid / 2) + 0)) + np.max((0, int(N_subgrid / 2) - Center_Z))] = Sub_Grid[:]

        if np.any(Sub_Grid[boundary] > 1) or np.any(np.isnan(Sub_Grid[boundary])):
            raise RuntimeError("Encountered invalid values in the grid during spreading.")

        if round(int(np.sum(Sub_Grid)) / int(np.sum(Sub_Grid_Initiale) + excess_ion_i)) != 1:
            raise RuntimeError("Spreading procedure did not conserve the photons. This should not happen.")

    # return Grid




def stacked_lyal_kernel(rr_al, lyal_array, LBox, nGrid, nGrid_min):
    """
    This function paints the lyal profile on a meshgrid whose size is the size where the lyal profile reaches zeros.
    Hence it is larger than LBox. It has a lower resolution than the Grid (nGrid_min = 64). We then chunk this large box into suboxes of sizes LBox and stack them.
    This ensures that despite a small LBox, we ensure full periodic boundary conditions and account for the wide spread of lyal profiles.
    rr_al : the comoving radius range
    lyal_array : the lyal profile (array)
    LBox,nGrid : the box size and grid rez of the current run.
    """
    # print(f"{rr_al.shape=}, {lyal_array.shape=}")
    profile_xal_HM = interp1d(rr_al, lyal_array, bounds_error=False, fill_value=0)  ##screening
    ind_lya_0 = np.min(np.where(lyal_array == 0))  ## indice where the lyman alpha profile gets to zero
    rr_al_max = rr_al[ind_lya_0]  ### max radius that we need to consider to fully include the lyman alpha profile
    box_extension = int(rr_al_max / (LBox / 2)) + 1

    # nGrid_min = 64
    if box_extension < 1:
        box_extension = 1

    elif box_extension % 2 == 0:
        box_extension += 1  ### this need to be even to make things work

    kernel_xal_HM = profile_to_3Dkernel(profile_xal_HM, box_extension * nGrid_min, box_extension * LBox)
    # kernel_xal_HM = profile_to_3Dkernel(profile_xal_HM, box_extension * nGrid_min, box_extension * LBox)
    # nGrid_extd = box_extension * nGrid_min
    # LBox_extd = box_extension * LBox  ## size and nbr of pix of the larger box

    stacked_xal_ker = np.zeros((nGrid_min, nGrid_min, nGrid_min))
    for ii in range(box_extension):  ## loop over the box_extension**3 subboxes and stack them
        for jj in range(box_extension):
            for kk in range(box_extension):
                stacked_xal_ker += kernel_xal_HM[
                    ii * nGrid_min:(ii + 1) * nGrid_min,
                    jj * nGrid_min:(jj + 1) * nGrid_min,
                    kk * nGrid_min:(kk + 1) * nGrid_min
                ]

    pix_lft = int(box_extension / 2) * nGrid_min  ### coordinate of the central subbox
    pix_rgth = (1 + int(box_extension / 2)) * nGrid_min
    ## remove the central box, to then add it later with full nGrid resolution
    stacked_xal_ker = stacked_xal_ker - kernel_xal_HM[pix_lft:pix_rgth, pix_lft:pix_rgth, pix_lft:pix_rgth]

    incr_rez = np.asarray(np.arange(0, nGrid) * nGrid_min / nGrid, int)  ## indices to then add

    kernel_xal_HM = profile_to_3Dkernel(profile_xal_HM, nGrid, LBox) + stacked_xal_ker[incr_rez, incr_rez, incr_rez]

    return kernel_xal_HM


def stacked_T_kernel(rr_T, T_array, LBox, nGrid, nGrid_min):
    """
    Same as stacked_lyal_kernel but for Temperature profiles.
    rr_T : the comoving radius range
    T_array : the Temp profile (array)
    LBox,nGrid : the box size and grid rez of the current run.
    """
    profile_T_HM = interp1d(rr_T, T_array, bounds_error=False, fill_value=0)  ##screening

    zero_K_indices = np.where(T_array < 1e-6)[0]
    if len(zero_K_indices) > 0:
        ind_T_0 = np.min(zero_K_indices)  ## indice where the T profile drops, xray haven't reached that scale
    else:
        ind_T_0 = -1  ## if T_array is always > 1e-6, we just take the whole profile...

    rr_T_max = rr_T[ind_T_0]  ### max radius that we need to consider to fully include the extended T profile
    box_extension = int(rr_T_max / (LBox / 2)) + 1

    # nGrid_min = 64
    if box_extension < 1:
        box_extension = 1

    elif box_extension % 2 == 0:
        box_extension += 1  ### this need to be even to make things work

    kernel_T_HM = profile_to_3Dkernel(profile_T_HM, box_extension * nGrid_min, box_extension * LBox)
    # nGrid_extd = box_extension * nGrid_min
    # LBox_extd = box_extension * LBox  ## size and nbr of pix of the larger box

    stacked_T_ker = np.zeros((nGrid_min, nGrid_min, nGrid_min))
    for ii in range(box_extension):  ## loop over the box_extension**3 subboxes and stack them
        for jj in range(box_extension):
            for kk in range(box_extension):
                stacked_T_ker += kernel_T_HM[ii * nGrid_min:(ii + 1) * nGrid_min, jj * nGrid_min:(jj + 1) * nGrid_min,
                                 kk * nGrid_min:(kk + 1) * nGrid_min]

    pix_lft = int(box_extension / 2) * nGrid_min  ### coordinate of the central subbox
    pix_rgth = (1 + int(box_extension / 2)) * nGrid_min
    ## remove the central box, to then add it later with full nGrid resolution
    stacked_T_ker = stacked_T_ker - kernel_T_HM[pix_lft:pix_rgth, pix_lft:pix_rgth, pix_lft:pix_rgth]

    incr_rez = np.asarray(np.arange(0, nGrid) * nGrid_min / nGrid, int)  ## indices to then add

    kernel_T_HM = profile_to_3Dkernel(profile_T_HM, nGrid, LBox) + stacked_T_ker[incr_rez, incr_rez, incr_rez]

    return kernel_T_HM
