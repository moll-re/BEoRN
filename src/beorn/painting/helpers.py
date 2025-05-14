"""Helpers for the painting module."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from scipy.interpolate import interp1d
import logging
logger = logging.getLogger(__name__)

from ..structs.parameters import Parameters


def profile_to_3Dkernel(profile: np.ndarray, nGrid: np.ndarray, LB: int) -> np.ndarray:
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
    x = np.linspace(-LB / 2, LB / 2, nGrid)  # y, z will be the same.
    rx, ry, rz = np.meshgrid(x, x, x, sparse=True)
    rgrid = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    kern = profile(rgrid)
    return kern





def spreading_excess_fast(parameters: Parameters, Grid_input, show_plot=False):
    """
    Last and fastest version of the function.
    Input : Grid_Storage, the cosmological mesh grid (X,X,X) with the ionized fractions, with overlap (pixels where x_ion>1). (X can be 256, 512 ..)
    A word regarding the elements of this function :
        - Binary_Grid : contains 1 where Grid_input>=1 and 0 elsewhere. This grid is used as input for scipy.measure.label. Format is (X,X,X).
        - Connected_regions : (X,X,X). Output of skimage.measure.label. Each pixel of it is labeled according to the ionized clump it belongs to.
        - x_ion_tot_i : total sum of ionizing fraction.
        - region_nbr, size_of_region : region label, and size of it . We use it to idenfity the very small regions (small_regions) with less than "pix_thresh" pixels. We treat them all together to speed up the process
        - Spread_Single :  spread the excess photons.
    """

    nGrid = len(Grid_input[0])
    Grid = np.copy(Grid_input)

    if parameters.simulation.thresh_pixel is None:
        # TODO - what are these balues???
        pix_thresh = 80 * (nGrid / 256) ** 3
    else:
        pix_thresh = parameters.simulation.thresh_pixel

    Binary_Grid = np.copy(Grid)
    Binary_Grid[np.where(Grid < 0.9999999)] = 0
    Binary_Grid[np.where(Grid >= 0.9999999)] = 1

    # The first region (i=0) is the still neutral IGM, in between the bubbles
    label_image = label(Binary_Grid)

    # Periodic boundary conditions for label_image
    # assign  same label to ionised regions that are connected through left/right, up/down, front/back box boundaries
    '''''''''
    t0_PBC = time.time()
    PBC_indices = np.where(label_image[0, :, :] * label_image[-1, :, :] > 0)  # neighbor ionised indices
    label_to_change, indice = np.unique(label_image[-1, PBC_indices[0], PBC_indices[1]], return_index=True)  # labels in the upper layer that should be changed to neighbor labels in the lower layer
    replacement_label = label_image[0, PBC_indices[0], PBC_indices[1]][indice]
    for il in range(len(label_to_change)):
        label_image[label_image == label_to_change[il]] = replacement_label[il]

    PBC_indices = np.where(label_image[:, 0, :] * label_image[:, -1, :] > 0)
    label_to_change, indice = np.unique(label_image[PBC_indices[0], -1, PBC_indices[1]], return_index=True)
    replacement_label = label_image[PBC_indices[0], 0, PBC_indices[1]][indice]
    for il in range(len(label_to_change)):
        label_image[label_image == label_to_change[il]] = replacement_label[il]

    PBC_indices = np.where(label_image[:, :, 0] * label_image[:, :, -1] > 0)
    label_to_change, indice = np.unique(label_image[PBC_indices[0], PBC_indices[1], -1], return_index=True)
    replacement_label = label_image[PBC_indices[0], PBC_indices[1], 0][indice]
    for il in range(len(label_to_change)):
        label_image[label_image == label_to_change[il]] = replacement_label[il]
    print('Imposing PBC on label_image took', print_time(time.time()-t0_PBC))
    '''''''''

    x_ion_tot_i = np.sum(Grid)
    logger.info(f'Current ionization fraction: {np.sum(Grid):.3f}')

    if x_ion_tot_i > Grid.size:
        logger.info('Universe is fully ionized.')
        # return a final grid with all pixels set to 1
        return np.ones_like(Grid_input)

    logger.debug(f'Universe not fully ionized : xHII is {x_ion_tot_i / Grid.size:.4e}.')

    region_nbr, size_of_region = np.unique(label_image, return_counts=True)
    logger.debug(f"Found {len(region_nbr)} connected regions.")
    label_max = np.max(label_image)

    small_regions = np.where(np.isin(label_image, region_nbr[np.where(size_of_region < pix_thresh)[0]]))
    # small_regions : Gridmesh indices gathering all the connected regions that have less than 10 pixels
    Small_regions_labels = region_nbr[np.where(size_of_region < pix_thresh)[0]]
    # labels of the small regions. Use this to exclude them from the for loop

    initial_excess = np.sum(Grid[small_regions] - 1)
    excess_ion = initial_excess

    logger.debug(f'there are {len(Small_regions_labels)} small connected regions with less than {pix_thresh} pixels. They contain a fraction {excess_ion / x_ion_tot_i:.4e} of the total ionisation fraction.')

    # Do the spreading for the small regions
    Grid = spread_single(parameters, Grid, small_regions)
    if np.any(Grid[small_regions] > 1):
        raise RuntimeError('small regions not correctly spread')

    all_regions_labels = np.array(range(1, label_max + 1))  # the remaining larges overlapping ionized regions
    large_regions_labels = all_regions_labels[np.where(np.isin(all_regions_labels, Small_regions_labels) == False)[0]]
    # indices of regions that have more than pix_thresh pixels

    # Then do the spreading individually for large regions
    for i, ir in enumerate(large_regions_labels):
        if show_plot and i % 100 == 0:
            # TODO - actually show the plot
            logger.debug(f'Applying spread_single on region {i}/{len(large_regions_labels)}')
        connected_indices = np.where(label_image == ir)
        Grid = spread_single(parameters, Grid, connected_indices)

    if np.any(Grid > 1):
        raise RuntimeError('Some grid pixels are still in excess.')

    X_Ion_Tot_f = np.sum(Grid)
    logger.debug(f'Final xion sum: {X_Ion_Tot_f:.3f}')
    if int(X_Ion_Tot_f) != int(x_ion_tot_i):
        raise RuntimeError('Spreading procedure did not conserve the photons. This should not happen.')

    return Grid




def spread_single(parameters: Parameters, Grid, connected_indices):
    """
    This spreads the excess ionizing photons for a given region.
    Input :
    - Grid : The meshgrid containing the ionizing fractions
    - Connected_indices : The indices of the ionized region from which you want to spread the overlaps. (excess_ion)
    - print_time : if it's not None, will print the time taken, along with the message contained in "print_time".

    Return : the same grid but with the excess ion fraction of the connected region spread around.

    Trick : we run distance_transform only for a sub-box centered on the connected region. This is particularly important for high resolution grids, when distance_transform_edt starts to take time (~s, but multilplied by the number of connected regions >1e4, starts taking time...)
            the size of the subbox is N_subgrid. It is called Sub_Grid.
    """

    nGrid = len(Grid[0])

    initial_excess = np.sum(Grid[connected_indices] - 1)
    Grid[connected_indices] = np.where(Grid[connected_indices] > 1, 1, Grid[connected_indices])
    excess_ion = initial_excess

    if initial_excess > 1e-8:
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

        if parameters.simulation.subgrid_approximation: 
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

    return Grid






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


