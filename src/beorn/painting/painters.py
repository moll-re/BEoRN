import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from astropy.convolution import convolve_fft

from .helpers import profile_to_3Dkernel, stacked_lyal_kernel, stacked_T_kernel, CONVOLVE_FFT_KWARGS

def paint_ionization_profile(
    output_grid: np.ndarray,
    radial_grid,
    x_HII_profile,
    nGrid, LBox,
    z,
    halo_grid: np.ndarray
) -> None:
    # TODO - describe how this modifies the output_grid in place
    profile_xHII = interp1d(
        x = radial_grid * (1 + z),
        y = x_HII_profile,
        bounds_error = False,
        fill_value = (1, 0)
    )
    kernel_xHII = profile_to_3Dkernel(profile_xHII, nGrid, LBox)
    # self.logger.debug(f"kernel_xHII has {np.sum(np.isnan(kernel_xHII))} NaN values")

    if not np.any(kernel_xHII > 0):
        ### if the bubble volume is smaller than the grid size,we paint central cell with ion fraction value
        # kernel_xHII[int(nGrid / 2), int(nGrid / 2), int(nGrid / 2)] = np.trapz(x_HII_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (LBox / nGrid / (1 + z)) ** 3
        output_grid += halo_grid * trapezoid(x_HII_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (LBox / nGrid / (1 + z)) ** 3

    else:
        renorm = trapezoid(x_HII_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (LBox / (1 + z)) ** 3 / np.mean(kernel_xHII)
        # extra_ion = put_profiles_group(Pos_Halos_Grid[indices], kernel_xHII * 1e-7 / np.sum(kernel_xHII)) * np.sum(kernel_xHII) / 1e-7 * renorm
        output_grid += convolve_fft(
            array = halo_grid,
            kernel = kernel_xHII * 1e-7 / np.sum(kernel_xHII),
            **CONVOLVE_FFT_KWARGS
        ) * np.sum(kernel_xHII) / 1e-7 * renorm
        # bubble_volume = trapezoid(4 * np.pi * radial_grid ** 2 * x_HII_profile, radial_grid)
        # print('bubble volume is ', len(indices) * bubble_volume,'pMpc, grid volume is', np.sum(extra_ion)* (LBox /nGrid/ (1 + z)) ** 3 )
        # Grid_xHII_i += extra_ion


def paint_alpha_profile(
    output_grid: np.ndarray,
    r_lyal,
    x_alpha_prof,
    nGrid,
    LBox,
    minimum_grid_size_lyal,
    z,
    truncate,
    halo_grid: np.ndarray
) -> None:

    ### We use this stacked_kernel functions to impose periodic boundary conditions when the lyal or T profiles extend outside the box size. Very important for Lyman-a.
    if isinstance(truncate, float):
        # truncate below a certain radius
        x_alpha_prof[r_lyal * (1 + z) < truncate] = x_alpha_prof[r_lyal * (1 + z) < truncate][-1]

    kernel_xal = stacked_lyal_kernel(
        r_lyal * (1 + z),
        x_alpha_prof,
        LBox,
        nGrid,
        nGrid_min = minimum_grid_size_lyal
    )

    if np.any(kernel_xal > 0):
        renorm = trapezoid(x_alpha_prof * 4 * np.pi * r_lyal ** 2, r_lyal) / (LBox / (1 + z)) ** 3 / np.mean(kernel_xal)

        output_grid += convolve_fft(
            array = halo_grid,
            kernel = kernel_xal * 1e-7 / np.sum(kernel_xal),
            **CONVOLVE_FFT_KWARGS
        ) * renorm * np.sum(kernel_xal) / 1e-7
        # we do this trick to avoid error from the fft when np.sum(kernel) is too close to zero.


def paint_temperature_profile(
    output_grid: np.ndarray,
    radial_grid,
    Temp_profile,
    nGrid,
    LBox,
    minimum_grid_size_heat,
    z,
    truncate,
    halo_grid: np.ndarray
) -> None:

    # TODO - truncation should not be handled by the PAINT function
    if isinstance(truncate, float):
        # truncate below a certain radius
        Temp_profile[radial_grid * (1 + z) < truncate] = Temp_profile[radial_grid * (1 + z) < truncate][-1]

    kernel_T = stacked_T_kernel(
        radial_grid * (1 + z),
        Temp_profile,
        LBox,
        nGrid,
        nGrid_min = minimum_grid_size_heat
    )

    if np.any(kernel_T > 0):
        renorm = trapezoid(Temp_profile * 4 * np.pi * radial_grid ** 2, radial_grid) / (LBox / (1 + z)) ** 3 / np.mean(kernel_T)

        output_grid += convolve_fft(
            array = halo_grid,
            kernel = kernel_T * 1e-7 / np.sum(kernel_T),
            **CONVOLVE_FFT_KWARGS
        ) * np.sum(kernel_T) / 1e-7 * renorm
