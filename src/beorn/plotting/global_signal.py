import numpy as np
import matplotlib.pyplot as plt
import tools21cm as t2c

from ..structs import GridDataMultiZ, Parameters

def draw_dTb_signal(ax: plt.Axes, grid: GridDataMultiZ, label=None, color=None):
    """
    TODO
    """
    z_range = grid.z[:]
    # TODO - fail on nan values instead of ignoring them
    mean_dtb = np.nanmean(grid.Grid_dTb, axis=(1,2,3))
    ax.plot(z_range, mean_dtb, lw=2, alpha=0.8, ls='-', color=color, label=label)
    ax.set_xlabel('z')
    ax.set_ylabel('dTb [mK]')


def draw_Temp_signal(ax: plt.Axes, grid: GridDataMultiZ, label=None, color=None):
    """
    TODO
    """
    z_range = grid.z[:]
    # TODO - fail on nan values instead of ignoring them
    mean_tk = np.nanmean(grid.Grid_Temp, axis=(1,2,3))
    ax.plot(z_range, mean_tk, lw=2, alpha=0.8, ls='-', color=color, label=label)
    ax.semilogy([], [])
    ax.set_xlim(6, 20)
    ax.set_ylabel('$T_{k}$ [K]')


def draw_xHII_signal(ax: plt.Axes, grid: GridDataMultiZ, label=None, color=None):
    """
    TODO
    """
    z_range = grid.z[:]
    # TODO - fail on nan values instead of ignoring them
    mean_x_HII = np.nanmean(grid.Grid_xHII, axis=(1,2,3))
    ax.plot(z_range, mean_x_HII, lw=2, alpha=0.8, ls='-', color=color, label=label)
    ax.set_xlim(6.3, 22)
    ax.set_ylabel('$x_{\mathrm{HII}}$')
    ax.set_xlabel('z')


def draw_dTb_power_spectrum_of_z(ax: plt.Axes, grid: GridDataMultiZ, parameters: Parameters, label=None, color=None, k_index=1):
    """
    TODO
    """
    z_range = grid.z[:]
    # TODO - fail on nan values instead of ignoring them
    mean_dtb = np.nanmean(grid.Grid_dTb, axis=(1,2,3))
    ps, bins = grid.power_spectrum(grid.Grid_dTb, parameters)
    k = bins[k_index]
    ps_k = ps[..., k_index]
    ps_c = ps_k * k ** 3 * mean_dtb ** 2 / (2 * np.pi ** 2)
    ax.semilogy(z_range, ps_c, label=f"{label} ({k=:.2f})", color=color)
    ax.set_ylim(1e-1, 1e3)
    ax.set_ylabel(rf'$(\overline{{dT_b}} \Delta_{{tot}})^{2}(k_0, z)$ [mK$^2$]')
    ax.set_xlabel('z')


def draw_dTb_power_spectrum_of_k(ax: plt.Axes, grid: GridDataMultiZ, parameters: Parameters, z_index: int, label=None, color=None):
    z = grid.z[z_index]
    current_grid = grid.Grid_dTb[z_index, ...]
    # TODO - fail on nan values instead of ignoring them
    mean_dtb = np.nanmean(current_grid)

    delta_quantity = current_grid / mean_dtb - 1
    bin_number = parameters.simulation.kbins.size
    box_dims = parameters.simulation.Lbox

    # TODO - is this the correct quantity?
    ps, bins = t2c.power_spectrum.power_spectrum_1d(delta_quantity, box_dims=box_dims, kbins=bin_number)
    ps_c = ps * bins ** 3 * mean_dtb ** 2 / (2 * np.pi ** 2)

    ax.semilogy(bins, ps_c, lw=1, ls='-', label=f"{label} (z={z:.2f})", color=color)
    ax.set_ylim(1e-1, 1e3)
    ax.set_ylabel(rf'$(\overline{{dT_b}} \Delta_{{tot}})^{2}(k, z_0)$ [mK$^2$]')
    ax.set_xlabel('z')
