import numpy as np
import matplotlib.pyplot as plt

from ..structs import GridDataMultiZ, Parameters

def draw_dTb_signal(ax: plt.Axes, grid: GridDataMultiZ, label=None, color=None):
    """
    TODO
    """
    z_range = grid.z[:]
    # TODO - fail on nan values instead of ignoring them
    mean_dtb = np.nanmean(grid.Grid_dTb, axis=(1,2,3))
    ax.plot(z_range, mean_dtb, lw=2, alpha=0.8, ls='-', color=color, label=label)
    ax.set_xlabel('z', fontsize=14)
    ax.set_ylabel('dTb [mK]', fontsize=14)

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
    ax.set_ylabel('$T_{k}$ [K]', fontsize=14)

def draw_xHII_signal(ax: plt.Axes, grid: GridDataMultiZ, label=None, color=None):
    """
    TODO
    """
    z_range = grid.z[:]
    # TODO - fail on nan values instead of ignoring them
    mean_x_HII = np.nanmean(grid.Grid_xHII, axis=(1,2,3))
    ax.plot(z_range, mean_x_HII, lw=2, alpha=0.8, ls='-', color=color, label=label)
    ax.set_xlim(6.3, 22)
    ax.set_ylabel('$x_{\mathrm{HII}}$', fontsize=14)
    ax.set_xlabel('z ', fontsize=14)

def draw_power_spectrum_of_z(ax: plt.Axes, grid: GridDataMultiZ, parameters: Parameters, label=None, color=None):
    """
    TODO
    """
    z_range = grid.z[:]
    # TODO - fail on nan values instead of ignoring them
    mean_dtb = np.nanmean(grid.Grid_dTb, axis=(1,2,3))
    ind_k = 1
    ps, bins = grid.power_spectrum(grid.Grid_dTb, parameters)
    k = bins[ind_k]
    ps_k = ps[..., ind_k]
    ps_c = ps_k * k ** 3 * mean_dtb ** 2 / (2 * np.pi ** 2)
    ax.semilogy(z_range, ps_c, lw=1, ls='-', label=label, color=color)
    ax.set_ylim(1e-1, 1e3)
    ax.set_ylabel('$\Delta_{21}^{2}(k,z)$ [mK]$^{2}$ ', fontsize=14)
    ax.set_xlabel('z ', fontsize=14)
