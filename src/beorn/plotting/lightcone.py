import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, LogNorm, Normalize
import numpy as np
import logging
logger = logging.getLogger(__name__)


from ..structs.lightcone import Lightcone

# Edit this gradient at https://eltos.github.io/gradient/#0:78E4FF-20:006DC2-49:001250-50:000000-51:562500-71.9:CF8400-100:FFEC33
#Good sharp gradient for dTb : https://eltos.github.io/gradient/#0:78E4FF-20:006DC2-49:001250-50:000000-51:562500-71.9:CF5D00-100:FFEC33
# is this an alternative?
COLOR_GRADIENT = LinearSegmentedColormap.from_list(
    'my_gradient',
    (
        (0.000, (0.471, 0.894, 1.000)),
        (0.200, (0.000, 0.427, 0.761)),
        (0.490, (0.000, 0.071, 0.314)),
        (0.500, (0.000, 0.000, 0.000)),
        (0.510, (0.337, 0.145, 0.000)),
        (0.719, (0.812, 0.518, 0.000)),
        (1.000, (1.000, 0.925, 0.200))
    )
)


def define_norm_cbar_label(data: np.ndarray, quantity: str) -> tuple:
    if quantity == 'Tk':
        norm, cmap, label = LogNorm(), plt.get_cmap('plasma'), r'$T_{\mathrm{k}} [K]$'
    elif quantity == 'lyal':
        norm, cmap, label = LogNorm(vmin=np.min(data[data > 0]), vmax=np.max(data)), plt.get_cmap('cividis'), r'$x_{\mathrm{al}}$'
    elif quantity == 'matter':
        norm, cmap, label = Normalize(vmin=-1,vmax=5),plt.get_cmap('viridis'), r'$\delta_{\mathrm{m}}$'
    elif quantity == 'bubbles':
        norm, cmap, label = Normalize(vmin=0,vmax=1),plt.get_cmap('binary'), r'$x_{\mathrm{HII}}$'
    elif quantity == 'Grid_dTb':
        norm, cmap, label = TwoSlopeNorm(vmin=np.min(data), vcenter=0, vmax=max(np.max(data),0.001)),COLOR_GRADIENT,'$\overline{dT}_{\mathrm{b}}$ [mK]'
    else:
        raise ValueError(f"Unknown quantity '{quantity}' for lightcone plotting.")
        norm = LogNorm(vmin=np.min(data) + 1, vmax=np.max(data) + 1)
    return norm, cmap, label


def plot_lightcone(lightcone: Lightcone, ax: plt.Axes, description: str, slice_number: int = None) -> None:
    # TODO - xticks should be frequencies
    logger.debug(f"Lightcone range is {lightcone.redshifts.min()} to {lightcone.redshifts.max()}")

    ax.set_title(description)

    scale_factors = 1 / (lightcone.redshifts + 1)
    lbox = lightcone.parameters.simulation.Lbox

    if slice_number is None:
        slice_number = lightcone.data.shape[0] // 2

    norm, cmap, label = define_norm_cbar_label(lightcone.data, lightcone.quantity)

    # xi = np.tile(lightcone.redshifts, lightcone.data.shape[1])
    # # yi = np.tile(np.linspace(0, lbox, lightcone.data.shape[1]).reshape(-1, 1), (1, lightcone.redshifts.size))
    # yi = np.array([np.linspace(0,128,lightcone.data.shape[1]) for i in range(xi.shape[1])]).T
    # zj = (
    #     lightcone.data[slice_number,1:,1:] +
    #     lightcone.data[slice_number,1:,:-1] +
    #     lightcone.data[slice_number,:-1,1:] +
    #     lightcone.data[slice_number,:-1,:-1]
    # ) / 4



    xi = np.array([lightcone.redshifts for i in range(lightcone.data.shape[1])])
    yi = np.array([np.linspace(0,128,lightcone.data.shape[1]) for i in range(xi.shape[1])]).T
    zj = (lightcone.data[slice_number,1:,1:]+lightcone.data[slice_number,1:,:-1]+lightcone.data[slice_number,:-1,1:]+lightcone.data[slice_number,:-1,:-1])/4
    im = ax.pcolormesh(xi, yi, zj, cmap=cmap, norm=norm)
    # im = ax.imshow(np.sum(lightcone.data, axis=0), cmap=cmap, norm=norm, origin='lower')

    ax.set_xlabel('a(t)')
    ax.set_ylabel('L (Mpc)')
    # ax.tick_params()

    # Add a secondary x-axis for redshift
    ax2 = ax.twiny()
    redshift_ticks = np.linspace(lightcone.redshifts[0], lightcone.redshifts[-1], 8)
    ax2.set_xticks(redshift_ticks, labels=np.round(redshift_ticks, 2))
    ax2.set_xlabel("z")
    # ax2.tick_params()

    # Plot the colorbar directly
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label)
