from pathlib import Path
import numpy as np

from ..structs.halo_catalog import HaloCatalog
from ..structs.parameters import Parameters


def load_halo_catalog(path: Path, parameters: Parameters) -> HaloCatalog:
    catalog = np.loadtxt(path)
    if catalog.shape == (0,):
        catalog = np.ndarray((0, 4))

    return HaloCatalog(
        # masses should be in Msun
        masses = catalog[:, 0] * parameters.cosmology.h,
        # shift to center the box
        positions = catalog[:, 1:] + parameters.simulation.Lbox / 2,
        parameters = parameters,
    )



def load_density_field(file: Path, LBox):
    """
    Parameters
    ----------
    file : String. Path to the pkdgrav density field
    LBox : Float, box size in Mpc/h
    nGrid : Float, number of grid pixels

    Returns
    ----------
    delta = rho_m/rho_mean-1
    3-D mesh grid. Size (nGrid,nGrid,nGrid)
    """
    dens = np.fromfile(file, dtype=np.float32)
    nGrid = round(dens.shape[0]**(1/3))
    pkd = dens.reshape(nGrid, nGrid, nGrid)
    pkd = pkd.T  ### take the transpose to match X_ion map coordinates
    # V_total = LBox ** 3
    # V_cell = (LBox / nGrid) ** 3
    # mass  = (pkd * rhoc0 * V_total).astype(np.float64)
    # rho_m = mass / V_cell
    # delta_b = (rho_m) / np.mean(rho_m, dtype=np.float64) - 1
    # since we divide by the mean, we can skip the mass calculation
    delta_b = pkd / np.mean(pkd, dtype=np.float64) - 1
    return delta_b
