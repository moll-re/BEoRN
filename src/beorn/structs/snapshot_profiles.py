from dataclasses import dataclass
import numpy as np

from .base_struct import BaseStruct

@dataclass(slots = True)
class GridData(BaseStruct):
    """
    Class representing the grid data for a single redshift snapshot.
    """

    z: float
    """Redshift of the snapshot."""

    delta_b: np.ndarray
    """Density field (delta_b) of the snapshot."""

    Grid_Temp: np.ndarray
    """Temperature field (T) of the snapshot."""

    Grid_dTb: np.ndarray
    """Brightness temperature field (dTb) of the snapshot."""

    Grid_dTb_no_reio: np.ndarray
    """Brightness temperature field (dTb) of the snapshot without reionization."""

    Grid_dTb_T_sat: np.ndarray
    """Brightness temperature field (dTb) of the snapshot with T_sat."""

    Grid_dTb_RSD: np.ndarray
    """Brightness temperature field (dTb) of the snapshot with RSD."""

    Grid_xHII: np.ndarray
    """Ionization fraction (xHII) of the snapshot."""

    Grid_xal: np.ndarray
    """Lyman alphy fraction (xal) of the snapshot."""

    Grid_xtot: np.ndarray
    """TODO"""

    # TODO some of these should not need to be precomputed - they could be made available as properties
