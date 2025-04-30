from dataclasses import dataclass
import numpy as np
from .base_struct import BaseStruct

@dataclass(slots = True)
class GridData(BaseStruct):
    """
    Class representing the grid data for a single redshift snapshot.
    """
    z: float
    delta_b: np.ndarray
    # temperature profiles
    Grid_Temp: np.ndarray
    Grid_dTb: np.ndarray
    Grid_dTb_no_reio: np.ndarray
    Grid_dTb_T_sat: np.ndarray
    Grid_dTb_RSD: np.ndarray
    # ionization profiles
    Grid_xHII: np.ndarray
    # Lyman alpha profiles
    Grid_xal: np.ndarray
    Grid_xtot: np.ndarray

    # TODO some of these should not need to be precomputed - they could be made available as properties
