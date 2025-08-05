from dataclasses import dataclass
import numpy as np

from .base_struct import BaseStruct
from .derived_quantities import GridPropertiesMixin

@dataclass(slots = True)
class GridData(BaseStruct, GridPropertiesMixin):
    """
    Class representing the grid data for a single redshift snapshot.
    """

    z: float
    """Redshift of the snapshot."""

    delta_b: np.ndarray
    """Density field (delta_b) of the snapshot."""

    Grid_Temp: np.ndarray
    """Temperature field (T) of the snapshot."""

    Grid_xHII: np.ndarray
    """Ionization fraction (xHII) of the snapshot."""

    Grid_xal: np.ndarray
    """Lyman alphy fraction (xal) of the snapshot."""
