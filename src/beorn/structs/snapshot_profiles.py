from dataclasses import dataclass
import numpy as np

from .base_struct import BaseStruct
from .base_quantities import GridBasePropertiesMixin
from .derived_quantities import GridDerivedPropertiesMixin

@dataclass(slots = True)
class GridData(BaseStruct, GridBasePropertiesMixin, GridDerivedPropertiesMixin):
    """
    Class representing the grid data for a single redshift snapshot.
    """

    z: float
    """Redshift of the snapshot."""
