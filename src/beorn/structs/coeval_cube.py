from dataclasses import dataclass

from .base_struct import BaseStruct
from .base_quantities import GridBasePropertiesMixin
from .derived_quantities import GridDerivedPropertiesMixin

@dataclass(slots = True)
class CoevalCube(BaseStruct, GridBasePropertiesMixin, GridDerivedPropertiesMixin):
    """
    Grid data for a single redshift snapshot. All grid data properties are implemented as base properties and derived properties in mixin classes. They contain the fundamental grids computed during the painting of the simulation as well as derived quantities computed from them.
    """

    z: float
    """Redshift of the snapshot."""


    def to_arrays(self) -> None:
        """Convert all grid data properties to numpy arrays. When being loaded from an HDF5 file, they are h5py datasets by default and cannot be communicated between MPI processes."""
        for field in self.__dataclass_fields__.values():
            value = getattr(self, field.name)
