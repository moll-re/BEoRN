"""Global description of the 3d data computed over multiple redshifts."""
from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np
import logging
import tools21cm as t2c
logger = logging.getLogger(__name__)

from .base_struct import BaseStruct
from .coeval_cube import CoevalCube
from .base_quantities import GridBasePropertiesMixin
from .derived_quantities import GridDerivedPropertiesMixin
from .parameters import Parameters


@dataclass
class TemporalCube(BaseStruct, GridBasePropertiesMixin, GridDerivedPropertiesMixin):
    """
    Collection of grid data over multiple redshifts. This is implemented such that an additional z dimension is added to each field of the similar 'CoevalCube' class.
    Appending a new redshift to this data automatically appends to the underlying hdf5 file.
    As such, this class reuses all the grid data properties (which are implemented as base properties and derived properties in mixin classes). Only the z dimension is added here.
    """

    z: np.ndarray = None
    """Array of redshifts for which the grid data is available."""

    @classmethod
    def create_empty(cls, parameters: Parameters, directory: Path, snapshot_number: int = None, **kwargs) -> "TemporalCube":
        """
        Creates an empty HDF5 file with the given file path. If the file already exists, it is not overwritten. In order to create an object of the correct size (to be used by parallel implementations), the snapshot_number must be provided. If not provided, an empty object is created without pre-allocated arrays but this cannot be used in parallel implementations.
        """
        path = cls.get_file_path(directory, parameters, **kwargs)
        if snapshot_number is None:
            logger.warning("snapshot_number is not provided. Creating an empty GridDataMultiZ without pre-allocated arrays. This cannot be used in parallel implementations.")
            ret = cls(
                z = None,
                parameters = parameters,
                delta_b = None,
                Grid_Temp = None,
                Grid_xHII = None,
                Grid_xal = None,
            )
        else:
            # initialize the arrays with the correct shape so that when initializing the cls object, a hdf5 file is created with datasets of the correct shape
            grid_shape = (snapshot_number, parameters.simulation.Ncell, parameters.simulation.Ncell, parameters.simulation.Ncell)
            z_size = (snapshot_number,)
            ret = cls(
                z = np.zeros(z_size),
                parameters = parameters,
                delta_b = np.zeros(grid_shape),
                Grid_Temp = np.zeros(grid_shape),
                Grid_xHII = np.zeros(grid_shape),
                Grid_xal = np.zeros(grid_shape),
            )

        # set after initialization to avoid reading from that file on construction
        ret._file_path = path
        ret.write()
        return ret


    def append(self, grid_snapshot: CoevalCube, index: int) -> None:
        """
        Append a new GridData (for another redshift snapshot) to the collection of grid data.
        """
        if not isinstance(grid_snapshot, CoevalCube):
            raise TypeError("grid_snapshot must be an instance of GridData")

        if self._file_path is None:
            raise ValueError("File path is not set. Cannot append data.")

        # NB: this could in theory have been made mpi-compatible: the h5py context can handle calls from different mpi ranks
        # but: this requires h5py to compiled against an mpi-compatible h5 backend
        # Instead, we use the precompiled h5py and simply assign a "master" process that handles the writing part centrally. No special care needed.
        with h5py.File(self._file_path, 'a') as hdf5_file:
            for f in grid_snapshot._writable_fields():
                value = getattr(grid_snapshot, f)

                if isinstance(value, (float, int, list)):
                    # Convert float to numpy array so that they can still be appended
                    value = np.array(value)

                elif isinstance(value, h5py.Dataset):
                    # If the value is already a h5py.Dataset, we can directly append it
                    value = value[:]

                elif isinstance(value, Parameters):
                    if f not in hdf5_file.keys():
                        self._to_h5_field(hdf5_file, f, value)
                    else:
                        logger.debug(f"Not overriding {f} in {self._file_path.name}")
                    continue

                elif not isinstance(value, np.ndarray):
                    logger.debug(f"Not appending {f} to {self._file_path.name} because type {type(value)} is not appendable.")
                    continue

                dataset = hdf5_file[f]
                dataset[index, ...] = value


    def power_spectrum(self, quantity: np.ndarray, parameters: Parameters) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the power spectrum of the given quantity over all redshifts.
        """
        bin_number = parameters.simulation.kbins.size
        box_dims = parameters.simulation.Lbox
        power_spectrum = np.zeros((self.z.size, bin_number))

        delta_quantity = quantity[:] / np.mean(quantity, axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis] - 1

        for i, z in enumerate(self.z):
            power_spectrum[i, ...], bins = t2c.power_spectrum.power_spectrum_1d(delta_quantity[i, ...], box_dims=box_dims, kbins=bin_number)

        return power_spectrum, bins


    def redshift_of_reionization(self, ionization_fraction: float = 0.5) -> int:
        """
        Compute the redshift of reionization, defined as the redshift at which the volume-averaged ionization fraction crosses the given threshold.
        Parameters
        ----------
        ionization_fraction : float
            The ionization fraction threshold to define the redshift of reionization. Default is 0.5.
        Returns
        -------
        int
            The index of the redshift at which the volume-averaged ionization fraction crosses the threshold.
        """
        if self.Grid_xHII is None:
            raise ValueError("Grid_xHII is not available.")

        xHII_mean = np.mean(self.Grid_xHII, axis=(1, 2, 3))
        reionization_index = np.argmin(np.abs(xHII_mean - ionization_fraction))
        return reionization_index
