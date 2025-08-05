"""Global description of the 3d data computed over multiple redshifts."""
from dataclasses import dataclass, fields
from pathlib import Path
import h5py
import numpy as np
import logging
import tools21cm as t2c
logger = logging.getLogger(__name__)

from .base_struct import BaseStruct
from .snapshot_profiles import GridData
from .derived_quantities import GridPropertiesMixin
from .parameters import Parameters


@dataclass
class GridDataMultiZ(BaseStruct, GridPropertiesMixin):
    """
    Collection of grid data over multiple redshifts. This is implemented such that an additional z dimension is added to each field of the GridData class.
    Appending a new redshift to this data automatically appends to the underlying hdf5 file.
    As such, once initialized (over a non-empty file)m this class has the SAME attributes as the GridData class, each with one additional axis at index 0.
    """

    def create(self, directory: Path, **kwargs) -> Path:
        """
        Creates an empty HDF5 file with the given file path. If the file already exists, it is not overwritten.
        """
        path = self.get_file_path(directory, self.parameters, **kwargs)
        # Create the file if it does not exist and raise an error if it does
        path.touch()
        self._file_path = path

        return path


    def append(self, grid_data: GridData) -> None:
        """
        Append a new GridData (for another redshift snapshot) to the collection of grid data.
        """
        if not isinstance(grid_data, GridData):
            raise TypeError("grid_data must be an instance of GridData")

        if self._file_path is None:
            raise ValueError("File path is not set. Cannot append data.")

        with h5py.File(self._file_path, 'a') as hdf5_file:
            for f in grid_data._writable_fields():
                value = getattr(grid_data, f)

                if isinstance(value, (float, int, list)):
                    # Convert float to numpy array so that they can still be appended
                    value = np.array(value)

                if isinstance(value, h5py.Dataset):
                    # If the value is already a h5py.Dataset, we can directly append it
                    value = value[:]

                if not isinstance(value, np.ndarray):
                    logger.debug(f"Not appending {f} to {self._file_path.name} because type {type(value)} is not appendable.")
                    continue

                if f not in hdf5_file:
                    # Create a new dataset if it doesn't exist
                    hdf5_file.create_dataset(
                        f,
                        data = value[np.newaxis, ...],
                        maxshape = (None, *value.shape)
                        # explicitly set the maxshape to allow for appending
                        )
                else:
                    # Append to the existing dataset
                    dataset = hdf5_file[f]
                    dataset.resize((dataset.shape[0] + 1, *dataset.shape[1:]))
                    dataset[-1] = value


    def power_spectrum(self, quantity: np.ndarray, parameters: Parameters) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the power spectrum of the given quantity over all redshifts.
        """
        bin_number = parameters.simulation.kbins.size
        box_dims = parameters.simulation.Lbox
        power_spectrum = np.zeros((self.z.size, bin_number))
        # self.z becomes available when the data is loaded
        for i, z in enumerate(self.z):
            delta_quantity = quantity[i, ...] / np.nanmean(quantity[i, ...]) - 1
            power_spectrum[i, ...], bins = t2c.power_spectrum.power_spectrum_1d(delta_quantity, box_dims=box_dims, kbins=bin_number)
        return power_spectrum, bins
