"""Global description of the 3d data computed over multiple redshifts."""
from dataclasses import dataclass, field, fields
from pathlib import Path
import h5py
import numpy as np
import logging
import tools21cm as t2c
logger = logging.getLogger(__name__)

from .base_struct import BaseStruct
from .snapshot_profiles import GridData
from .parameters import Parameters


@dataclass
class GridDataMultiZ(BaseStruct):
    """
    Collection of grid data over multiple redshifts. This is implemented such that an additional z dimension is added to each field of the GridData class.
    Appending a new redshift to this data automatically appends to the underlying hdf5 file.
    As such, once initialized (over a non-empty file)m this class has the SAME attributes as the GridData class, each with one additional axis at index 0.
    """

    def append(self, grid_data: GridData, parameters: Parameters, directory: Path):
        """
        Append a new redshift to the collection of grid data.
        """
        if not isinstance(grid_data, GridData):
            raise TypeError("grid_data must be an instance of GridData")

        self._append_to_hdf5(grid_data, parameters, directory)


    def _append_to_hdf5(self, grid_data: GridData, parameters: Parameters, directory: Path):
        """
        Append the grid data to the underlying HDF5 file.
        """
        path = self.get_file_path(directory = directory, parameters = parameters)
        with h5py.File(path, 'a') as hdf5_file:
            for field in fields(grid_data):
                key = field.name
                value = getattr(grid_data, key)

                if isinstance(value, (float, int, list)):
                    # Convert float to numpy array so that they can still be appended
                    value = np.array(value)

                if not isinstance(value, np.ndarray):
                    logger.debug(f"Skipping field {key} because type {type(value)} is not appendable.")
                    continue

                if key not in hdf5_file:
                    # Create a new dataset if it doesn't exist
                    hdf5_file.create_dataset(
                        key,
                        data = value[np.newaxis, ...],
                        maxshape = (None, *value.shape)
                        # explicitly set the maxshape to allow for appending
                        )
                else:
                    # Append to the existing dataset
                    dataset = hdf5_file[key]
                    dataset.resize((dataset.shape[0] + 1, *dataset.shape[1:]))
                    dataset[-1] = value


    def power_spectrum(self, quantity: np.ndarray, parameters: Parameters) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the power spectrum of the given quantity over all redshifts.
        """
        kbins = parameters.simulation.kbin
        # kbins = parameters.simulation.kbins
        box_dims = parameters.simulation.Lbox
        power_spectrum = np.zeros((len(self.z), parameters.simulation.kbin))
        # self.z becomes available when the data is loaded
        for i, z in enumerate(self.z):
            delta_quantity = quantity[i, ...] / np.mean(quantity[i, ...]) - 1
            power_spectrum[i, ...], bins = t2c.power_spectrum.power_spectrum_1d(delta_quantity, box_dims=box_dims, kbins=kbins)
        return power_spectrum, bins
