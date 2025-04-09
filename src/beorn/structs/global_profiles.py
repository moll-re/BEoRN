"""Global description of the 3d data computed over multiple redshifts."""
from dataclasses import dataclass, fields
from pathlib import Path
import h5py
import numpy as np
import logging
logger = logging.getLogger(__name__)

from .base_struct import BaseStruct
from .snapshot_profiles import GridData



@dataclass
class GridDataMultiZ(BaseStruct):
    """
    Collection of grid data over multiple redshifts. This is smartly implemented to add an additional z dimension to the fields defined in the GridData class.
    Appending a new redshift to this data automatically appends to the underlying hdf5 file.
    """
    file_root: Path

    # Make all the hdf5 datasets available as attributes. Essentially this reuses all the fields of the GridData class.
    # def __getattr__(self, name: str) -> h5py.Dataset:
    #     """
    #     Dynamically access datasets in the HDF5 file as attributes.
    #     """
    #     hdf5_file = h5py.File(self.file_path, 'r')
    #     if name in hdf5_file:
    #         return hdf5_file[name][:]
    #     else:
    #         hdf5_file.close()
    #         raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __post_init__(self):
        """
        Dynamically add attributes to the class for type checking and code completion.
        """
        if not self.file_path.exists():
            # Create the file if it doesn't exist
            f = h5py.File(self.file_path, 'w')
            f.close()
            logger.debug(f"Created new file {self.file_path}")

        hdf5_file = h5py.File(self.file_path, 'r')
        for dataset_name in hdf5_file.keys():
            setattr(self, dataset_name, hdf5_file[dataset_name])



    @property
    def file_path(self) -> Path:
        """
        Property to access the file path.
        """
        return self.file_root / f"global_grid_data_{self.parameters.unique_hash()}.h5"


    def append(self, grid_data: GridData, z: float):
        """
        Append a new redshift to the collection of grid data.
        """
        if not isinstance(grid_data, GridData):
            raise TypeError("grid_data must be an instance of GridData")
        # self.grid_data.append(grid_data)
        self._append_to_hdf5(grid_data)
        # TODO: also add the z value


    def _append_to_hdf5(self, grid_data: GridData):
        """
        Append the grid data to the underlying HDF5 file.
        """
        with h5py.File(self.file_path, 'a') as hdf5_file:
            for field in fields(grid_data):
                key = field.name
                value = getattr(grid_data, key)
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
