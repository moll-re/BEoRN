"""Base data structure for derived data classes containing data required at different stages of the simulation."""
from pathlib import Path
from abc import ABC
from dataclasses import dataclass, is_dataclass, asdict, fields
import h5py
import logging
logger = logging.getLogger(__name__)
from .parameters import Parameters


# kw_only allows use to specify an optional field even though subclasses have required fields
@dataclass(kw_only=True)
class BaseStruct(ABC):
    """
    Base class for derived data classes containing data required at different stages of the simulation.
    The implementation is such that "loading" this object creates attributes but keeps the actual data on on the disk to be loaded on demand. 
    """

    _file_path: Path = None

    @classmethod
    def get_file_path(cls, directory: Path, parameters: Parameters, **kwargs) -> str:
        """
        Returns the file path for the HDF5 file associated with this object. The file name is generated based on the class name, parameters, and any additional keyword arguments.
        """
        prefix = cls.__name__
        kwargs_string = "_".join([f"{key}={value}" for key, value in kwargs.items()])
        file_name = f"{prefix}_{parameters.unique_hash()}_{kwargs_string}.h5"
        return directory / file_name


    def __post_init__(self):
        """
        Dynamically add attributes to the class for type checking and code completion. All the available hdf5 datasets are now available as attributes
        """
        if self._file_path is not None:
            # Do not use a context manager here, because we want to keep the file open
            hdf5_file = h5py.File(self._file_path, 'r')
            for dataset_name in hdf5_file.keys():
                setattr(self, dataset_name, hdf5_file[dataset_name])

        logger.debug(f"Read data from {self._file_path}")
        


    def write(self, file_path: Path = None, directory: Path = None, parameters: Parameters = None, **kwargs):
        """
        Write the content of this dataclass into an HDF5 file. Can be called without any arguments to write to the default file path, or with a specific file path, or with additional keyword arguments to customize the file name.
        """
        if file_path and (directory or parameters or kwargs):
            raise ValueError("Either provide a file path or a directory and parameters, but not both.")
        if file_path is None:
            file_path = self.get_file_path(directory, parameters, **kwargs)
            file_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(file_path, 'w') as h5file:
            for field in fields(self):
                attr = field.name
                value = getattr(self, attr)
                self.to_h5_field(h5file, attr, value)

        logger.info(f"Data written to {file_path}")


    @classmethod
    def read(cls, file_path: Path = None, directory: Path = None, parameters: Parameters = None, **kwargs):
        """
        Reads the content of a specified HDF5 file and populate the dataclass. Can be called with a specific file path, or with a base directory and parameters to infer the file path.
        """
        if file_path and (directory or parameters or kwargs):
            raise ValueError("Either provide a file path or a directory and parameters, but not both.")
        if file_path is None:
            file_path = cls.get_file_path(directory, parameters, **kwargs)

        # initialize the dataclass with dummy values and the file path -> it will be populated in the __post_init__ method
        data = {f.name: None for f in fields(cls)}
        data['_file_path'] = file_path
        return cls(**data)




    def to_h5_field(self, file, attr: str, value: object):
        """
        Write the content of the dataclass into an HDF5 file.
        """
        # TODO this might fail with ipython auto-reload

        # simple types can be stored as attributes
        if isinstance(value, (int, float, str)):
            file.attrs[attr] = value

        # lists, tuples and arrays can be stored as datasets
        elif isinstance(value, (list, tuple)) or hasattr(value, 'shape'):
            file.create_dataset(attr, data=value)

        # finally dataclasses and dictionaries can be stored as groups and subgroups
        elif is_dataclass(value):
            data = asdict(value)
            self.to_h5_field(file, attr, data)

        elif isinstance(value, dict):
            sub_group = file.create_group(attr)
            for k, v in value.items():
                self.to_h5_field(sub_group, k, v)

        elif isinstance(value, Path):
            file.attrs[attr] = str(value)

        elif value is None:
            # skip None values
            pass

        else:
            raise TypeError(f"Unsupported data type for attribute '{attr}': {type(value)}")
