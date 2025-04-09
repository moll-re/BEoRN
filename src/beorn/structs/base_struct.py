"""Base data structure for derived data classes containing data required at different stages of the simulation."""
from pathlib import Path
from abc import ABC
from dataclasses import dataclass, is_dataclass, asdict, fields
import json
import h5py
import logging
logger = logging.getLogger(__name__)
from .parameters import Parameters, ParametersEncoder


# Don't allow this to class to be instantiated directly
@dataclass
class BaseStruct(ABC):
    """
    Base class for derived data classes containing data required at different stages of the simulation.
    """
    parameters: Parameters

    def write_file(self, file_path):
        """
        Write the content of the dataclass into an HDF5 file.
        """
        with h5py.File(file_path, 'w') as h5file:
            for field in fields(self):
                attr = field.name
                value = getattr(self, attr)
                self.to_h5_field(h5file, attr, value)
        logger.info(f"Data written to {file_path}")


    @classmethod
    def read_file(cls, file_path):
        """
        Reads the content of a specified HDF5 file and populate the dataclass.
        """
        with h5py.File(file_path, 'r') as h5file:
            data = {}
            for key in h5file.attrs:
                data[key] = h5file.attrs[key]
            for key in h5file.keys():
                data[key] = h5file[key][()]
        return cls(**data)


    def write(self, directory: Path, **kwargs):
        """
        Write the content of the dataclass into an HDF5 file.
        """
        file_name = self.get_file_name(self.parameters, type(self), **kwargs)
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / file_name
        self.write_file(file_path)
        return file_path


    @classmethod
    def read(cls, directory: Path, parameters: Parameters, **kwargs) -> object:
        """
        Reads the one file that is associated with the parameters and the object type. (e.g. `GridData` or `GridDataMultiZ`)
        """
        file_name = cls.get_file_name(parameters, cls, **kwargs)
        file_path = directory / file_name
        return cls.read_file(file_path)


    @classmethod
    def get_file_name(cls, parameters: Parameters, object_type: type, **kwargs) -> str:
        prefix = f"{object_type.__name__}"
        kwargs_string = "_".join([f"{key}={value}" for key, value in kwargs.items()])
        file_name = f"{prefix}_{parameters.unique_hash()}_{kwargs_string}.h5"
        return file_name


    def to_h5_field(self, file, attr: str, value: object):
        """
        Write the content of the dataclass into an HDF5 file.
        """
        if isinstance(value, Parameters):
            file.attrs[attr] = json.dumps(value, cls=ParametersEncoder)

        # simple types can be stored as attributes
        elif isinstance(value, (int, float, str)):
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


        else:
            raise TypeError(f"Unsupported data type for attribute '{attr}': {type(value)}")
