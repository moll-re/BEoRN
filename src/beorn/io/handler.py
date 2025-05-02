from pathlib import Path
import logging
import pickle
from typing import TypeVar

from ..structs.parameters import Parameters
from ..structs.base_struct import BaseStruct

BaseStructDerived = TypeVar("BaseStructDerived", bound = BaseStruct)

class Handler:
    logger = logging.getLogger(__name__)
    def __init__(self, file_root: Path):
        self.file_root = file_root
        self.file_root.mkdir(exist_ok = True)
        self.logger.debug(f"Using persistence directory at {self.file_root}")


    def get_file_name(self, parameters: Parameters, object_type: type, **kwargs) -> str:
        prefix = f"{object_type.__name__}"
        kwargs_string = "_".join([f"{key}={value}" for key, value in kwargs.items()])
        file_name = f"{prefix}_{parameters.unique_hash()}_{kwargs_string}.pkl"
        return file_name
    

    def write_file(self, parameters: Parameters, obj: object, **kwargs) -> Path:
        # TODO replace any

        file_name = self.get_file_name(parameters, type(obj), **kwargs)
        file_path = self.file_root / file_name
        with (file_path).open("wb") as f:
            pickle.dump(obj, f)
            # TODO think about writing the (potentially large) numpy arrays as separate npy files
        
        self.logger.info(f"Wrote {type(obj).__name__} to persistence ({file_path})")
        return file_path


    # TODO: type hinting is off I think
    def load_file(self, parameters: Parameters, cls: BaseStructDerived, **kwargs) -> BaseStructDerived:
        """
        Loads the instance of the class from the persistence directory. This is a convenience method that replaces the need to call cls.read() directly.
        """
        return cls.read(directory=self.file_root, parameters=parameters, **kwargs)
