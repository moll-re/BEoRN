import numpy as np

from . import twentyone_cm_fast
from . import pkdgrav
from . import thesan

from ..structs.halo_catalog import HaloCatalog
from ..structs.parameters import Parameters



def load_halo_catalog(parameters: Parameters, redshift_index: int) -> HaloCatalog:
    """
    Load halo catalog from the specified path based on the input type in parameters.

    Parameters
    ----------
    path : str or Path
        Path to the halo catalog file.
    parameters : Parameters
        Parameters object containing cosmological and simulation parameters.

    Returns
    -------
    HaloCatalog
        Halo catalog object.
    """
    path = parameters.simulation.halo_catalogs[redshift_index]

    if parameters.simulation.input_type == "21cmFAST":
        return twentyone_cm_fast.load_halo_catalog(path, parameters)
    elif parameters.simulation.input_type == "pkdgrav":
        return pkdgrav.load_halo_catalog(path, parameters)
    elif parameters.simulation.input_type == "thesan":
        return thesan.load_halo_catalog(path, redshift_index, parameters)
    else:
        raise ValueError(f"Unknown halo catalog type: {parameters.simulation.input_type}. Supported types are: 21cmFAST, pkdgrav, thesan.")


def load_density_field(parameters: Parameters, redshift_index: int) -> np.ndarray:
    """
    Load the density field based on the index and parameters.

    Parameters
    ----------
    index : int
        Index of the density field to load.
    parameters : Parameters
        Parameters object containing cosmological and simulation parameters.

    Returns
    -------
    np.ndarray
        3D meshgrid of delta_b = rho/mean_rho - 1, the baryonic density contrast field.
    """
    LBox = parameters.simulation.Lbox
    nGrid = parameters.simulation.Ncell

    if parameters.simulation.density_fields is None:
        raise ValueError('No density fields provided.')

    field_path = parameters.simulation.density_fields[redshift_index]

    if parameters.simulation.input_type == 'pkdgrav':
        return pkdgrav.load_density_field(field_path, LBox)
    elif parameters.simulation.input_type == '21cmFAST':
        return twentyone_cm_fast.load_density_field(field_path, LBox)
    elif parameters.simulation.input_type == 'thesan':
        return thesan.load_density_field(field_path, parameters)
    else:
        raise TypeError('param.sim.dens_field_type should be either 21cmFAST, pkdgrav, or thesan.')
